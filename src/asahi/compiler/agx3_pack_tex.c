/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 texture sampling, texel/image loads, storage image writes, coordinate
 * staging, and cube scratch allocation.
 */

#include "agx3_pack_internal.h"
#include "agx3_copy_core.h"
#include "agx3_pixel_core.h"
#include "agx3_stage_core.h"
#include "agx3_tex_core.h"
#include "util/half_float.h"

#include <limits.h>

/* Whether agx3_emit_texture_sample() publishes this sample's CEM face
 * coordinates with the sample's result slot rather than a staging slot: a cube
 * sample whose texture and sampler operands are both non-register. */
bool
agx3_tex_cube_groups_result_slot(const agx_instr *I)
{
   if (I->op != AGX_OPCODE_TEXTURE_SAMPLE ||
       (I->dim != AGX_DIM_CUBE && I->dim != AGX_DIM_CUBE_ARRAY) ||
       I->nr_srcs < 5)
      return false;
   return I->src[3].type != AGX_INDEX_REGISTER &&
          I->src[4].type != AGX_INDEX_REGISTER;
}

/* Coordinate word c of a texel fetch or sample: that word of its coordinate
 * vector, or its own scalar source when it takes split coordinates. */
static agx_index
agx3_tex_coord(const agx_instr *I, unsigned c)
{
   if (I->agx3_split_coords)
      return c ? I->src[c + 5] : I->src[0];
   agx_index coord = I->src[0];
   coord.value += 2 * c;
   coord.channels_m1 = 0;
   return coord;
}

/* Whether staging coordinate word c is the last read of its register by this
 * fetch or sample. Split components may name one register twice; only the
 * final read may discard it. Words are read in order, x, y, then the third
 * word. */
static bool
agx3_tex_coord_discard(const agx_instr *I, unsigned c, unsigned count)
{
   const agx_index coord = agx3_tex_coord(I, c);
   if (!coord.discard)
      return false;
   for (unsigned later = c + 1; later < count; ++later) {
      if (agx3_tex_coord(I, later).value == coord.value)
         return false;
   }
   return true;
}

static unsigned
agx3_fetch_nr_srcs(const agx_instr *I)
{
   return I->agx3_split_coords ? 8 : 6;
}

/* Consecutive raw 2D reads can consume the same immutable coordinate pair.
 * Do not cross even a single intervening instruction: it may overwrite the
 * GPR pair or publish another sampler-private operand. */
static bool
agx3_fetch_coordinates_match(const agx_instr *a, const agx_instr *b)
{
   if (!b || a->op != AGX_OPCODE_TEXTURE_LOAD || b->op != a->op ||
       a->nr_srcs != agx3_fetch_nr_srcs(a) ||
       b->nr_srcs != agx3_fetch_nr_srcs(b) || a->nr_dests != 1 ||
       a->dim != AGX_DIM_2D || b->dim != AGX_DIM_2D ||
       a->lod_mode != AGX_LOD_MODE_LOD_MIN || b->lod_mode != a->lod_mode ||
       a->src[1].type != AGX_INDEX_IMMEDIATE ||
       !agx3_tex_imm(b->src[1], a->src[1].value) ||
       a->offset || b->offset || a->shadow || b->shadow ||
       a->sparse || b->sparse || a->query_lod || b->query_lod ||
       (!a->agx3_split_coords && agx_channels(a->src[0]) != 2) ||
       (!b->agx3_split_coords && agx_channels(b->src[0]) != 2))
      return false;

   /* Matching physical register numbers do not imply matching values if the
    * first fetch replaces any part of the second fetch's coordinate input. */
   const unsigned dst = a->dest[0].value;
   for (unsigned c = 0; c < 2; ++c) {
      const agx_index x = agx3_tex_coord(a, c), y = agx3_tex_coord(b, c);
      if (!agx3_tex_plain_reg(x) || !agx3_tex_plain_reg(y) ||
          x.size != AGX_SIZE_32 || y.size != AGX_SIZE_32 ||
          x.value != y.value)
         return false;
      if (agx3_tex_plain_reg(a->dest[0]) &&
          dst + agx_index_size_16(a->dest[0]) > x.value && dst < x.value + 2)
         return false;
   }
   return true;
}

/* Split coordinate word c: x and y use the wide scalar COPY, which reads the
 * whole register file; a sample index or layer keeps PACK_SMPIR's seven-bit
 * source. */
static bool
agx3_split_coord_ok(agx_index coord, unsigned c)
{
   return agx3_tex_plain_reg(coord) && coord.size == AGX_SIZE_32 &&
          agx_channels(coord) == 1 && !(coord.value & 1) &&
          coord.value + 2 <= (c == 2 ? 128 : 256);
}

/* The coordinate operands of a texel fetch: a vector of `count` words, or,
 * for a staged M5 2D or multisampled read, `count` independent scalars. */
static bool
agx3_fetch_coords_ok(const agx_instr *I, bool m5, bool staged, unsigned count)
{
   if (I->agx3_split_coords) {
      if (!m5 || !staged ||
          (I->dim != AGX_DIM_2D && I->dim != AGX_DIM_2D_MS &&
           I->dim != AGX_DIM_2D_ARRAY) ||
          (count == 2) != agx_is_null(I->src[7]))
         return false;
      for (unsigned c = 0; c < count; ++c) {
         if (!agx3_split_coord_ok(agx3_tex_coord(I, c), c))
            return false;
      }
      return true;
   }
   return agx3_tex_plain_reg(I->src[0]) && I->src[0].size == AGX_SIZE_32 &&
          agx_channels(I->src[0]) == count &&
          (staged ? !(I->src[0].value & 1) &&
                       I->src[0].value + 2 * count <= 128
                  : !(I->src[0].value & 3) && I->src[0].value <= 60);
}

/* Native386's G18 multi-read witnesses keep all outstanding coordinate
 * pairs disjoint. Slot numbers synchronize publications; they do not select
 * a separate coordinate record without explicit offsets in COPY and SMP. */
static bool
agx3_g18g_plain_fetch_block(const agx_instr *I)
{
   return I->op == AGX_OPCODE_TEXTURE_LOAD && I->dim == AGX_DIM_2D &&
          I->nr_srcs == 6 && I->nr_dests == 1 &&
          I->src[3].type == AGX_INDEX_IMMEDIATE &&
          (I->lod_mode == AGX_LOD_MODE_AUTO_LOD ||
           I->lod_mode == AGX_LOD_MODE_LOD_MIN) &&
          agx3_tex_imm(I->src[1], 0) &&
          I->dest[0].size == AGX_SIZE_32 &&
          !I->offset && !I->shadow && !I->query_lod && !I->sparse &&
          I->gather == AGX_GATHER_NONE;
}

static bool
agx3_emit_texture_fetch_g17g(struct util_dynarray *emission,
                        enum agx_isa_encoding enc, const agx_instr *I,
                        uint8_t slot, uint16_t wait_mask, bool final,
                        bool stage_coordinates, bool separate_write_coords,
                        bool image_read_bank0, bool g18g,
                        uint16_t ms_array_scratch, bool pixel_entry,
                        uint32_t *pixel_entry_offset, const char **why)
{
   const bool immediate = I->nr_srcs == 6 &&
      I->src[3].type == AGX_INDEX_IMMEDIATE;
   const bool immediate_lod = I->lod_mode == AGX_LOD_MODE_LOD_MIN &&
      I->src[1].type == AGX_INDEX_IMMEDIATE && !I->src[1].memory &&
      !I->src[1].abs && !I->src[1].neg && I->src[1].value <= 31;
   /* G17G's integer level publication reads one half: Apple's uint level
    * is clamped to 16 bits first and a ushort level publishes as-is
    * (gpu/m5-fetch-lod-varying-reference.metal), so Mesa's 16-bit fetch
    * level stages directly. */
   const bool register_lod =
      I->lod_mode == AGX_LOD_MODE_LOD_MIN &&
      agx3_tex_plain_reg(I->src[1]) &&
      (I->src[1].size == AGX_SIZE_32 ||
       (enc == AGX_ISA_ENCODING_AGX3C && I->src[1].size == AGX_SIZE_16)) &&
      agx_channels(I->src[1]) == 1;
   const bool ms_2d = I->dim == AGX_DIM_2D_MS;
   const bool ms_array = I->dim == AGX_DIM_2D_MS_ARRAY;
   const bool array = I->dim == AGX_DIM_2D_ARRAY;
   const bool three_d = I->dim == AGX_DIM_3D;
   /* G18G stages an array fetch's layer: SMP's direct f31/f37 = 3 layer
    * names a uniform half, and
    * for a per-lane layer Apple's G18G compiler stages it with PACK_SMPIR in
    * every read and sample (agx3_tex_stage_array_layer_g18g), a shape the
    * G18G transcoder carries while it refuses the direct f77 = 1 form. So a
    * G18G array fetch takes G18G's staged block: X at 4, Y at 0, the layer
    * at 2 -- the layer word two halves above Y, as in every G18G witness,
    * including reads on slots 2 and 3 (m6-fetch-array-two). A runtime integer
    * level shares that word with the layer through PACK_SMPIR (native541). */
   const bool g18g_array = g18g && array && stage_coordinates;
   const bool g18g_ms = g18g && (ms_2d || ms_array) && stage_coordinates;
   const bool m5_indexed = enc == AGX_ISA_ENCODING_AGX3C && !immediate;
   const bool bindless_image = I->op == AGX_OPCODE_IMAGE_LOAD && !immediate &&
      agx3_tex_plain_reg(I->src[3]) && I->src[3].size == AGX_SIZE_64 &&
      agx_channels(I->src[3]) == 1 && !(I->src[3].value & 1) &&
      I->src[3].value <= 252 &&
      !(I->src[3].cache && (I->src[3].discard || I->src[3].kill)) &&
      agx3_tex_plain_uniform64(I->src[2]);
   /* Gallium rebinding supplies the zero-extended native table index and
    * sampler zero; the old descriptor byte offset is not admitted here. */
   const bool native_indexed_fetch = I->op == AGX_OPCODE_TEXTURE_LOAD &&
      !immediate && I->src[3].size == AGX_SIZE_64 &&
      (agx3_tex_plain_reg(I->src[3]) ||
       agx3_tex_plain_uniform64(I->src[3])) &&
      agx3_tex_imm(I->src[2], 0);
   const bool g18g_indexed_scalar = g18g &&
      (bindless_image || native_indexed_fetch) &&
      (I->dim == AGX_DIM_2D || array || three_d || ms_2d || ms_array) &&
      I->mask == 1 &&
      (register_lod ||
       ((I->lod_mode == AGX_LOD_MODE_AUTO_LOD || immediate_lod) &&
        agx3_tex_imm(I->src[1], 0)));
   /* Apple's G17P lane-indexed image read publishes coordinates and uint LOD
    * on staging slot 2, then issues the texture result on slot 1. Keep those
    * two independent scoreboard domains; reusing slot 1 for both is not a
    * native dependency shape.
    *
    * G17G does the opposite, and it matters: its staging bank is per slot,
    * so publishing the level on slot 2 while the sample runs on slot 1
    * leaves the sample reading an untouched block -- every run-time-level
    * texel fetch came back zero, whatever the texture held. Apple's own
    * G17G fragments (gpu/m5-fetch-lod-reference.metal and
    * m5-fetch-array-lod-reference.metal) publish the level on the sample's
    * own slot and wait for it there. */
   const uint8_t staging_slot =
      ms_array && !g18g ? 1
         : (register_lod && enc == AGX_ISA_ENCODING_AGX3B) ? 2 : slot;
   /* One named term per condition, as in agx3_emit_texture_sample: a fetch
    * that will not encode says which part of its shape has no witness, so a
    * refusal in a CTS log points straight at the work. */
   const char *outside = NULL;
#define M5_FETCH_OUTSIDE(cond, reason)                                         \
   do {                                                                        \
      if (!outside && (cond))                                                   \
         outside = (reason);                                                    \
   } while (0)

   M5_FETCH_OUTSIDE(I->nr_dests != 1 || I->nr_srcs != 6, "operand count");
   M5_FETCH_OUTSIDE(g18g && array && !stage_coordinates,
                    "G18G array fetch without staged coordinates");
   /* The staged block is keyed by the staging slot like every G17G fetch's
    * (the halves below are block-relative; the allocation reserves the
    * block, see block_base where the fetch is packed), and Apple's G18G
    * reads on slots 2 and 3 keep the same rule: the layer word sits two
    * halves above SMP's Y (m6-fetch-array-two). A runtime level shares that
    * word with the layer in the integer PACK_SMPIR form (native541). */
   /* A per-lane level shares the array block's layer word, packed with the
    * layer in one PACK_SMPIR (agx3_tex_stage_array_typed_lod_g18g, the
    * integer form Apple's G18G read(coord, layer, lod) emits). */
   /* Native541 retains the Y/layer record at halves0/2 for array reads in
    * read/write programs, including indexed images. G18 writes and vertex
    * outputs have their own allocator ranges; the final extent check proves
    * these do not overlap this read record. Fixed records are retired below
    * before reuse, independent of the scoreboard slot selected for a read. */
   M5_FETCH_OUTSIDE(!agx3_tex_plain_reg(I->dest[0]) ||
                       (I->dest[0].size != AGX_SIZE_16 &&
                        I->dest[0].size != AGX_SIZE_32) ||
                       agx_channels(I->dest[0]) != 4 ||
                       I->dest[0].value >= 128 ||
                       (I->dest[0].size == AGX_SIZE_32 &&
                        (I->dest[0].value & 1)),
                    "result register shape");
   M5_FETCH_OUTSIDE(!agx3_tex_plain_reg(I->src[0]) ||
                       I->src[0].size != AGX_SIZE_32 ||
                       agx_channels(I->src[0]) !=
                          ((I->dim == AGX_DIM_2D_ARRAY || ms_array || ms_2d ||
                            three_d)
                              ? 3
                              : 2),
                    "coordinate register shape");
   M5_FETCH_OUTSIDE(stage_coordinates
                       ? ((I->src[0].value & 1) ||
                          I->src[0].value + 2 * agx_channels(I->src[0]) > 128)
                       : ((I->src[0].value & 3) || I->src[0].value > 60),
                    "coordinate register alignment");
   M5_FETCH_OUTSIDE((ms_array || ms_2d)
                       ? !agx3_tex_imm(I->src[1], 0)
                       : (!(I->lod_mode == AGX_LOD_MODE_AUTO_LOD &&
                            agx3_tex_imm(I->src[1], 0)) &&
                          !immediate_lod && !register_lod),
                    "level operand shape");
   M5_FETCH_OUTSIDE((ms_array || ms_2d) &&
                       I->lod_mode != AGX_LOD_MODE_AUTO_LOD,
                    "multisample level mode");
   M5_FETCH_OUTSIDE(register_lod && !stage_coordinates,
                    "runtime level without staged coordinates");
   M5_FETCH_OUTSIDE(!(agx3_tex_imm(I->src[2], 0) ||
                      ((ms_array || ms_2d) &&
                       agx3_tex_plain_uniform64(I->src[2])) ||
                      bindless_image),
                    "sampler operand shape");
   M5_FETCH_OUTSIDE(immediate
                       ? I->src[3].value > 15
                       : ((enc != AGX_ISA_ENCODING_AGX3B && !m5_indexed) ||
                          (I->src[3].type != AGX_INDEX_REGISTER &&
                           I->src[3].type != AGX_INDEX_UNIFORM) ||
                          (I->src[3].size != AGX_SIZE_64 &&
                           !((ms_array || ms_2d) &&
                             I->src[3].size == AGX_SIZE_32)) ||
                          agx_channels(I->src[3]) != 1 ||
                          (I->src[3].value & 1) || I->src[3].value > 252),
                    "texture operand shape");
   M5_FETCH_OUTSIDE(I->src[3].memory || I->src[3].abs || I->src[3].neg,
                    "texture operand modifiers");
   M5_FETCH_OUTSIDE(g18g && I->op == AGX_OPCODE_IMAGE_LOAD && !immediate &&
                       I->src[3].size != AGX_SIZE_64,
                    "G18 image handle was not lowered to a native index");
   M5_FETCH_OUTSIDE(!agx3_tex_imm(I->src[4], 0) || !agx_is_null(I->src[5]),
                    "trailing operands");
   M5_FETCH_OUTSIDE(I->dim != AGX_DIM_2D && I->dim != AGX_DIM_2D_ARRAY &&
                       !ms_2d && !ms_array && !three_d,
                    "texture dimension");
   M5_FETCH_OUTSIDE(enc == AGX_ISA_ENCODING_AGX3B
                       ? (!I->mask || (I->mask & ~15u))
                       : (I->mask != 1 && I->mask != 15),
                    "result mask");
   M5_FETCH_OUTSIDE(I->offset || I->shadow || I->query_lod ||
                       I->gather != AGX_GATHER_NONE || I->sparse,
                    "fetch modifier");
   /* Native G18 runtime-indexed MS/MS-array reads retain the staged
    * sample/layer ABI with either a register or uniform descriptor. Both
    * image and texture loads use it; their zero-level AUTO_LOD shape has
    * already been checked above, separately from ordinary texel fetches. */
   M5_FETCH_OUTSIDE(m5_indexed &&
                        (!stage_coordinates ||
                         (I->mask != 15 && !g18g_indexed_scalar) ||
                        I->dest[0].size != AGX_SIZE_32 ||
                        (g18g_ms
                           ? (I->op != AGX_OPCODE_TEXTURE_LOAD &&
                              I->op != AGX_OPCODE_IMAGE_LOAD)
                           : ((g18g && bindless_image
                                  ? !(register_lod ||
                                      ((immediate_lod ||
                                        I->lod_mode == AGX_LOD_MODE_AUTO_LOD) &&
                                       agx3_tex_imm(I->src[1], 0)))
                                  : (I->op != AGX_OPCODE_TEXTURE_LOAD ||
                                     !immediate_lod || I->src[1].value)) ||
                              (I->dim != AGX_DIM_2D &&
                               I->dim != AGX_DIM_2D_ARRAY && !three_d)))),
                    "G17G register-indexed fetch shape");
   M5_FETCH_OUTSIDE(register_lod && enc == AGX_ISA_ENCODING_AGX3B && slot != 1,
                    "runtime level on a late slot");
#undef M5_FETCH_OUTSIDE

   if (outside) {
      if (why)
         *why = outside;
      return false;
   }

   /* A read/write image stages its read coordinates at halves16/18, keeping
    * them disjoint from imgwci's halves0/2. Native read-only indexed images,
    * like ordinary texture loads, use the bank0 coordinate ABI, and so does
    * a G18G image read in a shader with no image write (image_read_bank0,
    * native run112 kick 19 launch 2). */
   const uint16_t staging_coord =
      g18g && stage_coordinates && agx3_g18g_plain_fetch_block(I)
         ? 8u * (slot - 1)
         : (I->op == AGX_OPCODE_IMAGE_LOAD && !bindless_image && !ms_array &&
            !image_read_bank0) || separate_write_coords ? 16 : 0;
   if (stage_coordinates) {
      uint16_t staging_wait = wait_mask;
      /* COPY carries one wait bit, while the image coordinates and heap
       * index may have several independent producers. A runtime LOD's
       * PACK_SMPIR can consume their combined mask; an immediate-level
       * read has no such instruction. Retire only those operand producers
       * before publishing its coordinates.
       */
      if (((enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_IMAGE_LOAD) ||
           enc == AGX_ISA_ENCODING_AGX3C) &&
          !register_lod && util_bitcount(staging_wait) > 1) {
         if (!agx3_retire_slots_noop(emission, enc, staging_wait)) {
            if (why)
               *why = "retiring the coordinate producers";
            return false;
         }
         staging_wait = 0;
      }
      if (register_lod) {
         uint8_t lod[AGX3_TEX_MAX_LENGTH], lod_len = 0;
         /* G18G reads the level at Y + 2 (AGX3_TEX_G18G_LEVEL_WORD); an
          * array block publishes it with the layer instead of the layer
          * alone below. */
         const bool staged_lod =
            g18g_array
               ? agx3_tex_stage_array_typed_lod_g18g(
                    I->src[0].value + 4, I->src[1].value,
                    AGX3_TEX_LEVEL_INTEGER, staging_slot, staging_wait,
                    I->src[0].cache, I->src[0].discard, I->src[1].cache,
                    I->src[1].discard, lod, &lod_len)
            : g18g
               ? agx3_tex_stage_typed_lod_g18g(
                    staging_coord, I->src[1].value, staging_slot,
                    /* Native539 retains live integer levels without the
                     * optional cache hint; native659 uses the same uncached
                     * publication. Preserve liveness when dropping the hint. */
                    staging_wait, false, I->src[1].discard,
                    AGX3_TEX_LEVEL_INTEGER, lod, &lod_len)
               : agx3_tex_stage_typed_lod(
                    enc, I->src[1].value, staging_slot, staging_wait,
                    I->src[1].cache, I->src[1].discard,
                    AGX3_TEX_LEVEL_INTEGER, lod, &lod_len);
         if (!staged_lod || !agx3_append(emission, lod, lod_len)) {
            if (why)
               *why = "staging the runtime level";
            return false;
         }
         staging_wait = 0;
      }

      const unsigned plain_coords =
         (ms_2d || ms_array || array) ? 2 : agx_channels(I->src[0]);
      for (unsigned i = 0; i < plain_coords; i++) {
         uint8_t copy[AGX3_TEX_MAX_LENGTH], copy_len = 0;
         /* G18G reads a 3D Z at Y + 4, not X + 4 (AGX3_TEX_G18G_Z_WORD). */
         const uint16_t staged_half =
            (g18g_array || g18g_ms)
               ? (i == 0 ? AGX3_TEX_G18G_ARRAY_X : AGX3_TEX_G18G_ARRAY_Y)
            : g18g && three_d && i == 2 ? staging_coord + AGX3_TEX_G18G_Z_WORD
                                        : staging_coord + 2 * i;
         if (!agx3_tex_stage_copy_hints(enc, staged_half,
                                  I->src[0].value + 2 * i,
                                  staging_slot, i == 0 ? staging_wait : 0,
                                  enc == AGX_ISA_ENCODING_AGX3C ? !I->src[0].discard : I->src[0].cache,
                                  I->src[0].discard,
                                  copy, &copy_len) ||
             !agx3_append(emission, copy, copy_len)) {
            if (why)
               *why = "staging a coordinate";
            return false;
         }
      }

      /* Native array fetches consume the layer through PACK_SMPIR's bank6
       * operand, independently of the x/y coordinate COPYs. A third ordinary
       * COPY leaves that operand unwritten and reads layer0. The non-MS
       * lowered layer is the low16 bits of the third coordinate word. */
      if (array && !(g18g_array && register_lod)) {
         uint8_t packed[AGX3_TEX_MAX_LENGTH], packed_len = 0;
         const bool staged_layer = g18g_array
            ? agx3_tex_stage_array_layer_g18g(
                 I->src[0].value + 4, staging_slot, 0, I->src[0].cache,
                 I->src[0].discard, packed, &packed_len)
            : agx3_tex_stage_array_layer(
                 enc, I->src[0].value + 4, staging_slot, 0,
                 I->src[0].cache, I->src[0].discard,
                 packed, &packed_len);
         if (!staged_layer ||
             !agx3_append(emission, packed, packed_len)) {
            if (why)
               *why = "staging the array layer";
            return false;
         }
      }

      /* For non-array MSAA, the third 32-bit coordinate is the sample index.
       * Publish it through native's dedicated integer PACK_SMPIR operand. */
      if (g18g_ms) {
         uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
         uint16_t sample = I->src[0].value + 4;
         if (why)
            *why = "G18 multisample sample extension";
         if (ms_array) {
            /* NIR packs sample/layer in low/high halves. Native G18
             * publishes a full 32-bit sample, so copy-and-clear the low
             * half after normalizing the high half into its own word. */
            sample = ms_array_scratch + 2;
            if (!ms_array_scratch || sample + 1 >= 128 ||
                !agx3_tex_normalize_cube_array_layer(
                   enc, ms_array_scratch, I->src[0].value + 5,
                   I->src[0].cache, false, bytes, &len) ||
                !agx3_append(emission, bytes, len) ||
                !agx3_copy_encode_reg16(
                   enc, sample, I->src[0].value + 4, false,
                   I->src[0].cache, I->src[0].discard, 0, bytes, &len) ||
                !agx3_append(emission, bytes, len) ||
                !agx3_clr(enc, sample + 1, true, false, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               return false;
         }
         if (why)
            *why = "G18 multisample sample publication";
         if (!agx3_tex_stage_ms_g18g(
                enc, sample, ms_array_scratch, ms_array, staging_slot,
                ms_array ? false : I->src[0].cache,
                ms_array ? true : I->src[0].discard, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            return false;
      }
   }

   const struct agx3_tex_fetch_desc d = {
      .dest = I->dest[0].value,
      .coord = (g18g_array || g18g_ms) ? AGX3_TEX_G18G_ARRAY_X
               : stage_coordinates ? staging_coord : I->src[0].value,
      .width = I->dest[0].size == AGX_SIZE_32 ? AGX3_TEX_W32 : AGX3_TEX_W16,
      .mask = I->mask,
      .texture = immediate ? I->src[3].value : 0,
      .texture_kind = immediate ? AGX3_TEX_OPERAND_IMMEDIATE :
         I->src[3].type == AGX_INDEX_UNIFORM ? AGX3_TEX_OPERAND_UNIFORM :
                                             AGX3_TEX_OPERAND_REGISTER,
      .texture_value = immediate ? 0 : I->src[3].value,
      .texture_kill = !immediate && !m5_indexed &&
         I->src[3].type == AGX_INDEX_REGISTER &&
         (I->src[3].discard || I->src[3].kill),
      .slot = slot,
      .wait_mask = stage_coordinates ? (1u << (staging_slot - 1)) : wait_mask,
      .coord_kill = stage_coordinates || I->src[0].kill,
      .final = final,
      .array = I->dim == AGX_DIM_2D_ARRAY || ms_array,
      .staged_array = g18g_array,
      .multisample = ms_2d || ms_array,
      .staged_sample = (ms_2d || ms_array) && stage_coordinates,
      .g18g_ms = g18g_ms,
      /* Native G18 MS reads name sampler zero for both dimensions and
       * every descriptor kind, including bindful 2D-MS. */
      .sampler_value = g18g_ms ? 0
         : (ms_2d || ms_array) && stage_coordinates ? 1 : 0,
      .three_d = three_d,
      .lod_x64 = immediate_lod ? I->src[1].value * 64 : 0,
      .register_lod = register_lod,
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t len = 0;
   if (!agx3_tex_encode_fetch(enc, &d, bytes, &len)) {
      if (why)
         *why = "no grounded fetch form for this descriptor";
      return false;
   }
   if (pixel_entry) {
      uint8_t entry[AGX3_PIXEL_MAX_LENGTH], entry_len = 0;
      if (!pixel_entry_offset ||
          !agx3_pixel_colour_entry(enc, entry, &entry_len))
         return false;
      *pixel_entry_offset = emission->size;
      if (!agx3_append(emission, entry, entry_len))
         return false;
   }
   return agx3_append(emission, bytes, len);
}

static bool
agx3_emit_texture_fetch(struct util_dynarray *emission,
                        enum agx_isa_encoding enc, const agx_instr *I,
                        uint8_t slot, uint16_t wait_mask, bool final,
                        bool stage_coordinates, bool separate_write_coords,
                        bool image_read_bank0, bool g18g,
                        uint16_t ms_array_scratch,
                        struct agx3_staging_banks *banks,
                        uint16_t outstanding,
                        uint16_t *staged_end, bool fragment, bool pixel_entry,
                        uint32_t *pixel_entry_offset, uint16_t heap_uniform_base,
                        struct agx3_fetch_group *group)
{
   /* G18G (agx3d) keeps j873's G17G-family fetch emitter, whose forms its
    * agx3d re-encode is grounded on. */
   if (enc == AGX_ISA_ENCODING_AGX3C && g18g) {
      const char *why = "operand publication";
      const bool encoded = agx3_emit_texture_fetch_g17g(emission, enc, I, slot, wait_mask,
          final, stage_coordinates, separate_write_coords, image_read_bank0,
          g18g, ms_array_scratch, pixel_entry, pixel_entry_offset, &why);
      if (!encoded)
         fprintf(stderr, "AGX3 fetch refused: %s (g18=%u stage=%u slot=%u scratch=%u)\n",
                 why, g18g, stage_coordinates, slot, ms_array_scratch);
      return encoded;
   }
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   const bool immediate = I->nr_srcs == agx3_fetch_nr_srcs(I) &&
      I->src[3].type == AGX_INDEX_IMMEDIATE;
   const bool immediate_lod = I->lod_mode == AGX_LOD_MODE_LOD_MIN &&
      I->src[1].type == AGX_INDEX_IMMEDIATE && !I->src[1].memory &&
      !I->src[1].abs && !I->src[1].neg && I->src[1].value <= 31;
   const bool register_lod =
      I->lod_mode == AGX_LOD_MODE_LOD_MIN &&
      agx3_tex_plain_reg(I->src[1]) &&
      (I->src[1].size == AGX_SIZE_32 ||
       (m5 && I->src[1].size == AGX_SIZE_16)) &&
      agx_channels(I->src[1]) == 1;
   const bool ms_2d = I->dim == AGX_DIM_2D_MS;
   const bool ms_array = I->dim == AGX_DIM_2D_MS_ARRAY;
   const bool array = I->dim == AGX_DIM_2D_ARRAY;
   const bool three_d = I->dim == AGX_DIM_3D;
   const bool combined_array_lod = register_lod && array;
   const bool bindless_image = I->op == AGX_OPCODE_IMAGE_LOAD && !immediate &&
      agx3_tex_plain_reg(I->src[3]) && I->src[3].size == AGX_SIZE_64 &&
      agx_channels(I->src[3]) == 1 && !(I->src[3].value & 1) &&
      I->src[3].value <= 252 &&
      !(I->src[3].cache && (I->src[3].discard || I->src[3].kill)) &&
      (agx3_tex_plain_uniform64(I->src[2]) || agx3_tex_imm(I->src[2], 0));
   /* Apple's lane-indexed image read publishes coordinates and uint LOD on
    * staging slot 2, then issues the texture result on slot 1. Keep those two
    * independent scoreboard domains; reusing slot 1 for both is not a native
    * dependency shape. */
   const uint8_t staging_slot = m5 ? slot : ms_array || combined_array_lod
      ? 1 : register_lod ? 2 : slot;
   /* A sparse read writes one extra register after its masked channels, so
    * agx_emit_texture reserves an eight-channel destination for it (see
    * agx_validate.c's sparse size rule). The encoded result mask stays the
    * four-bit channel mask; the residency component is implicit. */
   const unsigned coord_count =
      (I->dim == AGX_DIM_2D_ARRAY || ms_array || ms_2d || three_d) ? 3 : 2;
   if (I->nr_dests != 1 || I->nr_srcs != agx3_fetch_nr_srcs(I) ||
       !agx3_tex_plain_reg(I->dest[0]) ||
       (I->dest[0].size != AGX_SIZE_16 &&
        I->dest[0].size != AGX_SIZE_32) ||
       agx_channels(I->dest[0]) != (I->sparse ? 8u : 4u) ||
       I->dest[0].value >= (m5 ? 256 : 128) ||
       I->dest[0].value + agx_index_size_16(I->dest[0]) > 256 ||
       (I->dest[0].size == AGX_SIZE_32 && (I->dest[0].value & 1)) ||
       !agx3_fetch_coords_ok(I, m5, stage_coordinates, coord_count) ||
       ((ms_array || ms_2d)
           ? !agx3_tex_imm(I->src[1], 0)
           : (!(I->lod_mode == AGX_LOD_MODE_AUTO_LOD &&
                agx3_tex_imm(I->src[1], 0)) &&
              !immediate_lod && !register_lod)) ||
       ((ms_array || ms_2d) && I->lod_mode != AGX_LOD_MODE_AUTO_LOD) ||
       (register_lod && !stage_coordinates) ||
       !(agx3_tex_imm(I->src[2], 0) ||
         ((ms_array || ms_2d) && agx3_tex_plain_uniform64(I->src[2])) ||
         bindless_image) ||
       (immediate ? I->src[3].value > 15 :
        ((enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C) ||
         (I->src[3].type != AGX_INDEX_REGISTER && I->src[3].type != AGX_INDEX_UNIFORM) ||
         (I->src[3].size != AGX_SIZE_64 &&
          !((ms_array || ms_2d) && I->src[3].size == AGX_SIZE_32)) ||
         agx_channels(I->src[3]) != 1 ||
         (I->src[3].value & 1) || I->src[3].value > 252)) ||
       I->src[3].memory || I->src[3].abs || I->src[3].neg ||
       !agx3_tex_imm(I->src[4], 0) ||
       !agx_is_null(I->src[5]) ||
       (I->dim != AGX_DIM_2D && I->dim != AGX_DIM_2D_ARRAY && !ms_2d &&
        !ms_array && !three_d) ||

       /* agx_instr::mask is four bits wide, so the residency component of a
        * sparse result never appears in it; the hardware writes that
        * component implicitly, after the masked channels. */
       (I->mask == 0 || (I->mask & ~15u)) ||
       I->offset || I->shadow ||
       I->query_lod || I->gather != AGX_GATHER_NONE ||
       /* Both encodings support sparse reads; sparse multisample has no
        * qualified form. */
       (I->sparse && (ms_2d || ms_array)))
      return false;

   /* A read/write image stages its read coordinates at halves16/18, keeping
    * them disjoint from imgwci's halves0/2, and its PACK_SMPIR operands at
    * half 22 (agx3_tex_stage_*_at()), clear of imgwci's data lanes and layer.
    * Native read-only indexed images, like ordinary texture loads, use the
    * bank0 coordinate ABI. */
   const uint16_t staging_base =
      /* Staged MS reads publish their sample through bank6 and consume
       * coordinates at bank0. Keep image loads on that same native ABI;
       * the bank allocator orders overlapping image writes. */
      (I->op == AGX_OPCODE_IMAGE_LOAD && !bindless_image && !ms_array &&
       !ms_2d) ||
            separate_write_coords ? 16 : 0;
   if (!m5 && register_lod && slot != 1)
      return false;

   /* A staged read consumes, and kills, its coordinate bank asynchronously,
    * so consecutive reads stage in disjoint banks, as Apple's do (smp f19=0
    * then f19=4). Callers that pass no `banks` keep one bank. Halves 4..7
    * carry PACK_SMPIR operands, so banks step by eight halves; plain 2D
    * reads, immediate-LOD 2D-array reads, and M5 multisample reads allocate
    * disjoint banks from `staging_base`, while other reads with PACK_SMPIR
    * operands keep the base bank and occupy halves 0..7. An outstanding read
    * sharing the halves is retired first. */
   const bool plain_bank = !ms_2d && !ms_array && !array && !register_lod;
   const bool array_bank = array && !register_lod;
   const bool m5_ms_banks = m5 && (ms_2d || ms_array);
   const bool banked_read = plain_bank || array_bank || m5_ms_banks;
   /* Fetch coordinates can name any aligned bank through half 60 (f19).
    * Use independent eight-half banks for all scoreboard slots (6 on AGX3B,
    * 8 on AGX3C) before recycling one. */
   const unsigned plain_banks =
      MIN2(m5 ? AGX3_M5_FETCH_BANKS : 6u, 1 + (60 - staging_base) / 8);
   /* Keep independent non-array MS publications in separate operand banks. */
   const unsigned ms_banks =
      m5_ms_banks && !ms_array ? AGX3_M5_MS_FETCH_BANKS : 4;
   const uint16_t bank_span =
      (m5_ms_banks || array_bank) ? 8 : plain_bank ? 2 * coord_count
                                                   : staging_base + 8;
   uint16_t staging_coord = staging_base;
   uint16_t bank_retire = 0;
   const bool reuse = group && group->reuse;
   if (reuse) {
      if (!m5 || !stage_coordinates || !banks || !plain_bank)
         return false;
      staging_coord = group->coord;
      /* Other sources and destination hazards still belong to this fetch;
       * only the already-completed coordinate publication is shared. The
       * coordinates are ready once this fetch's wait retires, so they carry
       * it instead of r0, whose producer may be outside the mask. */
      if (wait_mask && !agx3_retire_wait_on(emission, enc, I->src[0],
                                            wait_mask))
         return false;
      wait_mask = 0;
   } else if (stage_coordinates && banks) {
      agx3_staging_release(banks, outstanding);

      /* Non-banked reads with PACK_SMPIR operands occupy halves 0..bank_span. */
      const uint16_t first = banked_read ? staging_base : 0;
      const unsigned max_banks =
         m5_ms_banks ? ms_banks :
         (plain_bank || array_bank) ? plain_banks : 1u;
      bool placed = false;
      for (unsigned c = 0; c < max_banks && !placed; ++c) {
         const uint16_t start = first + 8 * c;
         const uint16_t conflict = agx3_staging_conflicts(
            banks, banked_read ? start : 0, bank_span);
         if (c == 0)
            bank_retire = conflict;
         if (!conflict) {
            staging_coord = banked_read ? start : staging_base;
            placed = true;
         }
      }
      if (placed)
         bank_retire = 0;
      else
         agx3_staging_forget(banks, bank_retire);
   }
   /* The r0 carrier reads and rewrites r0, so it is sound only while r0 is
    * ready or its producer is in the mask. When banked reads fill all banks,
    * the next one retires the first bank's read, and r0 may be the
    * destination of a later read still in flight: in the stencil resolve of
    * dEQP-VK.renderpasses.renderpass2.depth_stencil_resolve.*.stencil_min,
    * five texel fetches were in flight, the second writing r0..r3, and the
    * carrier waited for the first alone. Carry the retirement on this
    * fetch's coordinates with the fetch's own wait, which covers every
    * pending write of them; the staging COPYs right after it wait for the
    * same producers. */
   if (bank_retire) {
      if (!agx3_retire_wait_on(emission, enc, I->src[0],
                               bank_retire | wait_mask))
         return false;
      wait_mask = 0;
   }
   if (staged_end)
      *staged_end = banked_read ? staging_coord + bank_span : bank_span;

   /* The plain 2D fetch publishes its per-lane level after the coordinates,
    * so no PACK_SMPIR leads the group to absorb a combined producer mask. */
   const bool plain_2d_lod =
      register_lod && !combined_array_lod && I->dim == AGX_DIM_2D;

   if (stage_coordinates && !reuse) {
      uint16_t staging_wait = wait_mask & (uint16_t)~bank_retire;
      /* COPY carries one wait bit, while the image coordinates and heap
       * index may have several independent producers. A runtime LOD's
       * PACK_SMPIR can consume their combined mask; an immediate-level
       * read has no such instruction. Retire only those operand producers
       * before publishing its coordinates.
       */
      if (enc == AGX_ISA_ENCODING_AGX3B &&
          ((I->op == AGX_OPCODE_IMAGE_LOAD && !register_lod) ||
           plain_2d_lod) &&
          util_bitcount(staging_wait) > 1) {
         if (!agx3_retire_slots_noop(emission, enc, staging_wait))
            return false;
         staging_wait = 0;
      }
      if (m5 && plain_2d_lod) {
         uint8_t lod[AGX3_TEX_MAX_LENGTH], lod_len = 0;
         if (!agx3_tex_stage_typed_lod_at(
                enc, staging_coord, I->src[1].value, staging_slot,
                staging_wait, false, I->src[1].discard,
                AGX3_TEX_LEVEL_INTEGER, lod, &lod_len) ||
             !agx3_append(emission, lod, lod_len))
            return false;
         staging_wait = 0;
      }
      if (combined_array_lod) {
         uint8_t packed[AGX3_TEX_MAX_LENGTH], packed_len = 0;
         /* The layer is read before the x/y COPYs: it may only discard a
          * register neither of them still reads. */
         const agx_index layer = agx3_tex_coord(I, 2);
         const bool layer_discard = layer.discard &&
            agx3_tex_coord(I, 0).value != layer.value &&
            agx3_tex_coord(I, 1).value != layer.value;
         if (!agx3_tex_stage_array_integer_lod_at(
                enc, staging_base, layer.value, I->src[1].value,
                staging_slot, staging_wait,
                m5 ? !layer_discard : layer.cache,
                layer_discard,
                m5 ? !I->src[1].discard : I->src[1].cache,
                I->src[1].discard,
                packed, &packed_len) ||
             !agx3_append(emission, packed, packed_len))
            return false;
         staging_wait = 0;
      } else if (register_lod && !plain_2d_lod) {
         uint8_t lod[AGX3_TEX_MAX_LENGTH], lod_len = 0;
         if (!agx3_tex_stage_typed_lod_at(
                enc, staging_base, I->src[1].value, staging_slot,
                staging_wait, m5 ? false : I->src[1].cache,
                I->src[1].discard, AGX3_TEX_LEVEL_INTEGER, lod, &lod_len) ||
             !agx3_append(emission, lod, lod_len))
            return false;
         staging_wait = 0;
      }

      const unsigned plain_coords =
         (ms_2d || ms_array || array) ? 2 : coord_count;
      /* The staging COPY carries one wait bit. A fetch whose operands come
       * from several scoreboard slots repeats the first coordinate copy, one
       * outstanding bit each; every repeat writes the same register from the
       * same source, so the last leaves the right value. The source stays
       * live until the final repeat. */
      uint16_t pending_wait = staging_wait;
      for (unsigned i = 0; i < plain_coords; i++) {
       const unsigned reps =
          (i == 0) ? MAX2(util_bitcount(pending_wait), 1u) : 1u;
       for (unsigned rep = 0; rep < reps; rep++) {
         uint16_t wait_bit = 0;
         if (pending_wait) {
            wait_bit = (uint16_t)(pending_wait & (~pending_wait + 1u));
            pending_wait &= (uint16_t)~wait_bit;
         }
         uint8_t copy[AGX3_TEX_MAX_LENGTH], copy_len = 0;
         const agx_index coord = agx3_tex_coord(I, i);
         const bool last =
            agx3_tex_coord_discard(I, i, coord_count) && rep + 1 == reps;
         /* Split coordinates may live anywhere in the register file: stage
          * them with the wide scalar COPY, as scalar publications do. */
         const bool copied = I->agx3_split_coords
            ? agx3_tex_stage_copy_scalar_m5(staging_coord + 2 * i, coord.value,
                                            staging_slot, wait_bit, !last, last,
                                            copy, &copy_len)
            : agx3_tex_stage_copy_hints(enc, staging_coord + 2 * i,
                                        coord.value, staging_slot, wait_bit,
                                        m5 ? !last : coord.cache, last,
                                        copy, &copy_len);
         if (!copied ||
             !agx3_append(emission, copy, copy_len))
            return false;
       }
      }

      /* A plain 2D fragment texel fetch stages its level after its
       * coordinates (coordinate copies, PACK_SMPIR, SMP) and into its own
       * operand bank; a level published elsewhere reads back as zero. */
      if (plain_2d_lod && !m5) {
         uint8_t lod[AGX3_TEX_MAX_LENGTH], lod_len = 0;
         if (!agx3_tex_stage_typed_lod_bank(
                enc, staging_coord + 6, I->src[1].value, staging_slot,
                staging_wait, I->src[1].cache, I->src[1].discard,
                AGX3_TEX_LEVEL_INTEGER, lod, &lod_len) ||
             !agx3_append(emission, lod, lod_len))
            return false;
         staging_wait = 0;
      }

      /* Native array fetches consume the layer through PACK_SMPIR's bank6
       * operand, independently of the x/y coordinate COPYs. A third ordinary
       * COPY leaves that operand unwritten and reads layer0. The non-MS
       * lowered layer is the low16 bits of the third coordinate word. */
      if (array && !combined_array_lod) {
         uint8_t packed[AGX3_TEX_MAX_LENGTH], packed_len = 0;
         const agx_index layer = agx3_tex_coord(I, 2);
         if (!agx3_tex_stage_array_layer_at(
                enc, staging_coord, layer.value, staging_slot, 0,
                layer.cache, layer.discard,
                packed, &packed_len) ||
             !agx3_append(emission, packed, packed_len))
            return false;
      }

      /* For non-array MSAA, the third 32-bit coordinate is the sample index.
       * Publish it through native's dedicated integer PACK_SMPIR operand. */
      if (ms_2d) {
         uint8_t packed[AGX3_TEX_MAX_LENGTH], packed_len = 0;
         const agx_index sample = agx3_tex_coord(I, 2);
         if (!agx3_tex_stage_ms_sample_at(
                enc, staging_coord, sample.value, staging_slot, 0,
                sample.cache, sample.discard,
                packed, &packed_len) ||
             !agx3_append(emission, packed, packed_len))
            return false;
      }
      if (ms_array) {
         uint8_t packed[AGX3_TEX_MAX_LENGTH], packed_len = 0;
         /* Keep the packed source alive through both half reads. The final
          * publication consumes its low sample half after the high layer
          * half has been normalized into a separate native GPR word. */
         if (!agx3_tex_pack_ms_array_layer(
                enc, I->src[0].value + 5, ms_array_scratch,
                I->src[0].cache, false, packed, &packed_len) ||
             !agx3_append(emission, packed, packed_len) ||
             !agx3_tex_stage_ms_array_sample_at(
                enc, staging_coord, I->src[0].value + 4, ms_array_scratch,
                staging_slot,
                I->src[0].cache, I->src[0].discard, packed, &packed_len) ||
             !agx3_append(emission, packed, packed_len))
            return false;
      }
   }

   if (stage_coordinates && banks &&
       !agx3_staging_hold(banks, banked_read ? staging_coord : 0, bank_span,
                          slot))
      return false;

   const struct agx3_tex_fetch_desc d = {
      .heap_uniform_base = heap_uniform_base,
      .dest_cache = m5 && I->dest[0].cache,
      .dest = I->dest[0].value,
      .coord = stage_coordinates ? staging_coord : I->src[0].value,
      .width = I->dest[0].size == AGX_SIZE_32 ? AGX3_TEX_W32 : AGX3_TEX_W16,
      .mask = I->mask,
      .sparse = I->sparse,
      .texture = immediate ? I->src[3].value : 0,
      .texture_kind = immediate ? AGX3_TEX_OPERAND_IMMEDIATE :
         I->src[3].type == AGX_INDEX_UNIFORM ? AGX3_TEX_OPERAND_UNIFORM :
                                             AGX3_TEX_OPERAND_REGISTER,
      .texture_value = immediate ? 0 : I->src[3].value,
      .texture_kill = enc != AGX_ISA_ENCODING_AGX3C && !immediate &&
         I->src[3].type == AGX_INDEX_REGISTER &&
         (I->src[3].discard || I->src[3].kill),
      .slot = slot,
      .wait_mask = reuse ? 0 : stage_coordinates
         ? (1u << (staging_slot - 1)) : wait_mask,
      .coord_kill = stage_coordinates ? !(group && group->keep)
                                       : I->src[0].kill,
      .final = final,
      .array = I->dim == AGX_DIM_2D_ARRAY || ms_array,
      .multisample = ms_2d || ms_array,
      .staged_sample = (ms_2d || ms_array) && stage_coordinates,
      /* M5 raw reads use sampler selector1 in fragment programs and0 in
       * compute programs, for both ordinary and multisampled images. */
      .sampler_value = m5 ? (fragment ? 1 : 0) :
         ((ms_2d || ms_array) && stage_coordinates ? 1 : 0),
      .three_d = three_d,
      .lod_x64 = immediate_lod ? I->src[1].value * 64 : 0,
      .register_lod = register_lod,
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t len = 0;
   /* Apple's fragment texel fetches enter the pixel transaction after all
    * operand staging and immediately before SMP, exactly as its implicit
    * derivative samples do.
    */
   if (pixel_entry) {
      uint8_t entry[AGX3_PIXEL_MAX_LENGTH], entry_len = 0;
      if (pixel_entry_offset == NULL ||
          !agx3_pixel_colour_entry(enc, entry, &entry_len))
         return false;

      *pixel_entry_offset = emission->size;
      if (!agx3_append(emission, entry, entry_len))
         return false;
   }

   if (!agx3_tex_encode_fetch(enc, &d, bytes, &len) ||
       !agx3_append(emission, bytes, len))
      return false;
   if (group)
      group->coord = staging_coord;
   return true;
}

/* Two bounded public storage-image forms: integer 2D state0 fed by an image
 * load or atomic result, and normalized formatted-float state1 fed by one
 * explicit-LOD sample. The native imgwci operation is the coherent image
 * write form; NIR's coherent bit does not select a different instruction.
 * The PBE descriptor selects RGBA8 versus RGB10A2.
 * The four slot1 COPYs route the packed vec4 into the exact imgwci data lanes.
 * The first COPY consumes the one-shot texture dependency when it is still
 * outstanding. An intervening bounds consumer may already have retired that
 * producer; either way native imgwci waits for the new staging-slot1 writes.
 * Validation happens before the first append, so a refusal leaves the output
 * transaction untouched. */
static bool
agx3_image_write_copy(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                      bool kill, uint8_t slot, uint16_t wait,
                      uint8_t out[AGX3_STAGE_MAX_LENGTH], uint8_t *len)
{
   return enc == AGX_ISA_ENCODING_AGX3C
      ? agx3_tex_stage_copy_hints(enc, dest, src, slot, wait, !kill, kill, out, len)
      : agx3_stage_copy_source(enc, dest, src, kill, slot, wait, out, len);
}

static bool
agx3_emit_storage_image_write(struct util_dynarray *emission,
                              enum agx_isa_encoding enc, enum agx_chip chip,
                              const agx_instr *I,
                              uint16_t wait_mask,
                              struct agx3_staging_banks *banks,
                              uint16_t bank_floor, uint16_t *outstanding,
                              uint8_t *write_slot,
                              uint16_t heap_uniform_base,
                              bool state0_pair)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   const unsigned slots = m5 ? 8 : 6;
   const unsigned bank_limit = m5 ? 256 : AGX3_STAGED_WRITE_LIMIT;
   const bool normalized = I->format == AGX_FORMAT_U8NORM ||
                           I->format == AGX_FORMAT_RGB10A2;
   const bool immediate_state = agx3_tex_imm(I->src[3], 0) &&
      I->src[4].type == AGX_INDEX_IMMEDIATE &&
      !I->src[4].memory && !I->src[4].abs && !I->src[4].neg &&
      ((I->src[4].value == 0 && I->format == AGX_FORMAT_I8) ||
       (I->src[4].value == 1 &&
        (normalized || I->format == AGX_FORMAT_I8)));
   /* A bindless store names its descriptor set's uniform halves as the PBE
    * heap (f79) and carries the PBE record index in a GPR (f92);
    * hk_g17p_lower_storage_records() converts the byte offset to that index.
    * Heap and index are independent operands, and the state operand's form
    * is independent of f70 (normalized data): int and uint encode alike. */
   const bool heap_state =
      I->src[3].type == AGX_INDEX_UNIFORM && I->src[3].size == AGX_SIZE_64 &&
      agx_channels(I->src[3]) == 1 && !(I->src[3].value & 3) &&
      I->src[3].value <= 508 && !I->src[3].memory && !I->src[3].abs &&
      !I->src[3].neg && !I->src[3].kill && !I->src[3].cache &&
      !I->src[3].discard &&
      agx3_tex_plain_reg(I->src[4]) &&
      I->src[4].size == AGX_SIZE_32 && agx_channels(I->src[4]) == 1 &&
      !(I->src[4].value & 1) && I->src[4].value <= 126 &&
      !(I->src[4].cache && (I->src[4].discard || I->src[4].kill));
   /* Gallium on G17G names a statically bound image unit with an immediate
    * state index: the image's PBE record follows its texture records in the
    * stage table (image_records pairs), so that operand is the record, and
    * the heap is the texture table pointer the G17G resource prefix carries
    * at uniform half 0. agx3_tex_encode_image_write_agx3c() has the matching
    * immediate kind: f79 0, f91 the record, f99/f100 immediate. */
   const bool table_state =
      m5 && agx3_tex_imm(I->src[3], 0) &&
      I->src[4].type == AGX_INDEX_IMMEDIATE && !I->src[4].memory &&
      !I->src[4].abs && !I->src[4].neg && I->src[4].value <= 15;
   /* Cube stores reach the packer as 2D arrays with the face folded into the
    * layer, and 1D stores as 2D. G17G writes a layered image as 3D, the layer
    * staged like z: Gallium describes its PBE as 3D (agx_batch_upload_pbe), a
    * single-level 3D slice being a 2D-array layer. The G17P PACK_IMGWIR layer
    * did not reach the G17G write (every layer landed in layer 0). G17C takes
    * the layer operand, and its array descriptors (Honeykrisp keeps them, and
    * Gallium does on G17C) need it: their layer count does not shrink with
    * the selected mip level, unlike 3D depth.
    */
   const bool layer_operand = !m5 || chip == AGX_CHIP_G17C;
   const bool array = I->dim == AGX_DIM_2D_ARRAY && layer_operand;
   const bool three_d = I->dim == AGX_DIM_3D ||
                        (!layer_operand && I->dim == AGX_DIM_2D_ARRAY);
   const unsigned coords = (array || three_d) ? 3 : 2;
   if ((!m5 && enc != AGX_ISA_ENCODING_AGX3B) || I->nr_dests != 0 ||
       (m5 && ((I->dim != AGX_DIM_2D && !array && !three_d) ||
               (!heap_state && !table_state) ||
               ((array || three_d) && I->format != AGX_FORMAT_I8) ||
               (I->format != AGX_FORMAT_I8 && I->format != AGX_FORMAT_I32 &&
                I->format != AGX_FORMAT_U8NORM))) ||
       I->nr_srcs != 5 ||
       (I->dim != AGX_DIM_2D && !array && !three_d) ||
       (wait_mask & ~((1u << slots) - 1)) != 0 ||
       !agx3_tex_plain_reg(I->src[0]) || I->src[0].size != AGX_SIZE_32 ||
       agx_channels(I->src[0]) != 4 || (I->src[0].value & 1) ||
       I->src[0].value > (m5 ? 120 : 248) || !agx3_tex_plain_reg(I->src[1]) ||
       I->src[1].size != AGX_SIZE_32 || agx_channels(I->src[1]) != coords ||
       (I->src[1].value & 1) || I->src[1].value + 2 * (coords - 1) > (m5 ? 126 : 254) ||
       !agx3_tex_imm(I->src[2], 0) ||
       (!immediate_state && !heap_state && !table_state)) {
      if (agx_compiler_debug & AGX_DBG_SHADERS)
         fprintf(stderr, "G17P imgwci shape refusal: wait=0x%x data=%u/%u/%u coord=%u/%u/%u dim=%u coherent=%u\n",
                 wait_mask, I->src[0].value, I->src[0].size,
                 agx_channels(I->src[0]), I->src[1].value, I->src[1].size,
                 agx_channels(I->src[1]), I->dim, I->coherent);
      return false;
   }

   /* imgwci consumes its staging bank asynchronously and signals its f12
    * slot when done, so a bank stays held until its write's slot has been
    * waited; consecutive writes (and a write after a texture read) stage in
    * distinct banks, as Apple's do. The resolved-PBE forms below keep bank 0
    * and slot 1. */
   if (m5 && util_bitcount(wait_mask) > 1) {
      if (!agx3_retire_slots_noop(emission, enc, wait_mask))
         return false;
      *outstanding &= (uint16_t)~wait_mask;
      wait_mask = 0;
   }
   const unsigned bank_span = 16;
   uint16_t bank = 0;
   uint8_t slot = 1;
   if (banks) {
      agx3_staging_release(banks, *outstanding);

      uint16_t retire = 0;
      bool placed = false;
      for (unsigned start = heap_state ? bank_floor : 0;
           start + bank_span <= bank_limit && !placed &&
           (heap_state || start == 0);
           start += m5 ? 16 : 8) {
         if (!agx3_staging_conflicts(banks, start, bank_span)) {
            bank = start;
            placed = true;
         }
      }
      if (!placed) {
         bank = heap_state ? bank_floor : 0;
         retire |= agx3_staging_conflicts(banks, bank, bank_span);
      }

      /* Stage on a slot no unrelated producer holds, rotating so writes in
       * flight together use different slots. */
      const uint16_t busy = *outstanding & (uint16_t)~(wait_mask | retire);
      if (heap_state) {
         slot = 0;
         for (unsigned i = 0; i < slots && !slot; ++i) {
            const uint8_t s = (uint8_t)((banks->next_write_slot + i) % slots + 1);
            if (!(busy & (1u << (s - 1))))
               slot = s;
         }
         if (!slot) {
            slot = (uint8_t)(banks->next_write_slot % slots + 1);
            retire |= 1u << (slot - 1);
         }
         banks->next_write_slot = slot % slots;
      } else if (busy & 1u) {
         retire |= 1u;
      }

      if (retire) {
         if (!agx3_retire_slots_noop(emission, enc, retire))
            return false;
         *outstanding &= (uint16_t)~retire;
         agx3_staging_forget(banks, retire);
      }
      /* The write's own slot also names the bank it now holds. */
      agx3_staging_forget(banks, 1u << (slot - 1));
      if (!agx3_staging_hold(banks, bank, bank_span, slot))
         return false;
      *outstanding |= 1u << (slot - 1);
   }

   struct util_dynarray staged;
   util_dynarray_init(&staged, NULL);
   uint8_t coord_bytes[AGX3_STAGE_MAX_LENGTH], coord_len = 0;
   uint8_t layer_bytes[AGX3_TEX_MAX_LENGTH], layer_len = 0;
   /* A 3D write stages z at half 4 and its data at 8/10/12/14; an array
    * write packs its layer at half 6 and its data at 4/10/12/14. A bank
    * offsets every half (f5). */
   const uint16_t data_dest[4] = {
      bank + (three_d ? 8 : 4), bank + (array || three_d ? 10 : 6),
      bank + 12, bank + 14,
   };
   /* imgwci reads integer coordinates from the first two halves of its bank,
    * published with COPY on the slot it waits for; the source stays live.
    * The scoreboard wait goes on the FIRST staged COPY of the group (the
    * coordinates), as for every other staged producer: a coordinate from an
    * outstanding device load must retire before it is copied. */
   if (!agx3_image_write_copy(enc, bank, I->src[1].value, false, slot,
                               wait_mask, coord_bytes, &coord_len) ||
       !agx3_append(&staged, coord_bytes, coord_len) ||
       !agx3_image_write_copy(enc, bank + 2, I->src[1].value + 2, false,
                               slot, 0, coord_bytes, &coord_len) ||
       !agx3_append(&staged, coord_bytes, coord_len) ||
       (three_d &&
        (!agx3_image_write_copy(enc, bank + 4, I->src[1].value + 4, false,
                                 slot, 0, coord_bytes, &coord_len) ||
         !agx3_append(&staged, coord_bytes, coord_len))) ||
       (array &&
        (!agx3_tex_stage_image_write_layer(enc, I->src[1].value + 4,
                                           bank + 6, slot, 0, false, false,
                                           layer_bytes, &layer_len) ||
         !agx3_append(&staged, layer_bytes, layer_len)))) {
      if (agx_compiler_debug & AGX_DBG_SHADERS)
         fprintf(stderr, "G17P imgwci coordinate compose refusal\n");
      goto fail;
   }

   for (unsigned c = 0; c < 4; ++c) {
      uint8_t bytes[AGX3_STAGE_MAX_LENGTH], len = 0;
      const unsigned source = I->src[0].value + 2 * c;
      const bool index_alias = m5 && heap_state && source == I->src[4].value;
      if (!agx3_image_write_copy(enc, data_dest[c], source,
                                  I->src[0].discard && !index_alias, slot, 0,
                                  bytes, &len) ||
          !agx3_append(&staged, bytes, len)) {
         if (agx_compiler_debug & AGX_DBG_SHADERS)
            fprintf(stderr, "G17P imgwci data compose refusal: component=%u\n", c);
         goto fail;
      }
   }

   uint8_t imgwci[AGX3_TEX_MAX_LENGTH], imgwci_len = 0;
   const struct agx3_image_write_desc d = {
      .wait_mask = wait_mask,
      .staging_bank = bank,
      .result_slot = slot,
      .image_kind = heap_state ? AGX3_TEX_OPERAND_REGISTER :
                                 AGX3_TEX_OPERAND_IMMEDIATE,
      .image_value = I->src[4].value,
      .image_kill = heap_state && (I->src[4].discard || I->src[4].kill),
      .normalized = normalized,
      .dim = array     ? AGX3_IMAGE_WRITE_2D_ARRAY
             : three_d ? AGX3_IMAGE_WRITE_3D
                       : AGX3_IMAGE_WRITE_2D,
   };
   bool encoded;
   if (m5) {
      encoded = agx3_tex_encode_image_write_agx3c(
         &d, I->src[3].value, slot, imgwci, &imgwci_len);
   } else if (heap_state) {
      /* Consume this write's COPYs, as the resolved PBE form below does. */
      encoded = agx3_tex_encode_image_write_2d(
         enc, &d, I->src[3].value, slot, imgwci, &imgwci_len);
   } else if (immediate_state && I->src[4].value == 1 && !normalized) {
      /* Hk's mixed indexed-read/R32-write table contains texture and sampler
       * heap pointers followed by one resolved PBE pointer at uniform half8.
       * Apple's read/write witness instead used a PBE heap at half4 with
       * descriptor index1. Copying that pair selected our sampler heap.
       * Both staging slots passed the real-data hardware gate; retain the
       * existing, shorter slot1 COPYs and consume that same slot here. */
      struct agx3_image_write_desc resolved = d;
      resolved.image_value = 0;
      encoded = agx3_tex_encode_image_write_2d(
         enc, &resolved, heap_uniform_base + 8, 1, imgwci, &imgwci_len);
   } else {
      const uint16_t base = heap_uniform_base + (d.image_value == 1 ? 4 : 0);
      const uint8_t staging_slot = d.image_value == 0 ? 1 : 2;
      /* hk's read/write state-0 ABI points state 0 at the texture/PBE pair
       * (agx_shader_key::g17p_storage_image_pair); record 0 there is the
       * texture, which as a PBE describes a far larger surface and sends the
       * stores past the image. Take the PBE as record 1 of the same heap. */
      struct agx3_image_write_desc resolved = d;
      if (state0_pair && d.image_value == 0)
         resolved.image_value = 1;
      encoded = agx3_tex_encode_image_write_2d(
         enc, &resolved, base, staging_slot, imgwci, &imgwci_len);
   }
   if (!encoded || !agx3_append(&staged, imgwci, imgwci_len) ||
       !agx3_append(emission, staged.data, staged.size)) {
      if (agx_compiler_debug & AGX_DBG_SHADERS) {
         fprintf(stderr, "G17P imgwci instruction compose refusal\n");
         fprintf(stderr, "imgwci state: m5=%u bank=%u slot=%u base=%u index=%u kind=%u kill=%u normalized=%u wait=%x dim=%u encoded=%u length=%u\n",
                 m5, bank, slot, I->src[3].value, d.image_value, d.image_kind,
                 d.image_kill, d.normalized, d.wait_mask, d.dim, encoded, imgwci_len);
      }
      goto fail;
   }
   *write_slot = slot;
   util_dynarray_fini(&staged);
   return true;

fail:
   util_dynarray_fini(&staged);
   return false;
}

/* G18G image writes.
 *
 * The first grounded one is native run112 kick 19 launch 1, Apple's
 * image_write(texture2d<uint, access::write>) of the Metal workloads file
 * -- the Metal twin of gallery case 10's imageStore -- which the J873 ran and
 * whose code metal-nt -arch applegpu_g18g reproduces byte for byte:
 *
 *   iadd32 staging[0] <- x + 17y   copy staging[2] <- x   copy staging[4] <- y
 *   imgwci 160340c00b002100000000000060   (data@0 X@2 Y@4; staged on slot 1,
 *                                          signals slot 1)
 *
 * Apple's G18G compiler gives that layout, the data below X, to every
 * one-component write at a thread-position coordinate (seven metal-nt
 * variants: uint, int, float, splat, read_write, texture(1), swapped
 * coordinates), and the same to a lone four-component write: data 0..6, X 8,
 * Y 10 (m6-image-write iw_cs_rgba32f, m6-image-write2 iw2_{f4,u4,i4}_t*).
 * A 3D write's z and a 2D array's PACK_IMGWIR layer sit at Y + 2
 * (iw_cs_rgba32ui_3d, iw4_f4_3d; iw_cs_r32i_array, iw4_f4_array). Mesa
 * stages exactly that in a sixteen-half bank: the data components at the
 * bank, X after them, Y and the third coordinate after X, all COPYed on the
 * write's staging slot, then the agx3c imgwci that agx3_transcode_agx3d()
 * re-encodes into Apple's G18G form (kick 19's bytes for the R32 write in
 * bank 0, iw_cs_rgba32f's for a lone RGBA float write).
 *
 * Several writes in flight each take their own bank and slot, as Apple's
 * four writes do (m6-image-write4 iw4_four: four operand sets, f12 1/3/4/2,
 * each write waiting on the slot its operands were staged on and releasing
 * them, f13 1). A write holds its bank until its slot is synchronised; the
 * slots rotate 1..4, the result slots Apple's G18G imgwci uses. In a
 * program that also reads textures or images the banks start above every
 * read's slot-keyed staging block (AGX3_G18G_READ_STAGING_LIMIT), so no read
 * restages a bank a write is still consuming and no write clobbers a read's
 * coordinates; a vertex shader's start above its outputs' UVSW staging
 * (AGX3_G18G_OUTPUT_STAGING_LIMIT).
 *
 * A vertex function's write is the same imgwci between its outputs' UVSWs:
 * metal-nt -arch applegpu_g18g, texture2d<float/uint, access::write> written
 * from a vertex function (m6-image-write-vs iwv_f4, iwv_u4_two), f5/f116/f118
 * naming X, Y and the data, f12/f60 the slots, f79 0, f91 0 and 1. A
 * fragment function's write sits inside the pixel transaction, ahead of its
 * late depth test (m6-image-write iw_frag_*: pixwait 1, depthf, pixrel 1
 * after the imgwci), supplied by the linked fragment program.
 *
 * The image is src[4]'s descriptor index into src[3]'s uniform heap base.
 * The legacy immediate-zero base selects Gallium's texture table or the
 * lowered state-0 write's resolved PBE. Native459 grounds runtime GPR and
 * uniform indices with explicit heap bases in compute and fragment writes.
 *
 * Refused, failing closed: unsupported execution stages or descriptor shapes,
 * a level other than zero, 16-bit data, 1D/cube/multisample
 * dims that were not lowered to 2D/2D array, and a data type
 * agx_emit_image_store() could not name. The format marker is
 * agx_emit_image_store()'s: AGX_FORMAT_I32 for kick 19's one-component R32
 * integer write (2D only), AGX_FORMAT_I8 for four integer components (f70
 * 0), AGX_FORMAT_U8NORM for four float components (f70 1). */
static bool
agx3_emit_storage_image_write_g18g(struct util_dynarray *emission,
                                   enum agx_isa_encoding enc,
                                   const agx_instr *I, uint16_t wait_mask,
                                   mesa_shader_stage stage,
                                   struct agx3_staging_banks *banks,
                                   uint16_t bank_floor,
                                   uint16_t bank_limit,
                                   uint16_t *outstanding)
{
   /* Native fragment references318/433 use the same typed IMGWCI layout.
    * Gallium's lowered hull program retains the TCS stage, with its I/O in
    * memory; typed image writes use this same non-pixel staging protocol.
    * Preserve the staging/operand checks below; fragment helper suppression
    * and late depth feedback are supplied by the lowered, linked program. */
   const bool admitted_stage =
      stage == MESA_SHADER_COMPUTE || stage == MESA_SHADER_VERTEX ||
      stage == MESA_SHADER_TESS_CTRL || stage == MESA_SHADER_FRAGMENT;
   const bool array = I->dim == AGX_DIM_2D_ARRAY;
   const bool three_d = I->dim == AGX_DIM_3D;
   const unsigned coords = (array || three_d) ? 3 : 2;
   const bool one = I->format == AGX_FORMAT_I32 && I->dim == AGX_DIM_2D;
   const bool float_data = I->format == AGX_FORMAT_U8NORM;
   const bool typed = I->format == AGX_FORMAT_I32 ||
                      I->format == AGX_FORMAT_I8 || float_data;
   const unsigned components = one ? 1 : 4;
   const unsigned bank_span = 16;

   const bool implicit_heap = agx3_tex_imm(I->src[3], 0);
   const bool uniform_heap = agx3_tex_plain_uniform64(I->src[3]) &&
                             !(I->src[3].value & 3);
   const agx_index index = I->src[4];
   const bool immediate_index =
      index.type == AGX_INDEX_IMMEDIATE && !index.memory &&
      !index.abs && !index.neg && !index.kill && !index.cache &&
      !index.discard && index.value <= AGX3_IMAGE_WRITE_G18G_MAX_INDEX;
   const bool register_index =
      agx3_tex_plain_reg(index) && index.size == AGX_SIZE_32 &&
      agx_channels(index) == 1 && !(index.value & 1) && index.value <= 126 &&
      !(index.cache && (index.discard || index.kill));
   const bool uniform_index =
      index.type == AGX_INDEX_UNIFORM && index.size == AGX_SIZE_32 &&
      agx_channels(index) == 1 && !(index.value & 1) && index.value <= 252 &&
      !index.memory && !index.abs && !index.neg && !index.kill &&
      !index.cache && !index.discard;

   if (enc != AGX_ISA_ENCODING_AGX3C || !admitted_stage || !banks ||
       bank_floor + bank_span > bank_limit ||
       bank_limit > AGX3_STAGED_WRITE_LIMIT ||
       I->nr_dests != 0 || I->nr_srcs != 5 ||
       (I->dim != AGX_DIM_2D && !array && !three_d) || !typed ||
       (wait_mask & ~UINT16_C(0xff)) ||
       !agx3_tex_plain_reg(I->src[0]) || I->src[0].size != AGX_SIZE_32 ||
       agx_channels(I->src[0]) != 4 || (I->src[0].value & 1) ||
       !agx3_tex_plain_reg(I->src[1]) || I->src[1].size != AGX_SIZE_32 ||
       agx_channels(I->src[1]) != coords || (I->src[1].value & 1) ||
       !agx3_tex_imm(I->src[2], 0) || (!implicit_heap && !uniform_heap) ||
       (!immediate_index && !register_index && !uniform_index)) {
      if (agx_compiler_debug & (AGX_DBG_SHADERS | AGX_DBG_REFUSE))
         fprintf(stderr, "G18G imgwci shape refusal: stage=%u wait=0x%x "
                 "data=%u/%u/%u coord=%u/%u/%u dim=%u format=%u "
                 "image=%u:%u\n",
                 stage, wait_mask, I->src[0].value, I->src[0].size,
                 agx_channels(I->src[0]), I->src[1].value, I->src[1].size,
                 agx_channels(I->src[1]), I->dim, I->format,
                 I->src[4].type, I->src[4].value);
      return false;
   }

   agx3_staging_release(banks, *outstanding);

   /* The first COPY carries the write's dependency. A COPY takes one wait
    * bit, so a mask with more is retired first, as the G17G fetch path
    * does. An unrelated producer still outstanding on the write's slot is
    * retired too: the COPYs signal that slot and imgwci waits on it. The
    * first write of a program takes slot 1, as kick 19. */
   uint16_t copy_wait = wait_mask, retire = 0;
   if (util_bitcount(copy_wait) > 1) {
      retire |= copy_wait;
      copy_wait = 0;
   }
   const uint8_t slot =
      (uint8_t)(banks->next_write_slot % AGX3_IMAGE_WRITE_G18G_MAX_SLOT + 1);
   const uint16_t slot_bit = (uint16_t)(1u << (slot - 1));
   if (*outstanding & ~copy_wait & slot_bit)
      retire |= slot_bit;
   /* The write's own slot also names the bank it now holds. */
   agx3_staging_forget(banks, (uint16_t)(retire | slot_bit));

   uint16_t bank = bank_floor;
   bool placed = false;
   for (unsigned start = bank_floor;
        start + bank_span <= bank_limit && !placed;
        start += bank_span) {
      if (!agx3_staging_conflicts(banks, start, bank_span)) {
         bank = start;
         placed = true;
      }
   }
   if (!placed) {
      const uint16_t held = agx3_staging_conflicts(banks, bank, bank_span);
      retire |= held;
      agx3_staging_forget(banks, held);
   }

   const uint16_t x_half = bank + 2 * components;
   const struct agx3_image_write_g18g_desc d = {
      .x_half = x_half,
      .components = (uint8_t)components,
      .float_data = float_data,
      .dim = array     ? AGX3_IMAGE_WRITE_2D_ARRAY
             : three_d ? AGX3_IMAGE_WRITE_3D
                       : AGX3_IMAGE_WRITE_2D,
      .image_index = index.value,
      .slot = slot,
      .image_kind = immediate_index ? AGX3_TEX_OPERAND_IMMEDIATE :
                    register_index ? AGX3_TEX_OPERAND_REGISTER :
                                     AGX3_TEX_OPERAND_UNIFORM,
      .uniform_base = uniform_heap ? I->src[3].value : 0,
      .image_kill = register_index && (index.discard || index.kill),
   };

   /* Validation happens before the first append, so a refusal leaves the
    * output transaction untouched. Coordinates stay live: a later address
    * calculation may reuse them. */
   struct util_dynarray staged;
   util_dynarray_init(&staged, NULL);
   bool ok = true;
   for (unsigned c = 0; c < components && ok; ++c) {
      uint8_t bytes[AGX3_STAGE_MAX_LENGTH], len = 0;
      ok = agx3_stage_copy_source(enc, bank + 2 * c, I->src[0].value + 2 * c,
                                  I->src[0].discard, slot,
                                  c == 0 ? copy_wait : 0, bytes, &len) &&
           agx3_append(&staged, bytes, len);
   }
   for (unsigned c = 0; c < 2 + three_d && ok; ++c) {
      uint8_t bytes[AGX3_STAGE_MAX_LENGTH], len = 0;
      ok = agx3_stage_copy_source(enc, x_half + 2 * c, I->src[1].value + 2 * c,
                                  false, slot, 0, bytes, &len) &&
           agx3_append(&staged, bytes, len);
   }
   /* The layer is the coordinate vector's last read, and takes its liveness
    * as the array texel fetch's PACK_SMPIR does: every Apple witness
    * discards it there (f27 1). */
   if (ok && array) {
      uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
      ok = agx3_tex_stage_image_write_layer(enc, I->src[1].value + 4,
                                            x_half + 4, slot, 0,
                                            I->src[1].cache,
                                            I->src[1].discard, bytes, &len) &&
           agx3_append(&staged, bytes, len);
   }
   if (ok) {
      uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
      ok = agx3_tex_encode_image_write_g18g(enc, &d, bytes, &len) &&
           agx3_append(&staged, bytes, len);
   }
   if (!ok) {
      if (agx_compiler_debug & (AGX_DBG_SHADERS | AGX_DBG_REFUSE))
         fprintf(stderr, "G18G imgwci compose refusal: data=%u coord=%u "
                 "bank=%u slot=%u wait=0x%x dim=%u image=%u\n",
                 I->src[0].value, I->src[1].value, bank, slot, copy_wait,
                 I->dim, I->src[4].value);
      util_dynarray_fini(&staged);
      return false;
   }

   if (retire) {
      if (!agx3_retire_slots_noop(emission, enc, retire)) {
         util_dynarray_fini(&staged);
         return false;
      }
      *outstanding &= (uint16_t)~retire;
   }
   ok = agx3_append(emission, staged.data, staged.size) &&
        agx3_staging_hold(banks, bank, bank_span, slot);
   util_dynarray_fini(&staged);
   if (!ok)
      return false;
   banks->next_write_slot = slot % AGX3_IMAGE_WRITE_G18G_MAX_SLOT;
   *outstanding |= slot_bit;
   return true;
}

/* A 3D sample with a bias, a minimum LOD and a runtime offset. */
bool
agx3_is_staged_3d_bias_min(const agx_instr *I)
{
   return I->op == AGX_OPCODE_TEXTURE_SAMPLE && I->dim == AGX_DIM_3D &&
          I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN && I->offset &&
          !I->offset_xyz;
}

/* A cube sample with its own CEM scratch (agx3_reserve_cube_scratch)
 * uses it in place of the shared scratch above the allocation. */
bool
agx3_cube_own_scratch(const agx_instr *I)
{
   return I->op == AGX_OPCODE_TEXTURE_SAMPLE &&
          (I->dim == AGX_DIM_CUBE || I->dim == AGX_DIM_CUBE_ARRAY) &&
          !I->agx3_cube_scalar_coords && I->nr_srcs == 7 &&
          I->src[6].type == AGX_INDEX_REGISTER;
}

const uint16_t *
agx3_cube_scratch(const agx_instr *I, const uint16_t shared[3],
                  uint16_t own[3])
{
   if (!agx3_cube_own_scratch(I))
      return shared;
   own[0] = I->src[6].value;
   own[1] = I->src[6].value + 4;
   own[2] = I->src[6].value + 6;
   return own;
}

/* Native g18-gather-reference371: the same four components and integer
 * textures use a full four-lane 2D gather, through immediate descriptors.
 * More complex witnessed shapes still need their own staging integration. */
static bool
agx3_g17g_basic_gather(const agx_instr *I)
{
   return I->op == AGX_OPCODE_TEXTURE_SAMPLE && I->dim == AGX_DIM_2D &&
          I->gather != AGX_GATHER_NONE && I->nr_srcs == 6 &&
          I->src[3].type == AGX_INDEX_IMMEDIATE &&
          I->src[4].type == AGX_INDEX_IMMEDIATE &&
          I->lod_mode == AGX_LOD_MODE_AUTO_LOD && I->mask == 15 &&
          I->dest[0].size == AGX_SIZE_32 &&
          /* Native371/373 compare and immediate-offset gathers use the
           * existing ordinary 2D coordinate/reference publication. */
          (!I->offset || I->offset_xyz > 0) &&
          !I->sparse && !I->query_lod && !I->agx3_tex_coords_staged;
}

/* Native408 publishes plain and immediate-offset array layers separately
 * from the coordinates. Runtime offsets use their merged layer word below. */
static bool
agx3_g17g_array_gather(const agx_instr *I)
{
   return I->op == AGX_OPCODE_TEXTURE_SAMPLE && I->dim == AGX_DIM_2D_ARRAY &&
          I->gather != AGX_GATHER_NONE && I->nr_srcs == 6 &&
          I->src[3].type == AGX_INDEX_IMMEDIATE &&
          I->src[4].type == AGX_INDEX_IMMEDIATE &&
          I->lod_mode == AGX_LOD_MODE_AUTO_LOD && I->mask == 15 &&
          I->dest[0].size == AGX_SIZE_32 &&
          (!I->offset || I->offset_xyz > 0) &&
          !I->sparse && !I->query_lod && !I->agx3_tex_coords_staged;
}

static bool
agx3_g17g_runtime_gather(const agx_instr *I)
{
   return I->op == AGX_OPCODE_TEXTURE_SAMPLE &&
          (I->dim == AGX_DIM_2D || I->dim == AGX_DIM_2D_ARRAY) &&
          I->gather != AGX_GATHER_NONE && I->offset && !I->offset_xyz &&
          I->nr_srcs == 6 &&
          I->src[3].type == AGX_INDEX_IMMEDIATE &&
          I->src[4].type == AGX_INDEX_IMMEDIATE &&
          I->lod_mode == AGX_LOD_MODE_AUTO_LOD && I->mask == 15 &&
          I->dest[0].size == AGX_SIZE_32 &&
          !I->sparse && !I->query_lod && !I->agx3_tex_coords_staged;
}

/* Native685r4 independently compiles the 2D float, integer, raw-depth and
 * comparison gathers in every stage, with constant and runtime offsets.
 * The indirect sampler uses the same 64-bit descriptor operand as Native653;
 * only its native register/uniform selector differs from the fixed sampler.
 * Admission of this shape in the stream packer remains G18-specific. */
static bool
agx3_g18g_indirect_gather(const agx_instr *I)
{
   return I->op == AGX_OPCODE_TEXTURE_SAMPLE && I->dim == AGX_DIM_2D &&
          I->gather != AGX_GATHER_NONE && I->nr_srcs == 6 &&
          I->nr_dests == 1 && I->src[3].type == AGX_INDEX_IMMEDIATE &&
          (I->src[4].type == AGX_INDEX_REGISTER ||
           I->src[4].type == AGX_INDEX_UNIFORM) &&
          I->src[4].size == AGX_SIZE_64 && agx_channels(I->src[4]) == 1 &&
          I->lod_mode == AGX_LOD_MODE_AUTO_LOD && I->mask == 15 &&
          I->dest[0].size == AGX_SIZE_32 &&
          !I->sparse && !I->query_lod && !I->agx3_tex_coords_staged;
}

/* Native413 covers projected cube and cube-array gathers in all three
 * stages, over immediate descriptors and without LOD/offset modifiers. */
bool
agx3_g17g_cube_gather(const agx_instr *I)
{
   return I->op == AGX_OPCODE_TEXTURE_SAMPLE &&
          (I->dim == AGX_DIM_CUBE || I->dim == AGX_DIM_CUBE_ARRAY) &&
          I->gather != AGX_GATHER_NONE && I->nr_srcs == 6 &&
          I->src[3].type == AGX_INDEX_IMMEDIATE &&
          I->src[4].type == AGX_INDEX_IMMEDIATE &&
          I->lod_mode == AGX_LOD_MODE_AUTO_LOD && I->mask == 15 &&
          I->dest[0].size == AGX_SIZE_32 && !I->offset &&
          !I->sparse && !I->query_lod && !I->agx3_tex_coords_staged;
}

static bool
agx3_g17g_plain_2d_sample(const agx_instr *I)
{
   return I->op == AGX_OPCODE_TEXTURE_SAMPLE && I->dim == AGX_DIM_2D &&
          I->nr_srcs == 6 && I->src[3].type == AGX_INDEX_IMMEDIATE &&
          I->src[4].type == AGX_INDEX_IMMEDIATE &&
          I->gather == AGX_GATHER_NONE && !I->query_lod && !I->sparse &&
          !I->agx3_tex_coords_staged &&
          I->lod_mode != AGX_LOD_MODE_LOD_GRAD &&
          I->lod_mode != AGX_LOD_MODE_LOD_GRAD_MIN &&
          I->lod_mode != AGX_LOD_MODE_AUTO_LOD_BIAS_MIN;
}

/* Whether a G17G plain 2D sample reads its operands from a staging block
 * (coordinates at +0/+2, comparison reference at +4, level/bias word at +6)
 * rather than naming its coordinate registers directly. */
bool
agx3_g17g_staged_block(const agx_instr *I, bool g18g)
{
   /* Repeated gathers republish coordinates before asynchronous SMPs.
    * Give each outstanding consumer its own block, as for comparisons. */
   return agx3_g17g_basic_gather(I) ||
          (g18g && agx3_g18g_indirect_gather(I) &&
           (!I->offset || I->offset_xyz > 0)) ||
          (agx3_g17g_plain_2d_sample(I) &&
           (I->shadow || I->lod_mode == AGX_LOD_MODE_LOD_MIN ||
            (I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS &&
             I->src[1].type != AGX_INDEX_IMMEDIATE)));
}

/* Whether the staged block carries a per-lane level or bias word. */
bool
agx3_g17g_register_lod(const agx_instr *I)
{
   return (I->lod_mode == AGX_LOD_MODE_LOD_MIN ||
           I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS) &&
          I->src[1].type != AGX_INDEX_IMMEDIATE;
}

static bool
agx3_g18g_binding_sample(const agx_instr *I)
{
   /* Native653 covers all nine immediate/register/uniform descriptor pairs
    * for ordinary 2D, array and 3D samples, with full or scalar results.
    * Register levels work in every execution stage; implicit/bias sampling
    * is fragment-only, as checked by the caller. Keep the existing gates
    * for offsets, gradients and sparse operations. Native809 adds all nine
    * descriptor pairs for full-vector cube colour and scalar depth compares,
    * with explicit levels or implicit fragment LOD. Native816 adds fragment
    * register bias for the same descriptor pairs and masks. */
   const bool cube = I->dim == AGX_DIM_CUBE || I->dim == AGX_DIM_CUBE_ARRAY;
   const bool comparison_dim = I->dim == AGX_DIM_2D ||
                               I->dim == AGX_DIM_2D_ARRAY || cube;
   return I->op == AGX_OPCODE_TEXTURE_SAMPLE &&
          (I->dim == AGX_DIM_2D || I->dim == AGX_DIM_2D_ARRAY ||
            I->dim == AGX_DIM_3D || cube) &&
           (!I->shadow || comparison_dim) &&
          I->nr_srcs == 6 && I->nr_dests == 1 &&
          ((I->lod_mode == AGX_LOD_MODE_AUTO_LOD &&
            agx3_tex_imm(I->src[1], 0)) ||
           ((I->lod_mode == AGX_LOD_MODE_LOD_MIN ||
              (I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS)) &&
            agx3_tex_plain_reg(I->src[1]) &&
            I->src[1].size == AGX_SIZE_16 && agx_channels(I->src[1]) == 1)) &&
           I->dest[0].size == AGX_SIZE_32 &&
           (I->shadow ? I->mask == 1 : cube ? I->mask == 15
                                             : (I->mask == 1 || I->mask == 15)) &&
           !I->offset && !I->sparse && !I->query_lod &&
          I->gather == AGX_GATHER_NONE && !I->agx3_tex_coords_staged &&
          !agx_texture_resource(I);
}

static bool
agx3_g18g_colour_sample(const agx_instr *I)
{
   /* Native607 grounds explicit levels in all stages; native610 adds
    * implicit/register-bias fragment sampling. Native615 extends those
    * profiles to cube arrays. The caller gates stages. */
   return I->op == AGX_OPCODE_TEXTURE_SAMPLE &&
          (I->dim == AGX_DIM_3D || I->dim == AGX_DIM_CUBE ||
           I->dim == AGX_DIM_CUBE_ARRAY) &&
          I->nr_srcs == 6 && I->nr_dests == 1 &&
          I->src[3].type == AGX_INDEX_IMMEDIATE &&
          I->src[4].type == AGX_INDEX_IMMEDIATE &&
          (I->lod_mode == AGX_LOD_MODE_LOD_MIN ||
           I->lod_mode == AGX_LOD_MODE_AUTO_LOD ||
           I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS) &&
          (agx3_tex_imm(I->src[1], 0) ||
           (agx3_tex_plain_reg(I->src[1]) &&
            I->src[1].size == AGX_SIZE_16 && agx_channels(I->src[1]) == 1)) &&
          ((I->dest[0].size == AGX_SIZE_32 &&
            (I->mask == 1 || I->mask == 15)) ||
           /* Native917: RGBA half cube results, with a register level
            * or fragment register bias. Keep the LOD operand half-sized. */
           (I->dest[0].size == AGX_SIZE_16 && I->dim == AGX_DIM_CUBE &&
            I->mask == 15 && agx3_tex_plain_reg(I->src[1]) &&
            (I->lod_mode == AGX_LOD_MODE_LOD_MIN ||
             I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS))) &&
          !I->shadow && !I->offset && !I->sparse && !I->query_lod &&
          I->gather == AGX_GATHER_NONE && !I->agx3_tex_coords_staged &&
          !agx_texture_resource(I);
}

static bool
agx3_emit_texture_sample_g17g(struct util_dynarray *emission,
                         enum agx_isa_encoding enc, const agx_instr *I,
                         uint8_t slot, uint16_t wait_mask, bool final,
                         bool sample_group_nonterminal, bool sample_group_end,
                         bool pixel_entry, uint32_t *pixel_entry_offset,
                         uint8_t native_coord_slot,
                         const uint16_t cem_scratch[3], bool vertex_stage,
                         bool compute_stage, uint16_t array_scratch,
                         uint16_t lod_relay, bool g18g, const char **why)
{
   const bool explicit_lod = I->lod_mode == AGX_LOD_MODE_LOD_MIN;
   const bool bias_min = I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN;
   const bool gradient_min = I->lod_mode == AGX_LOD_MODE_LOD_GRAD_MIN;
   const bool min_lod = bias_min || gradient_min;
   /* An implicit-LOD sample with a constant bias: the same 12-bit x64
    * immediate operand as an explicit level, selected by f139 = 1 instead
    * of 2, and keeping the implicit-LOD marker.
    *
    * A zero bias is not encoded at all: Apple's own compiler folds it into
    * the plain implicit form, and this packer follows it, so the operand
    * stays the immediate zero the implicit form already requires. Minifloat
    * -0.0 folds the same way. */
   const bool zero_bias = I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS &&
                          I->src[1].type == AGX_INDEX_IMMEDIATE &&
                          agx_minifloat_decode(I->src[1].value) == 0.0f;
   const bool biased_lod = I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS &&
                           I->src[1].type == AGX_INDEX_IMMEDIATE && !zero_bias;
   /* The same sample with a per-lane bias instead: PACK_SMPIR publishes the
    * scalar exactly as it does for a per-lane level, and only f139 moves
    * (3 rather than 4). */
   const bool register_bias = I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS &&
                              I->src[1].type != AGX_INDEX_IMMEDIATE;
   const bool implicit_lod = I->lod_mode == AGX_LOD_MODE_AUTO_LOD || zero_bias;
   /* Explicit gradients publish four scalars into architecture-specific
    * banks. G18's X coordinate is at 16; older encodings name bank 0. */
   const bool gradient_lod = I->lod_mode == AGX_LOD_MODE_LOD_GRAD || gradient_min;
   /* Gather uses the implicit-LOD selector for 2D, array and cube forms,
    * including compute where it reads level zero. Map the IR component
    * values 1/3/5/7 explicitly to the native two-bit selector. */
   const bool gather = I->gather != AGX_GATHER_NONE;
   const bool shadow = I->shadow;
   const bool query_lod = I->query_lod;
   const bool offset = I->offset;
   enum agx3_tex_gather gather_component = AGX3_TEX_GATHER_NONE;
   switch (I->gather) {
   case AGX_GATHER_NONE: break;
   case AGX_GATHER_R: gather_component = AGX3_TEX_GATHER_R; break;
   case AGX_GATHER_G: gather_component = AGX3_TEX_GATHER_G; break;
   case AGX_GATHER_B: gather_component = AGX3_TEX_GATHER_B; break;
   case AGX_GATHER_A: gather_component = AGX3_TEX_GATHER_A; break;
   default: return agx3_tex_decline(I, __FILE__, __LINE__);
   }
   const bool immediate_lod =
      (explicit_lod || biased_lod) && I->src[1].type == AGX_INDEX_IMMEDIATE;
   const float immediate_level = immediate_lod ?
      agx_minifloat_decode(I->src[1].value) : 0.0f;
   const float immediate_level_x64 = immediate_level * 64.0f;
   /* f29 is a *signed* 12-bit LOD in 1/64 steps. Sweeping Apple's own
    * `t.sample(s, uv, bias(k))` over k = +-{1/64, 0.25, 0.5, 1, 1.5, 2, 4,
    * 31, 32, 33, 63, 64} gives f29 = k * 64 in two's complement -- bias(-1.0)
    * is 4032, bias(-32.0) is 2048 -- saturating at 2047 (+31.984375) and 2048
    * (-32). A minifloat immediate spans [-31, 31], so it always lands inside
    * that range, but the sign has to survive into the field. */
   const bool integral_x64 =
      immediate_level_x64 >= -2048.0f && immediate_level_x64 <= 2047.0f &&
      immediate_level_x64 == (float)(int16_t)immediate_level_x64;
   /* An explicit level is unsigned; only a bias may be negative. */
   const bool valid_immediate_lod =
      immediate_lod && immediate_level >= 0.0f && integral_x64;
   const bool valid_immediate_bias = immediate_lod && integral_x64;
   const uint16_t immediate_x64_field =
      (uint16_t)(((int)immediate_level_x64) & 0xfff);
   const bool cube_array = I->dim == AGX_DIM_CUBE_ARRAY;
   const bool array = I->dim == AGX_DIM_2D_ARRAY || cube_array;
   const bool three_d = I->dim == AGX_DIM_3D;
   const bool cube = I->dim == AGX_DIM_CUBE || cube_array;
   const bool shadow_level_zero = shadow && !cube_array &&
      valid_immediate_lod && immediate_level == 0.0f;
   /* A per-lane level and a per-lane bias share one staging form, so they
    * share one operand test: a live single-channel 16-bit register. */
   const bool scalar_lod_reg = agx3_tex_plain_reg(I->src[1]) &&
      I->src[1].size == AGX_SIZE_16 && agx_channels(I->src[1]) == 1;
   const bool bias_min_operand = agx3_tex_plain_reg(I->src[1]) &&
      I->src[1].size == AGX_SIZE_32 && agx_channels(I->src[1]) == 1 &&
      (I->src[1].value & 1) == 0 && I->src[1].value < 127;
   /* A 2D gradient is four 32-bit scalars in one aligned run, which the four
    * staging copies name two halves apart. */
   /* A 2D gradient is four 32-bit scalars in one aligned run; a 3D one is
    * six, and agx_emit_tex()'s collect already lays both out in the
    * interleaved order the staging banks want. */
   const unsigned gradient_channels = three_d ? 6u : 4u;
   const bool gradient_operand = agx3_tex_plain_reg(I->src[1]) &&
      I->src[1].size == AGX_SIZE_32 &&
      agx_channels(I->src[1]) == gradient_channels + gradient_min &&
      (I->src[1].value & 1) == 0 &&
       I->src[1].value + (2 * (gradient_channels + gradient_min)) - 2 < 128;
   /* G17G's fragment corpus (see agx3_tex_core.c sample_ok) grounds the
    * plain 2D immediate/per-lane level and bias, the immediate offset and
    * the depth comparison with AGX3B's field values; arrays, cubes, 3D,
    * gathers, LOD queries, minimum LOD and gradients stay AGX3B-only. */
   /* gpu/m5-shadow-array-cube-reference.metal adds the depth2d_array
    * comparisons (implicit, register bias and register level, fragment and
    * vertex) and the vertex-stage 2D register level. The same reference's
    * depthcube and depthcube_array comparisons ground the cube shapes: the
    * implicit LOD, a per-lane bias and a per-lane level, over immediate
    * descriptor operands, with the coordinate staging the AGX3C cube
    * publication already encodes. A cube sample without a comparison has no
    * witness and stays refused. */
   const bool gpu1_gather = enc == AGX_ISA_ENCODING_AGX3C &&
      (agx3_g17g_basic_gather(I) || agx3_g17g_runtime_gather(I) ||
       agx3_g17g_array_gather(I) || agx3_g17g_cube_gather(I) ||
       (g18g && agx3_g18g_indirect_gather(I)));
   const bool gpu1_plain_2d = enc == AGX_ISA_ENCODING_AGX3C &&
      !three_d && !cube && !gather && !query_lod && !min_lod &&
      I->nr_srcs == 6 &&
      I->src[3].type == AGX_INDEX_IMMEDIATE &&
      I->src[4].type == AGX_INDEX_IMMEDIATE;
   const bool gpu1_cube = (gpu1_gather && cube) || (enc == AGX_ISA_ENCODING_AGX3C &&
      cube && shadow && !three_d && !gather && !query_lod && !min_lod &&
      I->nr_srcs == 6 &&
      I->src[3].type == AGX_INDEX_IMMEDIATE &&
      I->src[4].type == AGX_INDEX_IMMEDIATE);
   const bool gpu1_array_register_lod =
      (gpu1_plain_2d && array && !cube) &&
      ((explicit_lod && !immediate_lod && scalar_lod_reg) ||
       (register_bias && scalar_lod_reg));
   /* Native g18-gradient-reference covers 2D and array gradients in vertex
    * and fragment stages, including signed immediate offsets. Other gradient
    * shapes retain their existing gates. */
   const bool g18g_gradient = g18g && gpu1_plain_2d && gradient_lod &&
                             !shadow && !min_lod && !native_coord_slot;
   const bool g18g_binding = g18g && enc == AGX_ISA_ENCODING_AGX3C &&
      agx3_g18g_binding_sample(I) &&
      ((!vertex_stage && !compute_stage) || explicit_lod);
   const bool g18g_colour_level = g18g_binding || (g18g &&
      enc == AGX_ISA_ENCODING_AGX3C && agx3_g18g_colour_sample(I) &&
      ((!vertex_stage && !compute_stage) || explicit_lod));
   /* Experimental combination: unbiased query selector is proven by949 (on
    * the M6); register-bias semantics still require independent hardware
    * readback. G18G only: no M5 run grounds this form. */
   const bool gpu1_query_lod = g18g && enc == AGX_ISA_ENCODING_AGX3C && query_lod &&
      I->nr_srcs == 6 && I->src[3].type == AGX_INDEX_IMMEDIATE &&
      I->src[4].type == AGX_INDEX_IMMEDIATE && !array && !cube &&
      !vertex_stage && !compute_stage && !min_lod && !I->sparse &&
      !native_coord_slot && (implicit_lod || register_bias);
   const bool lod_enc_ok =
      enc == AGX_ISA_ENCODING_AGX3B || gpu1_plain_2d || gpu1_cube ||
      g18g_colour_level || gpu1_query_lod;
   const bool valid_lod = bias_min ?
      (enc == AGX_ISA_ENCODING_AGX3B && bias_min_operand) : explicit_lod ?
      ((lod_enc_ok && (valid_immediate_lod || scalar_lod_reg)) ||
       (enc == AGX_ISA_ENCODING_AGX3C && valid_immediate_lod &&
        immediate_level_x64 == 0.0f)) :
      biased_lod
         ? (lod_enc_ok && valid_immediate_bias &&
            immediate_level_x64 != 0.0f)
         : register_bias
         ? (lod_enc_ok && scalar_lod_reg)
         : gradient_lod
         ? ((enc == AGX_ISA_ENCODING_AGX3B || g18g_gradient) && gradient_operand)
         : (implicit_lod &&
            (zero_bias || agx3_tex_imm(I->src[1], 0)));
   const bool texture_immediate =
      I->nr_srcs == 6 && I->src[3].type == AGX_INDEX_IMMEDIATE;
   const bool sampler_immediate =
      I->nr_srcs == 6 && I->src[4].type == AGX_INDEX_IMMEDIATE;
   const bool immediate = texture_immediate && sampler_immediate;
   /* agx_emit_tex() publishes the packed offset on the instruction rather
    * than through src[5], because a comparison sample needs that operand for
    * its reference value. src[5] still carries the immediate when there is no
    * comparator, and the two must agree. */
   const bool immediate_offset = offset && I->offset_xyz > 0 &&
      (shadow || (I->src[5].type == AGX_INDEX_IMMEDIATE &&
                  I->src[5].value == I->offset_xyz &&
                  !I->src[5].memory && !I->src[5].abs && !I->src[5].neg &&
                  !I->src[5].cache && !I->src[5].discard && !I->src[5].kill));
   const bool register_offset = offset && !I->offset_xyz && gather &&
      agx3_tex_plain_reg(I->src[5]) && I->src[5].size == AGX_SIZE_32 &&
      agx_channels(I->src[5]) == (shadow ? 2 : 1) &&
      !(I->src[5].value & 1) && I->src[5].value + (shadow ? 2 : 0) <= 126;
   /* AGX3B coordinates are copied to the fixed staging bank before SMP. COPY
    * and CEM each name a seven-bit source half-register, so the entire vector
    * must end below half 128. AGX3C retains its independently recovered direct
    * coordinate bound. */
   const unsigned max_coord = enc == AGX_ISA_ENCODING_AGX3B
                                 ? 128 - (2 * agx_channels(I->src[0]))
                                 : 60;
   /* A sample outside the grounded profile is worth naming the reason, the
    * way the Gallium draw profile does: this is a thirty-term conjunction
    * and bisecting it by inspection is how an afternoon disappears. */
   const char *outside = NULL;
#define M5_SMP_OUTSIDE(cond, reason)                                           \
   do {                                                                        \
      if (!outside && (cond))                                                  \
         outside = (reason);                                                    \
   } while (0)

   M5_SMP_OUTSIDE(I->nr_dests != 1 || I->nr_srcs != 6, "operand count");
   M5_SMP_OUTSIDE(!agx3_tex_plain_reg(I->dest[0]) ||
                     (I->dest[0].size != AGX_SIZE_16 &&
                      I->dest[0].size != AGX_SIZE_32) ||
                     agx_channels(I->dest[0]) != 4 ||
                     I->dest[0].value >= 128 ||
                     (I->dest[0].size == AGX_SIZE_32 &&
                      (I->dest[0].value & 1)),
                  "destination shape");
   M5_SMP_OUTSIDE(!agx3_tex_plain_reg(I->src[0]) ||
                     I->src[0].size != AGX_SIZE_32,
                  "coordinate is not a plain 32-bit register");
   M5_SMP_OUTSIDE(agx_channels(I->src[0]) !=
                     (query_lod ? (three_d || cube ? 3 : 2)
                      : cube_array ? 4
                      : (array || three_d || cube) ? 3
                                                   : 2),
                  "coordinate channel count");
   M5_SMP_OUTSIDE((I->src[0].value & 3) || I->src[0].value > max_coord,
                  "coordinate register out of the staging bound");
   M5_SMP_OUTSIDE(!valid_lod, "level-of-detail shape");
   M5_SMP_OUTSIDE(!agx3_tex_imm(I->src[2], 0), "source 2 is not immediate 0");
   M5_SMP_OUTSIDE(texture_immediate
                     ? I->src[3].value > 15
                     : ((I->src[3].type != AGX_INDEX_REGISTER &&
                         I->src[3].type != AGX_INDEX_UNIFORM) ||
                        I->src[3].size != AGX_SIZE_64 ||
                        agx_channels(I->src[3]) != 1 ||
                        (I->src[3].value & 1)),
                  "texture descriptor operand");
   M5_SMP_OUTSIDE(sampler_immediate
                     ? I->src[4].value > 15
                     : ((I->src[4].type != AGX_INDEX_REGISTER &&
                         I->src[4].type != AGX_INDEX_UNIFORM) ||
                        I->src[4].size != AGX_SIZE_64 ||
                        agx_channels(I->src[4]) != 1 ||
                        (I->src[4].value & 1)),
                  "sampler descriptor operand");
   M5_SMP_OUTSIDE(I->src[3].memory || I->src[3].abs || I->src[3].neg ||
                     I->src[4].memory || I->src[4].abs || I->src[4].neg,
                  "descriptor operand modifier");
   M5_SMP_OUTSIDE(shadow && (!agx3_tex_plain_reg(I->src[5]) ||
                             I->src[5].size != AGX_SIZE_32 ||
                             I->src[5].value > 126),
                  "comparison reference is not a plain 32-bit register");
   M5_SMP_OUTSIDE(shadow && agx_channels(I->src[5]) !=
                              (register_offset ? 2 : 1),
                  "comparison reference channel count");
   M5_SMP_OUTSIDE(shadow && offset && !immediate_offset && !register_offset,
                  "comparison with an unrepresentable offset");
   M5_SMP_OUTSIDE(!shadow && offset && !immediate_offset && !register_offset,
                  "offset is neither immediate nor a register");
   M5_SMP_OUTSIDE(!shadow && !offset && !agx_is_null(I->src[5]),
                  "source 5 is not null");
   M5_SMP_OUTSIDE(I->dim != AGX_DIM_2D && I->dim != AGX_DIM_2D_ARRAY &&
                     !three_d && !cube,
                  "dimension");
   /* gpu/m5-shadow-array-offset-reference.metal adds the depth2d_array
    * comparisons with a constant texel offset, in the implicit, per-lane
    * bias and per-lane level shapes, so an offset array is no longer the
    * unwitnessed case it was. */
   M5_SMP_OUTSIDE(array && enc != AGX_ISA_ENCODING_AGX3B &&
                     !gpu1_gather && !g18g_colour_level &&
                     !(enc == AGX_ISA_ENCODING_AGX3C && valid_immediate_lod &&
                       immediate_level_x64 == 0.0f) &&
                     !(gpu1_cube && (implicit_lod || scalar_lod_reg)) &&
                     !(gpu1_plain_2d && (!offset || immediate_offset) &&
                       (implicit_lod || gpu1_array_register_lod || g18g_gradient)),
                  "array on this encoding");
   /* f76 = 3 is orthogonal to f139; only the gradient is held back,
    * because gradient3d stages six scalars and that layout has no
    * witness. See agx3_tex_core.c for the five-row table. */
   M5_SMP_OUTSIDE(three_d && ((enc != AGX_ISA_ENCODING_AGX3B &&
                               !g18g_colour_level && !gpu1_query_lod) ||
                              (!valid_immediate_lod && !implicit_lod &&
                               !biased_lod && !register_bias &&
                               !explicit_lod && !gradient_lod && !bias_min)),
                  "3D on this encoding");
   M5_SMP_OUTSIDE(cube && ((enc != AGX_ISA_ENCODING_AGX3B && !gpu1_cube &&
                            !g18g_colour_level) ||
                           (!implicit_lod && !explicit_lod && !register_bias)),
                  "cube on this encoding");
   M5_SMP_OUTSIDE(I->mask == 0 || (I->mask & ~15u), "result mask");
   M5_SMP_OUTSIDE(I->sparse, "sparse residency");
   M5_SMP_OUTSIDE(min_lod && (array || cube || vertex_stage ||
                              (three_d && offset)),
                  "minimum level-of-detail shape");
   M5_SMP_OUTSIDE(query_lod && ((enc != AGX_ISA_ENCODING_AGX3B && !gpu1_query_lod) ||
                                (!implicit_lod && !register_bias) ||
                                shadow || offset || gather ||
                                I->dest[0].size != AGX_SIZE_32 ||
                                (I->mask & ~3u)),
                  "level-of-detail query shape");
   /* Offset arrays use the native staged layer. 3D carries the third
    * signed nibble in the same immediate, while cubes have no offset. */
   M5_SMP_OUTSIDE(offset && (cube || (enc != AGX_ISA_ENCODING_AGX3B &&
                                      !(((gpu1_plain_2d || gpu1_gather) &&
                                         immediate_offset) ||
                                        (gpu1_gather && register_offset)))),
                  "texel offset on this encoding");
   /* A depth comparison is orthogonal to the LOD shape on a plain 2D
    * sample. The array and cube comparisons use their own packed layer /
    * CEM operands, with implicit LOD or the witnessed explicit zero. */
   M5_SMP_OUTSIDE(shadow && (three_d ||
                             (enc != AGX_ISA_ENCODING_AGX3B &&
                              !gpu1_plain_2d && !gpu1_cube &&
                              !gpu1_gather && !g18g_binding) ||
                             I->mask != (gather ? 15u : 1u)),
                  "comparison on this encoding");
   M5_SMP_OUTSIDE(shadow && (cube || (array && !offset && !vertex_stage)) &&
                     !gpu1_array_register_lod &&
                     !g18g_binding &&
                     !(gpu1_cube && (scalar_lod_reg || implicit_lod)) &&
                     !implicit_lod && !shadow_level_zero,
                  "array/cube comparison level-of-detail shape");
   M5_SMP_OUTSIDE(gather && (I->lod_mode != AGX_LOD_MODE_AUTO_LOD || three_d ||
                             (enc != AGX_ISA_ENCODING_AGX3B && !gpu1_gather)),
                  "gather shape");
#undef M5_SMP_OUTSIDE

   if (outside) {
      if (why)
         *why = outside;
      return agx3_tex_decline(I, __FILE__, __LINE__);
   }


   /* The caller lowers additional required dependencies before staging. */
   if (wait_mask & (wait_mask - 1))
      return agx3_tex_decline(I, __FILE__, __LINE__);

   const bool register_operands = !immediate &&
      (I->src[3].type == AGX_INDEX_REGISTER ||
       I->src[4].type == AGX_INDEX_REGISTER);

   /* G17P's immediate cube corpus uses Apple's fixed r0/r7/r2 CEM
    * destinations, so the coordinate vector has to stay clear of them.
    * Every other cube sample -- G17P's dynamic descriptor forms and all of
    * G17G -- writes the allocator-owned bank reserved above instead, whose
    * base is always a live even half above the shader's registers. */
   const bool fixed_cem = immediate && enc == AGX_ISA_ENCODING_AGX3B;
   if (cube && fixed_cem) {
      const unsigned coord_first = I->src[0].value;
      const unsigned coord_last = coord_first + (cube_array && !query_lod ? 8 : 6);
      if (coord_first < 2 || (coord_first < 16 && coord_last > 14))
         return agx3_tex_decline(I, __FILE__, __LINE__);
   } else if (cube &&
              (cem_scratch == NULL || cem_scratch[0] == 0 ||
               (cem_scratch[0] & 1) || cem_scratch[2] > 190)) {
      return agx3_tex_decline(I, __FILE__, __LINE__);
   }

   /* Native register-indexed array and cube samples publish their coordinate /
    * face operand bank on slot 6 and wait slot 6 in SMP, independently of the
    * sample result slot. The dynamic cube witness groups both descriptor
    * loads and face publication on that transaction. Uniform/immediate
    * operands use the ordinary result slot. */
   const bool native_staged_coords = native_coord_slot != 0;
   const uint8_t stage_slot = native_staged_coords
                                 ? native_coord_slot
                                 : enc == AGX_ISA_ENCODING_AGX3B &&
                                      (array || cube) && register_operands ? 6
                                                                        : slot;
   /* Every native G17P witness using AGX3B names coordinate bank 0 in SMP,
    * including ordinary implicit 2D samples whose materialized coordinates
    * live in unrelated GPRs.  For example, the attachment-only f_retain_coords
    * witness copies r4/r5 to staging[0]/[2], reports ShaderInfo D=8, and
    * samples bank 0.  The staging address is an ABI location, not the source
    * GPR number.  AGX3c retains its independently decoded coordinate field. */
   const uint16_t coord_bank = (enc == AGX_ISA_ENCODING_AGX3B ||
                                native_staged_coords || g18g_binding || explicit_lod ||
                                register_bias || array || three_d || cube ||
                                gradient_lod || shadow)
                                  ? 0
                                  : I->src[0].value;
   /* G17G gives every in-flight staged sample its own eight-half block,
    * rotated by scoreboard slot. Apple's three-comparison fragment
    * (gpu/m5-shadow-2d-reference.metal) stages one coordinate pair three
    * times, into halves 0/2, 8/10 and 16/18, its three references into 4,
    * 12 and 20 and (bias variant) the bias word into 6, 14 and 22, then
    * issues the samples on slots 1, 2, 3 naming f19 = 0, 8, 16, each
    * waiting on its own slot. Two samples sharing one block would let the
    * later staging overwrite the earlier operands before its SMP consumed
    * them: here the three references would all compare against the last.
    * Apple shares a block only when its contents are identical (the
    * two-texture colour fragment samples one staged pair twice); a slot is
    * never shared between in-flight samples, so it keys the block. */
   const bool g17g_block =
      enc == AGX_ISA_ENCODING_AGX3C && agx3_g17g_staged_block(I, g18g);
   const uint16_t base = g17g_block ? 8u * (stage_slot - 1) : 0;
   /* G18G stages an array sample in its own block layout: X at 4, Y at 0
    * (SMP f56 = 0) and the layer word at 2 (agx3_tex_stage_array_layer_g18g).
    * The G18G transcoder moves SMP's f19 to 4 but no staging write, so the
    * G17G layout (X 0, Y 2, layer 6) would sample the wrong halves. Apple's
    * G18G witnesses cover the implicit, per-lane level and per-lane bias
    * samples of a texture2d_array with a per-lane layer, and reads on later
    * slots keep the layer word two halves above Y (m6-fetch-array-two); an
    * array sample drains the in-flight staged samples and owns bank 0, as
    * on G17G. Native G18 depth2d_array references also use this layout,
    * including per-lane level/bias and immediate texel offsets; their
    * comparison reference is explicitly named at half 6. */
   const bool g18g_array = g18g && array && !cube &&
                           (!offset || (shadow && immediate_offset)) &&
                           !gather && !query_lod && !min_lod && !gradient_lod &&
                           !three_d && !native_staged_coords && !g17g_block;
   const bool runtime_gather = gpu1_gather && register_offset;
   const bool g18g_runtime_gather = g18g && runtime_gather;
   const bool fixed_gather = runtime_gather || (gpu1_gather && (array || cube));
   const bool g18g_fixed_gather = g18g && fixed_gather;
   /* Native413: depth fragment uses X/reference6/4; all other gathers
    * use X4, with reference6 when comparing. Y is always0 on G18.
    * A non-array cube comparison that is not a gather keeps X/reference at
    * 6/4 in every stage: those are the halves its G18 SMP names
    * (agx3_transcode.c sets f19 = 6 and f58 = 4 for every non-array cube
    * comparison), as in native809's fragment depthcube profiles and the
    * pre-810 encoding that passed the cube shadow-LOD cases on hardware
    * (242/256). Only colour cubes and cube arrays use the X4 layout here.
    * Since 810 the g18g_binding profile also covers cube comparisons, so
    * g18g_colour_level alone would stage X at 4 and the reference at 6
    * while SMP still read X at 6 and the reference at 4. */
   const bool cube_x4 = gpu1_gather && cube
                          ? (!shadow || vertex_stage || compute_stage)
                          : cube_array || (g18g_colour_level && cube && !shadow);
   if (g18g && array && !cube && !g18g_array && !g18g_gradient &&
       !g18g_fixed_gather) {
      if (why)
         *why = "G18G array sample outside the witnessed staged block";
      return agx3_tex_decline(I, __FILE__, __LINE__);
   }
   const uint16_t coord = fixed_gather && !cube ? (g18g ? 4 : 0)
                          : g18g_gradient ? AGX3_TEX_G18G_GRADIENT_X
                          : g18g_array ? AGX3_TEX_G18G_ARRAY_X
                          : g17g_block ? base : coord_bank;
   /* A per-lane level or bias in a rotated block goes through Apple's
    * register relay (agx3_tex_pack_lod_word) into the block's +6 word; the
    * bank-6 stager stays byte-exact for the unrotated single-sample form. */
   const bool g17g_lod_relay =
      g17g_block && base != 0 && agx3_g17g_register_lod(I);
   if (g17g_lod_relay && (lod_relay == 0 || (lod_relay & 1) || lod_relay >= 128))
      return agx3_tex_decline(I, __FILE__, __LINE__);
   /* G17G marks a staged source with f27 on its last use and f25 when it is
    * read again (a later block's copies); other G17G stagers keep their
    * witnessed last-use form. */
   /* Native382 reuses one staged pair across gathers. Our repeated
    * publication must retain source registers until their actual last use;
    * otherwise later copies overwrite the shared pair with discarded data. */
   const bool preserve_coords = g17g_block || gpu1_gather || g18g_colour_level;
   const bool coord_discard = enc == AGX_ISA_ENCODING_AGX3B
                                 ? I->src[0].discard
                                 : !preserve_coords || I->src[0].discard;
   const bool coord_cache = enc == AGX_ISA_ENCODING_AGX3B
                               ? I->src[0].cache
                               : !coord_discard;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t len = 0;

   /* G17G publishes a noncube array's layer and register level or bias as
    * one pair in every stage (3700004a2500aa0a400260302100 in the fragment
    * depth2d_array witnesses). */
   const bool combined_array_lod = array && !cube &&
      (offset || vertex_stage || enc == AGX_ISA_ENCODING_AGX3C) &&
      ((explicit_lod && !immediate_lod) || register_bias);
   /* A G17G cube carries its per-lane level in the same PACK_SMPIR that
    * publishes the cube Z, after the CEM group, rather than staging it on
    * its own beforehand -- see agx3_tex_stage_cube_z_lod. Staging it
    * separately through the plain 2D form leaves every lane reading level
    * zero, which is what made KHR-GL30.ext_texture_shadow_lod's cube cases
    * render one level and one face for the whole quad. */
   const bool cube_paired_lod = cube && enc == AGX_ISA_ENCODING_AGX3C &&
      ((explicit_lod && !immediate_lod) || register_bias) && scalar_lod_reg;
   if (bias_min) {
      if (!agx3_tex_stage_bias_min(
             enc, I->src[1].value, I->src[1].value + 1, stage_slot, wait_mask,
             I->src[1].cache, I->src[1].discard, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
      wait_mask = 0;
   } else if (gradient_min) {
      if (!agx3_tex_stage_typed_lod(
             enc, I->src[1].value + 2 * gradient_channels, stage_slot,
             wait_mask, I->src[1].cache, I->src[1].discard,
             AGX3_TEX_LEVEL_MINIMUM, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
      wait_mask = 0;
   } else if (combined_array_lod) {
      const bool packed = g18g_array
         ? agx3_tex_stage_array_lod_g18g(
              I->src[0].value + 4, I->src[1].value, stage_slot, wait_mask,
              I->src[0].cache, I->src[0].discard,
              I->src[1].cache, I->src[1].discard, bytes, &len)
         : agx3_tex_stage_array_lod(
              enc, I->src[0].value + 4, I->src[1].value, stage_slot, wait_mask,
              I->src[0].cache, I->src[0].discard,
              I->src[1].cache, I->src[1].discard, bytes, &len);
      if (!packed || !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
      wait_mask = 0;
   } else if (((explicit_lod && !immediate_lod) || register_bias) &&
              !g17g_lod_relay && !cube_paired_lod) {
      /* G17G's PACK_SMPIR witnesses discard the published scalar and never
       * carry the cache hint, like its coordinate copies below. G18G reads
       * the word at Y + 2 (AGX3_TEX_G18G_LEVEL_WORD). */
      const bool staged = g18g
         ? agx3_tex_stage_typed_lod_g18g(coord, I->src[1].value, stage_slot,
                                         wait_mask, false, true,
                                         AGX3_TEX_LEVEL_FLOAT, bytes, &len)
         : agx3_tex_stage_lod(enc, I->src[1].value, stage_slot, wait_mask,
                              enc == AGX_ISA_ENCODING_AGX3B && I->src[1].cache,
                              enc != AGX_ISA_ENCODING_AGX3B || I->src[1].discard,
                              bytes, &len);
      if (!staged || !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
      wait_mask = 0;
   }

   /* CEM uses fixed r0/r2/r7 scratch registers. Publish the comparison first
    * so a perfectly valid allocator choice in one of those registers cannot
    * be clobbered by the cube-coordinate calculation. Apple's cube and
    * cube-array comparison shaders likewise publish the reference before the
    * final CEM axes. G18 COPY requires an explicit lifetime hint: a reference
    * reused by later samples stays live until its final discard-marked use. */
   if (shadow && cube &&
       (!agx3_tex_stage_copy_hints(
           enc, g18g && cube_x4 ? 6 : 4,
           I->src[5].value, stage_slot, 0,
           I->src[5].cache || (g18g && !I->src[5].discard),
           I->src[5].discard, bytes, &len) ||
        !agx3_append(emission, bytes, len)))
      return agx3_tex_decline(I, __FILE__, __LINE__);

   if (cube) {
      /* Native cube sampling consumes CEM's normalized coordinates and face,
       * not the raw XYZ vector. CEM has no wait fields. A register-descriptor
       * witness may leave only slot 6 active: the finalizers/face producer
       * join that transaction and SMP waits for the complete group. */
      const uint16_t staged_wait = (uint16_t)(1u << (stage_slot - 1));
      if (wait_mask && (!register_operands || wait_mask != staged_wait))
         return agx3_tex_decline(I, __FILE__, __LINE__);

      const uint16_t fixed_scratch[3] = {0, 14, 4};
      const uint16_t *scratch = fixed_cem ? fixed_scratch : cem_scratch;
      for (unsigned axis = 0; axis < 3; ++axis) {
         const bool packed = fixed_cem
            ? agx3_tex_cube_cem_calc(
                 enc, I->src[0].value, I->src[0].value + 2,
                 I->src[0].value + 4, axis,
                 axis == 2 && I->src[0].discard, bytes, &len)
            : agx3_tex_cube_cem_calc_to(
                 enc, I->src[0].value, I->src[0].value + 2,
                 I->src[0].value + 4, scratch[axis], axis,
                 axis == 2 && I->src[0].discard, bytes, &len);
         if (!packed ||
             !agx3_append(emission, bytes, len))
            return agx3_tex_decline(I, __FILE__, __LINE__);

         if (axis == 0 &&
             (!(fixed_cem
                   ? agx3_tex_cube_cem_rcp(enc, bytes, &len)
                   : agx3_tex_cube_cem_rcp_to(enc, scratch[0], bytes, &len)) ||
              !agx3_append(emission, bytes, len)))
            return agx3_tex_decline(I, __FILE__, __LINE__);
      }

      for (unsigned component = 0; component < 2; ++component) {
         const bool packed = g18g
            ? agx3_tex_cube_cem_finalize_g18g(
                 enc, component, cube_x4, scratch[0], scratch[component + 1],
                 stage_slot, bytes, &len)
            : fixed_cem
            ? agx3_tex_cube_cem_finalize(enc, component, component == 1,
                                         bytes, &len)
            : agx3_tex_cube_cem_finalize_to(
                 enc, component, scratch[0], scratch[component + 1],
                 stage_slot, bytes, &len);
         if (!packed ||
             !agx3_append(emission, bytes, len))
            return agx3_tex_decline(I, __FILE__, __LINE__);
      }
   } else if (!native_staged_coords) {
      if (!agx3_tex_stage_copy_hints(enc,
                                    gpu1_query_lod && g18g && three_d && !register_bias ? 8 : coord,
                                    I->src[0].value, stage_slot,
                                    wait_mask, coord_cache, coord_discard,
                                    bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
      if (!agx3_tex_stage_copy_hints(enc,
                                    (g18g_array || g18g_gradient || g18g_fixed_gather ||
                                     (gpu1_query_lod && g18g && three_d && !register_bias))
                                               ? AGX3_TEX_G18G_ARRAY_Y
                                               : coord + 2,
                                    I->src[0].value + 2,
                                    stage_slot, 0, coord_cache, coord_discard,
                                    bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
   }

   if (three_d &&
       (!agx3_tex_stage_copy_hints(
           enc, gpu1_query_lod && g18g && !register_bias ? 4 :
                coord + (g18g ? AGX3_TEX_G18G_Z_WORD : AGX3_TEX_G17G_Z_WORD),
           I->src[0].value + 4, stage_slot, 0,
           (g18g_colour_level || gpu1_query_lod) ? coord_cache : I->src[0].cache,
           (g18g_colour_level || gpu1_query_lod) ? coord_discard : I->src[0].discard, bytes, &len) ||
        !agx3_append(emission, bytes, len)))
      return agx3_tex_decline(I, __FILE__, __LINE__);

   /* The driver's noncube array ABI stages the layer into the sampler-private
    * bank and selects it with SMP f77=2.  Apple's direct f31/f37 form belongs
    * to the incompatible APPLE_FRAGMENT profile and is retained only by the
    * standalone texture encoder.
    */
   if (array && !cube && !query_lod && !combined_array_lod &&
       !register_offset &&
       (vertex_stage || shadow || implicit_lod || offset ||
        enc == AGX_ISA_ENCODING_AGX3C) &&
       (!((g18g_array || g18g_gradient || g18g_fixed_gather)
             ? agx3_tex_stage_array_layer_g18g(
                  I->src[0].value + 4, stage_slot, 0, I->src[0].cache,
                  I->src[0].discard, bytes, &len)
             : agx3_tex_stage_array_layer(
                  enc, I->src[0].value + 4, stage_slot, 0, I->src[0].cache,
                  I->src[0].discard, bytes, &len)) ||
        !agx3_append(emission, bytes, len)))
      return agx3_tex_decline(I, __FILE__, __LINE__);

   if (cube) {
      /* G17G normalizes a cube-array comparison's integer layer into a
       * general register first; the publication then names that register
       * rather than the IR's raw layer half, which is the operand Apple's
       * own depthcube_array fragment publishes. */
      const bool normalize_layer =
         enc == AGX_ISA_ENCODING_AGX3C &&
         (shadow || gpu1_gather || g18g_colour_level) && cube_array &&
         array_scratch && (!g18g || cube_paired_lod);
      if (normalize_layer) {
         if (!agx3_tex_normalize_cube_array_layer(
                enc, array_scratch, I->src[0].value + 6, I->src[0].cache,
                I->src[0].discard, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            return agx3_tex_decline(I, __FILE__, __LINE__);
      }
      const uint16_t cube_array_layer =
         normalize_layer ? array_scratch : I->src[0].value + 6;
      const bool layer_cache = normalize_layer ? false : I->src[0].cache;
      const bool layer_discard = normalize_layer ? true : I->src[0].discard;
      bool (* const stage_z_lod)(enum agx_isa_encoding, bool, uint16_t,
         uint16_t, uint8_t, uint16_t, bool, bool, bool, bool, uint8_t *,
         uint8_t *) = g18g ? agx3_tex_stage_cube_z_lod_g18g
                          : agx3_tex_stage_cube_z_lod_g17g;
      bool (* const stage_z)(enum agx_isa_encoding, uint16_t, uint8_t,
         uint16_t, bool, bool, uint8_t *, uint8_t *) =
            g18g ? agx3_tex_stage_cube_z_g18g : agx3_tex_stage_cube_z;
      const bool packed = g18g &&
         (shadow || gpu1_gather || g18g_colour_level) && cube_array
         ? agx3_tex_stage_cube_array_g18g(
              enc, cem_scratch[0] + 2, cube_array_layer,
              cube_paired_lod ? I->src[1].value : 0, cube_paired_lod,
              stage_slot, 0, layer_cache, layer_discard, I->src[1].cache,
              I->src[1].discard, bytes, &len)
         : shadow && cube_array && cube_paired_lod
         ? stage_z_lod(
              enc, true, cube_array_layer, I->src[1].value, stage_slot, 0,
              layer_cache, layer_discard, I->src[1].cache,
              I->src[1].discard, bytes, &len)
         : (shadow || gpu1_gather) && cube_array
         ? agx3_tex_stage_cube_array_shadow(
              enc, cube_array_layer, stage_slot, 0,
              layer_cache, layer_discard, bytes, &len)
         : cube_paired_lod
         ? stage_z_lod(
              enc, false, fixed_cem ? 2 : cem_scratch[0] + 2, I->src[1].value,
              stage_slot, 0, false, true, I->src[1].cache, I->src[1].discard,
              bytes, &len)
         : stage_z(enc, fixed_cem ? 2 : cem_scratch[0] + 2,
                                 stage_slot, 0, false, true,
                                 bytes, &len);
      if (!packed || !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
   }

   if (register_offset) {
      if (array &&
          (!agx3_tex_pack_ms_array_layer(
              enc, I->src[0].value + 4, array_scratch,
              I->src[0].cache, I->src[0].discard, bytes, &len) ||
           !agx3_append(emission, bytes, len)))
         return agx3_tex_decline(I, __FILE__, __LINE__);
      if (!agx3_tex_stage_gather_offset_at(
             enc, g18g_runtime_gather ? 2 : 6,
             I->src[5].value + (shadow ? 2 : 0), array, array_scratch,
             stage_slot, I->src[5].cache, I->src[5].discard, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
   }

   if (shadow && !cube) {
      const bool compare_discard = enc == AGX_ISA_ENCODING_AGX3B
                                      ? I->src[5].discard
                                      : !preserve_coords || I->src[5].discard;
      const bool compare_cache = enc == AGX_ISA_ENCODING_AGX3B
                                    ? I->src[5].cache
                                    : !compare_discard;
      /* G18G array comparisons keep X/Y at 4/0, their packed layer/LOD
       * at half 2 and the comparison reference at half 6. The native
       * m5_shadowoff_array_level G18 output uses this exact layout.
       * G18G names the reference's half (SMP f58, set by the transcoder):
       * X + 4 is its Y + 2 word, which a staged level or bias takes, so the
       * reference moves one word up then, as in Apple's m6-shadow-lod. */
      const bool g18g_level_word =
         g18g && ((explicit_lod && !immediate_lod) || register_bias);
      if (!agx3_tex_stage_copy_hints(enc,
                                     (g18g_array || g18g_fixed_gather) ? 6
                                     : g18g ? coord + (g18g_level_word ? 6 : 4)
                                          : base + 4,
                                     I->src[5].value,
                                     stage_slot, 0, compare_cache,
                                     compare_discard, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
   }

   /* The rotated level/bias word: pack it in the relay register, then COPY
    * it into the block on the sample's slot, as Apple's three-comparison
    * bias(reg) fragment does for each of its blocks. */
   if (g17g_lod_relay) {
      const bool lod_discard = I->src[1].discard;
      if (!agx3_tex_pack_lod_word(enc, lod_relay, I->src[1].value,
                                  !lod_discard, lod_discard, bytes, &len) ||
          !agx3_append(emission, bytes, len) ||
          !agx3_tex_stage_copy_hints(enc,
                                     base + (g18g ? AGX3_TEX_G18G_LEVEL_WORD
                                                  : AGX3_TEX_G17G_LEVEL_WORD),
                                     lod_relay, stage_slot, 0,
                                     false, true, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
   }

   /* The gradient scalars follow the coordinates, in the interleaved order
    * (ddx.x, ddy.x, ddx.y, ddy.y) the native kernels stage them in. The
    * collect that produced them is laid out the same way, so the run is
    * copied straight through. */
   if (gradient_lod) {
      static const uint16_t bank[6] = {
         AGX3_TEX_GRADIENT_BANK_DDX_X, AGX3_TEX_GRADIENT_BANK_DDY_X,
         AGX3_TEX_GRADIENT_BANK_DDX_Y, AGX3_TEX_GRADIENT_BANK_DDY_Y,
         AGX3_TEX_GRADIENT_BANK_DDX_Z, AGX3_TEX_GRADIENT_BANK_DDY_Z,
      };

      for (unsigned i = 0; i < gradient_channels; ++i) {
         const uint16_t gradient_bank = g18g_gradient ? 4 + (2 * i) : bank[i];
         if (!agx3_tex_stage_copy_hints(enc, gradient_bank,
                                        I->src[1].value + (2 * i), stage_slot,
                                        0, I->src[1].cache, I->src[1].discard,
                                        bytes, &len) ||
             !agx3_append(emission, bytes, len))
            return agx3_tex_decline(I, __FILE__, __LINE__);
      }
   }

   if (pixel_entry) {
      uint8_t entry[AGX3_PIXEL_MAX_LENGTH], entry_len = 0;
      if (pixel_entry_offset == NULL ||
          !agx3_pixel_colour_entry(enc, entry, &entry_len))
         return agx3_tex_decline(I, __FILE__, __LINE__);

      *pixel_entry_offset = emission->size;
      if (!agx3_append(emission, entry, entry_len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
   }

   const struct agx3_tex_sample_desc d = {
      /* The tex core's G18G sample forms (agx3_tex_sample_desc::g18g). */
      .g18g = g18g,
      .dest = I->dest[0].value,
      .coord = coord,
      .width = I->dest[0].size == AGX_SIZE_32 ? AGX3_TEX_W32 : AGX3_TEX_W16,
      .mask = I->mask,
      .texture = texture_immediate ? I->src[3].value : 0,
      .sampler = sampler_immediate ? I->src[4].value : 0,
      .texture_kind = texture_immediate ? AGX3_TEX_OPERAND_IMMEDIATE :
         I->src[3].type == AGX_INDEX_UNIFORM ? AGX3_TEX_OPERAND_UNIFORM : AGX3_TEX_OPERAND_REGISTER,
      .sampler_kind = sampler_immediate ? AGX3_TEX_OPERAND_IMMEDIATE :
         I->src[4].type == AGX_INDEX_UNIFORM ? AGX3_TEX_OPERAND_UNIFORM : AGX3_TEX_OPERAND_REGISTER,
      .texture_value = texture_immediate ? 0 : I->src[3].value,
      .sampler_value = sampler_immediate ? 0 : I->src[4].value,
      .texture_kill = !texture_immediate && I->src[3].kill,
      .sampler_kill = !sampler_immediate && I->src[4].kill,
      .gather = gather_component,
      .shadow = shadow,
      .query_lod = query_lod,
      .offset = offset,
      .offset_xyz = offset ? I->offset_xyz : 0,
      .offset_register = register_offset,
      .min_lod = min_lod,
      .lod = biased_lod       ? AGX3_TEX_LOD_BIAS
             : immediate_lod  ? AGX3_TEX_LOD_LEVEL
             : explicit_lod   ? AGX3_TEX_LOD_REGISTER
             : register_bias || bias_min ? AGX3_TEX_LOD_BIAS_REGISTER
             : gradient_lod   ? AGX3_TEX_LOD_GRADIENT
                              : AGX3_TEX_LOD_AUTO,
      .lod_x64 = immediate_lod ? immediate_x64_field : 0,
      .array = array,
      .array_layer = array && !query_lod && enc == AGX_ISA_ENCODING_AGX3B
                        ? I->src[0].value + (cube_array ? 6 : 4) : 0,
      .three_d = three_d,
      .cube = cube,
      .slot = slot,
      .wait_mask = (uint16_t)(1u << (stage_slot - 1)),
      /* CEM consumes and discards the source XYZ before SMP. Apple's cube
       * and cube-array samples consequently set SMP f27 even though the
       * sampler reads CEM's private coordinate bank rather than the original
       * AGX IR source. Ordinary dimensions retain their existing last-use
       * mapping. */
      .coord_kill = cube || enc == AGX_ISA_ENCODING_AGX3C ? true
                         : native_staged_coords
                              ? true
                              : I->src[0].kill,
      .final = final,
      .sample_group_nonterminal = sample_group_nonterminal,
      .sample_group_end = sample_group_end,
      .vertex_stage = vertex_stage,
      /* Native373 gathers and native607 explicit colour samples have no
       * fragment helper-group terminator in compute execution stages. */
      .compute_stage = compute_stage && (gpu1_gather || g18g_colour_level),
      .sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER,
   };
   if (!agx3_tex_encode_sample(enc, &d, bytes, &len) ||
       !agx3_append(emission, bytes, len))
      return agx3_tex_decline(I, __FILE__, __LINE__);
   return true;
}

/* Filtered 2D samples use scoreboard-signalling native staging producers.
 * Plain implicit 2D keeps its coordinate layout. Explicit per-lane LOD and
 * 2D-array samples stage coordinates into bank 0; their auxiliary scalar is
 * packed into bank 6. All input waits attach to the first producer; the sample
 * waits for their shared slot.
 */
static bool
agx3_emit_texture_sample(struct util_dynarray *emission,
                         enum agx_isa_encoding enc, const agx_instr *I,
                         uint8_t slot, uint16_t wait_mask, bool final,
                         bool sample_group_nonterminal, bool sample_group_end,
                         bool pixel_entry, uint32_t *pixel_entry_offset,
                         uint8_t native_coord_slot,
                         const uint16_t cem_scratch[3], bool vertex_stage,
                         bool compute_stage, uint16_t array_scratch,
                         uint16_t bank_base, uint16_t g17g_lod_relay, bool g18g,
                         bool publication_last, uint16_t heap_uniform_base)
{
   /* G18G (agx3d) keeps j873's G17G-family sampler, whose forms its agx3d
    * re-encode is grounded on. */
   if (enc == AGX_ISA_ENCODING_AGX3C && g18g)
      return agx3_emit_texture_sample_g17g(emission, enc, I, slot, wait_mask, final,
          sample_group_nonterminal, sample_group_end, pixel_entry,
          pixel_entry_offset, native_coord_slot, cem_scratch, vertex_stage, compute_stage, array_scratch,
          g17g_lod_relay, g18g, NULL);

   const bool published = agx_texture_resource(I) != 0;
   const bool scalar_cube = I->agx3_cube_scalar_coords;
   const bool split = I->agx3_split_coords;
   const unsigned source_count = scalar_cube || split ? 8
                                 : agx3_cube_own_scratch(I) ? 7 : 6;
   uint16_t own_cem_scratch[3];
   cem_scratch = agx3_cube_scratch(I, cem_scratch, own_cem_scratch);
   const bool explicit_lod = I->lod_mode == AGX_LOD_MODE_LOD_MIN;
   const bool bias_min = I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN;
   const bool gradient_min = I->lod_mode == AGX_LOD_MODE_LOD_GRAD_MIN;
   const bool min_lod = bias_min || gradient_min;
   /* An implicit-LOD sample with a constant bias: the same 12-bit x64
    * immediate operand as an explicit level, selected by f139 = 1 instead
    * of 2, and keeping the implicit-LOD marker.
    *
    * A zero bias is not encoded at all: Apple's own compiler folds it into
    * the plain implicit form, and this packer follows it, so the operand
    * stays the immediate zero the implicit form already requires. Minifloat
    * -0.0 folds the same way. */
   const bool zero_bias = I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS &&
                          I->src[1].type == AGX_INDEX_IMMEDIATE &&
                          agx_minifloat_decode(I->src[1].value) == 0.0f;
   const bool biased_lod = I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS &&
                           I->src[1].type == AGX_INDEX_IMMEDIATE && !zero_bias;
   /* The same sample with a per-lane bias instead: PACK_SMPIR publishes the
    * scalar exactly as it does for a per-lane level, and only f139 moves
    * (3 rather than 4). */
   const bool register_bias = I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS &&
                              I->src[1].type != AGX_INDEX_IMMEDIATE;
   const bool implicit_lod = I->lod_mode == AGX_LOD_MODE_AUTO_LOD || zero_bias;
   /* Gradient operands share the sample coordinate base. */
   const bool gradient_lod = I->lod_mode == AGX_LOD_MODE_LOD_GRAD || gradient_min;
   /* Gather uses the implicit-LOD selector for 2D, array and cube forms,
    * including compute where it reads level zero. Map the IR component
    * values 1/3/5/7 explicitly to the native two-bit selector. */
   const bool gather = I->gather != AGX_GATHER_NONE;
   const bool shadow = I->shadow;
   const bool query_lod = I->query_lod;
   const bool offset = I->offset;
   enum agx3_tex_gather gather_component = AGX3_TEX_GATHER_NONE;
   switch (I->gather) {
   case AGX_GATHER_NONE: break;
   case AGX_GATHER_R: gather_component = AGX3_TEX_GATHER_R; break;
   case AGX_GATHER_G: gather_component = AGX3_TEX_GATHER_G; break;
   case AGX_GATHER_B: gather_component = AGX3_TEX_GATHER_B; break;
   case AGX_GATHER_A: gather_component = AGX3_TEX_GATHER_A; break;
   default: return agx3_tex_decline(I, __FILE__, __LINE__);
   }
   const bool immediate_lod =
      (explicit_lod || biased_lod) && I->src[1].type == AGX_INDEX_IMMEDIATE;
   const float immediate_level = immediate_lod ?
      agx_minifloat_decode(I->src[1].value) : 0.0f;
   const float immediate_level_x64 = immediate_level * 64.0f;
   /* f29 is a *signed* 12-bit LOD in 1/64 steps. Sweeping Apple's own
    * `t.sample(s, uv, bias(k))` over k = +-{1/64, 0.25, 0.5, 1, 1.5, 2, 4,
    * 31, 32, 33, 63, 64} gives f29 = k * 64 in two's complement -- bias(-1.0)
    * is 4032, bias(-32.0) is 2048 -- saturating at 2047 (+31.984375) and 2048
    * (-32). A minifloat immediate spans [-31, 31], so it always lands inside
    * that range, but the sign has to survive into the field. */
   const bool integral_x64 =
      immediate_level_x64 >= -2048.0f && immediate_level_x64 <= 2047.0f &&
      immediate_level_x64 == (float)(int16_t)immediate_level_x64;
   /* An explicit level is unsigned; only a bias may be negative. */
   const bool valid_immediate_lod =
      immediate_lod && immediate_level >= 0.0f && integral_x64;
   const bool valid_immediate_bias = immediate_lod && integral_x64;
   const uint16_t immediate_x64_field =
      (uint16_t)(((int)immediate_level_x64) & 0xfff);
   const bool cube_array = I->dim == AGX_DIM_CUBE_ARRAY;
   const bool array = I->dim == AGX_DIM_2D_ARRAY || cube_array;
   const bool three_d = I->dim == AGX_DIM_3D;
   const bool cube = I->dim == AGX_DIM_CUBE || cube_array;
   /* A vertex stage has no derivatives, so a `texture()` depth comparison
    * arrives as an explicit level zero. The cube array adds only its layer,
    * staged through the face+layer word of the implicit cube-array
    * comparison. */
   const bool shadow_level_zero =
      shadow && valid_immediate_lod && immediate_level == 0.0f;
   /* A per-lane level and a per-lane bias share one staging form, so they
    * share one operand test: a live single-channel 16-bit register. */
   const bool scalar_lod_reg = agx3_tex_plain_reg(I->src[1]) &&
      I->src[1].size == AGX_SIZE_16 && agx_channels(I->src[1]) == 1;
   const bool packed_lod_reg = enc == AGX_ISA_ENCODING_AGX3C &&
      I->memory_is_packed && agx3_tex_plain_reg(I->src[1]) &&
      I->src[1].size == AGX_SIZE_32 && agx_channels(I->src[1]) == 1 &&
      !(I->src[1].value & 1) && I->src[1].value < 128;
   /* The C scalar publisher consumes its half-float source; a reusable
    * scalar is first materialized as a packed word. Composite layer/LOD
    * and face/LOD publishers also encode a live half-float operand. */
   const bool native_scalar_lod_reg = packed_lod_reg || (scalar_lod_reg &&
      (enc == AGX_ISA_ENCODING_AGX3B ||
       (enc == AGX_ISA_ENCODING_AGX3C && I->src[1].value < 128 &&
        ((array || cube) || (!I->src[1].cache && I->src[1].discard)))));
   const bool bias_min_operand = agx3_tex_plain_reg(I->src[1]) &&
      I->src[1].size == AGX_SIZE_32 && agx_channels(I->src[1]) == 1 &&
      (I->src[1].value & 1) == 0 && I->src[1].value < 127;
   /* A 2D gradient is four 32-bit scalars in one aligned run; a 3D one is
    * six, and agx_emit_tex()'s collect already lays both out in the
    * interleaved order the staging banks want. */
   const unsigned gradient_channels = three_d ? 6u : 4u;
   const bool gradient_operand = agx3_tex_plain_reg(I->src[1]) &&
      I->src[1].size == AGX_SIZE_32 &&
      agx_channels(I->src[1]) == gradient_channels + gradient_min &&
      (I->src[1].value & 1) == 0 &&
      I->src[1].value + (2 * (gradient_channels + gradient_min)) - 2 < 128;
   const bool valid_lod = bias_min ?
      bias_min_operand : explicit_lod ?
      (valid_immediate_lod || native_scalar_lod_reg) :
      biased_lod
         ? (valid_immediate_bias &&
            immediate_level_x64 != 0.0f)
         : register_bias
         ? native_scalar_lod_reg
         : gradient_lod
         ? gradient_operand
         : (implicit_lod &&
            (zero_bias || agx3_tex_imm(I->src[1], 0)));
   const bool texture_immediate =
      I->nr_srcs == source_count && I->src[3].type == AGX_INDEX_IMMEDIATE;
   const bool sampler_immediate =
      I->nr_srcs == source_count && I->src[4].type == AGX_INDEX_IMMEDIATE;
   const bool immediate = texture_immediate && sampler_immediate;
   /* agx_emit_tex() publishes the packed offset on the instruction rather
    * than through src[5], because a comparison sample needs that operand for
    * its reference value. src[5] still carries the immediate when there is no
    * comparator, and the two must agree. */
   const bool immediate_offset = offset && I->offset_xyz > 0 &&
      (shadow || (I->src[5].type == AGX_INDEX_IMMEDIATE &&
                  I->src[5].value == I->offset_xyz &&
                  !I->src[5].memory && !I->src[5].abs && !I->src[5].neg &&
                  !I->src[5].cache && !I->src[5].discard && !I->src[5].kill));
   /* AGX3C stages an array gather's runtime offset like an array sample's:
    * one bank-6 word with the layer (native texwit6 g_arr/g_arr_y). */
   const bool m5_array_gather = enc == AGX_ISA_ENCODING_AGX3C && gather &&
      array;
   const bool register_offset = offset && !I->offset_xyz && gather &&
      !m5_array_gather &&
      agx3_tex_plain_reg(I->src[5]) && I->src[5].size == AGX_SIZE_32 &&
      agx_channels(I->src[5]) == (shadow ? 2 : 1) &&
      !(I->src[5].value & 1) && I->src[5].value + (shadow ? 2 : 0) <= 126;
   /* Runtime offsets are contiguous 32-bit register components. A comparison,
    * if present, precedes the 2D pair in the collected operand. A runtime
    * 2D offset combined with register LOD/bias arrives as one prepacked word
    * instead, because those operands share sampler bank 6. */
   const unsigned offset_components = 2;
   const unsigned staged_offset_components =
      offset_components + (shadow ? 1 : 0);
   const bool staged_component_offset = offset && !gather && !three_d && !array &&
      I->offset_xyz == 0 &&
      agx3_tex_plain_reg(I->src[5]) && I->src[5].size == AGX_SIZE_32 &&
      agx_channels(I->src[5]) == staged_offset_components &&
      (I->src[5].value & 1) == 0 &&
      I->src[5].value + 2 * staged_offset_components <= 256;
   const unsigned packed_offset_components = 1 + (shadow ? 1 : 0);
   const bool staged_packed_offset = offset &&
      (!gather || m5_array_gather) && I->offset_xyz == 0 &&
      (array || explicit_lod || register_bias || min_lod) &&
      agx3_tex_plain_reg(I->src[5]) && I->src[5].size == AGX_SIZE_32 &&
      agx_channels(I->src[5]) == packed_offset_components &&
      (I->src[5].value & 1) == 0 &&
      I->src[5].value + 2 * packed_offset_components <= 128;
   const bool staged_offset =
      staged_component_offset || staged_packed_offset;
   /* Both encodings' COPY sources name seven-bit GPR half-registers.
    * SMP names a separate staging bank, not those source GPRs. */
   const unsigned max_coord = 128 - (2 * agx_channels(I->src[0]));
   const bool published_shape =
      ((enc == AGX_ISA_ENCODING_AGX3B && I->dim == AGX_DIM_2D) ||
       (enc == AGX_ISA_ENCODING_AGX3C && three_d && !shadow && !offset)) &&
      !query_lod &&
      !I->src[0].memory && !I->src[0].abs && !I->src[0].neg &&
      !I->src[0].cache && !I->src[0].discard &&
      I->src[0].size == AGX_SIZE_32 &&
      agx_channels(I->src[0]) == ((three_d || shadow) ? 3 : 2) &&
      (implicit_lod || immediate_lod) &&
      (agx_is_null(I->src[5]) || (!shadow && immediate_offset)) &&
      (!offset || immediate_offset);
   if (scalar_cube) {
      if (enc != AGX_ISA_ENCODING_AGX3B || I->nr_srcs != 8 ||
          I->dim != AGX_DIM_CUBE || shadow || offset || query_lod || gather ||
          I->sparse || I->lod_mode != AGX_LOD_MODE_AUTO_LOD ||
          native_coord_slot || published)
         return agx3_tex_decline(I, __FILE__, __LINE__);
      for (unsigned s = 0; s < 3; ++s) {
         agx_index src = I->src[s ? s + 5 : 0];
         if (!agx3_tex_plain_reg(src) || src.size != AGX_SIZE_32 ||
             agx_channels(src) != 1 || (src.value & 1) || src.value > 126)
            return agx3_tex_decline(I, __FILE__, __LINE__);
      }
   }
   if (split) {
      /* Split x and y stage through the wide scalar COPY. No other operand
       * names their registers (agx3_split_texture_coordinates() keeps such a
       * component in the vector): a level, bias or offset staged before them
       * could otherwise discard one first. */
      if (enc != AGX_ISA_ENCODING_AGX3C || I->nr_srcs != 8 ||
          I->dim != AGX_DIM_2D || published || query_lod ||
          native_coord_slot || !agx_is_null(I->src[7]))
         return agx3_tex_decline(I, __FILE__, __LINE__);
      for (unsigned c = 0; c < 2; ++c) {
         const agx_index coord = agx3_tex_coord(I, c);
         if (!agx3_split_coord_ok(coord, c))
            return agx3_tex_decline(I, __FILE__, __LINE__);
         for (unsigned s = 1; s < 6; ++s) {
            const agx_index other = I->src[s];
            if (agx3_tex_plain_reg(other) &&
                other.value < coord.value + 2 &&
                coord.value < other.value + agx_index_size_16(other))
               return agx3_tex_decline(I, __FILE__, __LINE__);
         }
      }
   }
   const bool coordinates_valid = published ? published_shape :
      scalar_cube || split ||
      (agx3_tex_plain_reg(I->src[0]) && I->src[0].size == AGX_SIZE_32 &&
       agx_channels(I->src[0]) == (query_lod ? (three_d || cube ? 3 : 2) :
                                  cube_array ? 4 :
                                  (array || three_d || cube) ? 3 : 2) &&
       !(I->src[0].value & 3) && I->src[0].value <= max_coord);
   if (I->nr_dests != 1 || I->nr_srcs != source_count ||
       !agx3_tex_plain_reg(I->dest[0]) ||
       (I->dest[0].size != AGX_SIZE_16 &&
        I->dest[0].size != AGX_SIZE_32) ||
       /* Sparse residency widens only the destination reservation: the extra
        * register the instruction writes after its masked channels. */
       agx_channels(I->dest[0]) != (I->sparse ? 8u : 4u) ||
       I->dest[0].value >= (enc == AGX_ISA_ENCODING_AGX3C ? 256 : 128) ||
       I->dest[0].value + agx_index_size_16(I->dest[0]) > 256 ||
       (I->dest[0].size == AGX_SIZE_32 && (I->dest[0].value & 1)) ||
       !coordinates_valid ||
       !valid_lod || !agx3_tex_imm(I->src[2], 0) ||
       (texture_immediate
           ? I->src[3].value > 15
           : ((I->src[3].type != AGX_INDEX_REGISTER &&
               I->src[3].type != AGX_INDEX_UNIFORM) ||
              I->src[3].size != AGX_SIZE_64 ||
              agx_channels(I->src[3]) != 1 || (I->src[3].value & 1))) ||
       (sampler_immediate
           ? I->src[4].value > 15
           : ((I->src[4].type != AGX_INDEX_REGISTER &&
               I->src[4].type != AGX_INDEX_UNIFORM) ||
              I->src[4].size != AGX_SIZE_64 ||
              agx_channels(I->src[4]) != 1 || (I->src[4].value & 1))) ||
       I->src[3].memory || I->src[3].abs || I->src[3].neg ||
       I->src[4].memory || I->src[4].abs || I->src[4].neg ||
       (!published && (shadow
           ? (register_offset
                 ? false
                 : staged_offset
                  ? false
                  : (!agx3_tex_plain_reg(I->src[5]) ||
                     I->src[5].size != AGX_SIZE_32 ||
                     agx_channels(I->src[5]) != 1 || I->src[5].value > 126 ||
                     (offset && !immediate_offset)))
           : offset ? (!immediate_offset && !register_offset && !staged_offset)
                     : !agx_is_null(I->src[5]))) ||
       (I->dim != AGX_DIM_2D && I->dim != AGX_DIM_2D_ARRAY && !three_d &&
        !cube) ||
       /* The dimension selector is independent of the supported LOD mode;
        * gradient and minimum operands retain their generation gates above. */
       (three_d && !valid_immediate_lod && !implicit_lod && !biased_lod &&
        !register_bias && !explicit_lod && !gradient_lod && !bias_min) ||
       (cube && !implicit_lod && !explicit_lod && !biased_lod &&
        !register_bias && !bias_min) ||
       (I->mask == 0 || (I->mask & ~15u)) ||
       /* Both encodings support sparse samples, excluding depth comparison
        * and LOD query. */
       (I->sparse && (shadow || query_lod)) ||
       /* A minimum LOD and a runtime offset share bank 6 only through the
        * packed word. */
       (min_lod && ((cube && !bias_min) || vertex_stage ||
                    (three_d && offset && !bias_min && !immediate_offset) ||
                    staged_component_offset)) ||
       (query_lod &&
        (!implicit_lod || shadow || offset ||
         gather || I->dest[0].size != AGX_SIZE_32 || (I->mask & ~3u))) ||
       /* Immediate 3D offsets carry their third signed nibble in f43.
        * Dynamic 3D offsets have already lowered to coordinate addition;
        * cubes have no offset form. */
       (offset && (cube ||
                    (enc != AGX_ISA_ENCODING_AGX3B &&
                     !(enc == AGX_ISA_ENCODING_AGX3C &&
                       (immediate_offset ||
                        (!array && !three_d &&
                         (staged_component_offset || register_offset)) ||
                        /* The complete bank-6 word of a 2D or 2D-array
                         * sample with its layer and/or register LOD. */
                        (staged_packed_offset &&
                         (!three_d || bias_min))))))) ||
       /* A depth comparison is orthogonal to the LOD shape on a plain 2D
        * sample. The array and cube comparisons use their own packed layer /
        * CEM operands, with implicit LOD or the witnessed explicit zero. */
       (shadow && (three_d ||
                   (enc == AGX_ISA_ENCODING_AGX3C &&
                    offset && !immediate_offset &&
                    !staged_component_offset &&
                    !register_offset && !staged_packed_offset) ||
                   I->mask != (gather ? 15u : 1u) ||
                   ((cube || (array && !offset && !vertex_stage)) &&
                    !implicit_lod && !shadow_level_zero &&
                    !(array && !cube && min_lod) &&
                    /* A comparison stages a register level or bias beside
                     * the operand its dimension already owns: the noncube
                     * array with its layer (agx3_tex_stage_array_lod), the
                     * cube with CEM's face, the cube array with both. A bias
                     * with a minimum exists for the cube only. A noncube
                     * array comparison gradient stages its scalars in banks
                     * 8..14, the layer in bank 6 and the reference in bank
                     * 4, so nothing collides. */
                    !((explicit_lod && !immediate_lod) || register_bias ||
                      (cube && bias_min) ||
                      (array && !cube && gradient_lod))))) ||
       (gather && (I->lod_mode != AGX_LOD_MODE_AUTO_LOD || three_d)))
      return agx3_tex_decline(I, __FILE__, __LINE__);

   /* The caller lowers additional required dependencies before staging. */
   if (wait_mask & (wait_mask - 1))
      return agx3_tex_decline(I, __FILE__, __LINE__);

   const bool register_operands = !immediate &&
      (I->src[3].type == AGX_INDEX_REGISTER ||
       I->src[4].type == AGX_INDEX_REGISTER);

   /* A cube register level or bias is published with CEM's face, after CEM
    * has run, so CEM must not overwrite it. */
   const bool cube_register_lod =
      cube && ((explicit_lod && !immediate_lod) || register_bias || bias_min);

   /* Every cube pipeline needs scratch outside the allocated live values.
    * Immediate descriptors do not make r0/r7/r2 free: a preceding gather
    * may still hold its result there when this CEM pipeline executes. */
   if (cube && (cem_scratch == NULL || (cem_scratch[0] & 1) ||
                cem_scratch[2] > 190))
      return agx3_tex_decline(I, __FILE__, __LINE__);

   /* Native register-indexed array and cube samples publish their coordinate /
    * face operand bank on slot 6 and wait slot 6 in SMP, independently of the
    * sample result slot. The dynamic cube witness groups both descriptor
    * loads and face publication on that transaction. Uniform/immediate
    * operands use the ordinary result slot. */
   const bool native_staged_coords = native_coord_slot != 0;
   const uint8_t stage_slot = native_staged_coords
                                 ? native_coord_slot
                                 : (array || cube) && register_operands ? 6
                                                                        : slot;
   /* Coordinates occupy a sampler-private range, independently of the
    * source GPRs. Apple's linked pipelines use four-half groups at 4/8/12/16
    * and at 84/128/132/140/152. The native witnesses are reproduced by
    * test_linked_staging_witnesses. Additional operands retain the existing
    * mode-specific ABI; only the relocatable profile can choose a new base.
    */
   const bool m5_fragment_3d = enc == AGX_ISA_ENCODING_AGX3C && three_d &&
      !vertex_stage && !compute_stage && implicit_lod;
   const bool m5_banked_3d = enc == AGX_ISA_ENCODING_AGX3C && three_d &&
      !shadow && !offset && !query_lod && !min_lod &&
      (((vertex_stage || compute_stage) && (immediate_lod || gradient_lod)) ||
       m5_fragment_3d);
   const bool combined_array_lod = array && !cube && !staged_offset &&
      ((explicit_lod && !immediate_lod) || register_bias || min_lod);
   const bool relocatable_array = array && !cube && !register_operands &&
      !query_lod && !staged_offset && !register_offset &&
      !combined_array_lod && !gradient_lod && !bias_min && !gradient_min &&
      !(bank_base & 7);
   if (bank_base != 0 &&
       ((enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C) ||
        native_staged_coords ||
        (array && !relocatable_array &&
         !(enc == AGX_ISA_ENCODING_AGX3B && vertex_stage &&
           explicit_lod && immediate_lod && !shadow && !offset)) ||
        (three_d && !m5_banked_3d &&
         !(enc == AGX_ISA_ENCODING_AGX3B && vertex_stage)) ||
        cube || (gradient_lod && !m5_banked_3d) || bias_min || gradient_min || query_lod ||
        (enc == AGX_ISA_ENCODING_AGX3B &&
         ((explicit_lod && !immediate_lod) || register_bias)) ||
        (enc == AGX_ISA_ENCODING_AGX3C &&
         (((explicit_lod && !immediate_lod) || register_bias) &&
          (bank_base & 7))) ||
        register_offset || (bank_base & 3) ||
        bank_base > AGX3_STAGING_HALFS -
           (m5_banked_3d && gradient_lod
               ? (vertex_stage ? 40 : 32) : (array || three_d) ? 8 : 4)))
      return agx3_tex_decline(I, __FILE__, __LINE__);
   const uint16_t coord = (enc == AGX_ISA_ENCODING_AGX3B ||
                           enc == AGX_ISA_ENCODING_AGX3C ||
                           native_staged_coords || explicit_lod ||
                           register_bias || array || three_d || cube ||
                           gradient_lod || shadow)
                             ? bank_base
                             : I->src[0].value;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t len = 0;

   const uint16_t offset_base = I->src[5].value + (shadow ? 2 : 0);
   /* The comparison 2D-array bias and the staged 3D bias read their minimum
    * from a packed register. */
   const bool register_minimum = enc == AGX_ISA_ENCODING_AGX3B && bias_min &&
      ((array && shadow) || (three_d && staged_packed_offset));

   const bool bank8_minimum = enc == AGX_ISA_ENCODING_AGX3C && bias_min &&
      ((array && shadow && !cube) || (three_d && staged_packed_offset));
   /* A runtime offset arrives as one NIR-packed bank-6 word that already
    * carries the layer and the bias, or the minimum: a gradient's, or a
    * non-array bias's. Only the 2D-array bias still stages its minimum. */
   const bool packed_array_word = array && staged_packed_offset;
   /* C's COPY names exactly one lifetime hint per source, as its other
    * staging publishers do: cache unless this is the last use. */
   const bool word_cache = enc == AGX_ISA_ENCODING_AGX3C
      ? !I->src[5].discard : I->src[5].cache;
   if (bias_min && array && !cube) {
      /* The minimum moves out of bank 6 so the bias can join the layer
       * there: to half 4 as Apple publishes it, or for a comparison, whose
       * reference owns half 4, to a packed register SMP names. The register
       * pack carries no wait, so it follows the staging that does. The bias
       * and minimum halves belong to one register, so only its last read
       * may discard it. */
      const bool minimum_last = shadow || packed_array_word;
      if (!shadow &&
          (!agx3_tex_stage_array_minimum_lod(
              enc, I->src[1].value + 1, stage_slot, wait_mask,
              I->src[1].cache, minimum_last && I->src[1].discard, bytes,
              &len) ||
           !agx3_append(emission, bytes, len)))
         return agx3_tex_decline(I, __FILE__, __LINE__);
      const bool bank6 = packed_array_word
         ? agx3_tex_stage_copy_hints(
              enc, 6, offset_base, stage_slot, shadow ? wait_mask : 0,
              word_cache, I->src[5].discard, bytes, &len)
         : agx3_tex_stage_array_lod(
              enc, I->src[0].value + 4, I->src[1].value, stage_slot,
              shadow ? wait_mask : 0, I->src[0].cache, I->src[0].discard,
              I->src[1].cache, !minimum_last && I->src[1].discard, bytes,
              &len);
      if (!bank6 || !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
      if (shadow &&
          (!(bank8_minimum
                ? agx3_tex_stage_minimum_bank8_m5(
                     I->src[1].value + 1, stage_slot, 0, !I->src[1].discard,
                     I->src[1].discard, bytes, &len)
                : agx3_tex_pack_minimum(enc, I->src[1].value + 1,
                                        array_scratch, I->src[1].cache,
                                        I->src[1].discard, bytes, &len)) ||
           !agx3_append(emission, bytes, len)))
         return agx3_tex_decline(I, __FILE__, __LINE__);
      wait_mask = 0;
   } else if (gradient_min && array && !cube) {
      if (!(packed_array_word
               ? agx3_tex_stage_copy_hints(
                    enc, 6, offset_base, stage_slot, wait_mask,
                    word_cache, I->src[5].discard, bytes, &len)
               : agx3_tex_stage_array_minimum_gradient(
                    enc, I->src[0].value + 4,
                    I->src[1].value + 2 * gradient_channels, stage_slot,
                    wait_mask,
                    enc == AGX_ISA_ENCODING_AGX3C ? !I->src[0].discard
                                                  : I->src[0].cache,
                    I->src[0].discard,
                    enc == AGX_ISA_ENCODING_AGX3C ? !I->src[1].discard
                                                  : I->src[1].cache,
                    I->src[1].discard, bytes, &len)) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
      wait_mask = 0;
   } else if (bias_min && three_d && staged_packed_offset) {
      /* A 3D word carries the Z nibble where a 2D one carries the minimum,
       * so the minimum goes to the packed register SMP names. The register
       * pack carries no wait, so it follows the staging that does. */
      if (!agx3_tex_stage_copy_hints(
             enc, 6, offset_base, stage_slot, wait_mask, word_cache,
             I->src[5].discard, bytes, &len) ||
          !agx3_append(emission, bytes, len) ||
          !(bank8_minimum
               ? agx3_tex_stage_minimum_bank8_m5(
                    I->src[1].value + 1, stage_slot, 0, !I->src[1].discard,
                    I->src[1].discard, bytes, &len)
               : agx3_tex_pack_minimum(enc, I->src[1].value + 1,
                                       array_scratch, I->src[1].cache,
                                       I->src[1].discard, bytes, &len)) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
      wait_mask = 0;
   } else if (bias_min && !cube && !staged_packed_offset) {
      if (!agx3_tex_stage_bias_min(
             enc, I->src[1].value, I->src[1].value + 1, stage_slot, wait_mask,
             I->src[1].cache, I->src[1].discard, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
      wait_mask = 0;
   } else if (gradient_min && !staged_packed_offset) {
      if (!agx3_tex_stage_typed_lod(
             enc, I->src[1].value + 2 * gradient_channels, stage_slot,
             wait_mask,
             enc == AGX_ISA_ENCODING_AGX3C ? !I->src[1].discard
                                           : I->src[1].cache,
             I->src[1].discard, AGX3_TEX_LEVEL_MINIMUM, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
      wait_mask = 0;
   } else if (staged_packed_offset) {
      if (!agx3_tex_stage_copy_hints(
             enc, 6, offset_base, stage_slot, wait_mask,
             word_cache, I->src[5].discard, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
      wait_mask = 0;
   } else if (combined_array_lod) {
      if (!agx3_tex_stage_array_lod(
             enc, I->src[0].value + 4, I->src[1].value, stage_slot, wait_mask,
             enc == AGX_ISA_ENCODING_AGX3C ? !I->src[0].discard
                                           : I->src[0].cache,
             I->src[0].discard,
             enc == AGX_ISA_ENCODING_AGX3C ? !I->src[1].discard
                                           : I->src[1].cache,
             I->src[1].discard, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
      wait_mask = 0;
   } else if (((explicit_lod && !immediate_lod) || register_bias) &&
              !cube_register_lod) {
      const bool staged = packed_lod_reg
         ? agx3_tex_stage_copy_hints(enc, bank_base + 6, I->src[1].value,
                                     stage_slot, wait_mask,
                                     !I->src[1].discard, I->src[1].discard,
                                     bytes, &len)
         : agx3_tex_stage_typed_lod_at(enc, bank_base, I->src[1].value,
                                      stage_slot, wait_mask,
                                      I->src[1].cache, I->src[1].discard,
                                      AGX3_TEX_LEVEL_FLOAT, bytes, &len);
      if (!staged ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
      wait_mask = 0;
   }

   /* CEM uses fixed r0/r2/r7 scratch registers. Publish the comparison first
    * so a perfectly valid allocator choice in one of those registers cannot
    * be clobbered by the cube-coordinate calculation. Apple's cube and
    * cube-array comparison shaders likewise publish the reference before the
    * final CEM axes. */
   if (shadow && cube && enc == AGX_ISA_ENCODING_AGX3B &&
       (!agx3_tex_stage_copy_hints(
           enc, 4, I->src[5].value, stage_slot, 0,
           enc == AGX_ISA_ENCODING_AGX3C ? !I->src[5].discard
                                         : I->src[5].cache,
           I->src[5].discard, bytes, &len) ||
        !agx3_append(emission, bytes, len)))
      return agx3_tex_decline(I, __FILE__, __LINE__);

   if (cube) {
      /* Native cube sampling consumes CEM's normalized coordinates and face,
       * not the raw XYZ vector. CEM has no wait fields. A register-descriptor
       * witness may leave only slot 6 active: the finalizers/face producer
       * join that transaction and SMP waits for the complete group. */
      const uint16_t staged_wait = (uint16_t)(1u << (stage_slot - 1));
      if (wait_mask && (!register_operands || wait_mask != staged_wait))
         return agx3_tex_decline(I, __FILE__, __LINE__);

      const uint16_t *scratch = cem_scratch;
      for (unsigned axis = 0; axis < 3; ++axis) {
         const bool discard[3] = {
            axis == 2 && I->src[0].discard,
            axis == 2 && (scalar_cube ? I->src[6].discard : I->src[0].discard),
            axis == 2 && (scalar_cube ? I->src[7].discard : I->src[0].discard),
         };
         const uint16_t cy = scalar_cube ? I->src[6].value : I->src[0].value + 2;
         const uint16_t cz = scalar_cube ? I->src[7].value : I->src[0].value + 4;
         /* Scalar cube coordinates are AGX3B-only and exclude shadow, so
          * the AGX3C reference form always sees uniform discard hints. */
         const bool packed = enc == AGX_ISA_ENCODING_AGX3C && shadow
            ? agx3_tex_cube_cem_calc_reference_to(
                 enc, I->src[0].value, cy, cz, scratch[axis], axis,
                 discard[0], I->src[5].value, bytes, &len)
            : agx3_tex_cube_cem_calc_hints_to(
                 enc, I->src[0].value, cy, cz, scratch[axis], axis, discard,
                 bytes, &len);
         if (!packed ||
             !agx3_append(emission, bytes, len))
            return agx3_tex_decline(I, __FILE__, __LINE__);

         if (axis == 0 &&
             (!agx3_tex_cube_cem_rcp_to(enc, scratch[0], bytes, &len) ||
              !agx3_append(emission, bytes, len)))
            return agx3_tex_decline(I, __FILE__, __LINE__);
      }

      for (unsigned component = 0; component < 2; ++component) {
         if (!agx3_tex_cube_cem_finalize_to(
                enc, component, scratch[0], scratch[component + 1],
                stage_slot, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            return agx3_tex_decline(I, __FILE__, __LINE__);
      }
   } else if (split) {
      /* Split coordinates may live anywhere in the register file: stage
       * them with the wide scalar COPY, as split texel fetches do. Nothing
       * after them reads their registers, so the later read of a shared
       * register discards it. */
      for (unsigned c = 0; c < 2; ++c) {
         const bool last = agx3_tex_coord_discard(I, c, 2);
         if (!agx3_tex_stage_copy_scalar_m5(coord + 2 * c,
                                            agx3_tex_coord(I, c).value,
                                            stage_slot, c ? 0 : wait_mask,
                                            !last, last, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            return agx3_tex_decline(I, __FILE__, __LINE__);
      }
   } else if (!native_staged_coords && !published) {
      if (!agx3_tex_stage_copy_hints(enc, coord, I->src[0].value, stage_slot,
                                    wait_mask,
                                    enc == AGX_ISA_ENCODING_AGX3C
                                       ? !I->src[0].discard
                                       : I->src[0].cache,
                                    I->src[0].discard,
                                    bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
      if (!agx3_tex_stage_copy_hints(enc, coord + 2, I->src[0].value + 2,
                                    stage_slot, 0,
                                    enc == AGX_ISA_ENCODING_AGX3C
                                       ? !I->src[0].discard
                                       : I->src[0].cache,
                                    I->src[0].discard,
                                    bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
   }

   if (three_d && !published &&
       (!agx3_tex_stage_copy_hints(
           enc, coord + 4, I->src[0].value + 4, stage_slot, 0,
           enc == AGX_ISA_ENCODING_AGX3C
              ? !I->src[0].discard : I->src[0].cache,
           I->src[0].discard, bytes, &len) ||
        !agx3_append(emission, bytes, len)))
      return agx3_tex_decline(I, __FILE__, __LINE__);

   /* The driver's noncube array ABI stages the layer into the sampler-private
    * bank and selects it with SMP f77=2.  Apple's direct f31/f37 form belongs
    * to the incompatible APPLE_FRAGMENT profile and is retained only by the
    * standalone texture encoder.
    */
   if (array && !cube && !query_lod && !combined_array_lod &&
       !register_offset &&
       !staged_offset &&
       (!agx3_tex_stage_array_layer_at(
           enc, bank_base, I->src[0].value + 4, stage_slot, 0,
           enc == AGX_ISA_ENCODING_AGX3C ? !I->src[0].discard
                                         : I->src[0].cache,
           I->src[0].discard, bytes, &len) ||
        !agx3_append(emission, bytes, len)))
      return agx3_tex_decline(I, __FILE__, __LINE__);

   if (cube) {
      const uint16_t face = cem_scratch[0] + 2;
      const uint16_t layer = I->src[0].value + 6;
      bool packed;
      if (cube_array && cube_register_lod) {
         /* A bias and minimum publish the minimum at half 4, then the bias
          * beside the face. Both halves belong to one register, so only the
          * second read may discard it. */
         packed = agx3_tex_pack_ms_array_layer(
                     enc, layer, array_scratch, I->src[0].cache,
                     I->src[0].discard, bytes, &len) &&
                  agx3_append(emission, bytes, len) &&
                  (!bias_min ||
                   (agx3_tex_stage_array_minimum_lod(
                       enc, I->src[1].value + 1, stage_slot, 0,
                       I->src[1].cache, false, bytes, &len) &&
                    agx3_append(emission, bytes, len))) &&
                  agx3_tex_stage_cube_array_lod(
                     enc, face, I->src[1].value, array_scratch, stage_slot, 0,
                     enc == AGX_ISA_ENCODING_AGX3C ? !I->src[1].discard
                                                   : I->src[1].cache,
                     I->src[1].discard, bytes, &len);
      } else if (bias_min) {
         packed = agx3_tex_pack_cube_face(enc, face, array_scratch, bytes,
                                          &len) &&
                  agx3_append(emission, bytes, len) &&
                  agx3_tex_stage_cube_bias_min(
                     enc, I->src[1].value, I->src[1].value + 1,
                     array_scratch, stage_slot, 0, I->src[1].cache,
                     I->src[1].discard, bytes, &len);
      } else if (cube_array && !(enc == AGX_ISA_ENCODING_AGX3C && query_lod)) {
         packed = agx3_tex_stage_cube_array_layer(
            enc, face, layer, stage_slot, 0,
            enc == AGX_ISA_ENCODING_AGX3C ? !I->src[0].discard
                                          : I->src[0].cache,
            I->src[0].discard, bytes, &len);
      } else if (cube_register_lod) {
         packed = agx3_tex_stage_cube_lod(
            enc, face, I->src[1].value, stage_slot, 0,
            enc == AGX_ISA_ENCODING_AGX3C ? !I->src[1].discard
                                          : I->src[1].cache,
            I->src[1].discard, bytes, &len);
      } else {
         packed = agx3_tex_stage_cube_z(enc, face, stage_slot, 0, false, true,
                                        bytes, &len);
      }
      if (!packed || !agx3_append(emission, bytes, len)) {
         /* The cube staging forms name half-registers in operand fields of
          * different widths, so a refusal here is about WHICH half the CEM
          * scratch and the layer landed on, not about the sample's shape.
          * Printing the shape alone sends the reader back to hardware. */
         char why[96];
         snprintf(why, sizeof(why),
                  "cube staging face=half%u layer=half%u slot=%u", face,
                  layer, stage_slot);
         agx3_report_refusal_why(I, __FILE__, __LINE__, why);
         return false;
      }
   }

   if (register_offset) {
      if (array &&
          (!agx3_tex_pack_ms_array_layer(
              enc, I->src[0].value + 4, array_scratch,
              I->src[0].cache, I->src[0].discard, bytes, &len) ||
           !agx3_append(emission, bytes, len)))
         return agx3_tex_decline(I, __FILE__, __LINE__);
      if (!agx3_tex_stage_gather_offset(
             enc, I->src[5].value + (shadow ? 2 : 0), array, array_scratch,
             stage_slot, I->src[5].cache, I->src[5].discard, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
   }

   /* The comparison reference belongs to this sample's own staging group,
    * at bank + 4 (see the sampler-private note above), not at a fixed half
    * 4 that later samples would overwrite. */
   if (shadow && !published &&
       (!cube || enc == AGX_ISA_ENCODING_AGX3C) &&
       (!agx3_tex_stage_copy_hints(
           enc, coord + 4, I->src[5].value, stage_slot, 0,
           /* C names exactly one hint; with a staged offset after it the
            * reference copy keeps the register (cache). */
           enc == AGX_ISA_ENCODING_AGX3C
              ? (staged_offset || !I->src[5].discard)
              : I->src[5].cache,
           staged_offset ? false : I->src[5].discard, bytes, &len) ||
        !agx3_append(emission, bytes, len)))
      return agx3_tex_decline(I, __FILE__, __LINE__);

   if (staged_component_offset) {
      if (!agx3_tex_stage_offset_2d(
             enc, offset_base, offset_base + 2, stage_slot, 0,
             I->src[5].cache, I->src[5].discard, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
   }

   /* The gradient scalars follow the coordinates, in the interleaved order
    * (ddx.x, ddy.x, ddx.y, ddy.y) the native kernels stage them in. The
    * collect that produced them is laid out the same way, so the run is
    * copied straight through. */
   if (gradient_lod) {
      static const uint16_t bank[6] = {
         AGX3_TEX_GRADIENT_BANK_DDX_X, AGX3_TEX_GRADIENT_BANK_DDY_X,
         AGX3_TEX_GRADIENT_BANK_DDX_Y, AGX3_TEX_GRADIENT_BANK_DDY_Y,
         AGX3_TEX_GRADIENT_BANK_DDX_Z, AGX3_TEX_GRADIENT_BANK_DDY_Z,
      };

      for (unsigned i = 0; i < gradient_channels; ++i) {
         if (!agx3_tex_stage_copy_hints(enc, bank_base + bank[i],
                                        I->src[1].value + (2 * i), stage_slot,
                                        0, enc == AGX_ISA_ENCODING_AGX3C
                                              ? !I->src[1].discard
                                              : I->src[1].cache,
                                        I->src[1].discard, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            return agx3_tex_decline(I, __FILE__, __LINE__);
      }
   }

   if (pixel_entry) {
      uint8_t entry[AGX3_PIXEL_MAX_LENGTH], entry_len = 0;
      if (pixel_entry_offset == NULL ||
          !agx3_pixel_colour_entry(enc, entry, &entry_len))
         return agx3_tex_decline(I, __FILE__, __LINE__);

      *pixel_entry_offset = emission->size;
      if (!agx3_append(emission, entry, entry_len))
         return agx3_tex_decline(I, __FILE__, __LINE__);
   }

   const struct agx3_tex_sample_desc d = {
      .heap_uniform_base = heap_uniform_base,
      .dest = I->dest[0].value,
      .coord = coord,
      .width = I->dest[0].size == AGX_SIZE_32 ? AGX3_TEX_W32 : AGX3_TEX_W16,
      .mask = I->mask,
      .sparse = I->sparse,
      .texture = texture_immediate ? I->src[3].value : 0,
      .sampler = sampler_immediate ? I->src[4].value : 0,
      .texture_kind = texture_immediate ? AGX3_TEX_OPERAND_IMMEDIATE :
         I->src[3].type == AGX_INDEX_UNIFORM ? AGX3_TEX_OPERAND_UNIFORM : AGX3_TEX_OPERAND_REGISTER,
      .sampler_kind = sampler_immediate ? AGX3_TEX_OPERAND_IMMEDIATE :
         I->src[4].type == AGX_INDEX_UNIFORM ? AGX3_TEX_OPERAND_UNIFORM : AGX3_TEX_OPERAND_REGISTER,
      .texture_value = texture_immediate ? 0 : I->src[3].value,
      .sampler_value = sampler_immediate ? 0 : I->src[4].value,
      .texture_kill = !texture_immediate && I->src[3].kill,
      .sampler_kill = enc != AGX_ISA_ENCODING_AGX3C &&
                        !sampler_immediate && I->src[4].kill,
      .gather = gather_component,
      .shadow = shadow,
      .query_lod = query_lod,
      .offset = offset,
      .offset_staged = staged_offset,
      .offset_staged_array = staged_packed_offset && array,
      .offset_xyz = offset ? I->offset_xyz : 0,
      .offset_register = register_offset,
      .min_lod = min_lod,
      .min_lod_register = register_minimum,
      .min_lod_value = register_minimum ? array_scratch : 0,
      .min_lod_kill = register_minimum,
      .lod = biased_lod       ? AGX3_TEX_LOD_BIAS
             : immediate_lod  ? AGX3_TEX_LOD_LEVEL
             : explicit_lod   ? AGX3_TEX_LOD_REGISTER
             : register_bias || bias_min ? AGX3_TEX_LOD_BIAS_REGISTER
             : gradient_lod   ? AGX3_TEX_LOD_GRADIENT
                              : AGX3_TEX_LOD_AUTO,
      .lod_x64 = immediate_lod ? immediate_x64_field : 0,
      .array = array,
      .array_layer = array && !query_lod
                        ? I->src[0].value + (cube_array ? 6 : 4) : 0,
      .three_d = three_d,
      .cube = cube,
      .slot = slot,
      .wait_mask = published ? wait_mask
                             : (uint16_t)(1u << (stage_slot - 1)),
      /* CEM consumes and discards the source XYZ before SMP. Apple's cube
       * and cube-array samples consequently set SMP f27 even though the
       * sampler reads CEM's private coordinate bank rather than the original
       * AGX IR source. Ordinary dimensions retain their existing last-use
       * mapping. */
      .coord_kill = published ? publication_last
                      : cube ? true
                         : native_staged_coords
                              ? true
                              : enc == AGX_ISA_ENCODING_AGX3C
                                   ? true
                              : I->src[0].kill,
      .final = final,
      .sample_group_nonterminal = sample_group_nonterminal,
      .sample_group_end = sample_group_end,
      .vertex_stage = vertex_stage,
      .compute_stage = compute_stage,
      .sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER,
   };
   return agx3_tex_encode_sample(enc, &d, bytes, &len) &&
          agx3_append(emission, bytes, len);
}

bool
agx3_reusable_cube_scratch(agx_context *ctx, unsigned maximum,
                           uint16_t scratch[3])
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble || !ctx->nir ||
       ctx->nir->info.stage != MESA_SHADER_FRAGMENT ||
       list_length(&ctx->blocks) != 1 || ctx->any_cf)
      return false;
   agx_foreach_instr_global(ctx, I) {
      switch (I->op) {
      case AGX_OPCODE_FFMA:
      case AGX_OPCODE_FMUL:
      case AGX_OPCODE_FADD:
      case AGX_OPCODE_FCMPSEL:
      case AGX_OPCODE_RSQRT:
      case AGX_OPCODE_SRSQRT:
      case AGX_OPCODE_RCP:
      case AGX_OPCODE_LOG2:
      case AGX_OPCODE_EXP2:
      case AGX_OPCODE_GET_SR:
      case AGX_OPCODE_CONVERT:
      case AGX_OPCODE_MOV:
      case AGX_OPCODE_BITOP:
      case AGX_OPCODE_INTERP_AGX3:
      case AGX_OPCODE_PERSP_MUL_AGX3:
      case AGX_OPCODE_RHW_INTERP_AGX3:
      case AGX_OPCODE_PERSP_RCP_AGX3:
      case AGX_OPCODE_EXPORT:
      case AGX_OPCODE_WAIT_PIX:
      case AGX_OPCODE_SIGNAL_PIX:
      case AGX_OPCODE_STOP:
      case AGX_OPCODE_PUBLISH_TEXTURE_AGX3:
      case AGX_OPCODE_PUBLISH_TEXTURE_SCALAR_AGX3:
         break;
      case AGX_OPCODE_TEXTURE_SAMPLE:
         if ((I->dim != AGX_DIM_2D && I->dim != AGX_DIM_CUBE) || I->shadow ||
             I->offset || I->query_lod || I->sparse ||
             I->gather != AGX_GATHER_NONE ||
             I->lod_mode != AGX_LOD_MODE_AUTO_LOD || I->nr_srcs < 6)
            return false;
         for (unsigned q = 3; q <= 4; ++q) {
            if (I->src[q].type != AGX_INDEX_UNIFORM &&
                I->src[q].type != AGX_INDEX_IMMEDIATE)
               return false;
         }
         break;
      default:
         return false;
      }
   }

   BITSET_DECLARE(live, AGX_NUM_REGS) = {0};
   BITSET_DECLARE(forbidden, AGX_NUM_REGS) = {0};
   BITSET_DECLARE(asynchronous_writes, AGX_NUM_REGS) = {0};
   agx_foreach_instr_global_rev(ctx, I) {
      if (I->op == AGX_OPCODE_TEXTURE_SAMPLE && I->dim == AGX_DIM_CUBE) {
         BITSET_OR(forbidden, forbidden, live);
         agx_foreach_reg_dest(I, d)
            BITSET_SET_COUNT(forbidden, I->dest[d].value,
                             agx_index_size_16(I->dest[d]));
         agx_foreach_reg_src(I, q)
            BITSET_SET_COUNT(forbidden, I->src[q].value,
                             agx_index_size_16(I->src[q]));
      }
      agx_foreach_reg_dest(I, d)
         BITSET_CLEAR_COUNT(live, I->dest[d].value,
                            agx_index_size_16(I->dest[d]));
      agx_foreach_reg_src(I, q)
         BITSET_SET_COUNT(live, I->src[q].value, agx_index_size_16(I->src[q]));
   }
   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_TEXTURE_SAMPLE && I->dim == AGX_DIM_CUBE) {
         BITSET_OR(forbidden, forbidden, asynchronous_writes);
         BITSET_DECLARE(result, AGX_NUM_REGS) = {0};
         agx_foreach_reg_dest(I, d)
            BITSET_SET_COUNT(result, I->dest[d].value,
                             agx_index_size_16(I->dest[d]));
         agx_block *block = list_first_entry(&ctx->blocks, agx_block, link);
         for (struct list_head *node = I->link.next;
              node != &block->instructions; node = node->next) {
            agx_instr *next = list_entry(node, agx_instr, link);
            bool consumes_result = false;
            agx_foreach_reg_src(next, q)
               consumes_result |= BITSET_TEST_COUNT(
                  result, next->src[q].value, agx_index_size_16(next->src[q]));
            if (consumes_result)
               break;
            agx_foreach_reg_dest(next, d)
               BITSET_SET_COUNT(forbidden, next->dest[d].value,
                                agx_index_size_16(next->dest[d]));
         }
      }
      if (agx3_is_async(I)) {
         agx_foreach_reg_dest(I, d)
            BITSET_SET_COUNT(asynchronous_writes, I->dest[d].value,
                             agx_index_size_16(I->dest[d]));
      }
   }
   const unsigned width[3] = {4, 2, 2};
   for (unsigned c = 0; c < 3; ++c) {
      bool found = false;
      unsigned limit = MIN2(maximum + 1, c == 0 ? 128 : 192);
      for (unsigned reg = 16; reg + width[c] <= limit; reg += 2) {
         if (BITSET_TEST_COUNT(forbidden, reg, width[c]))
            continue;
         scratch[c] = reg;
         BITSET_SET_COUNT(forbidden, reg, width[c]);
         found = true;
         break;
      }
      if (!found)
         return false;
   }
   return true;
}

enum agx3_pack_result
agx3_pack_tex_pre_instr(struct agx3_pack_state *st)
{
   struct util_dynarray *emission = st->emission;
   const enum agx_isa_encoding enc = st->enc;
   const agx_instr *I = st->I;

   if (I->op == AGX_OPCODE_PACK_SAMPLER_LOD_AGX3) {
      if (enc != AGX_ISA_ENCODING_AGX3C || I->nr_dests != 1 ||
          I->nr_srcs != 1 || I->saturate ||
          !agx3_tex_plain_reg(I->dest[0]) ||
          I->dest[0].size != AGX_SIZE_32 ||
          agx_channels(I->dest[0]) != 1 ||
          !agx3_tex_plain_reg(I->src[0]) ||
          I->src[0].size != AGX_SIZE_16 ||
          agx_channels(I->src[0]) != 1)
         AGX3_REFUSE_RET(I);
      uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
      const bool encoded = agx3_tex_pack_float_lod_m5(
         I->src[0].value, I->dest[0].value,
         I->src[0].cache, I->src[0].discard, st->wait, bytes, &len);
      if (!encoded || !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   return AGX3_PACK_UNHANDLED;
}

enum agx3_pack_result
agx3_pack_tex_pub_instr(struct agx3_pack_state *st)
{
   agx_context *ctx = st->ctx;
   struct util_dynarray *emission = st->emission;
   const enum agx_isa_encoding enc = st->enc;
   const agx_instr *I = st->I;

   if (agx_is_texture_publication(I)) {
      if (I->agx3_tex_coords_staged) {
         if (enc != AGX_ISA_ENCODING_AGX3B || !st->native_coord_slot ||
             !I->imm || I->imm > st->resource_capacity ||
             !st->publication_values[I->imm])
            AGX3_REFUSE_RET(I);
         const unsigned value = st->publication_values[I->imm] - 1;
         const uint16_t base = st->resource_bases[value];
         agx3_staging_release(&st->staging_banks, st->outstanding_here);
         if (st->resources[value].halves != 4 ||
             !agx3_staging_hold(&st->staging_banks, base, 4, st->native_coord_slot))
            AGX3_REFUSE_RET(I);
         ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, base + 4);
         return AGX3_PACK_EMITTED;
      }
      const bool scalar = I->op == AGX_OPCODE_PUBLISH_TEXTURE_SCALAR_AGX3;
      const bool m5_scalar = enc == AGX_ISA_ENCODING_AGX3C && scalar &&
         I->nr_srcs == 3 && !agx_is_null(I->src[2]);
      if ((!m5_scalar && enc != AGX_ISA_ENCODING_AGX3B) || !st->slot ||
          I->nr_srcs != (scalar ? 3 : 2) ||
          !I->imm || I->imm > st->resource_capacity ||
          !st->publication_values[I->imm])
         AGX3_REFUSE_RET(I);
      for (unsigned c = 0; c < I->nr_srcs; ++c) {
         const bool optional = c == (scalar ? 2 : 1);
         if (optional && agx_is_null(I->src[c]))
            continue;
         if (!agx3_tex_plain_reg(I->src[c]) ||
             I->src[c].size != AGX_SIZE_32 ||
             agx_channels(I->src[c]) != ((!scalar && c == 0) ? 2 : 1))
            AGX3_REFUSE_RET(I);
      }
      const unsigned value = st->publication_values[I->imm] - 1;
      const uint16_t base = st->resource_bases[value];
      const uint16_t span = st->resources[value].halves;
      agx3_staging_release(&st->staging_banks, st->outstanding_here);
      const uint16_t retire = st->wait |
         agx3_staging_conflicts(&st->staging_banks, base, span);
      /* COPY's wait bit does not enforce the required G17P producer
       * dependency. Keep that wait on its qualified carrier, while
       * the publication owns an independent completion event.
       */
      if (retire) {
         /* The mask covers this publication's source producers. An
          * unrelated r0 generation can still be pending, so carry
          * the wait on the first coordinate instead of modifying r0.
          * A wide M5 coordinate may live above half 192, which only
          * the halfword identity reaches. */
         if (!agx3_retire_wait_on(emission, enc, I->src[0], retire))
            AGX3_REFUSE_RET(I);
      }
      st->outstanding &= (uint16_t)~retire;
      st->outstanding |= (uint16_t)(1u << (st->slot - 1));
      agx3_staging_forget(&st->staging_banks, retire);
      for (unsigned c = 0; c < span / 2; ++c) {
         const agx_index src = I->src[scalar ? c : (c < 2 ? 0 : 1)];
         const uint16_t reg = src.value + (!scalar && c == 1 ? 2 : 0);
         uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
         /* Scalar coordinates/reference may alias one physical GPR.
          * The IR publication reads all sources together; its discard
          * annotation must land on the last native read of that GPR.
          */
         bool last = true;
         for (unsigned later = c + 1; later < span / 2; ++later) {
            const agx_index next_src =
               I->src[scalar ? later : (later < 2 ? 0 : 1)];
            const uint16_t next_reg =
               next_src.value + (!scalar && later == 1 ? 2 : 0);
            last &= next_reg != reg;
         }
         const bool wide_scalar = m5_scalar;
         const bool copied = wide_scalar
            ? agx3_tex_stage_copy_scalar_m5(
                 base + 2 * c, reg, st->slot, 0, !(src.discard && last),
                 src.discard && last, bytes, &len)
            : agx3_tex_stage_copy_hints(
                 enc, base + 2 * c, reg, st->slot, 0,
                 m5_scalar ? !(src.discard && last) : src.cache,
                 src.discard && last, bytes, &len);
         if (!copied || !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
      }
      if (!agx3_staging_hold(&st->staging_banks, base, span, st->slot))
         AGX3_REFUSE_RET(I);
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, base + span);
      return AGX3_PACK_EMITTED;
   }

   if (I->agx3_tex_coord_stage) {
      if (enc != AGX_ISA_ENCODING_AGX3B || st->native_coord_slot == 0 ||
          I->nr_dests != 1 || I->dest[0].size != AGX_SIZE_32 ||
          agx_channels(I->dest[0]) != 1)
         AGX3_REFUSE_RET(I);

      if (!st->native_coord_values[st->instr_index])
         AGX3_REFUSE_RET(I);
      const uint16_t staging_dest =
         st->resource_bases[st->native_coord_values[st->instr_index] - 1] +
         2 * (uint16_t)I->agx3_tex_coord_component;
      /* An unpublished sample reads its fixed staging bank through
       * completion, which the resource allocator does not model. Retire
       * any such reader before overwriting its coordinates. */
      agx3_staging_release(&st->staging_banks, st->outstanding_here);
      const uint16_t staging_war =
         agx3_staging_conflicts(&st->staging_banks, staging_dest, 2) & ~st->wait;
      if (staging_war) {
         if (!agx3_retire_slots_noop(emission, enc, staging_war))
            AGX3_REFUSE_RET(I);
         st->outstanding &= (uint16_t)~staging_war;
         agx3_staging_forget(&st->staging_banks, staging_war);
      }
      uint8_t bytes[AGX3_ALU_MAX_LENGTH];
      uint8_t len = 0;
      if (I->op == AGX_OPCODE_PERSP_MUL_AGX3) {
         if (I->nr_srcs != 2 || I->saturate || I->imm > 0x7f ||
             !agx3_vary_reg32(I->src[0]) ||
             !agx3_vary_reg32(I->src[1]) ||
             agx_channels(I->src[0]) != 1 ||
             agx_channels(I->src[1]) != 1 || I->src[0].kill ||
             I->src[1].kill ||
             (I->src[0].cache && I->src[0].discard) ||
             (I->src[1].cache && I->src[1].discard))
            AGX3_REFUSE_RET(I);
         const struct agx3_persp_mul_desc desc = {
            .stage_dest = true,
            .dest = staging_dest,
            .slot = st->native_coord_slot,
            .raw = I->src[0].value,
            .reciprocal = I->src[1].value,
            .raw_liveness = I->src[0].discard ? AGX3_OPERAND_KILL
               : I->src[0].cache ? AGX3_OPERAND_LIVE
                                 : AGX3_OPERAND_UNMARKED,
            .reciprocal_liveness =
               I->src[1].discard ? AGX3_OPERAND_KILL
               : I->src[1].cache ? AGX3_OPERAND_LIVE
                                 : AGX3_OPERAND_UNMARKED,
            .vary_index = I->imm,
            .wait_mask = st->wait,
         };
         if (!agx3_persp_mul_encode(enc, &desc, bytes, &len))
            AGX3_REFUSE_RET(I);
      } else if (I->op == AGX_OPCODE_FADD) {
         if (I->nr_srcs != 2 || I->saturate)
            AGX3_REFUSE_RET(I);
         uint16_t src[2];
         enum agx3_alu_hint hints[2];
         for (unsigned s = 0; s < 2; ++s) {
            if (!agx3_vary_reg32(I->src[s]) ||
                agx_channels(I->src[s]) != 1 || I->src[s].abs ||
                I->src[s].neg || I->src[s].kill ||
                (I->src[s].cache && I->src[s].discard))
               AGX3_REFUSE_RET(I);
            src[s] = I->src[s].value;
            hints[s] = agx3_post_ra_hint(I->src[s]);
         }
         if (!agx3_alu_fadd_stage(enc, staging_dest,
                                  st->native_coord_slot, src, hints, st->wait,
                                  bytes, &len))
            AGX3_REFUSE_RET(I);
      } else if (I->op == AGX_OPCODE_FFMA) {
         if (I->nr_srcs != 3 || I->saturate)
            AGX3_REFUSE_RET(I);
         const unsigned first =
            I->src[1].type == AGX_INDEX_UNIFORM ||
                  I->src[1].type == AGX_INDEX_IMMEDIATE
               ? 1
               : 0;
         const unsigned order[3] = {first, first ^ 1, 2};
         const enum agx3_stage_fma_source source0 =
            I->src[first].type == AGX_INDEX_UNIFORM
               ? AGX3_STAGE_FMA_UNIFORM
            : I->src[first].type == AGX_INDEX_IMMEDIATE
               ? AGX3_STAGE_FMA_IMMEDIATE
               : AGX3_STAGE_FMA_REGISTER;
         uint16_t src[3];
         bool live[3], discard[3];
         for (unsigned i = 0; i < 3; ++i) {
            const agx_index value = I->src[order[i]];
            const bool immediate =
               i == 0 && source0 == AGX3_STAGE_FMA_IMMEDIATE;
            const bool special =
               i == 0 && source0 != AGX3_STAGE_FMA_REGISTER;
            if (value.size != (immediate ? AGX_SIZE_16 : AGX_SIZE_32) ||
                agx_channels(value) != 1 || value.memory || value.abs ||
                value.neg || value.kill || value.value >= 256 ||
                (!immediate && (value.value & 1)) ||
                (value.cache && value.discard) ||
                (special ? value.cache || value.discard
                         : value.type != AGX_INDEX_REGISTER))
               AGX3_REFUSE_RET(I);
            src[i] = value.value;
            live[i] = value.cache;
            discard[i] = value.discard;
         }
         if (!agx3_stage_fma(enc, staging_dest, st->native_coord_slot, src,
                             source0, live, discard, st->wait, bytes, &len))
            AGX3_REFUSE_RET(I);
      } else {
         AGX3_REFUSE_RET(I);
      }

      if (!agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   return AGX3_PACK_UNHANDLED;
}

enum agx3_pack_result
agx3_pack_tex_instr(struct agx3_pack_state *st)
{
   agx_context *ctx = st->ctx;
   struct util_dynarray *emission = st->emission;
   const enum agx_isa_encoding enc = st->enc;
   const agx_instr *I = st->I;

   if (I->op == AGX_OPCODE_TEXTURE_LOAD ||
       I->op == AGX_OPCODE_IMAGE_LOAD) {
      /* Native TEX reads staging coordinates, independently of GPRs.
       * This also applies to the background reload's pixel coordinates. */
      /* Apple's argument-buffer vertex read publishes uint2 coordinates
       * before SMP.EP4. G17P indexed imageLoad uses texelFetch's
       * read-only texture heap and coordinate ABI. Legacy immediate
       * IMAGE_LOAD reads keep the unstaged form on AGX3B. M5 vertex
       * image reads also require coordinate publication; without it a
       * storage-texel read followed by a write stores incorrect values. */
      const bool vertex_fetch_stages_coordinates =
         (enc == AGX_ISA_ENCODING_AGX3B ||
          enc == AGX_ISA_ENCODING_AGX3C) &&
         ctx->stage == MESA_SHADER_VERTEX &&
         (I->op == AGX_OPCODE_TEXTURE_LOAD ||
          I->op == AGX_OPCODE_IMAGE_LOAD ||
          enc == AGX_ISA_ENCODING_AGX3C);
      const bool staged =
         /* G18 image/texture fetches publish their operands even in
          * lowered vertex programs used for GS/TES rasterization. */
         (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d) ||
         (enc == AGX_ISA_ENCODING_AGX3B &&
          I->dim == AGX_DIM_2D_MS_ARRAY) ||
         vertex_fetch_stages_coordinates ||
         ((I->op == AGX_OPCODE_TEXTURE_LOAD ||
           enc == AGX_ISA_ENCODING_AGX3B) &&
          ctx->stage == MESA_SHADER_VERTEX &&
          I->src[3].type != AGX_INDEX_IMMEDIATE) ||
         ctx->stage == MESA_SHADER_FRAGMENT ||
         ctx->stage == MESA_SHADER_MESH ||
         (enc == AGX_ISA_ENCODING_AGX3C &&
          ctx->stage == MESA_SHADER_TASK) ||
         agx3_stage_runs_as_kernel(ctx->stage);
      /* imgwci takes its coordinates from staging halves 0/2, so in a
       * shader that also writes images a read stages its coordinates at
       * halves 16/18 (smp f19=16), as Apple does: an unwaited write
       * COPY could replace them before the read consumes them. */
      const bool separate_write_coords =
         enc == AGX_ISA_ENCODING_AGX3B && st->image_writes_left != 0 &&
         ((st->g17p_storage_image_protocol &&
           I->op == AGX_OPCODE_TEXTURE_LOAD) ||
          I->op == AGX_OPCODE_IMAGE_LOAD) &&
         (I->dim == AGX_DIM_2D || I->dim == AGX_DIM_2D_ARRAY ||
          I->dim == AGX_DIM_3D);
      /* A fetch whose level is a live register, which is the shape
       * whose level stages into the sampler-private bank. */
      const bool register_fetch_lod =
         I->op == AGX_OPCODE_TEXTURE_LOAD &&
         I->lod_mode == AGX_LOD_MODE_LOD_MIN &&
         I->src[1].type == AGX_INDEX_REGISTER;
      /* A staged read consumes, and kills, its coordinate bank
       * asynchronously, so every stage that stages coordinates
       * (including vertex fetches and background, clear and
       * end-of-tile programs) uses the bank allocator: a later read
       * must not restage a bank an earlier one still consumes. With at
       * most one read in flight it returns the base bank and emits no
       * retirement. */
      const bool staged_bank_reads = staged;
      if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d) {
      if (staged) {
         const unsigned coord_base =
            (I->op == AGX_OPCODE_IMAGE_LOAD &&
             I->src[2].type == AGX_INDEX_IMMEDIATE &&
             I->dim != AGX_DIM_2D_MS_ARRAY &&
             !st->g18g_image_read_bank0) || separate_write_coords
               ? 16
               : 0;
         /* G17G keys a staged block by scoreboard slot, exactly as the
          * sample path below does: a fetch issued on slot N publishes
          * and reads its operands 8 * (N - 1) halves up the staging
          * file. The stagers and the fetch agree on that block by
          * themselves -- both carry the slot -- but the *allocation*
          * did not, so a fetch that landed on a late slot wrote past
          * the sampler-private size and read an untouched block. That
          * returns zero for every coordinate without faulting, which
          * is how a 2D-array texel fetch came back empty while the
          * same shader's array comparison and every plain 2D fetch
          * (always slot 1, base 0) were correct. */
         const unsigned block_base =
            enc == AGX_ISA_ENCODING_AGX3C ? 8u * (st->slot - 1) : 0;
         const unsigned coord_count =
            (I->dim == AGX_DIM_2D_MS ||
             I->dim == AGX_DIM_2D_MS_ARRAY ||
             I->dim == AGX_DIM_2D_ARRAY)
               ? 2
               : agx_channels(I->src[0]);
         ctx->agx3_staging_halfs = MAX2(
            ctx->agx3_staging_halfs,
            block_base + coord_base + 2 * coord_count);
         /* G18G's 3D Z word is Y + 4, the block's last two halves. */
         if (ctx->key->dev.emit_agx3d && I->dim == AGX_DIM_3D)
            ctx->agx3_staging_halfs =
               MAX2(ctx->agx3_staging_halfs, block_base + coord_base + 8);
         /* PACK_SMPIR publishes the non-array MS sample at half4.
          * Native MS fragments reserve eight halves, not just the
          * four occupied by the x/y coordinate COPYs. */
         if (I->dim == AGX_DIM_2D_ARRAY || I->dim == AGX_DIM_2D_MS ||
             I->dim == AGX_DIM_2D_MS_ARRAY)
            ctx->agx3_staging_halfs =
               MAX2(ctx->agx3_staging_halfs, block_base + 8);
         /* A per-lane level is published into bank 6 of the same
          * block. A plain 2D fetch reserves only its two coordinate
          * words, so that bank fell outside the sampler-private
          * allocation and the fetch read zero -- the same failure
          * this path already hit on G17P, which the dimension-based
          * widening above does not cover for 2D. Nothing faults: the
          * hardware allocates staging from this size and a write
          * beyond it simply never reaches the sampler. */
         if (I->lod_mode == AGX_LOD_MODE_LOD_MIN &&
             agx3_tex_plain_reg(I->src[1]))
            ctx->agx3_staging_halfs =
               MAX2(ctx->agx3_staging_halfs, block_base + coord_base + 8);
         /* Compact writes own halves 0..15. Do not infer disjointness
          * from the IR shape alone: check the selected read bank. */
         if (st->g18g_compact_image_banks && block_base + coord_base < 16)
            AGX3_REFUSE_RET(I);
         /* The whole block, for the G18G image-write bank floor
          * (AGX3_G18G_READ_STAGING_LIMIT). */
         st->g18g_other_staging_end = MAX2(st->g18g_other_staging_end,
                                       block_base + coord_base + 8);
         st->g18g_read_staging_end = MAX2(st->g18g_read_staging_end,
                                      block_base + coord_base + 8);
      }
      else
         ctx->agx3_staging_known = false;
      }
      assert(st->texture_ops_left > 0);
      const bool final = --st->texture_ops_left == 0;
      /* G18's non-plain fetches publish into fixed coordinate records.
       * Native541's concurrent array reads use distinct Y/layer
       * offsets, not distinct scoreboard slots alone. Retire previous
       * consumers before publishing this fixed record. Complete this
       * read below before a later sample or write can reuse the record. */
      const bool g18_fixed_fetch_record =
         enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
         staged && !agx3_g18g_plain_fetch_block(I);
      if (g18_fixed_fetch_record) {
         if (!agx3_retire_slots_noop(emission, enc,
                                     st->wait | st->outstanding_here))
            AGX3_REFUSE_RET(I);
         st->outstanding &= (uint16_t)~st->outstanding_here;
      }
      const uint16_t fetch_wait =
         g18_fixed_fetch_record ? 0 :
         st->wait | (((enc == AGX_ISA_ENCODING_AGX3B ||
                   ctx->key->dev.emit_agx3d) &&
                  (I->dim == AGX_DIM_2D_MS ||
                   I->dim == AGX_DIM_2D_MS_ARRAY))
                    ? (st->outstanding_here & 1u)
                    : 0);
      /* A fragment texel fetch with a per-lane level stages that level
       * into the sampler-private bank; Apple enters the pixel
       * transaction before such an SMP. Keep the same straight-line
       * gate placement rule as the implicit-derivative samples.
       * Native G18 fragments also enter before reading a multisample
       * texture when the destination itself is single-sampled. */
      const bool g18_ms_fragment_fetch =
         enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
         ctx->stage == MESA_SHADER_FRAGMENT &&
         !ctx->key->fs.ignore_tib_dependencies &&
         ctx->agx3_colour_link.straight_line &&
         (I->dim == AGX_DIM_2D_MS || I->dim == AGX_DIM_2D_MS_ARRAY);
      const bool open_pixel_fetch =
         ((enc == AGX_ISA_ENCODING_AGX3B &&
         ctx->stage == MESA_SHADER_FRAGMENT &&
         !ctx->key->fs.ignore_tib_dependencies &&
         ctx->agx3_colour_link.straight_line &&
         register_fetch_lod && I->dim == AGX_DIM_2D) ||
          st->g18_sample_colour_stores || g18_ms_fragment_fetch) &&
         !st->g17p_colour_open;
      /* A slot may wrap while its previous read still owns this
       * explicit block. Retire that consumer before republishing. */
      if (ctx->key->dev.emit_agx3d && staged &&
          agx3_g18g_plain_fetch_block(I) &&
          (st->outstanding_here & (1u << (st->slot - 1))) &&
          !agx3_retire_slots_noop(emission, enc,
                                    (uint16_t)(1u << (st->slot - 1))))
         AGX3_REFUSE_RET(I);
      uint16_t staged_end = 0;
      const bool share_fetch_coords =
         enc == AGX_ISA_ENCODING_AGX3C &&
         ctx->stage == MESA_SHADER_COMPUTE && staged;
      const agx_instr *next = (st->instr_index + 1) < st->count ? st->ordered[(st->instr_index + 1)] : NULL;
      st->fetch_group.reuse = share_fetch_coords && st->fetch_group.keep;
      st->fetch_group.keep = share_fetch_coords && next &&
         I->link.next == &next->link &&
         agx3_fetch_coordinates_match(I, next);
      if (!agx3_emit_texture_fetch(emission, enc, I, st->slot, fetch_wait,
                                   final, staged, separate_write_coords,
                                   st->g18g_image_read_bank0,
                                   ctx->key->dev.emit_agx3d,
                                   st->ms_array_scratch,
                                   staged_bank_reads ? &st->staging_banks
                                                     : NULL,
                                   st->outstanding_here, &staged_end,
                                   ctx->stage == MESA_SHADER_FRAGMENT,
                                   open_pixel_fetch,
                                   &ctx->agx3_colour_link.entry_offset,
                                   ctx->key->g17p_texture_uniform_base,
                                   share_fetch_coords ? &st->fetch_group : NULL))
         AGX3_REFUSE_RET(I);
      if (g18_fixed_fetch_record) {
         const uint16_t consumer = (uint16_t)(1u << (st->slot - 1));
         if (!agx3_retire_slots_noop(emission, enc, consumer))
            AGX3_REFUSE_RET(I);
         st->outstanding &= (uint16_t)~consumer;
      }
      if (open_pixel_fetch) {
         ctx->agx3_colour_link.entry_emitted = true;
         ctx->agx3_colour_link.entry_offset -= st->initial_size;
         st->g17p_colour_open = true;
      }
      if (staged) {
         /* The coordinate bank, or the eight halves of a read with
          * PACK_SMPIR operands: native MS fragments reserve eight
          * halves, not just the four occupied by the x/y COPYs. */
         ctx->agx3_staging_halfs =
            MAX2(ctx->agx3_staging_halfs, staged_end);
         st->g18g_read_staging_end = MAX2(st->g18g_read_staging_end, staged_end);
      } else {
         ctx->agx3_staging_known = false;
      }
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_IMAGE_WRITE && ctx->key->dev.emit_agx3d) {
      if (!agx3_emit_storage_image_write_g18g(
              emission, enc, I, st->wait, ctx->stage, &st->staging_banks,
              st->g18g_image_write_floor,
              st->g18g_compact_image_banks ? 16 : AGX3_STAGED_WRITE_LIMIT,
              &st->outstanding))
         AGX3_REFUSE_RET(I);
      /* Every write holds a sixteen-half bank; the first, in bank 0,
       * reserves 16 halves (specTPR f22 8, native kick 19 6: a
       * superset). */
      ctx->agx3_staging_halfs =
         MAX2(ctx->agx3_staging_halfs, MAX2(st->staging_banks.end, 16));
      st->g18g_image_write_placed = true;
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_IMAGE_WRITE) {
      uint8_t write_slot;
      if (!agx3_emit_storage_image_write(emission, enc,
                                         ctx->key->dev.chip, I, st->wait,
                                         &st->staging_banks,
                                         st->image_write_bank_floor,
                                         &st->outstanding, &write_slot,
                                         ctx->key->g17p_texture_uniform_base,
                                         ctx->key->g17p_storage_image_pair))
         AGX3_REFUSE_RET(I);
      st->image_write_slots &= st->outstanding;
      st->image_write_slots |= 1u << (write_slot - 1);
      ctx->agx3_staging_halfs =
         MAX2(ctx->agx3_staging_halfs, MAX2(st->staging_banks.end, 16));
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_TEXTURE_SAMPLE) {
      /* Geometry/TCS lowering retains the API stage in kernel NIR;
       * TES may likewise retain its vertex-stage identity. Sampling
       * must use the native execution stage, not fragment semantics. */
      const bool vertex_sample_stage =
         ctx->stage == MESA_SHADER_VERTEX ||
         (ctx->key->dev.emit_agx3d &&
          ctx->stage == MESA_SHADER_TESS_EVAL);
      const bool compute_sample_stage =
         ctx->stage == MESA_SHADER_COMPUTE ||
         (ctx->key->dev.emit_agx3d &&
          (ctx->stage == MESA_SHADER_GEOMETRY ||
           ctx->stage == MESA_SHADER_TESS_CTRL));
      /* Native gradients and runtime/array gathers use fixed blocks that
       * overlap other sampler publications. Retire their prior users
       * before replacing any operand. */
      const bool g18g_gradient_sample = ctx->key->dev.emit_agx3d &&
         I->lod_mode == AGX_LOD_MODE_LOD_GRAD;
      const bool gpu1_fixed_gather =
         enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
         (agx3_g17g_runtime_gather(I) || agx3_g17g_array_gather(I) ||
          agx3_g17g_cube_gather(I) ||
          (ctx->key->dev.emit_agx3d && agx3_g18g_indirect_gather(I) &&
           I->offset && !I->offset_xyz));
      const bool g18g_indirect_gather = ctx->key->dev.emit_agx3d &&
         agx3_g18g_indirect_gather(I);
      const bool g18g_colour_level_sample = ctx->key->dev.emit_agx3d &&
         (agx3_g18g_colour_sample(I) || agx3_g18g_binding_sample(I));
      if (g18g_gradient_sample && ctx->stage != MESA_SHADER_VERTEX &&
          ctx->stage != MESA_SHADER_FRAGMENT)
         AGX3_REFUSE_RET(I);
      if (g18g_gradient_sample || gpu1_fixed_gather ||
          g18g_colour_level_sample) {
         const uint16_t live = st->outstanding_here;
         if (live && !agx3_retire_slots_noop(emission, enc, live))
            AGX3_REFUSE_RET(I);
         st->wait &= (uint16_t)~live;
         st->outstanding &= (uint16_t)~live;
      }
      const bool staged_sample =
         !I->query_lod && !I->agx3_tex_coords_staged &&
         !agx_texture_resource(I);
      {
         /* A texture sample's planned scoreboard wait is retired on a
          * carrier instruction, not encoded in the sample's own staging
          * copies: on G17P their wait field does not take effect. Only
          * this sample's wait moves, so unrelated work is not
          * serialised. */
         /* A LOD query is not a staged sample: it has its own operand
          * and slot protocol, so its wait stays where it is. */
         /* PERSP_MUL/FADD coordinate publications already belong to a
          * native transaction. SMP consumes that transaction through
          * its coordinate-slot wait below; moving the wait to a
          * carrier would retire it before the sample can consume it.
          * Only the late COPY publication needs the carrier. */
         const uint16_t live = staged_sample ? st->wait : 0;
         if (live) {
            if (!agx3_retire_result_wait(emission, enc, I, live))
               AGX3_REFUSE_RET(I);
            st->wait &= (uint16_t)~live;
         }
      }
      assert(st->texture_ops_left > 0);
      const bool final = --st->texture_ops_left == 0;
      assert(st->texture_samples_remaining > 0);
      st->texture_samples_remaining--;
      const bool sample_group_member =
         I->lod_mode != AGX_LOD_MODE_LOD_GRAD &&
         I->lod_mode != AGX_LOD_MODE_LOD_GRAD_MIN;
      /* A fragment SMP whose f156 group is still open (nonterminal)
       * keeps the quad's helper lanes for what follows; a terminating
       * or lone member does not. So the last member stays open when a
       * quad derivative or quad shuffle follows it (G17P and G17G
       * hardware result; Apple's encoding of this shape is not
       * decoded). */
      const bool keep_quad_open =
         agx_isa_is_agx3(enc) &&
         ctx->stage == MESA_SHADER_FRAGMENT && sample_group_member &&
         st->sample_group_members_remaining == 1 &&
         (st->quad_consumer_after_last_member ||
          (ctx->key->dev.emit_agx3d && st->g17p_quad_op_after_group));
      const bool sample_group_nonterminal = keep_quad_open ||
         (st->g17p_sample_group && sample_group_member &&
          st->sample_group_members_remaining > 1) ||
         /* G17G's plain 2D fragment samples form the same group: the
          * three-comparison and two-texture witnesses clear f156 and
          * f150 on every member but the last. Other G17G shapes keep
          * their per-op terminator until witnessed. */
         (st->g17g_fragment_group && sample_group_member &&
          st->sample_group_members_remaining > 1 &&
          agx3_g17g_plain_2d_sample(I)) ||
         /* Compute gathers use level zero without a fragment group. */
         (enc == AGX_ISA_ENCODING_AGX3B &&
          compute_sample_stage &&
          I->gather != AGX_GATHER_NONE);
      bool sample_group_end = false;
      if (sample_group_member) {
         assert(st->sample_group_members_remaining > 0);
         sample_group_end = --st->sample_group_members_remaining == 0 &&
                            ((st->g17p_sample_group && !keep_quad_open) ||
                             (st->g17g_fragment_group &&
                              I->lod_mode == AGX_LOD_MODE_LOD_MIN &&
                              I->src[1].type == AGX_INDEX_IMMEDIATE &&
                              I->dim == AGX_DIM_2D));
      }
      /* Implicit-derivative fragment programs enter the pixel
       * transaction right before their first SMP, after coefficient
       * interpolation: entering later lets a scissor edge retire
       * uncovered lanes before SMP forms its derivative quad.
       * Neo explicit-LOD and gradient samples are exempt. G17G's
       * explicit-level witnesses acquire at the same point too. The entry is
       * published in the link metadata so agx_fast_link drops the split
       * epilog's duplicate gate; it is never placed inside one
       * control-flow arm. */
      const bool implicit_derivatives =
         I->lod_mode == AGX_LOD_MODE_AUTO_LOD ||
         I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS ||
         I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN;
      const bool native_staged_coords = I->agx3_tex_coords_staged;
      const bool open_pixel =
         (enc == AGX_ISA_ENCODING_AGX3B ||
          enc == AGX_ISA_ENCODING_AGX3C) &&
         ctx->stage == MESA_SHADER_FRAGMENT &&
         !ctx->key->fs.ignore_tib_dependencies &&
         ctx->agx3_colour_link.straight_line &&
         (implicit_derivatives || enc == AGX_ISA_ENCODING_AGX3C) &&
         !st->g17p_colour_open;
      /* Implicit LOD needs the 2x2 quad's derivatives, which exist
       * only in fragment shaders. Compute and vertex explicit-level
       * samples encode like fragment ones and stage a runtime level the
       * same way. */
      const bool explicit_level = ctx->stage != MESA_SHADER_FRAGMENT &&
         I->lod_mode == AGX_LOD_MODE_LOD_MIN &&
         (I->src[1].type == AGX_INDEX_IMMEDIATE ||
          compute_sample_stage);
      /* An explicit gradient carries its own derivative, so it needs no
       * quad either; Apple's own witness for the form is a compute
       * kernel. */
      const bool compute_gradient = compute_sample_stage &&
         (I->lod_mode == AGX_LOD_MODE_LOD_GRAD ||
          I->lod_mode == AGX_LOD_MODE_LOD_GRAD_MIN);
      /* Native runtime-indexed vertex pipelines ground both a per-lane
       * explicit level and explicit gradients in every noncube sample
       * dimension, including depth comparisons and array layers. A G17P
       * mesh program is VDM-launched like a vertex program and binds its
       * textures through the vertex path, so it samples the same way. */
      const bool vertex_explicit_lod =
         agx3_vertex_explicit_lod_stage(ctx->stage, enc) &&
         (I->lod_mode == AGX_LOD_MODE_LOD_MIN ||
          I->lod_mode == AGX_LOD_MODE_LOD_GRAD);
      /* Relocation is an operand-ABI property. Other forms retain
       * their fixed operand locations and wait only for overlapping
       * live ranges before publishing into those locations. */
      const bool m5_fragment_3d =
         ctx->stage == MESA_SHADER_FRAGMENT &&
         I->lod_mode == AGX_LOD_MODE_AUTO_LOD;
      const bool m5_banked_3d =
         enc == AGX_ISA_ENCODING_AGX3C && I->dim == AGX_DIM_3D &&
         !I->shadow && !I->offset && !I->query_lod &&
         (((ctx->stage == MESA_SHADER_VERTEX ||
            ctx->stage == MESA_SHADER_COMPUTE) &&
           ((I->lod_mode == AGX_LOD_MODE_LOD_MIN &&
             I->src[1].type == AGX_INDEX_IMMEDIATE) ||
            I->lod_mode == AGX_LOD_MODE_LOD_GRAD)) ||
          m5_fragment_3d);
      const bool relocatable_2d_array =
         (enc == AGX_ISA_ENCODING_AGX3B ||
          (enc == AGX_ISA_ENCODING_AGX3C && !ctx->key->dev.emit_agx3d)) &&
         I->dim == AGX_DIM_2D_ARRAY && !I->query_lod &&
         !(I->offset && !I->offset_xyz) &&
         I->src[3].type != AGX_INDEX_REGISTER &&
         I->src[4].type != AGX_INDEX_REGISTER &&
         (I->lod_mode == AGX_LOD_MODE_AUTO_LOD ||
          ((I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS ||
            I->lod_mode == AGX_LOD_MODE_LOD_MIN) &&
           I->src[1].type == AGX_INDEX_IMMEDIATE));
      const bool relocatable_coordinates =
         (enc == AGX_ISA_ENCODING_AGX3B ||
          enc == AGX_ISA_ENCODING_AGX3C) && st->slot != 0 &&
         !native_staged_coords &&
         (m5_banked_3d || relocatable_2d_array || I->dim == AGX_DIM_2D ||
          (enc == AGX_ISA_ENCODING_AGX3B &&
           (I->dim == AGX_DIM_3D ||
            (I->dim == AGX_DIM_2D_ARRAY && !I->shadow && !I->offset)) &&
           ctx->stage == MESA_SHADER_VERTEX &&
           ctx->key->g17p_uniform_dma &&
           I->lod_mode == AGX_LOD_MODE_LOD_MIN &&
           I->src[1].type == AGX_INDEX_IMMEDIATE)) &&
         !I->query_lod && !(I->offset && !I->offset_xyz) &&
         (m5_banked_3d || relocatable_2d_array ||
          I->lod_mode == AGX_LOD_MODE_AUTO_LOD ||
          ((I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS ||
            I->lod_mode == AGX_LOD_MODE_LOD_MIN) &&
           (I->src[1].type == AGX_INDEX_IMMEDIATE ||
            enc == AGX_ISA_ENCODING_AGX3C)));
      const bool staged_scalar =
         (I->lod_mode == AGX_LOD_MODE_LOD_MIN &&
          I->src[1].type != AGX_INDEX_IMMEDIATE) ||
         I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN ||
         (I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS &&
          I->src[1].type != AGX_INDEX_IMMEDIATE);
      const bool staged_compare = I->shadow;
      const bool staged_sample_offset = I->offset && I->offset_xyz == 0;
      const bool published = agx_texture_resource(I) != 0;
      const unsigned published_resource = published
         ? st->publication_values[agx_texture_resource(I)] - 1 : 0;
      const uint16_t sample_span =
         published ? agx_index_size_16(I->src[0]) :
              (gpu1_fixed_gather
                 ? (ctx->key->dev.emit_agx3d && !I->shadow ? 6 : 8)
              : (I->dim == AGX_DIM_CUBE ||
               I->dim == AGX_DIM_CUBE_ARRAY) ? 8
              : enc == AGX_ISA_ENCODING_AGX3C &&
                I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN &&
                ((I->dim == AGX_DIM_2D_ARRAY && I->shadow) ||
                 (I->dim == AGX_DIM_3D && staged_sample_offset))
                   ? 10
              : native_staged_coords ? 4
              : vertex_explicit_lod &&
                 (I->lod_mode != AGX_LOD_MODE_LOD_MIN ||
                  I->src[1].type != AGX_INDEX_IMMEDIATE)
                 ? (I->lod_mode == AGX_LOD_MODE_LOD_GRAD
                       ? (I->dim == AGX_DIM_3D ? 40 : 24)
                       : 16)
              : (I->lod_mode == AGX_LOD_MODE_LOD_GRAD ||
                 I->lod_mode == AGX_LOD_MODE_LOD_GRAD_MIN)
                   ? (I->dim == AGX_DIM_3D ? 32 : 16)
              : (staged_scalar || staged_sample_offset ||
                 (I->dim == AGX_DIM_2D_ARRAY && !I->query_lod) ||
                 I->dim == AGX_DIM_3D)
                 ? 8
              : staged_compare ? 6
              : (!ctx->key->dev.emit_agx3d ||
                 agx3_g17g_basic_gather(I) || g18g_indirect_gather) ? 4
                                             : I->src[0].value + 4);
      uint16_t bank_base = published ? st->resource_bases[published_resource] : 0;
      if ((enc == AGX_ISA_ENCODING_AGX3B ||
           enc == AGX_ISA_ENCODING_AGX3C) && !published) {
         /* Texel fetches and filtered samples publish into the same
          * sampler-private file. A sample must not overwrite a bank
          * still consumed by a fetch (and vice versa). */
         agx3_staging_release(&st->staging_banks, st->outstanding_here);
         if (relocatable_coordinates) {
            assert(staged_sample);
            if (!agx3_allocate_staging(emission, enc, I, &st->staging_banks,
                                       &st->outstanding, st->slot, sample_span,
                                       st->fixed_gradient_staging,
                                       &bank_base))
               AGX3_REFUSE_RET(I);
         } else {
            const uint16_t retire = agx3_staging_conflicts(
               &st->staging_banks, 0, sample_span);
            if (retire) {
               /* Ordinary samples retired their complete GPR wait
                * above. No staging has issued yet, so the result
                * remains ready for a resource-only carrier. Native
                * coordinate/query transactions do not have this proof. */
               const bool retired = staged_sample
                  ? agx3_retire_result_wait(emission, enc, I, retire)
                  : agx3_retire_slots_noop(emission, enc, retire);
               if (!retired)
                  AGX3_REFUSE_RET(I);
               st->outstanding &= (uint16_t)~retire;
               if (st->slot)
                  st->outstanding |= (uint16_t)(1u << (st->slot - 1));
               agx3_staging_forget(&st->staging_banks, retire);
            }
         }
      }
      uint16_t sample_wait = st->wait;
      if (native_staged_coords) {
         if (!st->native_coord_slot)
            AGX3_REFUSE_RET(I);
         const uint16_t coord_wait =
            (uint16_t)(1u << (st->native_coord_slot - 1));
         if (!(sample_wait & coord_wait))
            AGX3_REFUSE_RET(I);

         /* The SMP has one witnessed wait generation. Coordinate
          * staging owns it; drain independent descriptor producers
          * first, then let SMP consume the intact helper-lane staging
          * transaction. */
         const uint16_t other_waits = sample_wait & ~coord_wait;
         if (other_waits &&
             !agx3_retire_slots_noop(emission, enc, other_waits))
            AGX3_REFUSE_RET(I);
         sample_wait = coord_wait;
      }
      /* CEM has no scoreboard-wait fields. Keep a register-descriptor
       * producer already assigned slot 6 active so CEM's finalizers and
       * face publication join its native transaction. Retire every
       * other producer before entering the compound pipeline. */
      if ((I->dim == AGX_DIM_CUBE ||
           I->dim == AGX_DIM_CUBE_ARRAY) &&
          sample_wait != 0) {
         const bool register_operands =
            I->src[3].type == AGX_INDEX_REGISTER ||
            I->src[4].type == AGX_INDEX_REGISTER;
         const uint16_t native_wait =
            register_operands && enc == AGX_ISA_ENCODING_AGX3B
               ? (sample_wait & (1u << 5)) : 0;
         const uint16_t retire = sample_wait & ~native_wait;
         if (retire && !agx3_retire_slots_noop(emission, enc, retire))
            AGX3_REFUSE_RET(I);
         sample_wait = native_wait;
      }
      /* A native staging COPY carries one producer wait. Lower only
       * the extra RAW/WAW dependencies assigned to this sample, using
       * the same grounded wait carrier as other compound instructions.
       * Do not drain unrelated outstanding slots or change allocation.
       * Previously encodable zero/one-wait samples remain unchanged. */
      if (sample_wait & (sample_wait - 1)) {
         uint16_t staging_wait = sample_wait & -sample_wait;
         if (published) {
            /* Keep a still-pending publication as SMP's direct
             * dependency, and retire independent descriptor producers
             * on carriers. Never add a remembered slot to the mask:
             * an earlier consumer may already have completed it.
             */
            const unsigned producer = st->resources[published_resource].producer;
            const uint16_t publication_wait =
               1u << (st->res[producer].slot - 1);
            if (sample_wait & publication_wait)
               staging_wait = publication_wait;
         }
         if (!agx3_retire_slots_noop(emission, enc,
                                     sample_wait & ~staging_wait))
            AGX3_REFUSE_RET(I);
         sample_wait = staging_wait;
      }
      /* A G17G staged block is keyed by the sample's slot. Round-robin
       * assignment wraps onto a slot that is still outstanding once
       * eight async results are in flight; retire that producer here
       * so restaging the block cannot overtake its consumer. */
      if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
          agx3_g17g_staged_block(I, ctx->key->dev.emit_agx3d) &&
          (st->outstanding_here & (1u << (st->slot - 1))) &&
          !(sample_wait & (1u << (st->slot - 1))) &&
          !agx3_retire_slots_noop(emission, enc,
                                  (uint16_t)(1u << (st->slot - 1))))
         AGX3_REFUSE_RET(I);
      if ((ctx->stage != MESA_SHADER_FRAGMENT && !explicit_level &&
           !compute_gradient && !vertex_explicit_lod &&
           !((enc == AGX_ISA_ENCODING_AGX3B ||
              (enc == AGX_ISA_ENCODING_AGX3C &&
               !ctx->key->dev.emit_agx3d)) &&
             I->gather != AGX_GATHER_NONE) &&
           !(ctx->key->dev.emit_agx3d &&
             (vertex_sample_stage || compute_sample_stage) &&
             (agx3_g17g_basic_gather(I) || gpu1_fixed_gather ||
              g18g_indirect_gather))) ||
          /* Keep the pixel-entry gate after all native coordinate
           * staging and immediately before SMP, matching Apple. */
          !agx3_emit_texture_sample(emission, enc, I, st->slot, sample_wait,
                                    final, sample_group_nonterminal,
                                    sample_group_end, open_pixel,
                                    &ctx->agx3_colour_link.entry_offset,
                                    st->native_coord_slot, st->cem_scratch,
                                    ctx->key->dev.emit_agx3d
                                       ? vertex_sample_stage
                                       : vertex_explicit_lod,
                                    ctx->key->dev.emit_agx3d
                                       ? compute_sample_stage
                                       : ctx->stage == MESA_SHADER_COMPUTE,
                                    st->ms_array_scratch, bank_base, st->g17g_lod_relay,
                                    ctx->key->dev.emit_agx3d,
                                    published && st->resource_last_use[
                                       published_resource] == st->instr_index,
                                    ctx->key->g17p_texture_uniform_base))
         AGX3_REFUSE_RET(I);
      /* Keep this fixed sampler block owned until its consumer has
       * retired, so later samples, fetches or image writes cannot
       * republish any of its operands. */
      if (g18g_gradient_sample || gpu1_fixed_gather ||
          g18g_colour_level_sample) {
         const uint16_t live = (uint16_t)(1u << (st->slot - 1));
         if (!agx3_retire_slots_noop(emission, enc, live))
            AGX3_REFUSE_RET(I);
         st->outstanding &= (uint16_t)~live;
      }
      if (open_pixel) {
         ctx->agx3_colour_link.entry_emitted = true;
         ctx->agx3_colour_link.entry_offset -= st->initial_size;
         st->g17p_colour_open = true;
      }
      /* ShaderInfo fields 38/42 describe the sampler-private compiler
       * footprint in bytes, not the highest GPR supplying an operand.
       * Fresh attachment-only 2D families report 8 bytes and publish
       * into staging[0]/[2], while cube, cube-array and cube-shadow
       * families report 16 bytes. Fragment gradient pairs report 32
       * bytes for 2D and 64 bytes for 3D, with or without an offset;
       * vertex gradients instead report 48 and 80 bytes. Earlier
       * 24-byte cube witnesses also wrote a device buffer whose
       * independent staging requirement dominated. Keep the extents
       * stage- and dimension-specific, independent of post-RA source
       * placement and scoreboard slot. */
      /* A G17G staged block sits at 8 * (slot - 1); the footprint must
       * reach past the highest half its stagers write. */
      const uint16_t g17g_block_base =
         enc == AGX_ISA_ENCODING_AGX3C &&
         agx3_g17g_staged_block(I, ctx->key->dev.emit_agx3d)
            ? 8u * (st->slot - 1)
            : 0;
      const unsigned sample_staging_end =
         /* ShaderInfo 38/42: G18 vertex/fragment need 52/36 bytes. */
         g18g_gradient_sample ? (vertex_explicit_lod ? 26 : 18) :
         (enc == AGX_ISA_ENCODING_AGX3C ? g17g_block_base : bank_base) +
         sample_span;
      ctx->agx3_staging_halfs =
         MAX2(ctx->agx3_staging_halfs,
              ctx->key->dev.emit_agx3d ? sample_staging_end
                                       : bank_base + sample_span);
      /* For the G18G image-write bank floor
       * (AGX3_G18G_READ_STAGING_LIMIT). */
      st->g18g_other_staging_end =
         MAX2(st->g18g_other_staging_end, sample_staging_end);
      st->g18g_read_staging_end =
         MAX2(st->g18g_read_staging_end, sample_staging_end);
      if ((enc == AGX_ISA_ENCODING_AGX3B ||
           (enc == AGX_ISA_ENCODING_AGX3C &&
            !ctx->key->dev.emit_agx3d)) && st->slot) {
         agx3_staging_release(&st->staging_banks, st->outstanding);
         if (!agx3_staging_hold(&st->staging_banks, bank_base,
                                sample_span,
                                st->slot))
            AGX3_REFUSE_RET(I);
      }
      return AGX3_PACK_EMITTED;
   }

   return AGX3_PACK_UNHANDLED;
}
