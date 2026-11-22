/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 fragment varyings, interpolation, vertex UVSW output protocols,
 * tilebuffer access, colour packing, pixel transactions, MRT, and ZS/discard.
 */

#include "agx3_pack_internal.h"
#include "agx3_copy_core.h"
#include "agx3_fltnorm_core.h"
#include "agx3_pixel_core.h"
#include "agx3_sr_core.h"
#include "agx3_stage_core.h"
#include "agx3_tile_core.h"
#include "agx3_vary_core.h"
#include "agx3_zs_core.h"

#include <limits.h>

static uint32_t agx3_g17p_colour_writeout(uint32_t writeout, unsigned pair);

static bool
agx3_map_vary_location(enum agx_interpolation interpolation,
                        enum agx3_vary_location *location)
{
   switch (interpolation) {
   case AGX_INTERPOLATION_CENTER:
      *location = AGX3_VARY_CENTER;
      return true;
   case AGX_INTERPOLATION_CENTROID:
      *location = AGX3_VARY_CENTROID;
      return true;
   default:
      return false;
   }
}

/* Projective RHW is private to reciprocal interpolation. Follow its physical
 * half-register lifetime through the CFG: unrelated branches do not change
 * the iterator encoding, and a full overwrite ends the old value's lifetime.
 * Track each half independently so an ordinary read after a partial overwrite
 * cannot accidentally be accepted as a read of an unrelated new value. */
static bool
agx3_private_rhw_use(const agx_context *ctx, const agx_block *block,
                     const agx_instr *producer)
{
   if (producer->nr_dests != 1 || producer->nr_srcs != 2 ||
       !agx3_vary_reg32(producer->dest[0]) ||
       agx_channels(producer->dest[0]) != 1 ||
       producer->src[0].type != AGX_INDEX_IMMEDIATE)
      return false;

   struct rhw_path {
      const agx_block *block;
      unsigned live_halves;
      const agx_instr *after;
   };
   struct util_dynarray paths;
   util_dynarray_init(&paths, NULL);
   uint8_t *visited = calloc(ctx->num_blocks, sizeof(*visited));
   if (!visited)
      return false;
   util_dynarray_append(&paths,
                       ((struct rhw_path){block, 3, producer}));
   const unsigned reg = producer->dest[0].value;
   bool consumer = false, valid = false;

   for (unsigned p = 0; p < util_dynarray_num_elements(&paths, struct rhw_path); ++p) {
      struct rhw_path path = *util_dynarray_element(&paths, struct rhw_path, p);
      bool after = path.after == NULL;
      unsigned live = path.live_halves;
      agx_foreach_instr_in_block(path.block, I) {
         if (!after) {
            after = I == path.after;
            continue;
         }
         agx_foreach_src(I, src) {
            agx_index value = I->src[src];
            if (!agx3_src_is_register(value) || value.memory)
               continue;
            unsigned overlap = 0;
            for (unsigned h = 0; h < 2; ++h) {
               if (value.value <= reg + h &&
                   value.value + agx_index_size_16(value) > reg + h)
                  overlap |= 1u << h;
            }
            if (!(overlap & live))
               continue;
            if (live != 3 || I->op != AGX_OPCODE_PERSP_RCP_AGX3 || src != 0 ||
                value.value != reg || value.size != AGX_SIZE_32 ||
                agx_channels(value) != 1 || I->imm != producer->src[0].value)
               goto out;
            consumer = true;
         }
         agx_foreach_dest(I, dst) {
            agx_index value = I->dest[dst];
            if (!agx3_src_is_register(value) || value.memory)
               continue;
            for (unsigned h = 0; h < 2; ++h) {
               if (value.value <= reg + h &&
                   value.value + agx_index_size_16(value) > reg + h)
                  live &= ~(1u << h);
            }
         }
         if (!live)
            break;
      }
      if (!live)
         continue;
      for (unsigned e = 0; e < ARRAY_SIZE(path.block->successors); ++e) {
         const agx_block *next = path.block->successors[e];
         if (!next)
            continue;
         assert(next->index < ctx->num_blocks);
         if (visited[next->index] & (1u << live))
            continue;
         visited[next->index] |= 1u << live;
         util_dynarray_append(&paths,
                             ((struct rhw_path){next, live, NULL}));
      }
   }
   valid = consumer;
out:
   free(visited);
   util_dynarray_fini(&paths);
   return valid;
}

static bool
agx3_interp_uses_coords(enum agx_isa_encoding enc, const agx_instr *I)
{
   return I->interpolation == AGX_INTERPOLATION_SAMPLE ||
          (I->interpolation == AGX_INTERPOLATION_CENTROID &&
           (enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C));
}

/* Explicit sample and G17P centroid coordinates are an opaque 32-bit token.
 * Require each consumer's token to be the SSA result of ITRCOORDS of the same
 * interpolation, so a legacy ITER sample-index register or an unrelated
 * arithmetic value cannot select the native f32interp coordinate leaf.
 *
 * This is a property of the SSA value, not of a physical register: register
 * allocation may copy, swap or spill and fill the token on its way to a
 * consumer, which preserves its 32 bits but not the register that ITRCOORDS
 * wrote. Check it before RA; the packer then only checks the encoding.
 */
bool
agx3_interp_coords_from_itrcoords(const agx_context *ctx)
{
   const enum agx_isa_encoding enc = ctx->key->dev.encoding;
   const agx_instr **defs = calloc(ctx->alloc, sizeof(*defs));
   if (!defs)
      return false;

   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_dest(I, d)
         defs[I->dest[d].value] = I;
   }

   bool valid = true;
   agx_foreach_instr_global(ctx, I) {
      if ((I->op != AGX_OPCODE_INTERP_AGX3 &&
           I->op != AGX_OPCODE_RHW_INTERP_AGX3) ||
          !agx3_interp_uses_coords(enc, I))
         continue;

      const agx_index coords = I->src[1];
      const agx_instr *producer =
         I->nr_srcs == 2 && coords.type == AGX_INDEX_NORMAL &&
               !coords.memory && coords.value < ctx->alloc
            ? defs[coords.value]
            : NULL;
      if (!producer || producer->op != AGX_OPCODE_ITRCOORDS_AGX3 ||
          producer->interpolation != I->interpolation ||
          coords.size != AGX_SIZE_32 || agx_channels(coords) != 1 ||
          coords.abs || coords.neg) {
         valid = false;
         break;
      }
   }

   free(defs);
   return valid;
}

static bool
agx3_emit_interp(struct util_dynarray *emission, enum agx_isa_encoding enc,
                 const agx_instr *I, uint16_t wait_mask, bool first,
                 bool fragment)
{
   if (wait_mask != 0 || I->nr_dests != 1 || I->nr_srcs != 2 ||
       !agx3_vary_reg32(I->dest[0]) || I->src[0].type != AGX_INDEX_IMMEDIATE)
      return false;

   /* Sample and centroid interpolation consume an ITRCOORDS token through
    * the same f32interp leaves on G17P and G17G (lane-mesh/cwit-interp). */
   const bool token = agx3_interp_uses_coords(enc, I);
   enum agx3_vary_location location;
   if (token) {
      if ((enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C) ||
          first || !agx3_vary_reg32(I->src[1]) || I->src[1].kill ||
          (I->src[1].cache && I->src[1].discard))
         return false;
   } else {
      if (I->src[1].type != AGX_INDEX_IMMEDIATE || I->src[1].value != 0 ||
          !agx3_map_vary_location(I->interpolation, &location))
         return false;
   }

   uint8_t bytes[AGX3_VARY_MAX_LENGTH];
   uint8_t len = 0;
   if (token && I->op == AGX_OPCODE_RHW_INTERP_AGX3) {
      if (I->saturate)
         return false;
      return agx3_vary_projective_rhw_sample(enc, I->dest[0].value,
                                             I->src[0].value, I->src[1].value,
                                             I->src[1].discard, bytes, &len) &&
             agx3_append(emission, bytes, len);
   }
   if (token) {
      return agx3_vary_interp_sample(enc, I->dest[0].value, I->src[0].value,
                                     I->perspective, I->src[1].value,
                                     I->src[1].discard, bytes, &len) &&
             agx3_append(emission, bytes, len);
   }

   if (I->op == AGX_OPCODE_RHW_INTERP_AGX3) {
      if (I->saturate || location != AGX3_VARY_CENTER)
         return false;
      return agx3_vary_projective_rhw(enc, I->dest[0].value,
                                     I->src[0].value, first, bytes, &len) &&
             agx3_append(emission, bytes, len);
   }
   /* The native G17P center_no_perspective scalar/vector programs use
    * f11=1 as well. This bit does not select perspective correction: the
    * VIT shade model and explicit RCP/MUL sequence do that. Keep the raw
    * codec able to reproduce both native f11 forms. */
   const bool native_mode = I->perspective ||
      (fragment && enc == AGX_ISA_ENCODING_AGX3B &&
       location == AGX3_VARY_CENTER);
   return agx3_vary_interp_at_start(enc, I->dest[0].value, I->src[0].value,
                                    native_mode, location, first, bytes, &len) &&
          agx3_append(emission, bytes, len);
}

static bool
agx3_emit_itrcoords(struct util_dynarray *emission, enum agx_isa_encoding enc,
                    const agx_instr *I, uint16_t wait_mask)
{
   if (I->nr_dests != 1 || I->nr_srcs != 1 || !agx3_vary_reg32(I->dest[0]) ||
       !agx3_src_is_register(I->src[0]) || I->src[0].memory ||
       I->src[0].size != AGX_SIZE_16 || agx_channels(I->src[0]) != 1 ||
       I->src[0].value >= 256 || I->src[0].abs || I->src[0].neg ||
       I->src[0].kill || (I->src[0].cache && I->src[0].discard))
      return false;

   uint8_t bytes[AGX3_VARY_MAX_LENGTH];
   uint8_t len = 0;
   /* Cache is an optional post-RA hint. The fresh ITRCOORDS witnesses cover
    * the unmarked and terminal source forms, so preserve the terminal bit and
    * deliberately leave a merely cached source unmarked. */
   bool encoded;
   switch (I->interpolation) {
   case AGX_INTERPOLATION_SAMPLE:
      encoded = agx3_vary_sample_coords(enc, I->dest[0].value,
                                        I->src[0].value, I->src[0].discard,
                                        wait_mask, bytes, &len);
      break;
   case AGX_INTERPOLATION_CENTROID:
      encoded = agx3_vary_centroid_coords(enc, I->dest[0].value,
                                          I->src[0].value, I->src[0].discard,
                                          wait_mask, bytes, &len);
      break;
   default:
      return false;
   }
   return encoded && agx3_append(emission, bytes, len);
}

static bool
agx3_emit_ldcf(struct util_dynarray *emission, enum agx_isa_encoding enc,
               const agx_instr *I, uint8_t slot, uint16_t wait_mask, bool first)
{
   if (wait_mask != 0 || I->nr_dests != 1 || I->nr_srcs != 1 ||
       !agx3_vary_reg32(I->dest[0]) ||
       I->src[0].type != AGX_INDEX_IMMEDIATE || I->channels != 1)
      return false;

   uint8_t bytes[AGX3_VARY_MAX_LENGTH];
   uint8_t len = 0;
   return agx3_vary_ldcf_at_start(enc, I->dest[0].value, I->src[0].value,
                                 slot, first, bytes, &len) &&
          agx3_append(emission, bytes, len);
}

/* A relocatable vertex program writes each user varying at a placeholder
 * UVS index (agx_shader_key::g17p_uvs_reloc_size). Record an immediate-index
 * UVSW about to be appended to the emission if its index is one of them, so
 * the linker can re-encode it for the fragment shader's layout
 * (agx3_vary_relocate_uvsw()). Offsets are from the start of the emission
 * until agx_compile makes them relative to the main program. */
static void
agx3_note_uvs_reloc(agx_context *ctx, const struct util_dynarray *emission,
                    uint16_t index, uint16_t value, uint8_t slot,
                    uint16_t wait_mask)
{
   const unsigned base = ctx->key->g17p_uvs_reloc_base;
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_VERTEX || ctx->is_preamble ||
       index < base || index - base >= ctx->key->g17p_uvs_reloc_size)
      return;

   assert(value <= UINT8_MAX && wait_mask <= UINT8_MAX);
   const struct agx3_uvs_reloc r = {
      .offset = emission->size,
      .index = index,
      .value = value,
      .slot = slot,
      .wait = wait_mask,
   };
   if (!ctx->agx3_uvs_relocs.mem_ctx)
      util_dynarray_init(&ctx->agx3_uvs_relocs, ctx);
   memcpy(util_dynarray_grow_bytes(&ctx->agx3_uvs_relocs, 1,
                                   AGX3_UVS_RELOC_SIZE),
          &r, AGX3_UVS_RELOC_SIZE);
   ctx->agx3_uvs_reloc_count++;
}

static bool
agx3_emit_uvsw(agx_context *ctx, struct util_dynarray *emission,
               enum agx_isa_encoding enc, const agx_instr *I,
               uint16_t staging_half, uint8_t slot, uint16_t wait_mask,
               bool defer_writeout)
{
   /* An unlinked vertex shader reads its user outputs' indices from the
    * linked varying layout: a 16-bit uniform, or a register when computed. */
   const agx_index index = I->src[0];
   const bool plain_half = !index.memory && !index.abs && !index.neg &&
      index.size == AGX_SIZE_16 && agx_channels(index) == 1;
   const bool register_index = agx3_src_is_register(index) && plain_half;
   const bool uniform_index = index.type == AGX_INDEX_UNIFORM && plain_half;
   const bool fused = I->op != AGX_OPCODE_ST_VARY;
   if (I->nr_dests != 0 || (!fused && I->nr_srcs != 2) ||
       (index.type != AGX_INDEX_IMMEDIATE && !register_index &&
        !uniform_index) ||
       ((register_index || uniform_index) && defer_writeout) ||
       (!fused && !agx3_vary_reg32(I->src[1])))
      return false;

   uint8_t bytes[MAX2(16, AGX3_VARY_MAX_LENGTH)];
   uint8_t len = 0;
   /* UVSW's f9=5/f5 operand is a staging-file address, not the GPR named
    * by ST_VARY IR. First transfer the source into that file and wait for
    * its actual producer; UVSW then waits for this staging write's slot. */
   if (fused) {
      const bool fma = I->op == AGX_OPCODE_ST_VARY_FMA_AGX3;
      if ((I->op != AGX_OPCODE_ST_VARY_MUL_AGX3 && !fma) ||
          I->nr_srcs != (fma ? 4 : 3))
         return false;
      uint16_t src[3] = {0};
      bool live[3] = {0}, discard[3] = {0};
      const bool uniform0 = I->src[1].type == AGX_INDEX_UNIFORM;
      for (unsigned s = 1; s < I->nr_srcs; ++s) {
         agx_index x = I->src[s];
         if (x.size != AGX_SIZE_32 || agx_channels(x) != 1 || x.abs ||
             x.neg || x.memory ||
             !(agx3_src_is_register(x) || (fma && s == 1 && uniform0)))
            return false;
         src[s - 1] = x.value;
         if (x.type != AGX_INDEX_UNIFORM) {
            discard[s - 1] = x.discard || x.kill;
            live[s - 1] = x.cache && !discard[s - 1];
         }
      }
      if (!agx3_stage_vertex_alu(enc, fma, staging_half, slot, src,
                                uniform0, live, discard, wait_mask, bytes, &len))
         return false;
   } else if (!agx3_stage_copy_source(enc, staging_half, I->src[1].value,
                                      I->src[1].kill, slot, wait_mask, bytes, &len)) {
      return false;
   }
   if (!agx3_append(emission, bytes, len))
      return false;
   if (defer_writeout)
      return true;
   const uint16_t staged = (uint16_t)(1u << (slot - 1));
   if (!register_index && !uniform_index)
      agx3_note_uvs_reloc(ctx, emission, index.value, staging_half, slot,
                          staged);
   return (register_index
              ? agx3_vary_uvsw_register_index(enc, staging_half, index.value,
                                              slot, true, staged, bytes, &len)
           : uniform_index
              ? agx3_vary_uvsw_uniform_index(enc, staging_half, index.value,
                                             slot, true, staged, bytes, &len)
              : agx3_vary_uvsw(enc, staging_half, index.value, slot, true,
                               staged, bytes, &len)) &&
          agx3_append(emission, bytes, len);
}

static int
agx3_tile_format_width(enum agx_format format)
{
   switch (format) {
   case AGX_FORMAT_I8:
   case AGX_FORMAT_U8NORM:
   case AGX_FORMAT_S8NORM:
   case AGX_FORMAT_SRGBA8:
      return AGX3_TILE_W8;
   case AGX_FORMAT_I16:
   case AGX_FORMAT_F16:
   case AGX_FORMAT_U16NORM:
   case AGX_FORMAT_S16NORM:
      return AGX3_TILE_W16;
   case AGX_FORMAT_I32:
   case AGX_FORMAT_RGB10A2:
   case AGX_FORMAT_RG11B10F:
   case AGX_FORMAT_RGB9E5:
      return AGX3_TILE_W32;
   default:
      return -1;
   }
}

/*
 * The register span a tile store or load names: one register per channel at
 * 16 and 32 bits; at 8 bits two channels per 16-bit register (`lstcol`
 * carries a width, not a format), so four channels read two registers, two
 * read one, and one reads the low byte of one. Other 8-bit shapes are
 * refused.
 */
static bool
agx3_tile_value_ok(agx_index value, uint8_t mask, int width)
{
   if (!agx3_src_is_register(value) || value.memory || value.abs || value.neg ||
       (value.size != AGX_SIZE_16 && value.size != AGX_SIZE_32) ||
       (value.value & (agx_size_align_16(value.size) - 1)))
      return false;

   const unsigned channels = util_bitcount(mask);
   if (width == AGX3_TILE_W8) {
      /* Both directions: the store's value is packed by fltnorm, and the
       * load's destination is what normflt then unpacks. */
      if (value.size != AGX_SIZE_16 ||
          agx_channels(value) != DIV_ROUND_UP(channels, 2))
         return false;
   } else if (agx_channels(value) != channels) {
      return false;
   }

   return value.value + agx_index_size_16(value) <= 256;
}

static bool
agx3_tile_sample_operand(agx_index sample, struct agx3_tile_sample *out)
{
   if (sample.abs || sample.neg || sample.memory ||
       sample.size != AGX_SIZE_16)
      return false;

   if (sample.type == AGX_INDEX_IMMEDIATE) {
      /* The whole-pixel form. Apple encodes it as immediate 1 at every width
       * and every format; 0xFF is Mesa's spelling of the same request.
       *
       * AGX_TRACE_TILE_COVERAGE found this immediate being emitted for
       * multisample renders and read it as a missing tag. It was not: the
       * 2x/4x stores WERE tagged, and the tag they carried encoded the
       * register alternative of this operand naming a half register nobody
       * defined. That is why making the untagged form unencodable produced
       * no refusal -- there was no untagged store. The tags are gone now;
       * a multisample store carries a live mask register instead.
       */
      if (sample.kill || sample.discard || sample.cache ||
          sample.value != AGX3_TILE_ALL_SAMPLES)
         return false;

      *out = AGX3_TILE_SAMPLE_1X;
      return true;
   }

   if (!agx3_src_is_register(sample) || sample.value >= 256)
      return false;

   /* Register allocation rewrites `kill` into the post-RA `discard` hint on
    * some paths, and the rest of this packer reads the pair together. f55 is
    * that hint for the sample operand. A mask read again by later stores may
    * carry the post-RA `cache` hint instead; like any other live mask it
    * leaves f55 clear, as on all but the last store of Apple's 2x/4x
    * multi-target fragments. */
   *out = agx3_tile_sample_reg(sample.value, sample.kill || sample.discard);
   return true;
}

/* msaa163: a G18G fragment-stage tile program (FS epilog or background)
 * compiled for two or four samples. Its tile stores take the
 * SR39 register mask (agx_nir_lower_g18_tile_samples), exactly Apple's G18 4x
 * per-pixel fragment: rld 210a42000000 ; lstcol 270605082901ad02000071440000.
 */
bool
agx3_g18_msaa4_tile_shader(const agx_context *ctx)
{
   return ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
          ctx->key->dev.emit_agx3d && ctx->stage == MESA_SHADER_FRAGMENT &&
          (ctx->key->fs.nr_samples == 2 || ctx->key->fs.nr_samples == 4);
}

/* Native454/513/567 cover register-sample reloads at each of these tuple
 * widths. Keep format and alignment checks shared by encoding and ordering. */
static unsigned
agx3_g18_msaa_tile_stride(const agx_instr *I)
{
   if (I->format == AGX_FORMAT_I32)
      return I->mask == 1 ? 4 : I->mask == 15 ? 16 : 0;
   if (I->mask != 15)
      return 0;
   return I->format == AGX_FORMAT_U8NORM ? 4
          : I->format == AGX_FORMAT_F16 ? 8 : 0;
}

static bool
agx3_emit_tile(struct util_dynarray *emission, enum agx_isa_encoding enc,
               const agx_instr *I, uint8_t slot, uint16_t wait_mask,
               bool first_load, bool last_load, bool quad_store,
               bool mrt_store, bool store_f13, bool g18_msaa4, bool g18g)
{
   const bool load = I->op == AGX_OPCODE_LD_TILE;
   if ((!load && I->op != AGX_OPCODE_ST_TILE) ||
       I->nr_dests != (load ? 1 : 0) || I->nr_srcs != (load ? 2 : 3) ||
       I->mask < 1 || I->mask > 15 || I->pixel_offset > UINT8_MAX ||
       I->explicit_coords)
      return false;

   const agx_index value = load ? I->dest[0] : I->src[0];
   const agx_index sample = load ? I->src[0] : I->src[1];
   const agx_index coords = load ? I->src[1] : I->src[2];
   const int width = agx3_tile_format_width(I->format);
   struct agx3_tile_sample sample_op;
   if (((I->format == AGX_FORMAT_RGB10A2 ||
         I->format == AGX_FORMAT_RGB9E5) &&
        (I->mask != 1 || value.size != AGX_SIZE_32)) ||
       width < 0 || !agx3_tile_value_ok(value, I->mask, width) ||
       !agx3_tile_sample_operand(sample, &sample_op) || !agx_is_null(coords))
      return false;

   /* msaa163: at four samples the 0xFF immediate would be encoded as the 1x
    * whole-pixel immediate and write sample 0 only; refuse it rather than
    * resolve a quarter of the colour. Stores take the witnessed AGX3c/AGX3d
    * register form. Native454/513/567 ground raw32 and packed reloads with
    * the same register-sample operand and their corresponding tuple sizes. */
   if (g18_msaa4) {
      const unsigned stride = agx3_g18_msaa_tile_stride(I);
      if (!sample_op.reg ||
          (load && (!stride || (I->pixel_offset % stride))))
         return false;
      sample_op.g17_register = true;
   }

   /* AGX3b f13 varies independently in high-register store witnesses.  The
    * low-register, single-sample subset is the only one with a stable zero
    * selection that production IR can infer today. */
   /* Fresh two-to-four-target BG and ordinary fragments ground f13=0 for full
    * raw32 payloads starting at half16/24 too. Apple's store-only fragments of
    * three to eight targets and multi-target BG builtins of other formats
    * ground it for every width and mask, reading halves up to 63 once more
    * than four targets are stored; in those fragments f13 follows the store
    * order, not the register (agx3_g17p_mrt_protocol). Other high-bank shapes
    * retain the existing refusal. The caller proves the complete store
    * graph. */
   const bool quad_high = !load &&
      ((value.value < 32 && quad_store &&
        ((I->format == AGX_FORMAT_I32 && (I->mask == 1 || I->mask == 15)) ||
         (I->format == AGX_FORMAT_U8NORM && I->mask == 15))) ||
       (mrt_store && value.value < 64));
   if (!load && enc == AGX_ISA_ENCODING_AGX3B && value.value >= 16 && !quad_high)
      return false;

   uint8_t bytes[AGX3_TILE_MAX_LENGTH];
   uint8_t length = 0;
   /* Every retained G17P multisample LLDCol uses the register sample mask
    * with neither the first-load marker nor an incoming scoreboard wait. The
    * coverage operation that produced the mask has already completed its
    * pixel transaction before the load consumes the architectural register. */
   uint16_t tile_wait = load && sample_op.reg ? 0 : wait_mask;
   /* No LLDCol witness carries an incoming wait. A load that must wait (its
    * destination still pending in an inline-blend program, for one) retires
    * that wait on its destination first; waiting early is always safe. */
   if (load && tile_wait) {
      if (!agx3_retire_result_wait(emission, enc, I, tile_wait))
         return false;
      tile_wait = 0;
   }

   /* A multisample clear can reuse its color tuple for later samples.
    * LSTCOL f13 must not terminate that tuple before its final use. */
   if (g18_msaa4 && !load)
      store_f13 = value.kill || value.discard;
   const bool ok =
      load ? agx3_tile_load(enc, value.value, (enum agx3_tile_width)width,
                            slot, I->mask, I->pixel_offset, sample_op,
                            tile_wait, sample_op.reg ? false : first_load,
                            sample_op.reg ? (g18_msaa4 && last_load) : last_load,
                            bytes, &length)
           : g18g ? agx3_tile_store_f13_g18g(enc, value.value,
                                            (enum agx3_tile_width)width,
                                            I->mask, I->pixel_offset,
                                            sample_op, tile_wait, store_f13,
                                            bytes, &length)
           : agx3_tile_store_f13(enc, value.value,
                                 (enum agx3_tile_width)width, I->mask,
                                 I->pixel_offset, sample_op, tile_wait,
                                 store_f13, bytes, &length);
   return ok && agx3_append(emission, bytes, length);
}

/*
 * The sRGB transfer mode of one `fltnorm`/`normflt` pack. agx_compile.c marks
 * the encoded lanes of an 8-bit sRGB attachment in the instruction's mask (bit
 * n = lane n); the alpha lane of RGBA8Unorm_sRGB stays linear. Apple's modes
 * (agx3_fltnorm_core.h): every lane encoded, or lane 0 encoded with a linear
 * lane 1. A linear lane 0 under an encoded lane 1 has no mode; refuse it.
 */
static bool
agx3_norm_srgb_mode(const agx_instr *I, bool two_lanes, uint8_t *srgb)
{
   const unsigned lanes = I->mask & (two_lanes ? 3 : 1);
   if (I->mask & ~lanes)
      return false;
   if (lanes && I->format != AGX_FORMAT_U8NORM)
      return false;
   *srgb = !lanes                       ? AGX3_FLTNORM_SRGB_NONE
           : lanes == 1 && two_lanes    ? AGX3_FLTNORM_SRGB_LANE0
           : (lanes & 1)                ? AGX3_FLTNORM_SRGB_BOTH
                                        : 0xff;
   return *srgb != 0xff;
}

/*
 * `fltnorm`: pack two 32-bit floats, clamped and scaled, into the two halves
 * of one 16-bit register, one per channel pair before an 8-bit normalized
 * tile store (agx_pack_normalized_agx3()). A null second source selects the
 * single-lane form for an odd trailing channel. Both sources are marked
 * still live: RA drops kill/discard, so last use is unknown here, and "live"
 * is the conservative hint.
 */
static bool
agx3_emit_fltnorm(struct util_dynarray *emission, enum agx_isa_encoding enc,
                  const agx_instr *I, uint16_t wait_mask, bool first_instruction,
                  struct agx3_fltnorm_desc *link_desc)
{
   if (I->nr_dests != 1 || I->nr_srcs != 2)
      return false;

   const agx_index dest = I->dest[0];
   const bool norm16 = I->format == AGX_FORMAT_U16NORM ||
                       I->format == AGX_FORMAT_S16NORM;
   if (I->format != AGX_FORMAT_U8NORM && I->format != AGX_FORMAT_S8NORM &&
       I->format != AGX_FORMAT_U16NORM && I->format != AGX_FORMAT_S16NORM)
      return false;
   const bool two_lanes = !agx_is_null(I->src[1]);
   if (!agx3_src_is_register(dest) || dest.memory || dest.size != AGX_SIZE_16 ||
       agx_channels(dest) != (norm16 && two_lanes ? 2 : 1) ||
       dest.value >= 256)
      return false;

   struct agx3_fltnorm_desc d = {
      .dest = dest.value,
      .two_lanes = two_lanes,
      .snorm = I->format == AGX_FORMAT_S8NORM ||
               I->format == AGX_FORMAT_S16NORM,
      .is_16bit = norm16,
      .wait_mask = wait_mask,
      .first_instruction = first_instruction,
   };
   if (!agx3_norm_srgb_mode(I, two_lanes, &d.srgb))
      return false;

   for (unsigned i = 0; i < 2; ++i) {
      if (i == 1 && !d.two_lanes)
         break;
      const agx_index src = I->src[i];
      if (src.memory || agx_channels(src) != 1)
         return false;

      /* All three operand classes are witnessed: a register (every
       * interpolated shader), an 8-bit AGX minifloat immediate (a constant
       * colour, which the optimizer folds straight into this source), and a
       * uniform (a colour read from a buffer). A register may also carry the
       * abs and neg modifiers. */
      if (agx3_src_is_register(src)) {
         if (src.size != AGX_SIZE_32)
            return false;
         d.src[i].kind = AGX3_FLTNORM_SRC_REGISTER;
         d.src[i].abs = src.abs;
         d.src[i].neg = src.neg;
      } else if (src.type == AGX_INDEX_IMMEDIATE) {
         if (src.value > 0xff)
            return false;
         d.src[i].kind = AGX3_FLTNORM_SRC_IMMEDIATE;
      } else if (src.type == AGX_INDEX_UNIFORM) {
         if (src.size != AGX_SIZE_32)
            return false;
         d.src[i].kind = AGX3_FLTNORM_SRC_UNIFORM;
      } else {
         return false;
      }

      d.src[i].value = src.value;
   }

   uint8_t bytes[AGX3_FLTNORM_MAX_LENGTH];
   uint8_t len = 0;
   if (!agx3_fltnorm_encode(enc, &d, bytes, &len) ||
       !agx3_append(emission, bytes, len))
      return false;
   if (link_desc)
      *link_desc = d;
   return true;
}

static bool
agx3_emit_rgb10a2(struct util_dynarray *emission, enum agx_isa_encoding enc,
                  const agx_instr *I,
                 uint16_t wait, bool first, bool g18g)
{
   const bool unpack = I->op == AGX_OPCODE_UNPACK_RGB10A2_AGX3;
   const uint8_t layout = I->format == AGX_FORMAT_RGB9E5    ? AGX3_PACKED_RGB9E5
                          : I->format == AGX_FORMAT_RG11B10F ? AGX3_PACKED_RG11B10
                                                             : AGX3_PACKED_RGB10A2;
   const unsigned lanes = layout == AGX3_PACKED_RGB10A2 ? 4 : 3;
   if (I->nr_dests != 1 || I->nr_srcs != (unpack ? 1 : 4))
      return false;
   const agx_index dest = I->dest[0];
   if (!agx3_src_is_register(dest) || dest.memory || dest.size != AGX_SIZE_32 ||
       agx_channels(dest) != (unpack ? lanes : 1))
      return false;
   uint8_t bytes[16], length = 0;
   if (unpack) {
      const agx_index src = I->src[0];
      if (!agx3_src_is_register(src) || src.memory || src.abs || src.neg ||
          src.size != AGX_SIZE_32 || agx_channels(src) != 1)
         return false;
      const struct agx3_rgb10a2_unpack_desc d = {
         .dest = dest.value, .source = src.value,
         .wait_mask = wait, .discard = src.discard, .layout = layout,
         .dest_hint = enc == AGX_ISA_ENCODING_AGX3C,
      };
      if (!(g18g ? agx3_rgb10a2_unpack_encode_g18g(enc, &d, bytes, &length)
                 : agx3_rgb10a2_unpack_encode(enc, &d, bytes, &length)))
         return false;
   } else {
      struct agx3_rgb10a2_desc d = {.dest = dest.value, .layout = layout,
                                   .first_instruction = first, .wait_mask = wait};
      for (unsigned i = 0; i < 4; ++i) {
         const agx_index src = I->src[i];
         if (i >= lanes) {
            if (!agx_is_null(src))
               return false;
            continue;
         }
         if (src.memory || agx_channels(src) != 1 ||
             (agx3_src_is_register(src) && src.size != AGX_SIZE_32))
            return false;
         d.kind[i] = agx3_src_is_register(src) ? AGX3_FLTNORM_SRC_REGISTER :
                     src.type == AGX_INDEX_IMMEDIATE ? AGX3_FLTNORM_SRC_IMMEDIATE : 0;
         d.source[i] = src.value;
         d.discard[i] = src.discard;
         d.cache[i] = src.cache;
         d.abs[i] = src.abs;
         d.neg[i] = src.neg;
      }
      if (!(g18g ? agx3_rgb10a2_encode_g18g(enc, &d, bytes, &length)
                 : agx3_rgb10a2_encode_for_encoding(enc, &d, bytes, &length)))
         return false;
   }
   return agx3_append(emission, bytes, length);
}

/*
 * `normflt`: the tilebuffer-read inverse of `fltnorm`. One 16-bit register of
 * packed normalized lanes in, a 32-bit float pair out.
 */
static bool
agx3_emit_normflt(struct util_dynarray *emission, enum agx_isa_encoding enc,
                  const agx_instr *I, uint16_t wait_mask)
{
   if (I->nr_dests != 1 || I->nr_srcs != 1)
      return false;

   const agx_index dest = I->dest[0];
   const agx_index src = I->src[0];

   if (!agx3_src_is_register(dest) || dest.memory || dest.size != AGX_SIZE_32 ||
       agx_channels(dest) > 2 || agx_channels(dest) < 1)
      return false;
   const bool norm16 = I->format == AGX_FORMAT_U16NORM ||
                       I->format == AGX_FORMAT_S16NORM;
   if (I->format != AGX_FORMAT_U8NORM && I->format != AGX_FORMAT_S8NORM &&
       I->format != AGX_FORMAT_U16NORM && I->format != AGX_FORMAT_S16NORM)
      return false;
   const bool two_lanes = agx_channels(dest) == 2;
   if (!agx3_src_is_register(src) || src.memory || src.abs || src.neg ||
       src.size != AGX_SIZE_16 ||
       agx_channels(src) != (norm16 && two_lanes ? 2 : 1))
      return false;

   struct agx3_normflt_desc d = {
      .dest = dest.value,
      .dest_hint = true,
      .src = src.value,
      .two_lanes = two_lanes,
      .snorm = I->format == AGX_FORMAT_S8NORM ||
               I->format == AGX_FORMAT_S16NORM,
      .is_16bit = norm16,
      .wait_mask = wait_mask,
   };
   if (!agx3_norm_srgb_mode(I, two_lanes, &d.srgb))
      return false;

   uint8_t bytes[AGX3_FLTNORM_MAX_LENGTH];
   uint8_t len = 0;
   return agx3_normflt_encode(enc, &d, bytes, &len) &&
          agx3_append(emission, bytes, len);
}

static bool
agx3_emit_pixel(struct util_dynarray *emission, enum agx_isa_encoding enc,
                const agx_instr *I, uint16_t wait_mask, bool last,
                unsigned colour_pair, struct agx3_colour_link_info *link, size_t initial_size)
{
   if (I->nr_dests != 0 || I->nr_srcs != 0 ||
       (last && I->op != AGX_OPCODE_SIGNAL_PIX))
      return false;

   uint8_t bytes[AGX3_PIXEL_MAX_LENGTH];
   uint8_t len = 0;
   uint32_t writeout = agx3_g17p_colour_writeout(I->writeout, colour_pair);
   if (I->op == AGX_OPCODE_WAIT_PIX && (writeout & 4) && link) {
      if (link->rt_query_write_acquired) {
         if (writeout != 12) return false;
         writeout = 8;
      } else if (writeout == 12 && link->rt_query_wait_count == 0) {
         link->rt_query_wait_count = 1;
         link->rt_query_wait_offset = emission->size - initial_size;
         link->rt_query_wait_mask = wait_mask;
      } else {
         link->rt_query_wait_unsupported = true;
      }
   }
   const bool ok =
      I->op == AGX_OPCODE_WAIT_PIX
         ? agx3_pixel_wait(enc, writeout, wait_mask, bytes, &len)
         : agx3_pixel_release(enc, writeout, last, wait_mask, bytes, &len);

   return ok && agx3_append(emission, bytes, len);
}

/* The staging address a user output's COPY writes and its UVSW reads. Apple's
 * half-2k scheme holds for every output it can address; beyond that the
 * addresses rotate, which the drained batch boundary makes safe. See
 * AGX3_G17P_STAGING_OUTPUTS. */
static uint16_t
agx3_g17p_output_staging_half(unsigned output)
{
   return (uint16_t)(2 * (output % AGX3_G17P_STAGING_OUTPUTS));
}

unsigned
agx3_g17p_vertex_output_count(agx_context *ctx, enum agx_isa_encoding enc)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || ctx->stage != MESA_SHADER_VERTEX ||
       ctx->is_preamble)
      return 0;

   agx_block *output_block = NULL;
   agx_foreach_block(ctx, block)
      output_block = block;
   bool outputs[AGX3_MAX_VERTEX_OUTPUTS] = {false};
   unsigned output_count = 0, maximum_output = 0, minimum_output = UINT_MAX;
   bool stopped = false;
   agx_foreach_block(ctx, block) {
      agx_foreach_instr_in_block(block, I) {
         if (block != output_block) {
            /* Earlier control flow and producers retain ordinary encoding
             * validation. The terminal output protocol owns only this final
             * block, with no conditional or earlier vertex writeout. */
            if (agx_is_vertex_store(I) || I->op == AGX_OPCODE_STOP)
               return 0;
            continue;
         }
         if (stopped)
            return 0;
         switch (I->op) {
         case AGX_OPCODE_STOP:
            stopped = true;
            break;
         case AGX_OPCODE_ST_VARY:
         case AGX_OPCODE_ST_VARY_MUL_AGX3:
         case AGX_OPCODE_ST_VARY_FMA_AGX3:
            if (I->nr_dests != 0 || I->nr_srcs < 2 ||
                I->src[0].type != AGX_INDEX_IMMEDIATE ||
                I->src[0].value >= AGX3_MAX_VERTEX_OUTPUTS ||
                outputs[I->src[0].value] ||
                (I->op == AGX_OPCODE_ST_VARY && !agx3_vary_reg32(I->src[1])) ||
                agx_channels(I->src[1]) != 1)
               return 0;
            outputs[I->src[0].value] = true;
            output_count++;
            maximum_output = MAX2(maximum_output, I->src[0].value);
            minimum_output = MIN2(minimum_output, I->src[0].value);
            break;
         case AGX_OPCODE_GET_SR:
            /* A system-value read is a pure producer, validated by the SR
             * encoder. Inside the bundle only the two IDs native vertex
             * programs read there are admitted; before it any SR is, such
             * as the subgroup invocation a ballot reads. */
            if (output_count && I->sr != AGX_SR_VERTEX_ID &&
                I->sr != AGX_SR_INSTANCE_ID)
               return 0;
            break;
         case AGX_OPCODE_SCOPED_BARRIER_AGX3:
            /* A memory fence does not overwrite vertex staging. The
             * nonterminal output path retires each COPY before it, and the
             * normal barrier encoder preserves the requested memory scope.
             */
            break;
         default:
            /* Once writeout begins only ST_VARY followed by STOP is safe.
             * Before it begins, side-effecting instructions are fine: they
             * stay in place and normal packing validates their encoding. */
            if (output_count && !agx_opcodes_info[I->op].can_eliminate)
               return 0;
            break;
         }
      }
   }
   const bool grounded_base =
      minimum_output == 0 || minimum_output == AGX3_UVS_POSITION_ENTRIES;
   const bool dense_wide = output_count >= 5 && grounded_base &&
      maximum_output + 1 - minimum_output == output_count;
   /* Sparse terminal bundles prove their own component/slot mapping below.
    * Nonterminal sparse exports still require the driver's existing key. */
   const bool sparse_high = output_count > 0 && maximum_output >= 4;
   return stopped && (dense_wide || sparse_high) ? output_count : 0;
}

/* The sources of a dense terminal store bundle cannot be overwritten before
 * their injected staging COPY reads retire. Each six-store batch must use all
 * six distinct scoreboard slots, assigned by batch position rather than
 * component index so sparse exports cannot collide; preserve output order so grouped
 * Vulkan varyings do not get silently renumbered. Original producers retire
 * on the first COPY and UVSWs retire the staging producers. */
bool
agx3_outputs_are_terminal(agx_context *ctx, unsigned count,
                          uint8_t order[AGX3_MAX_VERTEX_OUTPUTS])
{
   unsigned next = 0;
   bool stopped = false;
   agx_foreach_instr_global(ctx, I) {
      if (stopped)
         return false;
      if (agx_is_vertex_store(I)) {
         if (next >= count || I->src[0].type != AGX_INDEX_IMMEDIATE)
            return false;
         const unsigned component = I->src[0].value;
         order[next++] = component;
      } else if (I->op == AGX_OPCODE_STOP) {
         if (next != count)
            return false;
         stopped = true;
      } else if (next) {
         return false;
      }
   }
   return next == count && stopped;
}

/* A dense run of scalar outputs, a materialized homogeneous 1.0, then the
 * last store (wide vertex programs): the late producer makes the
 * stores nonterminal, but every earlier value is already in its staging word
 * before the producer can reuse a GPR. Recognized by shape at any width up to
 * AGX3_MAX_VERTEX_OUTPUTS: stores dense and in order, the sole late producer a
 * MOV_IMM of 1.0f feeding only the final store, then STOP. */
bool
agx3_is_firefox_dense_output_suffix(agx_context *ctx, unsigned count)
{
   if (count < 5 || count > AGX3_MAX_VERTEX_OUTPUTS)
      return false;

   const unsigned last = count - 1;

   unsigned next = 0;
   bool started = false, saw_one = false, stopped = false;
   agx_index one = agx_null();

   agx_foreach_instr_global(ctx, I) {
      if (stopped)
         return false;

      if (!started && I->op != AGX_OPCODE_ST_VARY)
         continue;
      started = true;

      if (agx_is_vertex_store(I)) {
         if (I->nr_dests != 0 || I->nr_srcs != 2 ||
             I->src[0].type != AGX_INDEX_IMMEDIATE ||
             I->src[0].value != next || next > last)
            return false;

         if ((next < last && saw_one) ||
             (next == last &&
              (!saw_one || !agx_is_equiv(I->src[1], one) ||
               I->src[1].size != one.size)))
            return false;

         ++next;
         continue;
      }

      if (I->op == AGX_OPCODE_MOV_IMM && next == last && !saw_one &&
          I->nr_dests == 1 && I->nr_srcs == 0 && !I->saturate &&
          agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
          I->imm == 0x3f800000) {
         one = I->dest[0];
         saw_one = true;
         continue;
      }

      if (I->op == AGX_OPCODE_STOP && next == count) {
         stopped = true;
         continue;
      }

      return false;
   }

   return next == count && saw_one && stopped;
}

static int
agx3_cmp_u8(const void *a, const void *b)
{
   return (int)*(const uint8_t *)a - (int)*(const uint8_t *)b;
}

/* A dense store bundle whose writes interleave with other work can still be
 * batched when nothing between the first store and STOP stages its own
 * operands: each staged scalar then keeps its component-addressed staging
 * word until its batch's UVSW. Stores retire their staging producer
 * immediately, as the six-output fallback does, so intervening producers may
 * reuse the slots. Texture and image operations and depth/coverage exports
 * publish to the same staging bank and keep refusing the bundle. */
bool
agx3_outputs_leave_staging_alone(agx_context *ctx)
{
   bool started = false;
   agx_foreach_instr_global(ctx, I) {
      if (agx_is_vertex_store(I)) {
         started = true;
         continue;
      }
      if (!started)
         continue;
      switch (I->op) {
      case AGX_OPCODE_TEXTURE_SAMPLE:
      case AGX_OPCODE_TEXTURE_LOAD:
      case AGX_OPCODE_IMAGE_LOAD:
      case AGX_OPCODE_IMAGE_WRITE:
      case AGX_OPCODE_ZS_EMIT:
      case AGX_OPCODE_ZS_EMIT_AGX3:
      case AGX_OPCODE_SAMPLE_MASK_AGX3:
         return false;
      default:
         break;
      }
   }
   return started;
}

bool
agx3_vertex_outputs_share_source(agx_context *ctx)
{
   agx_index first = agx_null();
   bool found = false;
   agx_foreach_instr_global(ctx, I) {
      if (!agx_is_vertex_store(I))
         continue;
      if (I->op != AGX_OPCODE_ST_VARY)
         return false;
      if (!found) {
         first = I->src[1];
         found = true;
      } else if (!agx_is_equiv(first, I->src[1]) ||
                 first.size != I->src[1].size ||
                 first.abs != I->src[1].abs || first.neg != I->src[1].neg) {
         return false;
      }
   }
   return found;
}

bool
agx3_g17p_8norm_format(enum agx_format format)
{
   return format == AGX_FORMAT_U8NORM || format == AGX_FORMAT_S8NORM;
}

/* Native RGBA8 shaders pack before the write fence: opaque shaders then
 * acquire class 12, while blending already acquired it before the read/load.
 * Defer only over FLTNORM in one straight-line block, with exactly the native
 * wait/load/store sequence. Other pixel operations and control flow retain
 * their original ordering.
 */
const agx_instr *
agx3_g17p_deferred_write_wait(agx_context *ctx, bool blend)
{
   const agx_instr *write = NULL;
   unsigned blocks = 0, phase = blend ? 0 : 2;
   agx_foreach_block(ctx, block) {
      if (++blocks != 1)
         return NULL;
      agx_foreach_instr_in_block(block, I) {
         if (agx3_is_cf_op(I) || I->op == AGX_OPCODE_SIGNAL_PIX)
            return NULL;
         if (I->op == AGX_OPCODE_WAIT_PIX) {
            if (phase == 0 && I->writeout == 8) {
               phase = 1;
            } else if (phase == 2 && I->writeout == (blend ? 4 : 12)) {
               write = I;
               phase = 3;
            } else {
               return NULL;
            }
         } else if (I->op == AGX_OPCODE_LD_TILE ||
                    I->op == AGX_OPCODE_ST_TILE) {
            const bool load = I->op == AGX_OPCODE_LD_TILE;
            if (phase != (load ? 1 : 3) ||
                !((agx3_g17p_8norm_format(I->format) && I->mask == 15) ||
                  ((I->format == AGX_FORMAT_RGB10A2 ||
                    I->format == AGX_FORMAT_RGB9E5) && I->mask == 1)) ||
                I->pixel_offset || I->explicit_coords)
               return NULL;
            phase = load ? 2 : 4;
         } else if (phase == 1 ||
                    (phase == 3 && I->op != AGX_OPCODE_FLTNORM_AGX3 &&
                     I->op != AGX_OPCODE_PACK_RGB10A2_AGX3)) {
            return NULL;
         }
      }
   }
   return phase == 4 ? write : NULL;
}

/* The M5/25G76 native constant RGBA8 fragment packs both pairs before its
 * entry/wait/store/release bracket. Qualify only that immediate, single-RT,
 * straight-line shape here; this is not admission of M5 sampled or blended
 * fragment transactions. The complete 60-byte witness is in the compiler
 * regression test and has completed repeated draws on J813. */
const agx_instr *
agx3_g17g_deferred_colour_wait(agx_context *ctx)
{
   const agx_instr *write = NULL;
   unsigned phase = 0;
   if (list_length(&ctx->blocks) != 1)
      return NULL;

   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_STOP)
         continue;
      if (phase == 0 && I->op == AGX_OPCODE_WAIT_PIX && I->writeout == 12) {
         write = I;
      } else if ((phase == 1 || phase == 2) &&
                 I->op == AGX_OPCODE_FLTNORM_AGX3 && I->nr_srcs == 2 &&
                 I->src[0].type == AGX_INDEX_IMMEDIATE &&
                 I->src[1].type == AGX_INDEX_IMMEDIATE) {
         /* The encoder still validates both immediate operands. */
      } else if (phase == 3 && I->op == AGX_OPCODE_ST_TILE &&
                 I->format == AGX_FORMAT_U8NORM && I->mask == 15 &&
                 !I->pixel_offset && !I->explicit_coords) {
         /* One terminal store to the captured RGBA8 tile partition. */
      } else {
         return NULL;
      }
      phase++;
   }
   return phase == 4 ? write : NULL;
}

/* Partitions 1 and 2 own class pairs 1 and 2; every other partition shares
 * partition 0's. */
static unsigned
agx3_g17p_mrt_class_pair(unsigned partition)
{
   return partition < 3 ? partition : 0;
}

/* Class pair of the one colour partition a non-MRT colour transaction (the
 * single-store colour or one-load/one-store blend protocol) touches.
 *
 * Pixel classes order fragments per tile partition, not per program: every
 * fragment accessing a partition must use that partition's classes, or
 * overlapping fragments touching the same tile bytes are not ordered against
 * each other (e.g. an RGBA8 attachment at tile byte 2 behind an R16 one is
 * partition 1, not 0). Returns false when the partition cannot be
 * identified, so the program fails closed. */
bool
agx3_g17p_lone_colour_pair(agx_context *ctx, unsigned *pair)
{
   const agx_instr *tile = NULL;
   *pair = 0;
   agx_foreach_instr_global(ctx, I) {
      if (I->op != AGX_OPCODE_LD_TILE && I->op != AGX_OPCODE_ST_TILE)
         continue;
      if (tile && (tile->pixel_offset != I->pixel_offset ||
                   tile->colour_target != I->colour_target))
         return false;
      tile = I;
   }
   if (!tile)
      return true;

   const BITSET_WORD *partitions = ctx->key->g17p_tile_partitions;
   if (__bitset_is_empty(partitions,
                         BITSET_WORDS(AGX_G17P_MAX_BYTES_PER_SAMPLE))) {
      *pair = agx3_g17p_mrt_class_pair(tile->colour_target);
      return true;
   }

   if (tile->pixel_offset >= AGX_G17P_MAX_BYTES_PER_SAMPLE ||
       !BITSET_TEST(partitions, tile->pixel_offset))
      return false;
   unsigned ordinal = 0;
   for (unsigned offset = 0; offset < tile->pixel_offset; ++offset)
      ordinal += BITSET_TEST(partitions, offset) ? 1 : 0;
   *pair = agx3_g17p_mrt_class_pair(ordinal);
   return true;
}

/* The IR's colour wait/release masks (8 read, 4 write, 12 both) name class
 * pair 0; move them to `pair`. Depth/coverage class 1 is unaffected. */
static uint32_t
agx3_g17p_colour_writeout(uint32_t writeout, unsigned pair)
{
   if (!pair || !writeout || (writeout & ~UINT32_C(0xc)))
      return writeout;
   return writeout << (2 * pair);
}

/* Per-sample colour stores. A multisample background reload writes every
 * sample of one tilebuffer partition after fetching it from the multisample
 * attachment: one store per sample, each with its own sample-mask register
 * (coverage AND the sample's one-hot bit), all at the same RT offset and
 * format. A fragment unrolled per sample stores the same way inside the
 * ordinary colour transaction: class 12 acquired once after the 0x200 entry,
 * one LSTCOL per sample, release 524 once.
 *
 * Admitted shapes: a background emits one two- or four-store set to
 * partition zero with no pixel transaction (one set per target, at disjoint
 * whole-sample tile ranges, for several targets); a colour epilog emits one
 * set to partition zero, at most one store per framebuffer sample, after its
 * class-12 wait. Stores of API-disabled samples (zero coverage) may be
 * missing. Within a target each store's mask must come from its own
 * instruction: two stores carry the same mask when they name the same
 * register and nothing redefined it in between (RA may reuse one register for
 * masks killed at their store). Returns the number of targets, or zero.
 */
unsigned
agx3_g17p_sample_store_protocol(agx_context *ctx, bool colour)
{
   /* A linked epilog has the same complete per-sample colour transaction
    * as a monolithic reload. Its source values come from exports; all tile
    * masks, formats, ordering and releases are still checked below. */
   const bool g18 = agx3_g18_msaa4_tile_shader(ctx) && colour;
   agx_block *tail = colour ? agx_reconverged_mrt_tail(ctx) : NULL;
   /* G18G admits only j873's G18 four-sample colour transaction. */
   if ((ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B &&
        !(agx_is_g18g(&ctx->key->dev)
             ? g18
             : ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C)) ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble ||
       ctx->key->fs.ignore_tib_dependencies == colour ||
       (list_length(&ctx->blocks) != 1 && !tail))
      return 0;

   /* Track the definition reaching each mask over the C register file.
    * The native tile leaf retains B's own 192-half operand ceiling. */
   const agx_instr *reaching[256] = {NULL};
   struct {
      uint16_t offset;
      enum agx_format format;
      unsigned component_mask;
      unsigned stores;
      uint16_t mask_regs[8];
      const agx_instr *mask_defs[8];
   } targets[AGX_MAX_RTS];
   unsigned target_count = 0;
   uint64_t written_B = 0;
   bool stopped = false, waited = false;
   /* A colour epilog may first complete one class-1 depth/stencil
    * transaction of every sample's write, as Apple's per-sample depth
    * fragments do before their colour stores: 1 waited, 2 written,
    * 3 released. */
   unsigned depth = 0, colour_signals = 0;

   agx_block *store_block = tail ? tail : agx_start_block(ctx);
   agx_foreach_instr_in_block(store_block, I) {
      if (stopped)
         return 0;

      if (I->op == AGX_OPCODE_STOP) {
         stopped = true;
         continue;
      }

      if (colour && !waited && I->op == AGX_OPCODE_WAIT_PIX &&
          I->writeout == 1 && depth == 0) {
         depth = 1;
         continue;
      }
      if (colour && I->op == AGX_OPCODE_ZS_EMIT_AGX3 &&
          (depth == 1 || depth == 2)) {
         depth = 2;
         continue;
      }
      if (colour && I->op == AGX_OPCODE_SIGNAL_PIX && I->writeout == 1 &&
          depth == 2) {
         depth = 3;
         continue;
      }

      if (I->op == AGX_OPCODE_WAIT_PIX && colour && !waited &&
          !target_count && I->writeout == 12 && (depth == 0 || depth == 3)) {
         waited = true;
         continue;
      }

      /* G18's generic tile lowering releases after each unrolled sample.
       * Native sample-ID and texture-reload fragments instead hold class12
       * across all four stores, then release524 once. Recognize the complete
       * group before coalescing those releases in the emitter below. */
      if (g18 && I->op == AGX_OPCODE_SIGNAL_PIX && I->writeout == 12 &&
          waited && target_count == 1 &&
          colour_signals < targets[0].stores) {
         colour_signals++;
         continue;
      }

      if (agx3_is_cf_op(I) || I->op == AGX_OPCODE_WAIT_PIX ||
          I->op == AGX_OPCODE_SIGNAL_PIX || I->op == AGX_OPCODE_LD_TILE)
         return 0;

      if (I->op != AGX_OPCODE_ST_TILE) {
         agx_foreach_dest(I, d) {
            const agx_index dest = I->dest[d];
            if (!agx3_src_is_register(dest))
               continue;

            const unsigned halfs = agx_index_size_16(dest);
            for (unsigned h = 0; h < halfs; ++h) {
               const unsigned reg = dest.value + h;
               if (reg < ARRAY_SIZE(reaching))
                  reaching[reg] = I;
            }
         }
         continue;
      }

      struct agx3_tile_sample sample;
      if (I->nr_srcs != 3 || (colour && I->pixel_offset != 0) ||
          I->explicit_coords || !agx_is_null(I->src[2]) ||
          !agx3_tile_sample_operand(I->src[1], &sample) || !sample.reg)
         return 0;

      unsigned t = 0;
      while (t < target_count && targets[t].offset != I->pixel_offset)
         t++;

      if (t == target_count) {
         /* A new target: its whole samples must not overlap another's. */
         const int width = agx3_tile_format_width(I->format);
         const unsigned size_B =
            width < 0 ? 0 : util_last_bit(I->mask) << width;
         const uint64_t bytes =
            BITFIELD64_MASK(size_B) << MIN2(I->pixel_offset, 63);
         if (t == ARRAY_SIZE(targets) || !size_B ||
             I->pixel_offset + size_B > 64 || (written_B & bytes))
            return 0;

         written_B |= bytes;
         targets[t].offset = I->pixel_offset;
         targets[t].format = I->format;
         targets[t].component_mask = I->mask;
         targets[t].stores = 0;
         target_count++;
      } else if (I->format != targets[t].format ||
                 I->mask != targets[t].component_mask) {
         return 0;
      }

      if (targets[t].stores >= ARRAY_SIZE(targets[t].mask_defs))
         return 0;

      /* Same register and the same definition still reaching it is the same
       * mask twice, whatever it holds. Each target writes a sample at most
       * once; background targets of one sample share its mask. */
      const agx_instr *def = sample.value < ARRAY_SIZE(reaching)
                                ? reaching[sample.value] : NULL;

      for (unsigned i = 0; i < targets[t].stores; ++i) {
         if (targets[t].mask_regs[i] == sample.value &&
             targets[t].mask_defs[i] == def)
            return 0;
      }

      targets[t].mask_regs[targets[t].stores] = sample.value;
      targets[t].mask_defs[targets[t].stores++] = def;
   }

   const unsigned samples =
      target_count ? targets[0].stores : 0;
   if (g18) {
      if (depth || target_count != 1 ||
          (samples != 2 && samples != 4) || samples != ctx->key->fs.nr_samples ||
          colour_signals != samples)
         return 0;
      /* Native G18 2x/4x clear, sample-ID and reload programs share one
       * class-12 transaction across signed/unsigned normalized and integer
       * lanes of every width. RGB565 uses three half lanes; packed RGB10A2
       * uses one word. Keep complete per-sample coverage checks above. */
      const enum agx_format format = targets[0].format;
      const unsigned mask = targets[0].component_mask;
      const bool full_mask = mask == 1 || mask == 3 || mask == 15;
      if (!(((format == AGX_FORMAT_U8NORM || format == AGX_FORMAT_S8NORM ||
              format == AGX_FORMAT_U16NORM || format == AGX_FORMAT_S16NORM ||
              format == AGX_FORMAT_I8 || format == AGX_FORMAT_I16 ||
              format == AGX_FORMAT_I32 || format == AGX_FORMAT_F16) &&
             full_mask) ||
            (format == AGX_FORMAT_F16 && mask == 7) ||
            (format == AGX_FORMAT_RGB10A2 && mask == 1)))
         return 0;
   }
   for (unsigned t = 0; t < target_count; ++t) {
      if (targets[t].stores != samples)
         return 0;
   }

   const bool complete = colour
      ? (ctx->key->fs.nr_samples == 2 || ctx->key->fs.nr_samples == 4 ||
         (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
          ctx->key->fs.nr_samples == 8)) &&
           samples > 0 && samples <= ctx->key->fs.nr_samples
      : samples == 2 || samples == 4;
   return stopped && waited == colour && complete &&
                (depth == 0 || depth == 3) &&
                (target_count == 1 || ctx->key->g17p_mrt_bg)
             ? target_count : 0;
}

/* Native464's complete G18 resident-MS reload acquires class252 once before
 * its texture reads, then stores partitions0/16/32 for every sample without
 * an in-body release. Prove the unrolled equivalent, including mask values
 * across register reuse. This is separate from the fragment and G17P rules. */
bool
agx3_g18_ms_mrt_background_protocol(agx_context *ctx)
{
   if (!agx3_g18_msaa4_tile_shader(ctx) || ctx->is_preamble ||
       ctx->key->secondary || !ctx->key->fs.ignore_tib_dependencies ||
       !ctx->key->g17p_mrt_bg)
      return false;

   unsigned stores = 0, tile_stride = 0;
#define G18_BG_REQUIRE(condition) do { \
   if (!(condition)) { \
      if (agx_compiler_debug & AGX_DBG_REFUSE) \
         fprintf(stderr, "M5_TILE_REJECT G18 MS MRT background line=%u stores=%u\n", \
                 __LINE__, stores); \
      return false; \
   } \
} while (0)
   G18_BG_REQUIRE(list_length(&ctx->blocks) == 1);
   struct {
      bool constant;
      uint16_t value;
      uint8_t coverage;
   } halves[192] = {0};
   const unsigned samples = ctx->key->fs.nr_samples;
   bool stopped = false;

   agx_foreach_instr_global(ctx, I) {
      G18_BG_REQUIRE(!stopped);
      if (I->op == AGX_OPCODE_STOP) {
         stopped = true;
         continue;
      }
      G18_BG_REQUIRE(!agx3_is_cf_op(I) && I->op != AGX_OPCODE_WAIT_PIX &&
                     I->op != AGX_OPCODE_SIGNAL_PIX &&
                     I->op != AGX_OPCODE_LD_TILE);
       if (I->op == AGX_OPCODE_ST_TILE) {
          struct agx3_tile_sample sample;
          const unsigned stride = I->mask == 1 ? 4 : I->mask == 15 ? 16 : 0;
          G18_BG_REQUIRE(stores < samples * 3 && I->nr_dests == 0 &&
                         I->nr_srcs == 3 && I->format == AGX_FORMAT_I32 &&
                         stride && (!tile_stride || tile_stride == stride) &&
                         !I->explicit_coords &&
                         agx_is_null(I->src[2]) &&
                         I->pixel_offset == (stores % 3) * stride &&
                         agx3_tile_sample_operand(I->src[1], &sample) &&
                         sample.reg && sample.value < ARRAY_SIZE(halves));
          G18_BG_REQUIRE(halves[sample.value].coverage == BITFIELD_BIT(stores / 3));
          tile_stride = stride;
         stores++;
         continue;
      }
      G18_BG_REQUIRE(agx_opcodes_info[I->op].can_eliminate ||
                     I->op == AGX_OPCODE_GET_SR_COVERAGE ||
                     I->op == AGX_OPCODE_TEXTURE_LOAD ||
                     I->op == AGX_OPCODE_TEXTURE_SAMPLE ||
                     I->op == AGX_OPCODE_PRELOAD ||
                     I->op == AGX_OPCODE_COLLECT || I->op == AGX_OPCODE_SPLIT);

      /* Evaluate sources before invalidating destinations: mask ANDs often
       * overwrite an input. Track low halves, the architectural mask width. */
      bool constant = false;
      uint16_t value = 0;
      uint8_t coverage = 0;
      const bool scalar = I->nr_dests == 1 &&
         agx3_src_is_register(I->dest[0]) &&
         agx_channels(I->dest[0]) == 1 && !I->dest[0].memory;
      if (scalar && I->op == AGX_OPCODE_MOV_IMM && I->nr_srcs == 0) {
         constant = true;
         value = I->imm;
      } else if (scalar && I->op == AGX_OPCODE_GET_SR_COVERAGE &&
                 I->sr == AGX_SR_INPUT_SAMPLE_MASK) {
         coverage = BITFIELD_MASK(samples);
      } else if (scalar && I->op == AGX_OPCODE_BITOP &&
                 I->nr_srcs == 2 && !I->saturate) {
         bool known[2] = {false};
         uint16_t values[2] = {0};
         uint8_t covers[2] = {0};
         for (unsigned s = 0; s < 2; ++s) {
            const agx_index src = I->src[s];
            if (src.abs || src.neg || src.memory || agx_channels(src) != 1)
               continue;
            if (src.type == AGX_INDEX_IMMEDIATE) {
               known[s] = true;
               values[s] = src.value;
            } else if (agx3_src_is_register(src) &&
                       src.value < ARRAY_SIZE(halves)) {
               known[s] = halves[src.value].constant;
               values[s] = halves[src.value].value;
               covers[s] = halves[src.value].coverage;
            }
         }
         if ((I->truth_table & 0xfu) == AGX3_SEL_TT_MOV) {
            constant = known[0];
            value = values[0];
            coverage = covers[0];
         } else if ((I->truth_table & 0xfu) == AGX3_SEL_TT_AND) {
            constant = known[0] && known[1];
            value = values[0] & values[1];
            if (covers[0] && known[1])
               coverage = covers[0] & values[1];
            else if (covers[1] && known[0])
               coverage = covers[1] & values[0];
         }
      }
      agx_foreach_dest(I, d) {
         const agx_index dest = I->dest[d];
         if (!agx3_src_is_register(dest))
            continue;
         for (unsigned h = 0; h < agx_index_size_16(dest); ++h) {
            const unsigned reg = dest.value + h;
            if (reg < ARRAY_SIZE(halves)) {
               halves[reg].constant = false;
               halves[reg].coverage = 0;
            }
         }
      }
      if (scalar && I->dest[0].value < ARRAY_SIZE(halves)) {
         const unsigned reg = I->dest[0].value;
         halves[reg].constant = constant;
         halves[reg].value = value;
         halves[reg].coverage = coverage;
      }
   }
   G18_BG_REQUIRE(stopped && stores == samples * 3);
#undef G18_BG_REQUIRE
   return true;
}

static int
agx3_g17p_mrt_partition(const struct agx3_g17p_mrt_protocol *p,
                        uint16_t offset)
{
   for (unsigned i = 0; i < p->count; ++i) {
      if (p->offsets[i] == offset)
         return i;
   }

   return -1;
}

/* Native1130 (Apple M6, verified capture 1130; run 1139): the tile operation
 * forms of Apple's single-sample mixed-format fragments and backgrounds, all
 * at the whole-pixel immediate sample with no explicit coordinate, each at its
 * attachment's offset (aligned to the lane width):
 *
 *  - lstcol: any lane mask of the attachment at 8-, 16- and 32-bit lanes (41
 *    width/mask pairs among the 169 fragment stores, full masks in the 208
 *    background stores), and one word (mask 1) for a packed RGB10A2 target
 *    (R11G11B10F reaches the tile as its raw W32 word);
 *  - lldcol: the four forms of the 26 fragment reads -- 8-bit lanes mask 15,
 *    16-bit lanes mask 7 or 15, one 32-bit word (mask 1).
 *
 * Signed or 16-bit normalized tile formats and sRGB bytes have no witness. */
static bool
agx3_g18_native_mixed_tile_form(const agx_instr *I)
{
   const bool load = I->op == AGX_OPCODE_LD_TILE;
   if ((!load && I->op != AGX_OPCODE_ST_TILE) ||
       I->nr_dests != (load ? 1 : 0) || I->nr_srcs != (load ? 2 : 3) ||
       I->explicit_coords || I->pixel_offset > UINT8_MAX || I->mask < 1 ||
       I->mask > 15)
      return false;

   const agx_index sample = load ? I->src[0] : I->src[1];
   if (sample.type != AGX_INDEX_IMMEDIATE ||
       sample.value != AGX3_TILE_ALL_SAMPLES ||
       !agx_is_null(load ? I->src[1] : I->src[2]))
      return false;

   const int width = agx3_tile_format_width(I->format);
   if (width < 0 || (I->pixel_offset % (1u << width)))
      return false;

   switch (I->format) {
   case AGX_FORMAT_RGB10A2:
      return I->mask == 1;
   case AGX_FORMAT_I8:
   case AGX_FORMAT_U8NORM:
   case AGX_FORMAT_I16:
   case AGX_FORMAT_F16:
   case AGX_FORMAT_I32:
      break;
   default:
      return false;
   }

   return !load || (width == AGX3_TILE_W8 && I->mask == 15) ||
          (width == AGX3_TILE_W16 && (I->mask == 7 || I->mask == 15)) ||
          (width == AGX3_TILE_W32 && I->mask == 1);
}

/* A fragment epilog or background of a G18 single-sample mixed-format tile
 * (the driver's key; agx_g18g_native_mixed_mask) whose every tile operation is
 * a native1130 form, a background reading nothing. */
static bool
agx3_g18_native_mixed_shader(const agx_context *ctx)
{
   if (!ctx->key->g18_native_mixed ||
       ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3C ||
       !ctx->key->dev.emit_agx3d || ctx->stage != MESA_SHADER_FRAGMENT ||
       ctx->is_preamble || ctx->key->fs.nr_samples > 1)
      return false;

   agx_foreach_instr_global(ctx, I) {
      if (I->op != AGX_OPCODE_LD_TILE && I->op != AGX_OPCODE_ST_TILE)
         continue;
      if (!agx3_g18_native_mixed_tile_form(I) ||
          (I->op == AGX_OPCODE_LD_TILE && ctx->key->fs.ignore_tib_dependencies))
         return false;
   }
   return true;
}

/* Native1130 (run 1139): lstcol f13 is the payload's last-use bit. Of the 169
 * stores of the 39 fragments, the two with f13 = 0 are exactly the two whose
 * payload registers a later store of the same fragment reads again (impl4's
 * RGBA16Sint pair sharing one payload, impl6's RG32Uint payload under a later
 * RGB10A2Uint store); every other store has f13 = 1. Claiming a last use that is
 * not one lets the register file drop a value still to be read. The payloads
 * are compared as the registers each store reads. */
static bool
agx3_g18_store_payload_read_later(const agx_context *ctx, const agx_instr *S)
{
   const agx_index value = S->src[0];
   if (!agx3_src_is_register(value))
      return true;
   const unsigned lo = value.value, hi = lo + agx_index_size_16(value);
   bool after = false;
   agx_foreach_instr_global(ctx, I) {
      if (I == S) {
         after = true;
         continue;
      }
      if (!after || I->op != AGX_OPCODE_ST_TILE)
         continue;
      const agx_index other = I->src[0];
      if (!agx3_src_is_register(other))
         continue;
      const unsigned olo = other.value, ohi = olo + agx_index_size_16(other);
      if (olo < hi && lo < ohi)
         return true;
   }
   return false;
}

/* One tile access of a native1130 mixed-format fragment: the whole-pixel
 * immediate sample (Apple's 1x immediate 1), the read forms of
 * agx3_g18_native_mixed_tile_form() at their target's offset with the
 * first/last read markers and slot of the scoreboard (f78/f74/f12), and the
 * store's last-use bit (f13). The pixel classes around it are the MRT
 * protocol's. A read waiting for an earlier result retires that wait first:
 * no lldcol witness carries one. */
static bool
agx3_emit_g18_mixed_tile(struct util_dynarray *emission,
                         enum agx_isa_encoding enc, const agx_context *ctx,
                         const agx_instr *I, uint8_t slot, uint16_t wait_mask,
                         bool first_load, bool last_load)
{
   const bool load = I->op == AGX_OPCODE_LD_TILE;
   if (enc != AGX_ISA_ENCODING_AGX3C || !agx3_g18_native_mixed_tile_form(I))
      return false;

   const agx_index value = load ? I->dest[0] : I->src[0];
   const int width = agx3_tile_format_width(I->format);
   struct agx3_tile_sample sample_op;
   if ((I->format == AGX_FORMAT_RGB10A2 && value.size != AGX_SIZE_32) ||
       !agx3_tile_value_ok(value, I->mask, width) ||
       !agx3_tile_sample_operand(load ? I->src[0] : I->src[1], &sample_op) ||
       sample_op.reg)
      return false;

   uint16_t tile_wait = wait_mask;
   if (load && tile_wait) {
      if (!agx3_retire_result_wait(emission, enc, I, tile_wait))
         return false;
      tile_wait = 0;
   }

   uint8_t bytes[AGX3_TILE_MAX_LENGTH];
   uint8_t length = 0;
   const bool ok =
      load ? agx3_tile_load_g18_mixed(enc, value.value,
                                      (enum agx3_tile_width)width, slot,
                                      I->mask, I->pixel_offset, sample_op,
                                      first_load, last_load, bytes, &length)
           : agx3_tile_store_lifetime(
                enc, value.value, (enum agx3_tile_width)width, I->mask,
                I->pixel_offset, sample_op, tile_wait,
                !agx3_g18_store_payload_read_later(ctx, I), bytes, &length);
   return ok && agx3_append(emission, bytes, length);
}

static struct agx3_g17p_mrt_protocol
agx3_g17p_mrt_refused(void)
{
   return (struct agx3_g17p_mrt_protocol){0};
}

bool
agx3_g17p_interleaved_zs(const agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_FRAGMENT ||
       (ctx->key->fs.nr_samples != 2 && ctx->key->fs.nr_samples != 4))
      return false;
   unsigned writes = 0;
   bool tile_seen = false, interleaved = false;
   agx_foreach_instr_global(ctx, I) {
      tile_seen |= I->op == AGX_OPCODE_ST_TILE;
      if (I->op == AGX_OPCODE_ZS_EMIT_AGX3) {
         if (I->zs != 1)
            return false;
         writes++;
         interleaved |= tile_seen;
      }
   }
   return interleaved && writes == ctx->key->fs.nr_samples;
}

struct agx3_g17p_mrt_protocol
agx3_g17p_mrt_protocol(agx_context *ctx)
{
   const bool interleaved_zs = agx3_g17p_interleaved_zs(ctx);
   struct agx3_g17p_mrt_protocol p = {0};
   const bool g18_ms = agx3_g18_msaa4_tile_shader(ctx) && ctx->key->secondary;
   const bool g18 = ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
                    ctx->key->dev.emit_agx3d &&
                    (ctx->key->fs.nr_samples <= 1 || g18_ms);
   /* Generic G18 image spills retain conditional memory accesses between
    * the fixed, unrolled resident sample groups. They do not use the G17P
    * native-spill descriptor ABI. */
   const bool multi_block = ctx->key->g17p_emrt || g18_ms;
   /* Native1130 (run 1139): a fragment of a G18 single-sample mixed-format
    * tile whose every tile operation is a native1130 form. */
   const bool g18_native_mixed =
      g18 && !g18_ms && agx3_g18_native_mixed_shader(ctx);
   agx_block *tail = g18 ? NULL : agx_reconverged_mrt_tail(ctx);
   /* G18G admits only j873's G18 shapes. */
   if ((ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B &&
        !(agx_is_g18g(&ctx->key->dev)
             ? g18
             : ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
                  ctx->key->fs.nr_samples <= 8)) ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble ||
       (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C && !g18 &&
        ctx->key->fs.ignore_tib_dependencies) ||
       !ctx->agx3_mrt_tile_ops_ordered ||
       (!g18 && !ctx->key->secondary && !ctx->key->fs.ignore_tib_dependencies &&
        (ctx->key->no_stop ||
         !(agx_get_fs_structure_flags() & AGX_FS_STRUCTURE_INLINE_BLEND))) ||
       (!multi_block && list_length(&ctx->blocks) != 1 && !tail)) {
      /* Not a refusal by itself -- a single-target shader legitimately has no
       * MRT protocol -- so report only when the key names two or more tile
       * partitions, which is when the protocol is the thing a later tile op
       * will need. Reporting every shader buries the one that matters. */
      if ((agx_compiler_debug & AGX_DBG_REFUSE) && g18 &&
          __bitset_count(ctx->key->g17p_tile_partitions,
                         BITSET_WORDS(AGX_G17P_MAX_BYTES_PER_SAMPLE)) >= 2) {
         fprintf(stderr,
                 "G17P MRT protocol declined: agx3b=%u frag=%u preamble=%u "
                 "ordered=%u secondary=%u ignore-tib=%u emrt=%u blocks=%u\n",
                 ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B,
                 ctx->stage == MESA_SHADER_FRAGMENT, ctx->is_preamble,
                 ctx->agx3_mrt_tile_ops_ordered, ctx->key->secondary,
                 ctx->key->fs.ignore_tib_dependencies, ctx->key->g17p_emrt,
                 list_length(&ctx->blocks));
      }
      return p;
   }

   const bool partitions =
      !ctx->key->fs.ignore_tib_dependencies &&
      !__bitset_is_empty(ctx->key->g17p_tile_partitions,
                         BITSET_WORDS(AGX_G17P_MAX_BYTES_PER_SAMPLE));
   if (partitions) {
      if (__bitset_count(ctx->key->g17p_tile_partitions,
                         BITSET_WORDS(AGX_G17P_MAX_BYTES_PER_SAMPLE)) >
          ARRAY_SIZE(p.offsets))
         return p;
      unsigned offset;
      BITSET_FOREACH_SET(offset, ctx->key->g17p_tile_partitions,
                         AGX_G17P_MAX_BYTES_PER_SAMPLE) {
         p.targets[p.count] = p.count;
         p.offsets[p.count++] = offset;
      }
   }

   bool depth_written = false, tile_ops = false, tile_loads = false;
   bool partial_stores = false;
   agx_foreach_block(ctx, block) {
      if (tail && block != tail)
         continue;
      agx_foreach_instr_in_block(block, I) {
         /* A depth export may finish its own class-1 transaction first. */
         if (I->op == AGX_OPCODE_ZS_EMIT_AGX3 && !tile_ops)
            depth_written = true;
         if ((!multi_block && agx3_is_cf_op(I)) ||
             (I->op == AGX_OPCODE_SIGNAL_PIX &&
              (!depth_written || I->writeout != 1 ||
               (tile_ops && !multi_block && !interleaved_zs))))
            return agx3_g17p_mrt_refused();
         if (I->op != AGX_OPCODE_LD_TILE && I->op != AGX_OPCODE_ST_TILE)
            continue;
         /* Native1094 (Apple M6, verified): rg32 fragments and load
          * backgrounds store each RG32 target with lstcol W32 mask 3
          * (f10=2, f26=3) at eight-byte offsets 0/8, under the same pixel
          * classes as the RGBA32 pair. mask2 fragments reload two RGBA8
          * targets (lldcol mask 15) and store them with partial U8NORM
          * masks (f26=10/15 and 5/5); 1076's masks3 grounds the
          * store-only partial form for three targets. */
         const bool g18_1x = g18 && !g18_ms;
         const bool g18_rg32_store = g18_1x && I->op == AGX_OPCODE_ST_TILE &&
                                     I->format == AGX_FORMAT_I32 && I->mask == 3;
         const bool g18_partial_store =
            g18_1x && I->op == AGX_OPCODE_ST_TILE &&
            I->format == AGX_FORMAT_U8NORM && I->mask >= 1 && I->mask < 15;
         if (g18 && !g18_rg32_store && !g18_partial_store &&
             !g18_native_mixed &&
             !(g18_ms ? agx3_g18_msaa_tile_stride(I)
                      : ((I->format == AGX_FORMAT_U8NORM && I->mask == 15) ||
                         (I->format == AGX_FORMAT_I32 &&
                          (I->mask == 1 || I->mask == 15)))))
            return agx3_g17p_mrt_refused();
         partial_stores |= g18_partial_store;
         /* A native1130 mixed layout has no common stride: its offsets
          * follow the attachment-index layout. */
         if (!g18_native_mixed &&
             (g18_ms || g18_rg32_store ||
              (g18 && I->format == AGX_FORMAT_I32 && I->mask == 15))) {
            const unsigned stride =
               g18_rg32_store ? 8 : agx3_g18_msaa_tile_stride(I);
            if ((p.ms_tile_stride && p.ms_tile_stride != stride) ||
                (I->pixel_offset % stride))
               return agx3_g17p_mrt_refused();
            p.ms_tile_stride = stride;
         }
         if (I->explicit_coords || I->pixel_offset > UINT8_MAX)
            return agx3_g17p_mrt_refused();

         tile_ops = true;
         tile_loads |= I->op == AGX_OPCODE_LD_TILE;
         unsigned i;
         for (i = 0; i < p.count; ++i) {
            if (p.offsets[i] == I->pixel_offset)
               break;
         }
         if (i == p.count) {
            if (partitions || p.count == ARRAY_SIZE(p.offsets))
               return agx3_g17p_mrt_refused();
            p.targets[p.count] = I->colour_target;
            p.offsets[p.count++] = I->pixel_offset;
         } else if (!partitions && p.targets[i] != I->colour_target) {
            return agx3_g17p_mrt_refused();
         }
      }
   }
   /* A lone attachment that is not attachment zero is still a multi-target
    * layout: it owns its own class, not class pair zero, as in Apple's
    * eight-slot fragment with only target 2 active. */
   /* Native id_blend_2x/4x uses the same class-12 lifetime for a single
    * target's unrolled sample loads/stores, acquiring read class 8 once
    * and write class 4 once. Keep the ordinary one-store path unchanged. */
   const bool single_target_blend =
      ctx->key->fs.nr_samples > 1 && tile_loads && !multi_block;

   /* A multisample eMRT epilog may have a single resident partition: hk
    * spills reloaded multisample attachments wider than
    * AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE, leaving one eight-byte
    * target in the tile and the others as image stores, which carry no pixel
    * class. Its tile side is the one-target unrolled sample set above (one
    * LSTCol per sample at partition zero inside one class-12 bracket), but
    * the spilled attachments' control flow keeps it out of
    * agx3_g17p_sample_store_protocol(). The multisample eMRT acquire below
    * covers it, including a load/store pair of a resident attachment that is
    * not written. */
   const bool single_target_emrt =
      ctx->key->fs.nr_samples > 1 && multi_block;

   if (!p.count || (p.count == 1 && p.targets[0] == 0 && !single_target_blend &&
                    !single_target_emrt && !interleaved_zs))
      return agx3_g17p_mrt_refused();

   /* Only the witnessed partial-mask shapes: native1094 mask2 reloads and
    * stores two targets; 1076's masks3 stores three with no reload. A
    * native1130 mixed fragment has partial masks of every format with and
    * without reads (36 of the 39 workloads). */
   if (partial_stores && !g18_native_mixed &&
       !((p.count == 2 && tile_loads) || (p.count == 3 && !tile_loads)))
      return agx3_g17p_mrt_refused();

   /* A background of four or more targets keeps the constructors' ascending,
    * transaction-free shape; see the note in agx_compile.c. */
   if (p.count >= 4 && ctx->key->fs.ignore_tib_dependencies)
      return agx3_g17p_mrt_refused();

   for (unsigned i = 0; i < p.count; ++i) {
      for (unsigned j = i + 1; j < p.count; ++j) {
         if (p.offsets[j] < p.offsets[i]) {
            const uint16_t tmp = p.offsets[i];
            p.offsets[i] = p.offsets[j];
            p.offsets[j] = tmp;
            const uint8_t target = p.targets[i];
            p.targets[i] = p.targets[j];
            p.targets[j] = target;
         }
      }
   }
   /* Native1130: a mixed fragment touches no byte of a target whose write
    * mask is empty, attachment zero's included: in five of the 39 fragments
    * the lowest stored offset is 2..8. Each target keeps its own classes. */
   if (p.offsets[0] != 0 && !g18_native_mixed)
       return agx3_g17p_mrt_refused();

   /* G18's native two-to-eight-target RGBA8 fragments use class 12 for
    * targets 0 and 3..7, class 48 for target 1, and class 192 for target 2.
    * Sparse writes keep the framebuffer target identity. Limit this path to
    * the witnessed four-byte RGBA8/R32 layout. Native426 R32Uint store/fetch
    * witnesses use the same classes; each loaded class's read half remains
    * held through its ordered stores. */
   if (g18) {
      if (!p.count || p.count > AGX_MAX_RTS)
         return agx3_g17p_mrt_refused();
      for (unsigned i = 0; i < p.count; ++i) {
         /* Absent framebuffer slots compact the tile offsets, while native
          * sparse-layout fragments retain their logical target's class.
          * A native1130 mixed layout fixes each offset itself. */
         if ((!g18_native_mixed &&
              p.offsets[i] != (p.ms_tile_stride ? p.ms_tile_stride : 4) * i) ||
             (g18_ms && p.targets[i] != i) || p.targets[i] >= AGX_MAX_RTS ||
             (i && p.targets[i] <= p.targets[i - 1]))
            return agx3_g17p_mrt_refused();
      }
   }

   p.multisample = ctx->key->fs.nr_samples > 1 &&
                   !ctx->key->fs.ignore_tib_dependencies;
   unsigned previous_load = p.count;
   uint8_t load_order[AGX_MAX_RTS * 8];
   unsigned load_instr_count = 0;
   uint8_t store_order[AGX_MAX_RTS * 8];
   unsigned store_instr_count = 0;
   agx_index store_coverage = agx_null();
   bool common_store_coverage = true;
   bool stores_started = false;
   bool stopped = false;
   agx_foreach_instr_global(ctx, I) {
      if (stopped)
         return agx3_g17p_mrt_refused();

      if (I->op == AGX_OPCODE_STOP) {
         stopped = true;
         continue;
      }

      if (I->op != AGX_OPCODE_LD_TILE && I->op != AGX_OPCODE_ST_TILE) {
         if (stores_started && !multi_block &&
             (!p.multisample ||
              (!agx_opcodes_info[I->op].can_eliminate &&
               !(interleaved_zs &&
                 (I->op == AGX_OPCODE_ZS_EMIT_AGX3 ||
                  (I->op == AGX_OPCODE_SIGNAL_PIX && I->writeout == 1))))))
            return agx3_g17p_mrt_refused();
         continue;
      }

      const int partition = agx3_g17p_mrt_partition(&p, I->pixel_offset);
      /* Four to eight targets of any formats share the class-12 lifetime. */
      const bool load = I->op == AGX_OPCODE_LD_TILE;
      if (p.count >= 4 &&
          (I->nr_srcs != (load ? 2 : 3) || I->nr_dests != (load ? 1 : 0) ||
           !agx_is_null(I->src[load ? 1 : 2]) || partition < 0))
         return agx3_g17p_mrt_refused();
      struct agx3_tile_sample sample;
      if (partition < 0 || I->nr_srcs < 2 ||
          !agx3_tile_sample_operand(
             I->src[I->op == AGX_OPCODE_LD_TILE ? 0 : 1], &sample) ||
          sample.reg != p.multisample)
         return agx3_g17p_mrt_refused();

      if (I->op == AGX_OPCODE_LD_TILE) {
         const unsigned max_loads =
            p.multisample ? ctx->key->fs.nr_samples : 1;
         if (++p.loads[partition] > max_loads ||
             (!p.multisample &&
              (stores_started || (unsigned)partition > previous_load)) ||
             load_instr_count >= ARRAY_SIZE(load_order))
            return agx3_g17p_mrt_refused();
         previous_load = partition;
         load_order[load_instr_count++] = partition;
      } else {
         stores_started = true;
         if (store_instr_count >= ARRAY_SIZE(store_order))
            return agx3_g17p_mrt_refused();
         if (agx_is_null(store_coverage))
            store_coverage = I->src[1];
         else
            common_store_coverage &= agx_is_equiv(store_coverage, I->src[1]);
         p.stores[partition]++;
         store_order[store_instr_count++] = partition;
      }
   }

   unsigned stored_partitions = 0;
   unsigned stores_per_partition = 0;
   for (unsigned i = 0; i < p.count; ++i) {
      if (!p.stores[i])
         continue;

      stored_partitions++;
      if (!stores_per_partition)
         stores_per_partition = p.stores[i];
      else if (p.stores[i] != stores_per_partition)
         return agx3_g17p_mrt_refused();
   }

   /* Independently validate the load-free coverage broadcast after RA.
    * Keep mixed masks, tile reads and multi-block shaders on the existing
    * complete per-sample protocol. */
   const bool g18_broadcast = g18_ms && stored_partitions >= 2 &&
      stores_per_partition == 1 && !load_instr_count && common_store_coverage &&
      list_length(&ctx->blocks) == 1;
   if (!stored_partitions ||
       (g18_ms && !g18_broadcast &&
        stores_per_partition != ctx->key->fs.nr_samples) ||
       (stores_per_partition != 1 &&
        !(p.multisample &&
          stores_per_partition <= ctx->key->fs.nr_samples)) ||
       store_instr_count != stored_partitions * stores_per_partition)
      return agx3_g17p_mrt_refused();

   unsigned loaded_partitions = 0;
   for (unsigned i = 0; i < p.count; ++i) {
      if (p.loads[i] && p.loads[i] != stores_per_partition)
         return agx3_g17p_mrt_refused();
      loaded_partitions += p.loads[i] != 0;
   }

   if (load_instr_count != loaded_partitions * stores_per_partition)
      return agx3_g17p_mrt_refused();

   if (p.multisample) {
      for (unsigned sample = 0; sample < stores_per_partition; ++sample) {
         unsigned at = sample * loaded_partitions;
         for (unsigned index = 0; index < p.count; ++index) {
            const unsigned partition = g18_ms ? index : p.count - 1 - index;
            if (p.loads[partition] && load_order[at++] != partition)
               return agx3_g17p_mrt_refused();
         }
      }
   }

   /* Each unrolled sample repeats the same descending target set. This is
    * the multi-target form of the already-grounded per-sample colour-store
    * protocol; no new instruction or modifier shape is introduced. */
   for (unsigned sample = 0; sample < stores_per_partition; ++sample) {
      unsigned at = sample * stored_partitions;
      for (int partition = p.count - 1; partition >= 0; --partition) {
         if (!p.stores[partition])
            continue;
         if (store_order[at++] != partition)
            return agx3_g17p_mrt_refused();
      }
   }

   unsigned store_count = 0;
   p.multisample_combined = p.multisample;
   for (int i = p.count - 1; i >= 0; --i) {
      p.multisample_combined &= p.stores[i] != 0;
      /* `partitions` already enumerates every framebuffer attachment, so its
       * ordinal is the identity; otherwise the attachment names itself.
       * Apple's sparse-target fragments all disagree with the ordinal: an
       * eight-slot fragment writing only targets 1 and 3 uses classes
       * {1, 3}, not {0, 1}. */
      const unsigned pair = agx3_g17p_mrt_class_pair(p.targets[i]);
      uint32_t read_class, write_class, combined_class;
      if ((p.stores[i] != stores_per_partition &&
           !(partitions && !p.stores[i] && !p.loads[i])) ||
          !agx3_pixel_colour_partition_classes(
             pair, &read_class, &write_class, &combined_class))
         return agx3_g17p_mrt_refused();
      p.has_loads |= p.loads[i];
      if (pair == 0 && p.loads[i] && !p.shared_loaded) {
         p.shared_loaded = true;
         p.shared_load_first = i;
      }
      if (!p.stores[i])
         continue;
      if (pair == 0 && !(p.classes & BITFIELD_BIT(0)))
         p.shared_first = i;
      if (pair == 0)
         p.shared_last = i;
      p.classes |= BITFIELD_BIT(pair);
      p.combined_classes |= combined_class;
      store_count++;
   }
   /* Multisample store-only fragments of up to eight targets carry no f13. */
   if (p.stores[1] && p.stores[2] && store_count > 5 && !p.multisample)
      p.f13_stores = store_count - 5;

   p.raw32_store = true;
   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_LD_TILE)
         p.raw32_store = false;
      if (I->op == AGX_OPCODE_ST_TILE) {
         const int partition = agx3_g17p_mrt_partition(&p, I->pixel_offset);
         p.raw32_store &= I->format == AGX_FORMAT_I32 && I->mask == 15 &&
                          partition >= 0 && I->pixel_offset == 16u * partition;
      }
   }
   p.native_mixed = g18_native_mixed;
   p.valid = stopped && stores_started;
   return p;
}

/* Fresh G17P BG constructors for two to four identical RGBA32 targets, four
 * R32 targets, and three or four RGBA8 targets have no PIXWAIT/PIXREL in the
 * shader and use loader release 0x200. They store ascending offsets, unlike
 * the ordinary four-target epilog's shared class-12 lifetime. */
bool
agx3_g17p_multi_background_protocol(const agx_context *ctx)
{
   const bool g18 = ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
                    ctx->key->dev.emit_agx3d && ctx->key->fs.nr_samples <= 1;
   if ((ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B &&
        !(agx_is_g18g(&ctx->key->dev)
             ? g18
             : ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
                  ctx->key->fs.nr_samples <= 8)) ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble ||
       !ctx->key->fs.ignore_tib_dependencies ||
       list_length(&ctx->blocks) != 1)
      return false;

   unsigned stores = 0, stride = 0;
   bool stopped = false;
   enum agx_format format = 0;
   BITSET_DECLARE(written_B, AGX_G17P_MAX_BYTES_PER_SAMPLE) = {0};
   /* Native1130: Apple's clear and reload backgrounds of the admitted mixed
    * layouts store each of the row's targets exactly once, with the row's
    * forms and no pixel transaction (clear 0/8/24/32, reload 0/8/16, resume
    * reload 8/0/24/32: disjoint bytes, so no order is implied). Mesa's own
    * backgrounds store them in ascending offset order; require that. */
   /* Native1130 (run 1139): the 39 clear and reload backgrounds of capture
    * 1130 have the same shape for every mixed layout: each attachment stored
    * once with its full lane form (masks 1, 3, 15, and 7 for B5G6R5's three
    * 16-bit lanes), disjoint, no pixel transaction and no read. */
   const bool g18_native_mixed =
      g18 && ctx->key->g17p_mrt_bg && agx3_g18_native_mixed_shader(ctx);
   int previous_offset = -1;
   agx_foreach_instr_global(ctx, I) {
      if (stopped || agx3_is_cf_op(I) || I->op == AGX_OPCODE_WAIT_PIX ||
          I->op == AGX_OPCODE_SIGNAL_PIX || I->op == AGX_OPCODE_LD_TILE)
         return false;
      if (I->op == AGX_OPCODE_STOP) {
         stopped = true;
      } else if (I->op == AGX_OPCODE_ST_TILE && ctx->key->g17p_mrt_bg &&
                 g18_native_mixed) {
         if ((int)I->pixel_offset <= previous_offset || stores >= AGX_MAX_RTS ||
             I->nr_srcs != 3 || I->nr_dests != 0 || I->explicit_coords ||
             I->src[1].type != AGX_INDEX_IMMEDIATE ||
             I->src[1].value != AGX3_TILE_ALL_SAMPLES ||
             !agx_is_null(I->src[2]) ||
             (I->mask != 1 && I->mask != 3 && I->mask != 15 &&
              !(I->mask == 7 &&
                agx3_tile_format_width(I->format) == AGX3_TILE_W16)))
            return false;
         previous_offset = I->pixel_offset;
         stores++;
      } else if (I->op == AGX_OPCODE_ST_TILE && ctx->key->g17p_mrt_bg) {
         if (g18) {
            /* Native1094 rg32 load backgrounds: two lstcol W32 mask 3
             * stores at ascending offsets 0/8 after the reloads, with no
             * pixel transaction, the same shape as the RGBA32 pair. */
            const unsigned width =
               I->format == AGX_FORMAT_I32 && I->mask == 15 ? 16 :
               (I->format == AGX_FORMAT_I32 && I->mask == 3) ? 8 :
               ((I->format == AGX_FORMAT_U8NORM && I->mask == 15) ||
                (I->format == AGX_FORMAT_I32 && I->mask == 1)) ? 4 : 0;
            if (!width || (stores && width != stride) ||
                I->pixel_offset != stores * width || stores >= AGX_MAX_RTS)
               return false;
            stride = width;
         }
         /* Apple's BG builtins for other layouts store each target once, in
          * no required order, so only require disjoint whole-pixel stores.
          * A multisample clear stores each target through the coverage
          * register, as the one-target multisample background does. Five
          * and eight RGBA32Float clears store up to offset 112 the same
          * way. */
         const int width = agx3_tile_format_width(I->format);
         const unsigned size_B =
            width < 0 ? 0 : util_last_bit(I->mask) << width;
         struct agx3_tile_sample sample;
         if (!size_B || stores >= AGX_MAX_RTS ||
             I->pixel_offset + size_B > AGX_G17P_MAX_BYTES_PER_SAMPLE ||
             BITSET_TEST_RANGE(written_B, I->pixel_offset,
                               I->pixel_offset + size_B - 1) ||
             I->nr_srcs != 3 || I->nr_dests != 0 ||
             I->explicit_coords ||
             !agx3_tile_sample_operand(I->src[1], &sample) ||
             sample.reg != (ctx->key->fs.nr_samples > 1) ||
             !agx_is_null(I->src[2]))
            return false;
         BITSET_SET_RANGE(written_B, I->pixel_offset,
                          I->pixel_offset + size_B - 1);
         stores++;
      } else if (I->op == AGX_OPCODE_ST_TILE) {
         const bool rgba8 = I->format == AGX_FORMAT_U8NORM && I->mask == 15;
         const unsigned width =
            rgba8 ? 4 : I->mask == 15 ? 16 : I->mask == 1 ? 4 : 0;
         if (!width || (stores && width != stride) || stores >= 4 ||
             (!rgba8 && I->format != AGX_FORMAT_I32) || I->nr_srcs != 3 ||
             I->nr_dests != 0 || I->explicit_coords ||
             I->src[1].type != AGX_INDEX_IMMEDIATE ||
             I->src[1].value != AGX3_TILE_ALL_SAMPLES ||
             !agx_is_null(I->src[2]) || I->pixel_offset != stores * width)
            return false;
         if (stores && I->format != format)
            return false;
         format = I->format;
         stride = width;
         stores++;
      } else if (!agx_opcodes_info[I->op].can_eliminate &&
                 I->op != AGX_OPCODE_TEXTURE_SAMPLE &&
                 I->op != AGX_OPCODE_PRELOAD &&
                 I->op != AGX_OPCODE_COLLECT && I->op != AGX_OPCODE_SPLIT) {
         return false;
      }
   }
   if (ctx->key->g17p_mrt_bg) {
      /* A sparse M5 attachment set can leave one live target at a nonzero
       * logical slot. It uses the same transaction-free background ABI. */
      const unsigned minimum =
         ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C ? 1 : 2;
      return stopped && stores >= minimum;
   }
   return stopped && ((stride == 16 && stores >= 2 && stores <= 4) ||
                      (stride == 4 && (stores == 4 ||
                       (stores == 3 && format == AGX_FORMAT_U8NORM))));
}

/* Prove complete shader-part shapes from post-RA IR. EXPORT is an ABI
 * declaration with no machine instruction; every declared source must be
 * supplied by one of the bounded uniform copies. Never infer this from bytes.
 */
void
agx3_g17p_colour_link_info(agx_context *ctx, bool opaque_protocol)
{
   struct agx3_colour_link_info info = {0};
   memset(&ctx->agx3_colour_link, 0, sizeof(ctx->agx3_colour_link));
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_FRAGMENT ||
       ctx->key->fs.ignore_tib_dependencies)
      return;

   unsigned blocks = 0, packs = 0, stores = 0, waits = 0;
   const bool copy_only = !ctx->key->secondary && ctx->key->no_stop;
   const bool pack_only = ctx->key->secondary && opaque_protocol;
   if (!copy_only && !pack_only)
      return;

   agx_foreach_block(ctx, block) {
      if (++blocks != 1)
         return;
      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_STOP)
            continue;
         if (copy_only) {
            if (I->op == AGX_OPCODE_EXPORT)
               continue;
            if (I->op != AGX_OPCODE_BITOP ||
                (I->truth_table & 0xfu) != AGX3_SEL_TT_MOV || I->saturate ||
                I->nr_dests != 1 || I->nr_srcs != 2 ||
                I->src[0].type != AGX_INDEX_UNIFORM || I->src[0].memory ||
                I->src[0].abs || I->src[0].neg ||
                I->src[0].size != AGX_SIZE_32 || agx_channels(I->src[0]) != 1 ||
                I->src[1].type != AGX_INDEX_IMMEDIATE || I->src[1].value != 0 ||
                !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
                I->dest[0].size != AGX_SIZE_32 || agx_channels(I->dest[0]) != 1 ||
                ((I->dest[0].value | I->src[0].value) & 1) ||
                I->dest[0].value >= 256 || I->src[0].value >= 512 ||
                info.copy_count == 4)
               return;
            for (unsigned j = 0; j < info.copy_count; ++j)
               if (info.copies[j].reg == I->dest[0].value)
                  return;
            info.copies[info.copy_count].reg = I->dest[0].value;
            info.copies[info.copy_count++].uniform = I->src[0].value;
         } else if (I->op == AGX_OPCODE_FLTNORM_AGX3) {
            if (++packs > 2 || stores)
               return;
         } else if (I->op == AGX_OPCODE_WAIT_PIX) {
            if (I->writeout != 12 || ++waits != 1 || stores)
               return;
         } else if (I->op == AGX_OPCODE_ST_TILE) {
            if (++stores != 1 || packs != 2 || waits != 1 ||
                I->nr_srcs != 3 ||
                I->src[0].size != AGX_SIZE_16 || agx_channels(I->src[0]) != 2 ||
                !agx3_src_is_register(I->src[0]) || I->src[0].memory ||
                I->src[1].type != AGX_INDEX_IMMEDIATE || I->src[1].value != 0xff ||
                !agx_is_null(I->src[2]) || I->explicit_coords || I->pixel_offset ||
                !agx3_g17p_8norm_format(I->format) || I->mask != 15)
               return;
            info.store_reg = I->src[0].value;
         } else {
            return;
         }
      }
   }
   if (copy_only) {
      unsigned exported = 0;
      agx_foreach_block(ctx, block) {
         agx_foreach_instr_in_block(block, I) {
            if (I->op != AGX_OPCODE_EXPORT)
               continue;
            if (I->nr_srcs != 1 || !agx3_src_is_register(I->src[0]) ||
                I->src[0].memory || I->src[0].size != AGX_SIZE_32 ||
                agx_channels(I->src[0]) != 1 || I->src[0].value != I->imm)
               return;
            unsigned j;
            for (j = 0; j < info.copy_count; ++j)
               if (info.copies[j].reg == I->src[0].value)
                  break;
            if (j == info.copy_count)
               return;
            exported |= 1u << j;
         }
      }
      info.copy_only = info.copy_count &&
         exported == (1u << info.copy_count) - 1;
   } else {
      info.opaque_pack = packs == 2 && stores == 1 && waits == 1;
   }
   ctx->agx3_colour_link = info;
}

enum agx3_pack_result
agx3_pack_vary_uvsw_instr(struct agx3_pack_state *st)
{
   agx_context *ctx = st->ctx;
   struct util_dynarray *emission = st->emission;
   const enum agx_isa_encoding enc = st->enc;
   const agx_instr *I = st->I;

   if (I->op == AGX_OPCODE_ITRCOORDS_AGX3) {
      if (ctx->stage != MESA_SHADER_FRAGMENT ||
          !agx3_emit_itrcoords(emission, enc, I, st->wait)) {
         /* The operand list cannot show the mode or the scoreboard
          * slots waited on, which are what the encoder gates on. */
         char why[48];
         snprintf(why, sizeof(why), "interpolation %u, wait 0x%x",
                  (unsigned)I->interpolation, (unsigned)st->wait);
         AGX3_REFUSE_WHY_RET(I, why);
      }
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_INTERP_AGX3 ||
       I->op == AGX_OPCODE_RHW_INTERP_AGX3) {
      if (I->op == AGX_OPCODE_RHW_INTERP_AGX3 &&
          ((enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C) ||
           ctx->stage != MESA_SHADER_FRAGMENT ||
           !agx3_private_rhw_use(ctx, st->block, I)))
         AGX3_REFUSE_RET(I);
      uint16_t interp_wait = st->wait;
      /* Implicit-coordinate interpolation does not read its destination.
       * Its wait therefore covers the old destination's producers, not
       * an arbitrary r0. A synthetic r0 identity can race an unrelated
       * load and write back the old pointer or only one of its halves.
       *
       * Carry exactly these WAW dependencies on a source-free LDIMM
       * into the destination that interpolation immediately replaces.
       * Join the required bits before writing either destination half.
       * No temporary register or additional dependency is needed. */
      if (interp_wait) {
         /* Sample interpolation can read a token tied to its result. */
         if (enc == AGX_ISA_ENCODING_AGX3B &&
             I->interpolation != AGX_INTERPOLATION_SAMPLE) {
            if (I->nr_dests != 1 ||
                !agx3_vary_reg32(I->dest[0]) ||
                agx_channels(I->dest[0]) != 1)
               AGX3_REFUSE_RET(I);
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
            if (!agx3_alu_compose(enc, AGX3_ALU_LDIMM,
                                    I->dest[0].value, 0, 0, 0, 0,
                                    interp_wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE_RET(I);
         } else if (!agx3_retire_result_wait(emission, enc, I, interp_wait)) {
            AGX3_REFUSE_RET(I);
         }
         st->outstanding &= (uint16_t)~interp_wait;
         interp_wait = 0;
      }
      /* G18G keeps j873's first-input marker on centre RHW reads only. */
      const bool first =
         (enc == AGX_ISA_ENCODING_AGX3B ||
          (enc == AGX_ISA_ENCODING_AGX3C &&
           (!ctx->key->dev.emit_agx3d ||
            (I->op == AGX_OPCODE_RHW_INTERP_AGX3 &&
             I->interpolation == AGX_INTERPOLATION_CENTER)))) &&
         ctx->stage == MESA_SHADER_FRAGMENT && !st->fragment_input_started &&
         !st->g17p_fragment_helper_protocol;
      if (!agx3_emit_interp(emission, enc, I, interp_wait, first,
                            ctx->stage == MESA_SHADER_FRAGMENT))
         AGX3_REFUSE_RET(I);
      st->fragment_input_started = true;
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_PERSP_RCP_AGX3) {
      if (I->nr_dests != 1 || I->nr_srcs != 1 || I->saturate ||
          I->imm > 0x7f || !agx3_vary_reg32(I->dest[0]) ||
          agx_channels(I->dest[0]) != 1 ||
          !agx3_vary_reg32(I->src[0]) ||
          agx_channels(I->src[0]) != 1 || I->src[0].kill ||
          (I->src[0].cache && I->src[0].discard))
         AGX3_REFUSE_RET(I);
      const struct agx3_persp_rcp_desc desc = {
         .dest = I->dest[0].value,
         .rhw = I->src[0].value,
         .coef_index = I->imm,
         .rhw_cache = I->src[0].cache,
         .wait_mask = st->wait,
      };
      /* Post-RA discard is optional liveness. This operand leaf has no
       * discard bit: those bits carry the coefficient index instead. */
      uint8_t bytes[AGX3_FLTNORM_MAX_LENGTH], len = 0;
      if (!agx3_persp_rcp_encode(enc, &desc, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_PERSP_MUL_AGX3) {
      if (I->nr_dests != 1 || I->nr_srcs != 2 || I->saturate ||
          I->imm > 0x7f || !agx3_vary_reg32(I->dest[0]) ||
          agx_channels(I->dest[0]) != 1 ||
          !agx3_vary_reg32(I->src[0]) ||
          !agx3_vary_reg32(I->src[1]) ||
          agx_channels(I->src[0]) != 1 ||
          agx_channels(I->src[1]) != 1 ||
          I->src[0].kill || I->src[1].kill ||
          (I->src[0].cache && I->src[0].discard) ||
          (I->src[1].cache && I->src[1].discard))
         AGX3_REFUSE_RET(I);
      const struct agx3_persp_mul_desc desc = {
         .dest = I->dest[0].value,
         .raw = I->src[0].value,
         .reciprocal = I->src[1].value,
         .raw_liveness = I->src[0].discard ? AGX3_OPERAND_KILL :
            I->src[0].cache ? AGX3_OPERAND_LIVE : AGX3_OPERAND_UNMARKED,
         .reciprocal_liveness = I->src[1].discard ? AGX3_OPERAND_KILL :
            I->src[1].cache ? AGX3_OPERAND_LIVE : AGX3_OPERAND_UNMARKED,
         .vary_index = I->imm,
         .dest_hint = !ctx->key->dev.emit_agx3d,
         .wait_mask = st->wait,
      };
      uint8_t bytes[AGX3_FLTNORM_MAX_LENGTH], len = 0;
      bool ok = agx3_persp_mul_encode(enc, &desc, bytes, &len);
      if (!ok && st->wait) {
         /* The f32mul forms carry waits only up to slot 6, but AGX3C
          * allocates slots 7 and 8 too. Carry such a wait on an
          * identity of the destination first, as for other producers
          * without a wait field for it, then multiply without one. */
         struct agx3_persp_mul_desc unwaited = desc;
         unwaited.wait_mask = 0;
         ok = agx3_retire_result_wait(emission, enc, I, st->wait) &&
              agx3_persp_mul_encode(enc, &unwaited, bytes, &len);
      }
      if (!ok || !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_LDCF) {
      const bool first = enc == AGX_ISA_ENCODING_AGX3B &&
         ctx->stage == MESA_SHADER_FRAGMENT && !st->fragment_input_started &&
         !st->g17p_fragment_helper_protocol;
      if (st->wait && !agx3_retire_slots_noop(emission, enc, st->wait))
         AGX3_REFUSE_RET(I);
      if (!agx3_emit_ldcf(emission, enc, I, st->slot, 0, first))
         AGX3_REFUSE_RET(I);
      st->fragment_input_started = true;
      return AGX3_PACK_EMITTED;
   }

   if (agx_is_vertex_store(I)) {
      /* Use a bounded staging allocation independent of GPR indices.
       * Each scalar is consumed immediately, before its slot/address
       * can be reused. A still-live general producer sharing the slot
       * must retire before the staging COPY reuses it. */
      const uint8_t stage_slot = st->g17p_terminal_outputs
         ? st->g17p_outputs_staged % st->nslots + 1
         : (st->g17p_position_protocol || st->g17p_user_output_protocol)
            ? I->src[0].value % st->nslots + 1 : st->next_uvs_slot;
      const uint16_t staging_half =
         st->g17p_user_output_protocol
            ? agx3_g17p_output_staging_half(I->src[0].value)
            : (uint16_t)(st->g18g_output_staging_base +
                         2 * (stage_slot - 1));
      const uint16_t staging_bit = 1u << (stage_slot - 1);
      uint16_t stage_wait;
      if (st->g17p_terminal_outputs) {
         /* The first batch drains original GPR producers. Each later
          * batch drains prior UVSWs before recycling their six slots.
          * Staging addresses remain distinct across both batches. */
         stage_wait = st->g17p_outputs_staged == 0 ? st->wait | st->outstanding_here :
            st->g17p_outputs_staged % st->nslots == 0 ? (1u << st->nslots) - 1 : 0;
      } else {
         stage_wait = st->wait | (st->g17p_user_output_protocol ? st->outstanding_here :
                               st->outstanding_here & staging_bit);
      }
      /* UVSW reuses the same staging file as asynchronous image
       * operations. Retire consumers of these two halves before the
       * coordinate/data COPY, even if they signal another slot. */
      agx3_staging_release(&st->staging_banks, st->outstanding_here);
      const uint16_t staging_conflicts =
         agx3_staging_conflicts(&st->staging_banks, staging_half, 2);
      if (staging_conflicts) {
         /* The first COPY's own wait retires the conflicts it already
          * waits on; only the others need a carrier before it. */
         const uint16_t carried = staging_conflicts & (uint16_t)~stage_wait;
         /* The r0 carrier reads and rewrites r0: sound only when r0
          * is ready or its producer is in the mask. In M5 vertex
          * shaders of texturelodoffset_pcoffset *shadow_vertex the
          * position load into r0 was outstanding when the first
          * output's COPY needed the sample's operand staging retired,
          * and the carrier waited for the sample alone (misplaced
          * vertices, intermittently, under concurrent load). Carry the
          * conflict on the stored register with this store's producer
          * wait, which covers every pending write of it (the COPY
          * waits for those producers anyway). */
         agx_index carrier = agx_null();
         if (enc == AGX_ISA_ENCODING_AGX3C) {
            for (unsigned s = 1; s < I->nr_srcs; ++s) {
               if (agx3_src_is_register(I->src[s]) &&
                   !I->src[s].memory && I->src[s].value < 256) {
                  carrier = I->src[s];
                  break;
               }
            }
         }
         if (carried &&
             !(carrier.type == AGX_INDEX_REGISTER
                  ? agx3_retire_wait_on(emission, enc, carrier,
                                        carried | st->wait)
                  : agx3_retire_slots_noop(emission, enc, carried)))
            AGX3_REFUSE_RET(I);
         st->outstanding &= (uint16_t)~carried;
         agx3_staging_forget(&st->staging_banks, staging_conflicts);
      }
      if (!agx3_emit_uvsw(ctx, emission, enc, I, staging_half, stage_slot,
                          stage_wait, st->g17p_position_protocol ||
                                      st->g17p_user_output_protocol))
         AGX3_REFUSE_RET(I);
      ctx->agx3_staging_halfs =
         MAX2(ctx->agx3_staging_halfs, staging_half + 2);
      /* For the G18G image-write bank floor
       * (AGX3_G18G_OUTPUT_STAGING_LIMIT). */
      st->g18g_other_staging_end =
         MAX2(st->g18g_other_staging_end, staging_half + 2);
      st->outstanding &= (uint16_t)~stage_wait;
      if (st->g17p_user_output_protocol) {
         /* Nonterminal stores can be followed by GPR/slot reuse, so
          * retain the existing immediate retirement there. A terminal
          * bundle has no intervening IR and uses unique slots per batch;
          * its original producers were all retired by the first COPY.
          * Each batch's UVSWs retire its staging reads instead. */
         if (!st->g17p_terminal_outputs &&
             !agx3_retire_slots_noop(emission, enc, staging_bit))
            AGX3_REFUSE_RET(I);
         /* Each batch writes the components it staged: a terminal
          * bundle in store order, other stores in component order. */
         st->g17p_output_order[st->g17p_outputs_staged] = I->src[0].value;
         ++st->g17p_outputs_staged;
         if (st->g17p_outputs_staged % st->nslots == 0 ||
             st->g17p_outputs_staged == st->g17p_output_count) {
            const unsigned first =
               ((st->g17p_outputs_staged - 1) / st->nslots) * st->nslots;
            if (!st->g17p_terminal_outputs) {
               qsort(st->g17p_output_order + first,
                     st->g17p_outputs_staged - first,
                     sizeof(st->g17p_output_order[0]), agx3_cmp_u8);
            }
            for (unsigned position = first;
                 position < st->g17p_outputs_staged; ++position) {
               const unsigned component = st->g17p_output_order[position];
               uint8_t bytes[AGX3_VARY_MAX_LENGTH], len = 0;
               const unsigned token =
                  (st->g17p_terminal_outputs ? position : component) % st->nslots;
               agx3_note_uvs_reloc(ctx, emission, component,
                                   agx3_g17p_output_staging_half(component),
                                   token + 1, 1u << token);
               if (!agx3_vary_uvsw(enc,
                                    agx3_g17p_output_staging_half(component),
                                    component, token + 1, true,
                                    1u << token, bytes, &len) ||
                   !agx3_append(emission, bytes, len))
                  AGX3_REFUSE_RET(I);
            }
         }
      }
      /* Match the complete Apple float2 VS's producer-before-writeout
       * schedule: stage x/y before Mesa reuses their GPRs for z/w,
       * then emit the four UVSWs only after all four values exist.
       * This is the existing exact straight-line shape guard, not a
       * fixed shader-body replacement. */
      if (st->g17p_position_protocol && ++st->g17p_positions_staged == 4) {
         for (unsigned component = 0; component < 4; ++component) {
            uint8_t bytes[AGX3_VARY_MAX_LENGTH], len = 0;
            if (!agx3_vary_uvsw(enc, component * 2, component,
                                 component + 1, true, 1u << component,
                                 bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE_RET(I);
         }
      }
      st->next_uvs_slot = (st->next_uvs_slot % st->nslots) + 1;
      return AGX3_PACK_EMITTED;
   }

   return AGX3_PACK_UNHANDLED;
}

enum agx3_pack_result
agx3_pack_frag_pixel_instr(struct agx3_pack_state *st)
{
   agx_context *ctx = st->ctx;
   struct util_dynarray *emission = st->emission;
   const enum agx_isa_encoding enc = st->enc;
   const agx_instr *I = st->I;

   if (I->op == AGX_OPCODE_WAIT_PIX ||
       I->op == AGX_OPCODE_SIGNAL_PIX) {
      if (st->g18_sample_colour_stores) {
         if (I->op == AGX_OPCODE_WAIT_PIX) {
            uint8_t bytes[AGX3_PIXEL_MAX_LENGTH], len = 0;
            if (!st->g17p_colour_open) {
               if (!agx3_pixel_colour_entry(enc, bytes, &len) ||
                   !agx3_append(emission, bytes, len))
                  AGX3_REFUSE_RET(I);
               ctx->agx3_colour_link.entry_emitted = true;
               ctx->agx3_colour_link.entry_offset =
                  emission->size - len - st->initial_size;
               st->g17p_colour_open = true;
            }
            if (!agx3_pixel_wait(enc, 12, st->wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE_RET(I);
         } else if (!agx3_retire_slots_noop(emission, enc, st->wait)) {
            AGX3_REFUSE_RET(I);
         }
         return AGX3_PACK_EMITTED;
      }
      if (st->interleaved_zs && I->op == AGX_OPCODE_SIGNAL_PIX) {
         if (!agx3_retire_slots_noop(emission, enc, st->wait))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
      /* A late depth test for fragment memory side effects stages the
       * pass-through Z value before acquiring class 1 in Apple's fresh
       * G17P atomic fragment. Hold this wait until ZS_EMIT can reproduce
       * that ordering; SIGNAL_PIX remains in its original position. */
      /* Native M5 covered-sample loops (zs_depth_4x and kin) take
       * class 1 once, ahead of the loop and its staging, instead. */
      if (st->has_zs_emit && !st->zs_sample_loop && I->op == AGX_OPCODE_WAIT_PIX &&
          I->writeout == 1) {
         if (st->interleaved_zs && st->g17p_zs_writes > 0) {
            if (!agx3_retire_slots_noop(emission, enc, st->wait))
               AGX3_REFUSE_RET(I);
            return AGX3_PACK_EMITTED;
         }
         if (st->g17p_deferred_zs_wait || st->wait != 0)
            AGX3_REFUSE_RET(I);
         st->g17p_deferred_zs_wait = I;
         return AGX3_PACK_EMITTED;
      }
      /* Multi-RT colour waits in the generic IR do not identify a tile
       * partition.  Retire their scoreboard dependencies here; the
       * exact partition-specific wait is emitted immediately beside
       * its LLDCol/LSTCol below. */
      if (st->g17p_mrt.valid && I->op == AGX_OPCODE_WAIT_PIX &&
          I->writeout != 1) {
         if ((st->g17p_mrt.has_loads || ctx->key->g17p_emrt) &&
             !st->g17p_colour_open) {
            uint8_t entry[AGX3_PIXEL_MAX_LENGTH];
            uint8_t entry_len = 0;
            if (!agx3_pixel_colour_entry(enc, entry, &entry_len) ||
                !agx3_append(emission, entry, entry_len))
               AGX3_REFUSE_RET(I);
            ctx->agx3_colour_link.entry_emitted = true;
            ctx->agx3_colour_link.entry_offset =
               emission->size - entry_len - st->initial_size;
            st->g17p_colour_open = true;
         }
         if (!agx3_retire_slots_noop(emission, enc, st->wait))
            AGX3_REFUSE_RET(I);
         /* The single combined acquire ahead of the stores is the
          * multisample eMRT form: Apple's 2x/4x multi-target fragments
          * hold every stored partition's classes across the whole
          * unrolled sample loop. A one-sample fragment brackets each
          * partition around its own tile access instead, which is what
          * the LLDCol/LSTCol path below emits, so leave the classes
          * unacquired here and let it do that. Anything else -- a
          * multisample eMRT epilog that does not store every partition
          * -- still fails closed. */
         if (ctx->key->g17p_emrt && !st->g17p_mrt_acquired &&
             st->g17p_mrt.multisample) {
            uint8_t bytes[AGX3_PIXEL_MAX_LENGTH], len;
            if (!st->g17p_mrt.multisample_combined ||
                !agx3_pixel_wait(enc, st->g17p_mrt.combined_classes, 0,
                                 bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE_RET(I);
            st->g17p_mrt_acquired = true;
         }
         return AGX3_PACK_EMITTED;
      }
      if (I == st->g17p_deferred_write || I == st->g17g_deferred_write) {
         /* The scoreboard has already retired these dependencies here.
          * Keep that retirement at the original instruction, independent
          * of where the pixel fence is emitted. */
         if (!agx3_retire_slots_noop(emission, enc, st->wait))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
      const bool colour_entry_wait =
         I->writeout == 1 ||
         /* Dead blend reads can leave separate class-8/class-4
          * waits around a store-only shader. Acquire before the
          * first surviving colour wait, preserving both waits. */
         (st->g17p_colour_protocol && (I->writeout & 0xc)) ||
         I->writeout == (st->g17p_blend_protocol ? 8 : 0xc);
      if (st->g17p_pixel_protocol && I->op == AGX_OPCODE_WAIT_PIX &&
          colour_entry_wait &&
          !st->g17p_colour_open) {
         uint8_t entry[AGX3_PIXEL_MAX_LENGTH];
         uint8_t entry_len = 0;
         if (!agx3_pixel_colour_entry(enc, entry, &entry_len) ||
             !agx3_append(emission, entry, entry_len))
            AGX3_REFUSE_RET(I);
         ctx->agx3_colour_link.entry_emitted = true;
         ctx->agx3_colour_link.entry_offset = emission->size - entry_len - st->initial_size;
         st->g17p_colour_open = true;
      }
      /* G18's 4x colour fragment ends like Apple's: the release after
       * its masked store carries the 0x200 bit (pixrel f14 0x20c). */
      const bool g18_msaa4_colour =
         ctx->stage == MESA_SHADER_FRAGMENT &&
         !ctx->key->fs.ignore_tib_dependencies &&
         ctx->key->dev.emit_agx3d &&
         (ctx->key->fs.nr_samples == 2 || ctx->key->fs.nr_samples == 4);
      /* G18G: j873's depth-only and 4x colour final release. */
      const bool final_release =
         agx_is_g18g(&ctx->key->dev) &&
         (st->has_zs_emit || g18_msaa4_colour) &&
         I->op == AGX_OPCODE_SIGNAL_PIX && I == st->g17g_final_signal &&
         !st->g17g_store_after_final_signal;
      /* The class-1 release closing a covered-sample loop, recorded
       * for agx_fast_link when it is the plain form. */
      const bool zs_loop_release =
         st->zs_sample_loop && I->op == AGX_OPCODE_SIGNAL_PIX &&
         I->writeout == 1 && st->wait == 0 && st->g17p_zs_writes > 0 &&
         I != st->g17p_final_release && !ctx->agx3_colour_link.zs_loop_release;
      const size_t release_at = emission->size;
      if (!agx3_emit_pixel(emission, enc, I, st->wait,
                           final_release || I == st->g17p_final_release,
                           st->g17p_lone_pair, &ctx->agx3_colour_link, st->initial_size))
         AGX3_REFUSE_RET(I);
      if (zs_loop_release && emission->size - release_at == AGX3_PIXEL_MAX_LENGTH) {
         ctx->agx3_colour_link.zs_loop_release = true;
         ctx->agx3_colour_link.zs_loop_release_offset =
            release_at - st->initial_size;
      }
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_FLTNORM_AGX3) {
      const bool first = (enc == AGX_ISA_ENCODING_AGX3B ||
                          st->g17g_deferred_write != NULL) &&
         ctx->stage == MESA_SHADER_FRAGMENT &&
         !ctx->key->fs.ignore_tib_dependencies &&
         emission->size == st->initial_size;
      struct agx3_fltnorm_desc *link_desc =
         ctx->agx3_colour_link.opaque_pack ?
            &ctx->agx3_colour_link.packs[st->g17p_link_packs_seen++] : NULL;
      if (!agx3_emit_fltnorm(emission, enc, I, st->wait, first, link_desc))
         AGX3_REFUSE_RET(I);
      ctx->agx3_fltnorm_start |= first;
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_PACK_RGB10A2_AGX3 ||
       I->op == AGX_OPCODE_UNPACK_RGB10A2_AGX3) {
      /* A secondary epilog follows the main shader: its local offset
       * zero is not the start of the linked program. G17P records the
       * marker and relocates it during linking; G18G (agx3d) epilogs
       * use the native non-leading form directly. Keep G18G
       * whole-program leading packs guarded by the encoder. */
      const bool first = (enc == AGX_ISA_ENCODING_AGX3B ||
                          !ctx->key->dev.emit_agx3d ||
                          !ctx->key->secondary) &&
         ctx->stage == MESA_SHADER_FRAGMENT &&
         !ctx->key->fs.ignore_tib_dependencies &&
         emission->size == st->initial_size;
      if ((enc != AGX_ISA_ENCODING_AGX3B &&
           enc != AGX_ISA_ENCODING_AGX3C) ||
          !agx3_emit_rgb10a2(emission, enc, I, st->wait, first,
                             ctx->key->dev.emit_agx3d))
         AGX3_REFUSE_RET(I);
      ctx->agx3_fltnorm_start |= first &&
         I->op == AGX_OPCODE_PACK_RGB10A2_AGX3;
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_NORMFLT_AGX3) {
      if (!agx3_emit_normflt(emission, enc, I, st->wait))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_LD_TILE || I->op == AGX_OPCODE_ST_TILE) {
      bool first = false, last = false;
      if (I->op == AGX_OPCODE_LD_TILE) {
         assert(st->tile_loads_left > 0);
         first = st->tile_loads_seen++ == 0;
         last = --st->tile_loads_left == 0;
      } else {
         assert(st->tile_stores_left > 0);
         last = --st->tile_stores_left == 0;
      }
      if (st->g17p_mrt.valid && st->g17p_mrt.multisample_combined) {
         /* Apple's 2x/4x fragments of two to eight targets acquire every
          * stored partition's classes in one PIXWAIT (60, or 252 from
          * three targets), access the partitions in descending order
          * through the coverage register and release those classes once
          * (msaa/native/persample/msmrt*.py). Blended fragments open the
          * same lifetime at their first load and keep it through the
          * final store. Fragments writing only some targets bracket each
          * partition as at one sample (mssubset.py). */
         uint8_t bytes[AGX3_PIXEL_MAX_LENGTH];
         uint8_t len = 0;
         const bool load = I->op == AGX_OPCODE_LD_TILE;
         /* A lone active sparse attachment retains its own classes;
          * only logical target zero owns the class-8/class-4 pair.
          * An eMRT epilog is excluded: its combined acquire has already
          * been emitted at the colour WAIT_PIX above, and splitting the
          * lifetime again here would acquire class 8 and class 4 a
          * second time inside it. */
         const bool single_blend =
            st->g17p_mrt.count == 1 && st->g17p_mrt.targets[0] == 0 &&
            st->g17p_mrt.has_loads && !ctx->key->g17p_emrt && !st->interleaved_zs;
         const bool acquire = single_blend
            ? (load ? first : st->g17p_mrt_stores_seen == 0)
            : !st->g17p_mrt_acquired &&
               (load ? first : (!st->g17p_mrt.has_loads &&
                                st->g17p_mrt_stores_seen == 0));
         const uint32_t acquire_classes = single_blend
            ? (load ? 8 : 4) : st->g17p_mrt.combined_classes;
         /* Native454's G18 three-target wide reloads retain the combined
          * acquire across all samples and end with PIXREL512, as do
          * physical native464's three-target float fragments. Native513
          * scalar reloads release their colour classes at final stores;
          * keep those classes held until this combined group's end. */
         const uint32_t release_classes =
            agx3_g18_msaa4_tile_shader(ctx) && st->g17p_mrt.count == 3 &&
                  st->g17p_mrt.has_loads && st->g17p_mrt.ms_tile_stride == 16
               ? 0 : st->g17p_mrt.combined_classes;
         if (!st->g17p_colour_open) {
            if (!agx3_pixel_colour_entry(enc, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE_RET(I);
            ctx->agx3_colour_link.entry_emitted = true;
            ctx->agx3_colour_link.entry_offset =
               emission->size - len - st->initial_size;
            st->g17p_colour_open = true;
         }
         if ((acquire &&
              (!agx3_pixel_wait(enc, acquire_classes, 0, bytes, &len) ||
               !agx3_append(emission, bytes, len))) ||
             !agx3_emit_tile(
                emission, enc, I, st->slot, st->wait, false,
                /* Native567's single-target register read
                 * terminates its load group in each sample
                 * body, including a statically unrolled one. */
                load && agx3_g18_msaa4_tile_shader(ctx) &&
                   st->g17p_mrt.count == 1,
                st->g17p_mrt.raw32_store,
                /* Native per-sample exports use halves
                 * 0..31 even for one/two targets. */
                true,
                /* f13 releases the payload, which the unrolled
                 * samples may share: set it only at the last use. On
                 * an earlier store the later samples read zero, on
                 * G17P as on M5. G18G keeps its G17P-style rule: f13
                 * on every extended-MRT store. */
                (!ctx->key->dev.emit_agx3d && !load &&
                 (I->src[0].kill || I->src[0].discard) &&
                 (enc == AGX_ISA_ENCODING_AGX3C ||
                  (ctx->key->g17p_emrt &&
                   I->src[0].value >= 16))) ||
                   (ctx->key->dev.emit_agx3d &&
                    ctx->key->g17p_emrt && !load &&
                    I->src[0].value >= 16),
                agx3_g18_msaa4_tile_shader(ctx),
                ctx->key->dev.emit_agx3d) ||
             (!load && last &&
              (!agx3_pixel_release(
                  enc, st->interleaved_zs ? 0 : release_classes,
                  true, 0, bytes, &len) ||
               !agx3_append(emission, bytes, len))))
            AGX3_REFUSE_RET(I);
         st->g17p_mrt_stores_seen += !load;
         st->g17p_mrt_acquired |= acquire;
         return AGX3_PACK_EMITTED;
      }
      if (st->g17p_mrt.valid) {
         const int partition =
            agx3_g17p_mrt_partition(&st->g17p_mrt, I->pixel_offset);
         uint32_t read_class, write_class, combined_class;
         if (partition < 0 ||
             !agx3_pixel_colour_partition_classes(
                agx3_g17p_mrt_class_pair(st->g17p_mrt.targets[partition]), &read_class,
                &write_class, &combined_class))
            AGX3_REFUSE_RET(I);

         uint8_t bytes[AGX3_PIXEL_MAX_LENGTH];
         uint8_t len = 0;
         if (!st->g17p_colour_open) {
            if (!agx3_pixel_colour_entry(enc, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE_RET(I);
            ctx->agx3_colour_link.entry_emitted = true;
            ctx->agx3_colour_link.entry_offset =
               emission->size - len - st->initial_size;
            st->g17p_colour_open = true;
         }

         const bool load = I->op == AGX_OPCODE_LD_TILE;
         /* Class 12 opens before the first load or store using it and
          * stays held until the last store; the other pairs bracket
          * their own loads and store. */
         const bool shared =
            agx3_g17p_mrt_class_pair(st->g17p_mrt.targets[partition]) == 0;
         const bool loaded = shared ? st->g17p_mrt.shared_loaded
                                    : st->g17p_mrt.loads[partition];
         const uint32_t acquire =
            load ? read_class : loaded ? write_class : combined_class;
         const bool acquire_class =
            !shared || partition == (load ? st->g17p_mrt.shared_load_first
                                          : st->g17p_mrt.shared_first);
         const bool release_class =
            !shared || partition == st->g17p_mrt.shared_last;
         /* No leading-store f13 hint on a payload a later attachment
          * reuses (it reads back as zero on G17P); the hint stays on
          * unique payloads. */
         const bool store_f13 =
            !load &&
            ((enc == AGX_ISA_ENCODING_AGX3C &&
              !ctx->key->dev.emit_agx3d) ||
             st->g17p_mrt_stores_seen++ < st->g17p_mrt.f13_stores) &&
            (I->src[0].kill || I->src[0].discard);
         /* A multisample store keeps its own coverage dependency. */
         const bool store_waits = !acquire_class || st->g17p_mrt.multisample;
         uint32_t query_acquire = acquire;
         if (acquire_class && shared && (acquire & 4)) {
            if (ctx->agx3_colour_link.rt_query_write_acquired) {
               if (acquire != 12) AGX3_REFUSE_RET(I);
               query_acquire = 8;
            } else if (acquire == 12 && ctx->agx3_colour_link.rt_query_wait_count == 0) {
               ctx->agx3_colour_link.rt_query_wait_count = 1;
               ctx->agx3_colour_link.rt_query_wait_offset = emission->size - st->initial_size;
               ctx->agx3_colour_link.rt_query_wait_mask = store_waits ? 0 : st->wait;
            } else {
               ctx->agx3_colour_link.rt_query_wait_unsupported = true;
            }
         }
         if (acquire_class &&
             (!agx3_pixel_wait(enc, query_acquire, store_waits ? 0 : st->wait,
                               bytes, &len) ||
              !agx3_append(emission, bytes, len)))
            AGX3_REFUSE_RET(I);
         /* Native1130 (run 1139): a mixed-format fragment's reads and
          * stores take their native1130 forms (agx3_emit_g18_mixed_tile);
          * the classes around them are this protocol's. */
         if (!(st->g17p_mrt.native_mixed
                  ? agx3_emit_g18_mixed_tile(emission, enc, ctx, I, st->slot,
                                             store_waits ? st->wait : 0,
                                             load && first, load && last)
                  : agx3_emit_tile(emission, enc, I, st->slot,
                                   store_waits ? st->wait : 0,
                                   ctx->key->dev.emit_agx3d &&
                                      load && first,
                                   ctx->key->dev.emit_agx3d &&
                                      load && last,
                                   st->g17p_mrt.raw32_store,
                                   st->g17p_mrt.count >= 3,
                                   store_f13,
                                   agx3_g18_msaa4_tile_shader(ctx),
                                   ctx->key->dev.emit_agx3d)) ||
             (!load && release_class &&
              (!agx3_pixel_release(enc, combined_class, last, 0, bytes,
                                  &len) ||
               !agx3_append(emission, bytes, len))))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
      if (I->op == AGX_OPCODE_ST_TILE && st->g17p_deferred_write) {
         if (!st->g17p_colour_open) {
            uint8_t entry[AGX3_PIXEL_MAX_LENGTH], entry_len = 0;
            if (!agx3_pixel_colour_entry(enc, entry, &entry_len) ||
                !agx3_append(emission, entry, entry_len))
               AGX3_REFUSE_RET(I);
            ctx->agx3_colour_link.entry_emitted = true;
            ctx->agx3_colour_link.entry_offset = emission->size - entry_len - st->initial_size;
            st->g17p_colour_open = true;
         }
         if (!agx3_emit_pixel(emission, enc, st->g17p_deferred_write, 0,
                              false, st->g17p_lone_pair, &ctx->agx3_colour_link, st->initial_size))
            AGX3_REFUSE_RET(I);
      }
      if (I->op == AGX_OPCODE_ST_TILE && st->g17g_deferred_write) {
         uint8_t entry[AGX3_PIXEL_MAX_LENGTH], entry_len = 0;
         if (!agx3_pixel_colour_entry(enc, entry, &entry_len) ||
             !agx3_append(emission, entry, entry_len) ||
             !agx3_emit_pixel(emission, enc, st->g17g_deferred_write, 0,
                              false, 0, &ctx->agx3_colour_link,
                              st->initial_size))
            AGX3_REFUSE_RET(I);
      }
      /* Native C multisample stores retain a reused payload with
       * f13=0 and release it only at its last use. B's raw32 stores
       * retain their own policy. */
      const bool store_f13 = enc == AGX_ISA_ENCODING_AGX3C &&
         ((!st->g17p_multi_background &&
           (!st->g17p_sample_colour_stores || ctx->key->fs.nr_samples <= 1)) ||
          I->src[0].kill || I->src[0].discard);
      if (!agx3_emit_tile(emission, enc, I, st->slot, st->wait, first, last,
                          st->g17p_multi_background ||
                             st->g17p_sample_colour_stores,
                          (st->g17p_multi_background ||
                           st->g17p_msaa_background_targets > 1) &&
                             ctx->key->g17p_mrt_bg,
                          !ctx->key->dev.emit_agx3d && store_f13,
                          agx3_g18_msaa4_tile_shader(ctx),
                          ctx->key->dev.emit_agx3d))
         AGX3_REFUSE_RET(I);

      /* Native1094 a2c (Apple M6, verified): a 2x/4x fragment whose
       * coverage write sits in the same part as its colour store ends
       * pixwait12 -> lstcol -> pixrel524 (671000040000) -> end, with one
       * release. The final SIGNAL_PIX after this store already emits
       * that release (final_release below); the protocol's own release
       * here would be a second pixrel524, which no native fragment and
       * no program Mesa 1109 links (813-case frontier) contains. */
      const bool g18_ir_final_release =
         enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
         (ctx->key->fs.nr_samples == 2 || ctx->key->fs.nr_samples == 4) &&
         st->has_coverage_write && st->g17g_final_signal &&
         !st->g17g_store_after_final_signal;
      if (st->g17p_pixel_protocol && I->op == AGX_OPCODE_ST_TILE && last &&
          !g18_ir_final_release) {
         uint8_t release[AGX3_PIXEL_MAX_LENGTH];
         uint8_t release_len = 0;
         if (!st->g17p_colour_open ||
             !agx3_pixel_release(
                enc, agx3_g17p_colour_writeout(0xc, st->g17p_lone_pair),
                true, 0, release, &release_len) ||
             !agx3_append(emission, release, release_len))
            AGX3_REFUSE_RET(I);
      }
      if (st->g17g_deferred_write && I->op == AGX_OPCODE_ST_TILE && last) {
         uint8_t release[AGX3_PIXEL_MAX_LENGTH], release_len = 0;
         if (!agx3_pixel_release(enc, 12, true, 0, release, &release_len) ||
             !agx3_append(emission, release, release_len))
            AGX3_REFUSE_RET(I);
      }
      if (st->g18_sample_colour_stores && I->op == AGX_OPCODE_ST_TILE && last) {
         uint8_t release[AGX3_PIXEL_MAX_LENGTH], release_len = 0;
         if (!st->g17p_colour_open ||
             !agx3_pixel_release(enc, 12, true, 0, release, &release_len) ||
             !agx3_append(emission, release, release_len))
            AGX3_REFUSE_RET(I);
      }
      return AGX3_PACK_EMITTED;
   }

   /*
    * Preamble uniform write. Apple's own constant program writes its
    * uniforms with an ordinary `copy` whose f9 is 3 -- the destination
    * moves from f2 into f3 and becomes a uniform half. Witnessed
    * byte-exactly against a write to uniform half 24; see
    * agx3_copy_core.h.
    *
    * Each emitted copy is one grounded 32-bit component. A scalar
    * 64-bit value is therefore two consecutive witnessed copies, just
    * like a two-component 32-bit value; no unobserved wide form is used.
    */
   if (I->op == AGX_OPCODE_UNIFORM_STORE && I->nr_srcs == 2 &&
       (I->mask == 1 || I->mask == 3 || I->mask == 7 ||
        I->mask == 15)) {
      const agx_index value = I->src[0], index = I->src[1];
      const unsigned components = util_bitcount(I->mask);
      const bool scalar16 = value.size == AGX_SIZE_16 && I->mask == 1 &&
                            agx_channels(value) == 1;
      const bool scalar64 = value.size == AGX_SIZE_64 && I->mask == 1 &&
                            agx_channels(value) == 1 &&
                            (value.value & 3) == 0;
      const bool vector32 = value.size == AGX_SIZE_32 &&
                            agx_channels(value) == components;
      const unsigned copies = scalar64 ? 2 : components;
      /* Apple widens a scalar half preamble result to one aligned
       * 32-bit copy. Permit that exact substitution only when both
       * files are pair-aligned and the extra uniform half is padding;
       * otherwise a neighbouring scalar result could be overwritten. */
      const bool padded16 = scalar16 && !(value.value & 1) &&
                            !(index.value & 1) && index.value + 2 <= 512 &&
                            !st->uniform_half_used[index.value + 1];

      if (agx3_src_is_register(value) && !value.memory && !value.abs &&
          !value.neg && (padded16 || scalar64 || vector32) &&
          index.type == AGX_INDEX_IMMEDIATE && !index.abs &&
          !index.neg && index.value + 2 * copies <= 512) {
         /*
          * Apple's witnessed uniform write carries no wait bits, and
          * the wait bank for this form is not witnessed, so retire any
          * scoreboard dependency with the band no-op first rather than
          * guessing a field. Both halves are witnessed encodings.
          */
         /*
          * One copy per 32-bit component, exactly as Apple's constant
          * program does: its four uniform writes land on halves 20, 22,
          * 24 and 26 -- consecutive 32-bit slots two halves apart.
          */
         bool ok = true;
         if (st->wait) {
            /* A different load may still be producing r0. Carry a
             * partial wait on the value being published, whose producer
             * is covered by the mask, on both AGX3 encodings. */
            uint8_t carrier[AGX3_ALU_MAX_LENGTH], carrier_len = 0;
            const bool retired = scalar16
               ? agx3_alu_bit_imm(enc, AGX3_ALU_BOR, true, value.value,
                                  value.value, 0, AGX3_ALU_HINT_NONE, st->wait,
                                  carrier, &carrier_len)
               : agx3_alu_band_identity_wait(enc, value.value, st->wait,
                                             carrier, &carrier_len);
            ok = retired && agx3_append(emission, carrier, carrier_len);
         }

         for (unsigned i = 0; ok && i < copies; ++i) {
            uint8_t bytes[AGX3_COPY_MAX_LENGTH];
            uint8_t len = 0;
            const bool last = i + 1 == copies;
            /* A half-register discard does not prove that its
             * paired upper half is dead. */
            const bool kill = !padded16 && last && value.kill;
            ok = (agx_is_g18g(&ctx->key->dev)
                     ? agx3_copy_encode_reg_to_uniform_g18g(
                          enc, index.value + 2 * i,
                          value.value + 2 * i, kill, 0, bytes, &len)
                     : agx3_copy_encode_reg_to_uniform(
                          enc, index.value + 2 * i,
                          value.value + 2 * i, kill, 0, bytes, &len)) &&
                 agx3_append(emission, bytes, len);
         }

         if (ok) {
            st->outstanding &= (uint16_t)~st->wait;
            return AGX3_PACK_EMITTED;
         }
      }
      AGX3_REFUSE_RET(I);
   }

   if (I->op == AGX_OPCODE_GET_COVERAGE_1X_AGX3) {
      /* SR2 is the one-sample coverage word. A multisample depth
       * export writes each sample's depth separately in Apple's
       * fragments, which this single update does not do. */
      if ((enc != AGX_ISA_ENCODING_AGX3B &&
           enc != AGX_ISA_ENCODING_AGX3C) ||
          ctx->stage != MESA_SHADER_FRAGMENT ||
          ctx->key->fs.nr_samples > 1 ||
          I->nr_dests != 1 || I->nr_srcs ||
          !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
          I->dest[0].size != AGX_SIZE_16 || agx_channels(I->dest[0]) != 1)
         AGX3_REFUSE_RET(I);
      const struct agx3_sr_desc d = {
         .dest = I->dest[0].value, .width = AGX3_SR_W16,
         .sr = 2, .slot = st->slot, .wait_mask = st->wait,
      };
      uint8_t bytes[AGX3_SR_MAX_LENGTH], len;
      if (!agx3_sr_encode(enc, &d, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      st->fragment_input_started = true;
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_ZS_EMIT_AGX3) {
      const bool depth = I->zs & 1, stencil = I->zs & 2;
      const bool g18_msaa4 = agx3_g18_msaa4_tile_shader(ctx);
      /* One word of a covered-sample loop, its class-1 wait already
       * emitted before the loop. */
      const bool sample_loop = I->zs & 4;
      /* Apple's 2x/4x per-sample depth/stencil writes stage each word,
       * and its depth, after the previous sample's, all under the one
       * class-1 wait taken before the first DEPTHF (msaa/native/zexp2
       * fs_id_{2,4}x, msaa/native/zmrt/zsten fs_id_s_4x). */
      const unsigned staging =
         ctx->key->dev.emit_agx3d && !g18_msaa4 ? 0 : st->g17p_zs_staging;
      if ((enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C) ||
          ctx->stage != MESA_SHADER_FRAGMENT || I->nr_dests ||
          I->nr_srcs != 1 ||
          (ctx->key->dev.emit_agx3d
              /* G18G keeps j873's G17G-family operand: one 16-bit
               * register pair per staged word. */
              ? (I->zs < 1 || I->zs > 3 ||
                 !agx3_src_is_register(I->src[0]) || I->src[0].memory ||
                 I->src[0].size != AGX_SIZE_16 ||
                 agx_channels(I->src[0]) != (depth ? 4 : 2) ||
                 (I->src[0].value & 1) ||
                 (g18_msaa4
                    ? (!depth || stencil || st->g17p_zs_writes >= 4 ||
                       (st->g17p_zs_writes == 0) !=
                          (st->g17p_deferred_zs_wait != NULL))
                    : (!st->g17p_deferred_zs_wait ||
                       ctx->key->fs.nr_samples > 1)))
              : (!(I->zs & 3) || I->zs > 7 ||
                 (sample_loop && ((I->zs & 3) == 3 || !st->zs_sample_loop ||
                                  enc != AGX_ISA_ENCODING_AGX3C ||
                                  st->g17p_deferred_zs_wait)) ||
                 !agx3_vary_reg32(I->src[0]) ||
                 agx_channels(I->src[0]) != (depth ? 2 : 1) ||
                 (!sample_loop &&
                  (st->g17p_zs_writes == 0) != (st->g17p_deferred_zs_wait != NULL)) ||
                 st->g17p_zs_writes >=
                    (enc == AGX_ISA_ENCODING_AGX3C ? 8 : 4))))
         AGX3_REFUSE_RET(I);

      uint8_t bytes[AGX3_STAGE_MAX_LENGTH];
      uint8_t len = 0;
      if (!st->g17p_colour_open) {
         if (!agx3_pixel_colour_entry(enc, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         ctx->agx3_colour_link.entry_emitted = true;
         ctx->agx3_colour_link.entry_offset =
            emission->size - len - st->initial_size;
         st->g17p_colour_open = true;
      }
      if (ctx->key->dev.emit_agx3d) {
         /* Apple's G17G copies carry no liveness flag: 0b 00 29 04 is
          * dest 0, source r0, no wait; the depth copy to staging 2 is
          * the coordinate copy already witnessed for fetches. */
         /* Native315 uses distinct staging pairs for its four G18
          * depth exports, still without a copy liveness flag. */
         if (!agx3_stage_copy_kill(enc, staging, I->src[0].value, false, st->slot,
                                   st->wait | st->outstanding_here, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         if (depth &&
             (!agx3_stage_copy_kill(enc, staging + 2, I->src[0].value + 2, false,
                                    st->slot, 0, bytes, &len) ||
              !agx3_append(emission, bytes, len)))
            AGX3_REFUSE_RET(I);
      } else {
      /* Native DEPTHF producers publish both staging dwords with f13.
       * Keep that flag independent of the allocator's GPR last-use.
       * Apple's stencil-only [[stencil]] fragment publishes its single
       * coverage/stencil dword without it. */
      if (!agx3_stage_copy_liveness(enc, staging, I->src[0].value,
                                  depth && enc == AGX_ISA_ENCODING_AGX3B, I->src[0].discard, st->slot,
                                  st->wait | st->outstanding_here, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      if (depth &&
          (!agx3_stage_copy_liveness(enc, staging + 2,
                                     I->src[0].value + 2, enc == AGX_ISA_ENCODING_AGX3B,
                                     I->src[0].discard, st->slot, 0,
                                     bytes, &len) ||
           !agx3_append(emission, bytes, len)))
         AGX3_REFUSE_RET(I);
      }
      if (st->g17p_deferred_zs_wait && !st->interleaved_zs &&
          !agx3_emit_pixel(emission, enc, st->g17p_deferred_zs_wait, 0, false,
                           0, &ctx->agx3_colour_link, st->initial_size))
         AGX3_REFUSE_RET(I);
      st->g17p_deferred_zs_wait = NULL;

      /* The loop restages the word each iteration; native keeps f13
       * clear on that DEPTHF. */
      const struct agx3_zs_desc zs = {
         .value = staging,
         .value_kill = !sample_loop,
         .slot = st->slot,
         .wait_mask = (uint16_t)(1u << (st->slot - 1)),
      };
      if (!agx3_zs_encode_depth_stencil(enc, &zs, depth, stencil, bytes,
                                        &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);

      st->outstanding = 0;
      st->g17p_zs_writes++;
      st->g17p_zs_staging += depth ? 4 : 2;
      ctx->agx3_staging_halfs =
         MAX2(ctx->agx3_staging_halfs, staging + (depth ? 4 : 2));
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_GET_SR_COVERAGE) {
      /* Coverage pseudo-registers, all read as W16: SR2 is the
       * active-lane source of the fragment helper-thread predicate,
       * SR34 the active thread index in the subgroup (f48 stays clear:
       * helpers may need it), SR36 the number of active threads. Others
       * are refused. */
      const bool active_thread =
         I->sr == AGX_SR_IS_ACTIVE_THREAD &&
         ctx->stage == MESA_SHADER_FRAGMENT;
      const bool sample_coverage =
         (I->sr == AGX_SR_INPUT_SAMPLE_MASK ||
          (enc == AGX_ISA_ENCODING_AGX3C &&
           I->sr == AGX_SR_COVERAGE_MASK)) &&
         ctx->stage == MESA_SHADER_FRAGMENT;
      /* A G17P tessellation-control program is a compute dispatch
       * (hk_launch_tess(); hk_shader.c gives it the compute
       * texture/sampler heaps), so SR34/SR36 read the same subgroup
       * state as in compute. So is the main geometry kernel
       * (hk_launch_gs_prerast()), which runs one GS invocation per
       * thread. M5 object (task) and mesh programs read SR34 the same
       * way: a native G17G first-thread election in an object
       * shader reads SR34 into a half register. A G17P mesh program is
       * a VDM-launched threadgroup whose subgroups Apple's own mesh
       * code reduces across (sRedumax in its write_uvb helper). */
      const bool subgroup_stage =
         ctx->stage == MESA_SHADER_COMPUTE ||
         ctx->stage == MESA_SHADER_VERTEX ||
         ctx->stage == MESA_SHADER_TESS_CTRL ||
         ctx->stage == MESA_SHADER_GEOMETRY ||
         ctx->stage == MESA_SHADER_FRAGMENT ||
         ctx->stage == MESA_SHADER_MESH ||
         (enc == AGX_ISA_ENCODING_AGX3C &&
          ctx->stage == MESA_SHADER_TASK);
      const bool active_index =
         I->sr == AGX_SR_ACTIVE_THREAD_INDEX_IN_SUBGROUP &&
         subgroup_stage;
      const bool active_count =
         I->sr == AGX_SR_TOTAL_ACTIVE_THREADS_IN_SUBGROUP &&
         subgroup_stage;
      /* msaa163: Apple's G18 4x fragment reads the covered-sample word
       * the same way, rld 210a42000000 = W16 SR39 into a half, and
       * passes it straight to lstcol. Of the agx3c-encoded compiles,
       * only a G18G two- or four-sample fragment reads it. */
      const bool g18_msaa4_coverage =
         enc == AGX_ISA_ENCODING_AGX3C && sample_coverage &&
         agx3_g18_msaa4_tile_shader(ctx);
      /* Native ballot-read643 compute, vertex and fragment programs
       * read SR34 as W16 on both G17G and G18G. The C encoder and
       * G18 field-vector translation reproduce all twelve reads;
       * keep SR36 and the other coverage gates unchanged. */
      if ((enc != AGX_ISA_ENCODING_AGX3B &&
           !(enc == AGX_ISA_ENCODING_AGX3C &&
             (active_index || active_thread ||
              (ctx->key->dev.emit_agx3d ? g18_msaa4_coverage
                                        : sample_coverage)))) ||
          !(active_thread || sample_coverage || active_index || active_count) ||
          I->nr_dests != 1 ||
          I->nr_srcs != 0 || !agx3_src_is_register(I->dest[0]) ||
          I->dest[0].memory || I->dest[0].size != AGX_SIZE_16 ||
          agx_channels(I->dest[0]) != 1)
         AGX3_REFUSE_RET(I);
      const struct agx3_sr_desc d = {
         .dest = I->dest[0].value, .width = AGX3_SR_W16,
         .sr = sample_coverage ? 39 : active_index ? 34 : active_count ? 36 : 2,
         .slot = st->slot,
         .wait_mask = st->wait,
         .first = enc == AGX_ISA_ENCODING_AGX3C && sample_coverage &&
            !ctx->key->fs.ignore_tib_dependencies &&
            emission->size == st->initial_size && !st->fragment_input_started &&
            !st->g17p_fragment_helper_protocol,
      };
      uint8_t bytes[AGX3_SR_MAX_LENGTH], len = 0;
      if (!agx3_sr_encode(enc, &d, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      st->fragment_input_started |= enc == AGX_ISA_ENCODING_AGX3C &&
         sample_coverage && !ctx->key->fs.ignore_tib_dependencies;
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_ZS_EMIT) {
      /* Fragment stores without early tests inject a pass-through depth
       * write so side effects obey late depth testing. Apple's fresh
       * G17P fragment-atomic pipeline stages that scalar in staging[0]
       * on slot1, then emits pixwait(1), depthf(depth, wait slot1), and
       * the ordinary pixrel. Admit precisely that depth-only graph. */
      const agx_index samples = I->src[0], depth = I->src[1];
      if (enc != AGX_ISA_ENCODING_AGX3B ||
          ctx->stage != MESA_SHADER_FRAGMENT || I->nr_dests != 0 ||
          I->nr_srcs != 2 || I->zs != 1 ||
          samples.type != AGX_INDEX_IMMEDIATE ||
          samples.size != AGX_SIZE_16 || samples.value != 0xff ||
          !agx3_vary_reg32(depth) || agx_channels(depth) != 1 ||
          !depth.discard || depth.kill || depth.cache ||
          !st->g17p_deferred_zs_wait)
         AGX3_REFUSE_RET(I);

      uint8_t bytes[AGX3_STAGE_MAX_LENGTH], len = 0;
      if (!st->g17p_colour_open) {
         if (!agx3_pixel_colour_entry(enc, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         ctx->agx3_colour_link.entry_emitted = true;
         ctx->agx3_colour_link.entry_offset =
            emission->size - len - st->initial_size;
         st->g17p_colour_open = true;
      }
      if (!agx3_stage_copy_kill(enc, 0, depth.value, true, 1, st->wait,
                                bytes, &len) ||
          !agx3_append(emission, bytes, len) ||
          !agx3_emit_pixel(emission, enc, st->g17p_deferred_zs_wait, 0,
                           false, 0, &ctx->agx3_colour_link, st->initial_size))
         AGX3_REFUSE_RET(I);
      const struct agx3_zs_desc d = {
         .value = 0, .value_kill = true, .slot = 1, .wait_mask = 1,
      };
      if (!agx3_zs_encode_depth_stencil(enc, &d, true, false, bytes,
                                         &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      st->outstanding &= (uint16_t)~st->wait;
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 2);
      st->g17p_deferred_zs_wait = NULL;
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_SAMPLE_MASK_AGX3) {
      /* G17G spells the same kill protocol; Apple's applegpu_g17g
       * discard fragment (test/native/g18g
       * oracle.discard.fragment.g17g.bin) stages the coverage word
       * before its colour-entry wait, then waits on class 1, kills
       * with MOVMSK and releases class 1 -- each instruction the
       * encoders below produce for AGX3c. G18G keeps this order;
       * the other agx3c GPUs use agx3c's. */
      const bool g18g_order = agx_is_g18g(&ctx->key->dev);
      /* G18's complete coverage-word path may target sample zero
       * with an immediate 1. MOVMSK only consumes the second source;
       * the logical target is already incorporated in that word.
       */
      const bool g18_full_target = ctx->key->dev.emit_agx3d &&
         I->nr_srcs == 2 && I->src[0].type == AGX_INDEX_IMMEDIATE &&
         I->src[0].size == AGX_SIZE_16 && I->src[0].value == 1;
      if ((enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C) ||
          ctx->stage != MESA_SHADER_FRAGMENT ||
          I->nr_dests || I->nr_srcs != 2 ||
          !(agx3_src_is_register(I->src[0]) ||
            (ctx->key->dev.emit_agx3d
                ? g18_full_target
                : (I->src[0].type == AGX_INDEX_IMMEDIATE &&
                   (I->src[0].value == 1 ||
                    (enc == AGX_ISA_ENCODING_AGX3C &&
                     I->src[0].value <= 255))))) || I->src[0].memory ||
          I->src[0].size != AGX_SIZE_16 || agx_channels(I->src[0]) != 1 ||
          !agx3_vary_reg32(I->src[1]) || agx_channels(I->src[1]) != 1)
         AGX3_REFUSE_RET(I);
      const bool force_early = I->imm & 1;
      /* Native1121 psdiscard (run 1128): the per-sample update word is
       * staged at staging 2 under scoreboard slot 2 and consumed by
       * movmsk 160325200000, inside the class-1 hold the shader took
       * at its first update; it carries no class transaction. */
      const bool second = I->imm & AGX_SAMPLE_MASK_AGX3_SECOND_STAGING;
      const unsigned stage_dest = second ? 2 : 0;
      const uint8_t stage_slot = second ? 2 : 1;
      if ((I->imm & ~(3 | AGX_SAMPLE_MASK_AGX3_SECOND_STAGING)) ||
          (force_early && !(I->imm & 2)) ||
          (second && ((I->imm & 3) || !ctx->key->dev.emit_agx3d)))
         AGX3_REFUSE_RET(I);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (g18g_order &&
          (!agx3_stage_copy_source(enc, stage_dest, I->src[1].value,
                                   I->src[1].discard, stage_slot,
                                   st->wait | st->outstanding_here, bytes,
                                   &len) ||
           !agx3_append(emission, bytes, len)))
         AGX3_REFUSE_RET(I);
      if (!st->g17p_colour_open) {
         uint8_t entry[AGX3_PIXEL_MAX_LENGTH], entry_len;
         if (!agx3_pixel_colour_entry(enc, entry, &entry_len) ||
             !agx3_append(emission, entry, entry_len))
            AGX3_REFUSE_RET(I);
         ctx->agx3_colour_link.entry_emitted = true;
         ctx->agx3_colour_link.entry_offset = emission->size - entry_len - st->initial_size;
         st->g17p_colour_open = true;
      }
      if (!g18g_order &&
          (!agx3_stage_copy_source(enc, stage_dest, I->src[1].value,
                                   I->src[1].discard, stage_slot,
                                   st->wait | st->outstanding_here, bytes,
                                   &len) ||
           !agx3_append(emission, bytes, len)))
         AGX3_REFUSE_RET(I);
      /* Native simple and sampled-alpha fragments produce and stage
       * SR2's complete coverage word before acquiring class1. Keep
       * this ordering within the atomic pseudo rather than inheriting
       * the AGX2 wait-before-mask-calculation schedule. */
      if ((I->imm & 2) && !force_early &&
          (!agx3_pixel_wait(enc, 1, 0, bytes, &len) ||
           !agx3_append(emission, bytes, len)))
         AGX3_REFUSE_RET(I);
      const struct agx3_zs_desc zs = {
         .value = stage_dest, .value_kill = true, .slot = stage_slot,
         .wait_mask = (uint16_t)BITFIELD_BIT(stage_slot - 1),
      };
      if (!agx3_zs_encode_sample_mask(enc, &zs, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      /* A no-tests discard does not acquire class 1 above, so it
       * must not release it either. Native G17P kill_fetch and
       * kill_memory emit these early MOVMSKs without PIXREL(1),
       * then bracket the eventual testing update with one wait and
       * release. Keep that release paired with its acquire. */
      if ((I->imm & 2) && !force_early &&
          (!agx3_pixel_release(enc, 1, false, 0, bytes, &len) ||
           !agx3_append(emission, bytes, len)))
         AGX3_REFUSE_RET(I);
      st->outstanding = 0;
      ctx->agx3_staging_halfs =
         MAX2(ctx->agx3_staging_halfs, stage_dest + 2);
      return AGX3_PACK_EMITTED;
   }

   /* Special-register read. */
   return AGX3_PACK_UNHANDLED;
}
