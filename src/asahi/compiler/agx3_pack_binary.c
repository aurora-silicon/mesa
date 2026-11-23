/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 binary emission: top-level driver, slot/resource scheduling,
 * per-category instruction dispatch, and branch/fixup finalization.
 *
 * The entire front end -- NIR lowering, optimization, register allocation and
 * scheduling -- is encoding-independent and reused verbatim from AGX2. Only the
 * final pack stage forks: where agx_pack_binary emits AGX2, this walks the same
 * scheduled agx_instr IR and dispatches each instruction to the modular AGX3
 * packers (agx3_pack_{cf,tex,frag,mem,subgroup,alu}.c).
 *
 * Any unmapped op or out-of-domain instruction fails the shader (never
 * aborts: this runs inside the UMD).
 */

#include "c11/threads.h"
#include "agx3_pack_internal.h"
#include "agx3_pixel_core.h"
#include "agx3_binder.h"
#include "agx3_compose/agx3_decode.h"
#include "agx3_g17g_array_eot.h"
#include "agx3_g17p_array_eot.h"
#include "agx3_g17p_quad_eot.h"
#include "agx3_sr_core.h"
#include "agx3_tex_core.h"

#include <limits.h>

#define AGX3_REFUSAL_SIZE 640
static tss_t agx3_refusal_key;
static once_flag agx3_refusal_once = ONCE_FLAG_INIT;

/*
 * This backend is fail-closed per instruction: a shader containing anything
 * outside the grounded domain is refused, never mis-encoded. A bare refusal is
 * hard to act on, so name the instruction that closed the door under the
 * compiler's existing shader-debug flag (AGX_MESA_DEBUG=shaders). This is the
 * difference between "AGX3 rejected this shader" and a usable bug report.
 */
static const char *
agx3_index_kind(const agx_index x)
{
   switch (x.type) {
   case AGX_INDEX_NULL:      return "null";
   case AGX_INDEX_NORMAL:    return "ssa";
   case AGX_INDEX_IMMEDIATE: return "imm";
   case AGX_INDEX_UNIFORM:   return "uni";
   case AGX_INDEX_REGISTER:  return "reg";
   default:                  return "?";
   }
}

static void
agx3_refusal_key_init(void)
{
   tss_create(&agx3_refusal_key, free);
}

static char *
agx3_refusal_buffer(bool create)
{
   call_once(&agx3_refusal_once, agx3_refusal_key_init);
   char *buf = tss_get(agx3_refusal_key);
   if (!buf && create) {
      buf = calloc(1, AGX3_REFUSAL_SIZE);
      if (buf && tss_set(agx3_refusal_key, buf) != thrd_success) {
         free(buf);
         buf = NULL;
      }
   }
   return buf;
}

const char *
agx3_last_pack_refusal(void)
{
   const char *buf = agx3_refusal_buffer(false);
   return buf ? buf : "";
}

void
agx3_report_refusal_why(const agx_instr *I, const char *file, unsigned line,
                        const char *why)
{
   char shape[512];
   int n = snprintf(shape, sizeof(shape), "d%u:%s%u",
                    I->nr_dests ? 16u << I->dest[0].size : 0u,
                    I->nr_dests ? agx3_index_kind(I->dest[0]) : "none",
                    I->nr_dests ? I->dest[0].value : 0u);

   for (unsigned i = 0; i < I->nr_srcs && n > 0 && n < (int)sizeof(shape); ++i) {
      const agx_index x = I->src[i];
      /* An immediate's `value` is a constant, not a register number, so it
       * gets a '#'. Without it `s0=16:imm32` reads as a 32-bit immediate when
       * it is in fact a 16-bit slot holding the minifloat 32 -- and a width
       * and a kind that print as one token is the same class of ambiguity as
       * a field left out entirely. */
      const bool immediate = x.type == AGX_INDEX_IMMEDIATE;
      /* A source's CHANNEL COUNT gates several texture paths -- a 2D fetch
       * wants two coordinate channels and an array or 3D fetch three, and
       * without it two refusals that differ only in dimensionality print
       * identically. Print it whenever it is not the scalar default. */
      const unsigned chans = agx_channels(x);
      n += snprintf(shape + n, sizeof(shape) - n, " s%u=%u:%s%s%u%s%s%s%s%s", i,
                    16u << x.size, agx3_index_kind(x), immediate ? "#" : "",
                    x.value,
                    x.abs ? ".abs" : "", x.neg ? ".neg" : "",
                    x.kill ? ".kill" : "", x.cache ? ".cache" : "",
                    x.discard ? ".discard" : "");
      if (chans != 1 && n > 0 && n < (int)sizeof(shape))
         n += snprintf(shape + n, sizeof(shape) - n, "x%u", chans);
   }

   /* Keep the operand shape in the first refusal too: an application's
    * failing shader is rarely reproducible, and its log is all there is. */
   char *first = agx3_refusal_buffer(true);
   if (first && !first[0])
      snprintf(first, AGX3_REFUSAL_SIZE,
               "%s at %s:%u%s%s [%s lod=%u dim=%u]",
               agx_opcodes_info[I->op].name, agx3_basename(file), line,
               why ? ", " : "",
               why ? why : "", shape, (unsigned)I->lod_mode, (unsigned)I->dim);

   if (!(agx_compiler_debug & AGX_DBG_SHADERS))
      return;

   /* mask, format, shift and nest decide gates that the operand list cannot
    * show. A control-flow refusal turns entirely on `nest` -- a break out of
    * one enclosing block and a break out of five printed identically without
    * it, which makes the refusal impossible to diagnose. (`cond` is
    * the icond/fcond union; which one it is follows from the opcode.)
    * A vector spill differs from a scalar one only in `mask`; and a shifted
    * integer subtract (`a - (b << N)`, which is how a multiply by a small
    * negative constant is strength-reduced) looks exactly like a plain
    * negated add without `shift`. Omitting them costs a wrong diagnosis. */
   fprintf(stderr,
           "%s:%u: no grounded AGX3 encoding for %s "
           "[%s%s%s cond=%u nest=%u chan=%u mask=0x%x fmt=%u "
           "shift=%u dim=%u lod=%u%s%s%s%s%s]%s%s\n",
           agx3_basename(file), line, agx_opcodes_info[I->op].name, shape,
           I->saturate ? " sat" : "", I->invert_cond ? " inv" : "",
           (unsigned)I->fcond, (unsigned)I->nest,
           I->nr_dests ? agx_channels(I->dest[0]) : 0u,
           (unsigned)I->mask, (unsigned)I->format, (unsigned)I->shift,
           (unsigned)I->dim, (unsigned)I->lod_mode,
           I->offset ? " off" : "", I->shadow ? " shadow" : "",
           I->query_lod ? " qlod" : "",
           I->gather != AGX_GATHER_NONE ? " gather" : "",
           I->coherent ? " coherent" : "",
           why ? " -- " : "", why ? why : "");
}

void
agx3_report_refusal(const agx_instr *I, const char *file, unsigned line)
{
   agx3_report_refusal_why(I, file, line, NULL);
}

/*
 * agx3_emit_texture_sample() screens an operand shape through some thirty
 * gates before it encodes anything, and the caller reports only "no grounded
 * AGX3 encoding for texture_sample". Two shapes that differ in one field --
 * a cube array where the cube encodes, a register descriptor where the
 * immediate one encodes -- then print the same line, so every guess about
 * which field mattered costs a round trip to hardware. Name the gate.
 */
bool
agx3_tex_decline(const agx_instr *I, const char *file, unsigned line)
{
   agx3_report_refusal_why(I, file, line, "texture operand gate");
   return false;
}

/* Instructions that read other lanes of the 2x2 quad, so need its helper
 * invocations alive when they execute. */
static bool
agx3_reads_quad_lanes(const agx_instr *I)
{
   switch (I->op) {
   case AGX_OPCODE_DFDX:
   case AGX_OPCODE_DFDY:
   case AGX_OPCODE_QUAD_SHUFFLE:
   case AGX_OPCODE_QUAD_SHUFFLE_XOR:
   case AGX_OPCODE_QUAD_SHUFFLE_UP:
   case AGX_OPCODE_QUAD_SHUFFLE_DOWN:
   case AGX_OPCODE_QUAD_REDUCE:
   case AGX_OPCODE_QUAD_PREFIX:
   case AGX_OPCODE_ICMP_QUAD_BALLOT:
   case AGX_OPCODE_FCMP_QUAD_BALLOT:
      return true;
   default:
      return false;
   }
}

/* An async producer whose result later consumers must wait on. */
bool
agx3_is_async(const agx_instr *I)
{
   if ((I->op == AGX_OPCODE_ATOMIC || I->op == AGX_OPCODE_LOCAL_ATOMIC) &&
       I->nr_dests == 1 && !agx_is_null(I->dest[0]))
      return true;

   /* The compound depth write publishes its staging word and consumes that
    * scoreboard generation with depthf. It needs an assigned slot even
    * though it has no general-register destination. */
   if (I->op == AGX_OPCODE_ZS_EMIT_AGX3)
      return true;

   return I->op == AGX_OPCODE_DEVICE_LOAD || I->op == AGX_OPCODE_LOCAL_LOAD ||
          I->op == AGX_OPCODE_LDCF || I->op == AGX_OPCODE_GET_SR ||
          I->op == AGX_OPCODE_GET_SR_COVERAGE ||
          I->op == AGX_OPCODE_GET_COVERAGE_1X_AGX3 ||
          I->op == AGX_OPCODE_TEXTURE_LOAD || I->op == AGX_OPCODE_IMAGE_LOAD ||
          I->op == AGX_OPCODE_TEXTURE_SAMPLE || I->op == AGX_OPCODE_LD_TILE ||
          agx_is_texture_publication(I) ||
          I->op == AGX_OPCODE_SCS_LOAD_AGX3 ||
          I->op == AGX_OPCODE_SCS_LOAD_UNIFORM_INDEXED_AGX3 ||
          I->op == AGX_OPCODE_SCS_LOAD_REGISTER_BASE_AGX3 ||
          I->op == AGX_OPCODE_SCS_LOAD_REGISTER_INDEXED_AGX3 ||
          I->op == AGX_OPCODE_LOAD_CPROG_AGX3 ||
          I->op == AGX_OPCODE_STACK_LOAD ||
          I->op == AGX_OPCODE_STACK_STORE;
}

static bool
agx3_slot_instr_reads(const struct agx3_slot_instr *consumer,
                      const struct agx3_slot_instr *producer)
{
   for (unsigned src = 0; src < consumer->nr_srcs; ++src) {
      for (unsigned dst = 0; dst < producer->nr_dsts; ++dst) {
         if (consumer->src[src] == producer->dst[dst])
            return true;
      }
   }

   return false;
}


/* The end-of-tile program Apple's builtin factory emits (68 bytes for 2D,
 * 84 for 2D array): PBE table pointer in uniform halves 0..3, imageblock
 * byte offset in halves 4..5, coordinates = group position * dispatched group
 * dimensions. The preceding pack_imgwir and coordinate select are part of
 * imgwblk's ABI. Formats differ only in the IMGWBLK lane-class selector, a
 * function of the store format: 8-bit integer 0x55, 8-bit normalized or packed
 * 32-bit 0x75, 16-bit normalized 0x95, 16-bit integer or float 0x15, raw
 * 32-bit 0x35 (RGB565/5551/4444 use 16-bit float lanes, RG32 integer raw
 * 32-bit, RGBA16 UNORM normalized). The exact post-RA graph is recognized
 * and replaced atomically, so no partial imgwblk escapes. */
enum agx3_g17p_rt0_eot_kind {
   AGX3_G17P_RT0_EOT_NONE,
   AGX3_G17P_RT0_EOT_SELECTOR_75,
   AGX3_G17P_RT0_EOT_SELECTOR_15,
   AGX3_G17P_RT0_EOT_SELECTOR_35,
   AGX3_G17P_RT0_EOT_SELECTOR_55,
   AGX3_G17P_RT0_EOT_NORM16,
};

/* The IMGWBLK lane class the RT0 end-of-tile key admits for a store of this
 * format. The 2D, 2D-array and multisample builtins share it. */
static enum agx3_g17p_rt0_eot_kind
agx3_g17p_rt0_eot_store_kind(const struct agx_shader_key *key,
                             enum agx_format store_format)
{
   const unsigned words = key->g17p_rt0_eot_words;
   switch (store_format) {
   case AGX_FORMAT_I8:
      return words == 1 ? AGX3_G17P_RT0_EOT_SELECTOR_55
                        : AGX3_G17P_RT0_EOT_NONE;
   case AGX_FORMAT_U16NORM:
   case AGX_FORMAT_S16NORM:
      return words == 1 || words == 2 ? AGX3_G17P_RT0_EOT_NORM16
                                      : AGX3_G17P_RT0_EOT_NONE;
   case AGX_FORMAT_I16:
   case AGX_FORMAT_F16:
      return words == 1 || words == 2 ? AGX3_G17P_RT0_EOT_SELECTOR_15
                                      : AGX3_G17P_RT0_EOT_NONE;
   case AGX_FORMAT_I32:
      if (key->g17p_rt0_eot_i32_four_words &&
          key->g17p_rt0_eot_i32_signed_four_words)
         return AGX3_G17P_RT0_EOT_NONE;
      if (key->g17p_rt0_eot_i32_four_words ||
          key->g17p_rt0_eot_i32_signed_four_words)
         return words == 4 ? AGX3_G17P_RT0_EOT_SELECTOR_35
                           : AGX3_G17P_RT0_EOT_NONE;
      return words == 1 || words == 2 ? AGX3_G17P_RT0_EOT_SELECTOR_35
                                      : AGX3_G17P_RT0_EOT_NONE;
   default:
      return AGX3_G17P_RT0_EOT_SELECTOR_75;
   }
}

static enum agx3_g17p_rt0_eot_kind
agx3_g17p_rt0_eot_kind(const agx_context *ctx, enum agx_isa_encoding enc)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C && ctx->key->m5_rt0_eot;
   /* G18G (agx3d) takes the G17G-family builtin through j873's key bit. */
   const bool g18g_eot = agx_is_g18g(&ctx->key->dev) && ctx->key->g17g_rt0_eot;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !m5 && !g18g_eot) ||
       ctx->stage != MESA_SHADER_COMPUTE || ctx->is_preamble || ctx->key->no_stop)
      return AGX3_G17P_RT0_EOT_NONE;

   unsigned instructions = 0, get_x = 0, get_y = 0, offsets = 0, stores = 0;
   unsigned x_reg = UINT_MAX, y_reg = UINT_MAX, offset_imm_reg = UINT_MAX;
   unsigned coord_reg = UINT_MAX, offset_reg = UINT_MAX;
   enum agx_format store_format = AGX_FORMAT_I8;

   agx_foreach_block(ctx, block) {
      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_STOP)
            continue;
         instructions++;

         if (I->op == AGX_OPCODE_GET_SR) {
            if (I->nr_dests != 1 || I->nr_srcs != 0 ||
                !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
                I->dest[0].size != AGX_SIZE_32 ||
                agx_channels(I->dest[0]) != 1)
               return false;

            if (I->sr == AGX_SR_THREAD_POSITION_IN_GRID_X) {
               get_x++;
               x_reg = I->dest[0].value;
            } else if (I->sr == AGX_SR_THREAD_POSITION_IN_GRID_Y) {
               get_y++;
               y_reg = I->dest[0].value;
            } else {
               return false;
            }
         } else if (I->op == AGX_OPCODE_MOV_IMM) {
            /* The imageblock offset reaches the program through its offset
             * uniform, so a target need not start at byte zero, and may
             * start past byte 64: Apple's builtin storing any one target of
             * eight RGBA32Float targets is this single-target program. */
            if (I->nr_dests != 1 || I->nr_srcs != 0 ||
                I->imm >= AGX_G17P_MAX_BYTES_PER_SAMPLE ||
                !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
                I->dest[0].size != AGX_SIZE_32 ||
                agx_channels(I->dest[0]) != 1)
               return false;
            offsets++;
            offset_imm_reg = I->dest[0].value;
         } else if (I->op == AGX_OPCODE_BLOCK_IMAGE_STORE) {
            if (I->nr_dests != 0 || I->nr_srcs != 4 ||
                (I->format != AGX_FORMAT_U8NORM &&
                 I->format != AGX_FORMAT_S8NORM &&
                 I->format != AGX_FORMAT_S16NORM &&
                 I->format != AGX_FORMAT_I8 &&
                 I->format != AGX_FORMAT_U16NORM &&
                 I->format != AGX_FORMAT_I16 &&
                 I->format != AGX_FORMAT_F16 &&
                 I->format != AGX_FORMAT_RGB10A2 &&
                 I->format != AGX_FORMAT_RG11B10F &&
                 I->format != AGX_FORMAT_I32) || I->dim != AGX_DIM_2D ||
                !I->explicit_coords ||
                I->src[0].type != AGX_INDEX_IMMEDIATE ||
                I->src[0].value != 0 || I->src[0].size != AGX_SIZE_16 ||
                I->src[1].type != AGX_INDEX_IMMEDIATE ||
                I->src[1].value != 0 || I->src[1].size != AGX_SIZE_16 ||
                !agx3_src_is_register(I->src[2]) || I->src[2].memory ||
                I->src[2].abs || I->src[2].neg ||
                I->src[2].size != AGX_SIZE_32 ||
                agx_channels(I->src[2]) != 1 ||
                !agx3_src_is_register(I->src[3]) || I->src[3].memory ||
                I->src[3].abs || I->src[3].neg ||
                I->src[3].size != AGX_SIZE_32 ||
                agx_channels(I->src[3]) != 2)
               return false;
            if ((I->format == AGX_FORMAT_I32) != ctx->key->g17p_rt0_eot_i32)
               return false;
            stores++;
            store_format = I->format;
            offset_reg = I->src[2].value;
            coord_reg = I->src[3].value;
         } else {
            return false;
         }
      }
   }

   if (instructions != 4 || get_x != 1 || get_y != 1 || offsets != 1 ||
       stores != 1 || x_reg != coord_reg || y_reg != coord_reg + 2 ||
       offset_imm_reg != offset_reg)
      return AGX3_G17P_RT0_EOT_NONE;

   /* G18G: the packed float MRT store is G17P-only. G18 signed RGBA32 uses
    * the same 2D store as unsigned/float RGBA32 (the old distinct signed
    * witness was the array program); G18 still passes the complete 2D
    * matcher. */
   if (g18g_eot && ctx->key->g17p_r11g11b10_mrt_eot)
      return AGX3_G17P_RT0_EOT_NONE;

   if (ctx->key->g17p_rt0_eot_i32_signed_four_words &&
       (store_format != AGX_FORMAT_I32 ||
        ctx->key->g17p_rt0_eot_words != 4))
      return AGX3_G17P_RT0_EOT_NONE;

   /* G18's native 2D/2DMS builtins select the imageblock class from its
    * physical store type. The G17G exceptions below select different types
    * for RGB565, RGBA16_UNORM and RG32 integer pairs on G18, corrupting
    * already-correct tile values during the store. */
   if (g18g_eot)
      return agx3_g17p_rt0_eot_store_kind(ctx->key, store_format);

   if (m5 && store_format != AGX_FORMAT_U8NORM &&
       store_format != AGX_FORMAT_I8 && store_format != AGX_FORMAT_U16NORM &&
       store_format != AGX_FORMAT_S16NORM && store_format != AGX_FORMAT_F16 &&
       store_format != AGX_FORMAT_S8NORM && store_format != AGX_FORMAT_I16 &&
       store_format != AGX_FORMAT_I32 && store_format != AGX_FORMAT_RGB10A2 &&
       store_format != AGX_FORMAT_RG11B10F)
      return AGX3_G17P_RT0_EOT_NONE;

   return agx3_g17p_rt0_eot_store_kind(ctx->key, store_format);
}

/* Apple's layered EOT computes the tile origin from threadgroup position
 * and dispatched group size before publishing the layer and XY image wires.
 * Match the complete allocated Mesa EOT graph before selecting the G17P
 * builtin or the G17G array wire sequence with its uniform PBE/offset ABI;
 * no general array block-store shape is implied. */
static enum agx3_g17p_rt0_eot_kind
agx3_g17p_layered_eot_kind(const agx_context *ctx, enum agx_isa_encoding enc)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C && ctx->key->m5_rt0_eot;
   const bool g18g_eot = agx_is_g18g(&ctx->key->dev) && ctx->key->g17g_rt0_eot;
   if ((!m5 && !g18g_eot && enc != AGX_ISA_ENCODING_AGX3B) ||
       ctx->stage != MESA_SHADER_COMPUTE || ctx->is_preamble || ctx->key->no_stop)
      return AGX3_G17P_RT0_EOT_NONE;

   enum agx_format store_format = AGX_FORMAT_U8NORM;

   unsigned instruction = 0;
   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_STOP)
         continue;

      switch (instruction++) {
      case 0:
      case 1:
      case 2: {
         const enum agx_sr expected[] = {
            AGX_SR_THREAD_POSITION_IN_GRID_X,
            AGX_SR_THREAD_POSITION_IN_GRID_Y,
            AGX_SR_THREADGROUP_POSITION_IN_GRID_Z,
         };
         if (I->op != AGX_OPCODE_GET_SR || I->nr_dests != 1 ||
             I->nr_srcs != 0 || I->sr != expected[instruction - 1] ||
             I->dest[0].type != AGX_INDEX_REGISTER ||
             I->dest[0].size != AGX_SIZE_32 || I->dest[0].memory ||
             agx_channels(I->dest[0]) != 1 ||
             I->dest[0].value != 2 * (instruction - 1))
            return AGX3_G17P_RT0_EOT_NONE;
         break;
      }
      case 3:
         if (I->op != AGX_OPCODE_MOV_IMM || I->nr_dests != 1 ||
             I->nr_srcs != 0 || I->imm != 0 ||
             I->dest[0].type != AGX_INDEX_REGISTER ||
             I->dest[0].size != AGX_SIZE_32 || I->dest[0].memory ||
             agx_channels(I->dest[0]) != 1 || I->dest[0].value != 6)
            return AGX3_G17P_RT0_EOT_NONE;
         break;
      case 4:
      case 5:
      case 6: {
         const unsigned lane = instruction - 5;
         if (I->op != AGX_OPCODE_BITOP ||
             (I->truth_table & 0xfu) != AGX3_SEL_TT_MOV || I->saturate ||
             I->nr_dests != 1 || I->nr_srcs != 2 ||
             I->dest[0].type != AGX_INDEX_REGISTER ||
             I->dest[0].size != AGX_SIZE_32 || I->dest[0].memory ||
             agx_channels(I->dest[0]) != 1 ||
             I->dest[0].value != 8 + 2 * lane ||
             I->src[0].type != AGX_INDEX_REGISTER ||
             I->src[0].size != AGX_SIZE_32 || I->src[0].memory ||
             I->src[0].abs || I->src[0].neg ||
             agx_channels(I->src[0]) != 1 ||
             I->src[0].value != 2 * lane ||
             I->src[1].type != AGX_INDEX_IMMEDIATE ||
             I->src[1].value != 0)
            return AGX3_G17P_RT0_EOT_NONE;
         break;
      }
      case 7:
         if (I->op != AGX_OPCODE_BLOCK_IMAGE_STORE || I->nr_dests != 0 ||
             I->nr_srcs != 4 ||
             (I->format == AGX_FORMAT_I32) != ctx->key->g17p_rt0_eot_i32 ||
             I->dim != AGX_DIM_2D_ARRAY || !I->explicit_coords ||
             I->src[0].type != AGX_INDEX_IMMEDIATE ||
             I->src[0].value != 0 || I->src[0].size != AGX_SIZE_16 ||
             I->src[1].type != AGX_INDEX_IMMEDIATE ||
             I->src[1].value != 0 || I->src[1].size != AGX_SIZE_16 ||
             I->src[2].type != AGX_INDEX_REGISTER ||
             I->src[2].size != AGX_SIZE_32 || I->src[2].memory ||
             !I->src[2].discard || I->src[2].abs || I->src[2].neg ||
             agx_channels(I->src[2]) != 1 || I->src[2].value != 6 ||
             I->src[3].type != AGX_INDEX_REGISTER ||
             I->src[3].size != AGX_SIZE_32 || I->src[3].memory ||
             !I->src[3].discard || I->src[3].abs || I->src[3].neg ||
             agx_channels(I->src[3]) != 3 || I->src[3].value != 8)
            return AGX3_G17P_RT0_EOT_NONE;
         store_format = I->format;
         break;
      default:
         return AGX3_G17P_RT0_EOT_NONE;
      }
   }

   if (instruction != 8 ||
       (m5 && store_format != AGX_FORMAT_U8NORM &&
        store_format != AGX_FORMAT_I8 && store_format != AGX_FORMAT_U16NORM &&
        store_format != AGX_FORMAT_S16NORM && store_format != AGX_FORMAT_F16 &&
        store_format != AGX_FORMAT_S8NORM && store_format != AGX_FORMAT_I16 &&
        store_format != AGX_FORMAT_I32 && store_format != AGX_FORMAT_RGB10A2 &&
        store_format != AGX_FORMAT_RG11B10F))
      return AGX3_G17P_RT0_EOT_NONE;

   return agx3_g17p_rt0_eot_store_kind(ctx->key, store_format);
}

/* Native EOT key {RGBA16F RT0 at partition 0, R8 RT1 at a nonzero partition}
 * (mask 3) has a distinct, complete 238-byte program.  As above, match the
 * whole post-RA graph before replacing it: this is the exact mixed-width
 * layout used by Quake's WBOIT pass. */
static bool
agx3_is_g17p_wboit_mrt_eot(const agx_context *ctx,
                           enum agx_isa_encoding enc)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || ctx->stage != MESA_SHADER_COMPUTE ||
       ctx->is_preamble || ctx->key->no_stop)
      return false;

   unsigned instructions = 0, get_x = 0, get_y = 0, stores = 0;
   unsigned x_reg = UINT_MAX, y_reg = UINT_MAX, coord_reg = UINT_MAX;
   unsigned zero_reg = UINT_MAX, eight_reg = UINT_MAX;

   agx_foreach_block(ctx, block) {
      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_STOP)
            continue;
         instructions++;

         if (I->op == AGX_OPCODE_GET_SR) {
            if (I->nr_dests != 1 || I->nr_srcs != 0 ||
                !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
                I->dest[0].size != AGX_SIZE_32 ||
                agx_channels(I->dest[0]) != 1)
               return false;

            if (I->sr == AGX_SR_THREAD_POSITION_IN_GRID_X) {
               get_x++;
               x_reg = I->dest[0].value;
            } else if (I->sr == AGX_SR_THREAD_POSITION_IN_GRID_Y) {
               get_y++;
               y_reg = I->dest[0].value;
            } else {
               return false;
            }
         } else if (I->op == AGX_OPCODE_MOV_IMM) {
            if (I->nr_dests != 1 || I->nr_srcs != 0 ||
                !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
                I->dest[0].size != AGX_SIZE_32 ||
                agx_channels(I->dest[0]) != 1)
               return false;

            if (I->imm == 0)
               zero_reg = I->dest[0].value;
            else if (I->imm == 8)
               eight_reg = I->dest[0].value;
            else
               return false;
         } else if (I->op == AGX_OPCODE_BLOCK_IMAGE_STORE) {
            if (I->nr_dests != 0 || I->nr_srcs != 4 ||
                I->dim != AGX_DIM_2D || !I->explicit_coords ||
                I->src[0].type != AGX_INDEX_IMMEDIATE ||
                I->src[0].value != 0 || I->src[0].size != AGX_SIZE_16 ||
                I->src[1].type != AGX_INDEX_IMMEDIATE ||
                I->src[1].value != stores || I->src[1].size != AGX_SIZE_16 ||
                !agx3_src_is_register(I->src[2]) || I->src[2].memory ||
                I->src[2].abs || I->src[2].neg ||
                I->src[2].size != AGX_SIZE_32 ||
                agx_channels(I->src[2]) != 1 ||
                !agx3_src_is_register(I->src[3]) || I->src[3].memory ||
                I->src[3].abs || I->src[3].neg ||
                I->src[3].size != AGX_SIZE_32 ||
                agx_channels(I->src[3]) != 2)
               return false;

            if ((stores == 0 &&
                 (I->format != AGX_FORMAT_I16 ||
                  I->src[2].value != zero_reg)) ||
                (stores == 1 &&
                 (I->format != AGX_FORMAT_U8NORM ||
                  I->src[2].value != eight_reg)) ||
                stores >= 2)
               return false;

            if (stores == 0)
               coord_reg = I->src[3].value;
            else if (I->src[3].value != coord_reg)
               return false;
            stores++;
         } else {
            return false;
         }
      }
   }

   return instructions == 6 && get_x == 1 && get_y == 1 && stores == 2 &&
          zero_reg != UINT_MAX && eight_reg != UINT_MAX &&
          x_reg == coord_reg && y_reg == coord_reg + 2;
}

/* Fresh applegpu_g17p builtin EOT with two RGBA8 attachments (mask 3) is a
 * distinct 120-byte program. Its two attachment entries use identical
 * format3/layout2/components4 keys; their vector positions select RT0/RT1,
 * while the imageblock byte offsets are 0/4. Match the complete allocated
 * graph before substituting it so no other MRT layout inherits these target
 * selectors or offsets. */
static bool
agx3_is_g17p_dual_word_mrt_eot(const agx_context *ctx,
                               enum agx_isa_encoding enc,
                               enum agx_format format)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || ctx->stage != MESA_SHADER_COMPUTE ||
       ctx->is_preamble || ctx->key->no_stop ||
       (format == AGX_FORMAT_I32 && !ctx->key->g17p_r11g11b10_mrt_eot) ||
       (format != AGX_FORMAT_I32 && format != AGX_FORMAT_U8NORM))
      return false;

   unsigned instructions = 0, get_x = 0, get_y = 0, stores = 0;
   unsigned x_reg = UINT_MAX, y_reg = UINT_MAX, coord_reg = UINT_MAX;
   unsigned zero_reg = UINT_MAX, four_reg = UINT_MAX;

   agx_foreach_block(ctx, block) {
      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_STOP)
            continue;
         instructions++;

         if (I->op == AGX_OPCODE_GET_SR) {
            if (I->nr_dests != 1 || I->nr_srcs != 0 ||
                !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
                I->dest[0].size != AGX_SIZE_32 ||
                agx_channels(I->dest[0]) != 1)
               return false;

            if (I->sr == AGX_SR_THREAD_POSITION_IN_GRID_X) {
               get_x++;
               x_reg = I->dest[0].value;
            } else if (I->sr == AGX_SR_THREAD_POSITION_IN_GRID_Y) {
               get_y++;
               y_reg = I->dest[0].value;
            } else {
               return false;
            }
         } else if (I->op == AGX_OPCODE_MOV_IMM) {
            if (I->nr_dests != 1 || I->nr_srcs != 0 ||
                !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
                I->dest[0].size != AGX_SIZE_32 ||
                agx_channels(I->dest[0]) != 1)
               return false;

            if (I->imm == 0)
               zero_reg = I->dest[0].value;
            else if (I->imm == 4)
               four_reg = I->dest[0].value;
            else
               return false;
         } else if (I->op == AGX_OPCODE_BLOCK_IMAGE_STORE) {
            if (I->nr_dests != 0 || I->nr_srcs != 4 ||
                I->format != format || I->dim != AGX_DIM_2D ||
                !I->explicit_coords ||
                I->src[0].type != AGX_INDEX_IMMEDIATE ||
                I->src[0].value != 0 || I->src[0].size != AGX_SIZE_16 ||
                I->src[1].type != AGX_INDEX_IMMEDIATE ||
                I->src[1].value != stores || I->src[1].size != AGX_SIZE_16 ||
                !agx3_src_is_register(I->src[2]) || I->src[2].memory ||
                I->src[2].abs || I->src[2].neg ||
                I->src[2].size != AGX_SIZE_32 ||
                agx_channels(I->src[2]) != 1 ||
                !agx3_src_is_register(I->src[3]) || I->src[3].memory ||
                I->src[3].abs || I->src[3].neg ||
                I->src[3].size != AGX_SIZE_32 ||
                agx_channels(I->src[3]) != 2 ||
                (stores == 0 && I->src[2].value != zero_reg) ||
                (stores == 1 && I->src[2].value != four_reg) || stores >= 2)
               return false;

            if (stores == 0)
               coord_reg = I->src[3].value;
            else if (I->src[3].value != coord_reg)
               return false;
            stores++;
         } else {
            return false;
         }
      }
   }

   return instructions == 6 && get_x == 1 && get_y == 1 && stores == 2 &&
          zero_reg != UINT_MAX && four_reg != UINT_MAX &&
          x_reg == coord_reg && y_reg == coord_reg + 2;
}

/* The formats each target selector of the multi-target EOT builtin is
 * derived for. The logical format, carried by the key, picks the selector;
 * the store format only has to be one that selector was derived for. */
static bool
agx3_g17p_mrt_eot_store_format_ok(enum agx_format format, uint8_t selector)
{
   switch (format) {
   case AGX_FORMAT_I8:
      return selector == 0x55;
   case AGX_FORMAT_U8NORM:
   case AGX_FORMAT_S8NORM:
   case AGX_FORMAT_RGB10A2:
   case AGX_FORMAT_RG11B10F:
      return selector == 0x75;
   case AGX_FORMAT_I16:
   case AGX_FORMAT_F16:
      return selector == 0x15;
   case AGX_FORMAT_U16NORM:
      return selector == 0x95 || selector == 0x15;
   case AGX_FORMAT_S16NORM:
      return selector == 0x95;
   case AGX_FORMAT_I32:
      return selector == 0x35 || selector == 0x75 || selector == 0x15;
   default:
      return false;
   }
}

/* Recognize EOT shaders with two to four raw32 targets, three to four RGBA8
 * targets, or two to eight targets with key-selected formats. The native
 * constructors use one store per dense attachment. Homogeneous targets have a
 * four-byte or sixteen-byte imageblock stride; key-selected targets take any
 * offset, which reaches the program through its offset uniform. */
static unsigned
agx3_g17p_multi_eot_count(const agx_context *ctx)
{
   const bool g18 = ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
                    ctx->key->dev.emit_agx3d;
   if ((ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B && !g18) ||
       ctx->stage != MESA_SHADER_COMPUTE || ctx->is_preamble ||
       ctx->key->no_stop ||
       (ctx->key->g17p_rt0_eot_multisample && !ctx->key->g17p_mrt_eot_count) ||
       list_length(&ctx->blocks) != 1)
      return false;

   const unsigned mixed_count = ctx->key->g17p_mrt_eot_count;
   if (mixed_count == 1 || mixed_count > AGX_MAX_RTS)
      return false;
   if (g18) {
      if (mixed_count < 2 ||
          (ctx->key->g17p_rt0_eot_multisample && mixed_count != 3))
         return false;
      for (unsigned rt = 0; rt < mixed_count; ++rt) {
         /* Native1121 (run 1130): 16-bit-lane targets store with IMGWBLK
          * f29 = 1 in every mixed end-of-tile program of the capture.
          * Native1130 (run 1139): 8-bit integer lanes store with f29 = 0 in
          * every end of tile of a mixed-format tile of capture 1130. */
         if (ctx->key->g17p_mrt_eot_selectors[rt] != 0x75 &&
             ctx->key->g17p_mrt_eot_selectors[rt] != 0x35 &&
             !(ctx->key->g17p_mrt_eot_selectors[rt] == 0x15 &&
               !ctx->key->g17p_rt0_eot_multisample) &&
             !(ctx->key->g17p_mrt_eot_selectors[rt] == 0x55 &&
               !ctx->key->g17p_rt0_eot_multisample &&
               ctx->key->g18_native_mixed))
            return false;
         if (ctx->key->g17p_rt0_eot_multisample &&
             ctx->key->g17p_mrt_eot_selectors[rt] != 0x35)
            return false;
      }
   }

   unsigned values[256];
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i)
      values[i] = UINT_MAX;
   unsigned x = UINT_MAX, y = UINT_MAX, stores = 0, stride = 0;
   bool stopped = false;
   agx_foreach_instr_global(ctx, I) {
      if (stopped)
         return false;
      if (I->op == AGX_OPCODE_STOP) {
         stopped = true;
      } else if (I->op == AGX_OPCODE_GET_SR || I->op == AGX_OPCODE_MOV_IMM) {
         if (I->nr_dests != 1 || I->nr_srcs ||
             !agx3_src_is_register(I->dest[0]) ||
             I->dest[0].size != AGX_SIZE_32 || agx_channels(I->dest[0]) != 1 ||
             I->dest[0].value >= ARRAY_SIZE(values) || I->dest[0].value % 2)
            return false;
         const unsigned reg = I->dest[0].value;
         /* A later constant must not replace either remembered coordinate.
          * Dword register spans must also be disjoint. */
         if ((x != UINT_MAX && reg < x + 2 && x < reg + 2) ||
             (y != UINT_MAX && reg < y + 2 && y < reg + 2))
            return false;
         values[reg] = UINT_MAX;
         if (I->op == AGX_OPCODE_MOV_IMM) {
            if (mixed_count ? I->imm >= AGX_G17P_MAX_BYTES_PER_SAMPLE
                            : (I->imm > 48 || I->imm % 4))
               return false;
            values[reg] = I->imm;
         } else if (I->sr == AGX_SR_THREAD_POSITION_IN_GRID_X && x == UINT_MAX) {
            x = reg;
         } else if (I->sr == AGX_SR_THREAD_POSITION_IN_GRID_Y && y == UINT_MAX) {
            y = reg;
         } else {
            return false;
         }
      } else if (I->op == AGX_OPCODE_BLOCK_IMAGE_STORE) {
         if (stores >= (mixed_count ? mixed_count : 4) || I->nr_dests ||
             I->nr_srcs != 4 ||
             (mixed_count
                 ? !agx3_g17p_mrt_eot_store_format_ok(
                      I->format, ctx->key->g17p_mrt_eot_selectors[stores])
                 : I->format != (ctx->key->g17p_rt0_eot_i32
                                    ? AGX_FORMAT_I32
                                    : AGX_FORMAT_U8NORM)) ||
             I->dim != AGX_DIM_2D ||
             !I->explicit_coords ||
             I->src[0].type != AGX_INDEX_IMMEDIATE || I->src[0].value != 0 ||
             I->src[0].size != AGX_SIZE_16 ||
             I->src[1].type != AGX_INDEX_IMMEDIATE || I->src[1].value != stores ||
             I->src[1].size != AGX_SIZE_16 ||
             !agx3_src_is_register(I->src[2]) || I->src[2].abs || I->src[2].neg ||
             I->src[2].size != AGX_SIZE_32 || agx_channels(I->src[2]) != 1 ||
             I->src[2].value >= ARRAY_SIZE(values) ||
             !agx3_src_is_register(I->src[3]) || I->src[3].abs || I->src[3].neg ||
             I->src[3].size != AGX_SIZE_32 || agx_channels(I->src[3]) != 2 ||
             x == UINT_MAX || y != x + 2 || I->src[3].value != x)
            return false;
         const unsigned offset = values[I->src[2].value];
         if (stores == 1)
            stride = offset;
         if (mixed_count ? offset == UINT_MAX
             : (stores == 0 && offset != 0) ||
                  (stores > 0 &&
                   ((stride != 4 && stride != 16) || offset != stores * stride)))
            return false;
         stores++;
      } else {
         return false;
      }
   }
   if (mixed_count)
      return stopped && stores == mixed_count ? stores : 0;
   /* A two-word packed-format pair has its own explicit format gate. Do
    * not let its four-byte stride enter through the new RGBA32 admission. */
   if (!ctx->key->g17p_rt0_eot_i32)
      return stopped && stride == 4 && stores >= 3 && stores <= 4 ? stores : 0;
   return stopped && ((stride == 16 && stores >= 2 && stores <= 4) ||
                      (stride == 4 && stores == 4)) ? stores : 0;
}

/* What a register currently holds, for the layered end-of-tile matcher. The
 * coordinate sources are tracked by role rather than by register number,
 * because register allocation reuses a system register's slot for a later tile
 * offset once the coordinate triple has been assembled. */
enum agx3_g17p_eot_role {
   AGX3_G17P_EOT_ROLE_NONE = 0,
   AGX3_G17P_EOT_ROLE_X,
   AGX3_G17P_EOT_ROLE_Y,
   AGX3_G17P_EOT_ROLE_LAYER,
};

/* The layered sibling of agx3_g17p_multi_eot_count().
 *
 * Apple's builtin end-of-tile factory emits a distinct program for 2D-ARRAY
 * attachments: it publishes the layer alongside the XY tile origin, so each
 * block store takes a three-component coordinate vector and AGX_DIM_2D_ARRAY.
 * The generic AGX3 packer has no encoding for an array block image store at
 * all -- it reports "no grounded AGX3 encoding for block_image_store ... dim=3"
 * -- so substituting the native text is the only way such a program can exist.
 * agx3_g17p_layered_eot_kind() already does this for one target; this
 * recognizes the same shape with two through eight.
 *
 * Registers are tracked rather than positionally fixed, because scheduling is
 * free to interleave the per-target offset constants with the stores.
 */
static unsigned
agx3_g17p_array_multi_eot_count(const agx_context *ctx, const char **why)
{
   const unsigned count = ctx->key->g17p_mrt_eot_count;
   const bool g18 = ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
                    ctx->key->dev.emit_agx3d;

#define REJECT(reason)                                                         \
   do {                                                                        \
      if (why)                                                                 \
         *why = (reason);                                                      \
      return 0;                                                                \
   } while (0)

   if ((ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B && !g18) ||
       ctx->stage != MESA_SHADER_COMPUTE || ctx->is_preamble ||
       ctx->key->no_stop || list_length(&ctx->blocks) != 1)
      REJECT("not a single-block G17P compute end-of-tile shader");

   if (count < 2 || count > AGX_MAX_RTS)
      REJECT("key does not name two to eight mixed targets");

   /* The G18 witness family below is single-sample RGBA8. Keep its admission
    * separate from G17P's broader selector and multisample replacement. */
   if (g18) {
      if (ctx->key->g17p_rt0_eot_multisample)
         REJECT("G18 array MRT witness is single-sample");
      for (unsigned rt = 0; rt < count; ++rt) {
         if (ctx->key->g17p_mrt_eot_selectors[rt] != 0x75)
            REJECT("G18 array MRT witness requires RGBA8 stores");
      }
   }

   /* values[r]: the immediate r holds, or UINT_MAX. role[r]: which coordinate
    * source r currently carries. Both are ordinary forward dataflow, so a
    * register that is reused after its value is dead simply takes its new
    * meaning from the instruction that rewrites it. */
   unsigned values[256];
   uint8_t role[256];
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i) {
      values[i] = UINT_MAX;
      role[i] = AGX3_G17P_EOT_ROLE_NONE;
   }

   unsigned coords = UINT_MAX, stores = 0;
   bool stopped = false;

   agx_foreach_instr_global(ctx, I) {
      if (stopped)
         REJECT("instruction after stop");

      if (I->op == AGX_OPCODE_STOP) {
         stopped = true;
         continue;
      }

      if (I->op != AGX_OPCODE_BLOCK_IMAGE_STORE) {
         /* Every producer writes one whole aligned 32-bit register. */
         if (I->nr_dests != 1 || !agx3_src_is_register(I->dest[0]) ||
             I->dest[0].size != AGX_SIZE_32 || agx_channels(I->dest[0]) != 1 ||
             I->dest[0].value >= ARRAY_SIZE(values) || I->dest[0].value % 2)
            REJECT("unexpected destination shape");

         const unsigned reg = I->dest[0].value;

         values[reg] = UINT_MAX;
         role[reg] = AGX3_G17P_EOT_ROLE_NONE;

         switch (I->op) {
         case AGX_OPCODE_GET_SR:
            if (I->nr_srcs)
               REJECT("get_sr with sources");
            if (I->sr == AGX_SR_THREAD_POSITION_IN_GRID_X)
               role[reg] = AGX3_G17P_EOT_ROLE_X;
            else if (I->sr == AGX_SR_THREAD_POSITION_IN_GRID_Y)
               role[reg] = AGX3_G17P_EOT_ROLE_Y;
            else if (I->sr == AGX_SR_THREADGROUP_POSITION_IN_GRID_Z)
               role[reg] = AGX3_G17P_EOT_ROLE_LAYER;
            else
               REJECT("unexpected system register");
            break;

         case AGX_OPCODE_MOV_IMM:
            if (I->nr_srcs || I->imm >= AGX_G17P_MAX_BYTES_PER_SAMPLE)
               REJECT("immediate is not a tile offset");
            values[reg] = I->imm;
            break;

         case AGX_OPCODE_BITOP: {
            /* The register move that assembles the coordinate triple. */
            if ((I->truth_table & 0xfu) != AGX3_SEL_TT_MOV || I->saturate ||
                I->nr_srcs != 2 || !agx3_src_is_register(I->src[0]) ||
                I->src[0].abs || I->src[0].neg ||
                I->src[0].size != AGX_SIZE_32 ||
                agx_channels(I->src[0]) != 1 ||
                I->src[0].value >= ARRAY_SIZE(values) ||
                I->src[1].type != AGX_INDEX_IMMEDIATE || I->src[1].value != 0)
               REJECT("bitop is not a plain register move");

            /* A move carries both meanings of its source. */
            const unsigned src = I->src[0].value;
            values[reg] = values[src];
            role[reg] = role[src];
            break;
         }

         default:
            REJECT("unexpected opcode");
         }

         continue;
      }

      if (stores >= count || I->nr_dests || I->nr_srcs != 4 ||
          !agx3_g17p_mrt_eot_store_format_ok(
             I->format, ctx->key->g17p_mrt_eot_selectors[stores]) ||
          I->dim != AGX_DIM_2D_ARRAY || !I->explicit_coords ||
          I->src[0].type != AGX_INDEX_IMMEDIATE || I->src[0].value != 0 ||
          I->src[0].size != AGX_SIZE_16 ||
          I->src[1].type != AGX_INDEX_IMMEDIATE ||
          I->src[1].value != stores || I->src[1].size != AGX_SIZE_16 ||
          !agx3_src_is_register(I->src[2]) || I->src[2].abs || I->src[2].neg ||
          I->src[2].size != AGX_SIZE_32 || agx_channels(I->src[2]) != 1 ||
          I->src[2].value >= ARRAY_SIZE(values) ||
          !agx3_src_is_register(I->src[3]) || I->src[3].abs || I->src[3].neg ||
          I->src[3].size != AGX_SIZE_32 || agx_channels(I->src[3]) != 3 ||
          I->src[3].value + 4 >= ARRAY_SIZE(values))
         REJECT("store is not a native array end-of-tile store");

      /* Every target stores through one coordinate triple holding the grid X,
       * grid Y and layer, in that order. Checking the roles here also rejects
       * any program that overwrote one of them between two stores. */
      const unsigned base = I->src[3].value;
      if (coords == UINT_MAX)
         coords = base;

      if (base != coords || role[base] != AGX3_G17P_EOT_ROLE_X ||
          role[base + 2] != AGX3_G17P_EOT_ROLE_Y ||
          role[base + 4] != AGX3_G17P_EOT_ROLE_LAYER)
         REJECT("coordinate triple is not (grid x, grid y, layer)");

      /* The offset reaches the program through its offset uniform, so it only
       * has to be a tile offset this shader established. */
      if (values[I->src[2].value] == UINT_MAX)
         REJECT("store offset is not a known immediate");

      stores++;
   }

   if (!stopped || stores != count)
      REJECT("target count does not match the key");

#undef REJECT

   return stores;
}

static enum agx3_pack_result
agx3_try_pack_prebaked_eot(agx_context *ctx, struct util_dynarray *emission,
                           enum agx_isa_encoding enc)
{
   /* A layered mixed-format end-of-tile program. Apple's array builtin differs
    * from the 2D one only in publishing the layer wire, and its per-target
    * IMGWBLK lane-class selector is the sole format-dependent byte, exactly as
    * in the 2D family. A multisample array attachment (BTNF dimension 8)
    * differs from the array text only in those selectors, whose low nibble
    * becomes the multisample dimension class 0 -- the same rule the one-target
    * layered program applies above. Unlike the 2D family it needs no
    * PACK_IMGWIR wire-width change; see agx3_g17p_array_eot.h. */
   const char *array_eot_why = NULL;
   const unsigned array_eot_count =
      agx3_g17p_array_multi_eot_count(ctx, &array_eot_why);
   if (array_eot_count) {
      if (enc == AGX_ISA_ENCODING_AGX3C) {
         /* Complete G17G builtins, with identical instruction fields in all
          * paired G18 captures. The ordinary transcode emits G18 bytes.
          * Native metadata uses four GPR dwords and sixteen staging bytes
          * per target, including the early stores in the seven/eight forms. */
         const struct agx3_g17g_array_eot_program *program =
            &agx3_g17g_array_eot[array_eot_count];
         ctx->max_reg = 7;
         ctx->agx3_staging_halfs = 8 * array_eot_count;
         memcpy(util_dynarray_grow_bytes(emission, 1, program->size),
                program->text, program->size);
         return AGX3_PACK_EMITTED;
      }
      const uint8_t *text = agx3_g17p_array_eot[array_eot_count].text;
      const unsigned size = agx3_g17p_array_eot[array_eot_count].size;
      const bool multisample = ctx->key->g17p_rt0_eot_multisample;
      uint8_t program[sizeof(agx3_g17p_array_eot_8)];

      assert(size <= sizeof(program));
      memcpy(program, text, size);

      for (unsigned rt = 0; rt < array_eot_count; ++rt) {
         const uint8_t selector = ctx->key->g17p_mrt_eot_selectors[rt];
         const bool store_samples = multisample &&
            !(ctx->key->g17p_mrt_eot_resolve_mask & BITFIELD_BIT(rt));
         program[agx3_g17p_array_eot[array_eot_count].selector[rt]] =
            store_samples ? (selector & 0xf0) : selector;
      }

      /* The matched synthetic graph used registers that are absent from the
       * replacement, so its register count does not describe what will run.
       * Apple's metadata for the array builtin asks for five GPR dwords and
       * sixteen staging bytes per target. */
      ctx->max_reg = 9;
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs,
                                     8 * array_eot_count);

      /* Six or more targets exceed agx3_append's 8-bit length. */
      memcpy(util_dynarray_grow_bytes(emission, 1, size), program, size);
      return AGX3_PACK_EMITTED;
   }

   const unsigned multi_eot_count = agx3_g17p_multi_eot_count(ctx);
   if (multi_eot_count && enc == AGX_ISA_ENCODING_AGX3C) {
      /* Native G17G BTNF RGBA8 EOT programs for two through eight targets,
       * re-encoded to G18 by the normal final pass. Each complete program
       * matches its G18 reference; capture308 also executes the eight-target
       * program. u0..3 points to dense PBEs, u(4+4*i) holds tile offset i.
       * The caller compacts sparse shader keys and PBE bindings together.
       * Reproducible references: J873 mrt-290-20261004/native, counts 2..8. */
      static const uint8_t rgba8_eot_2[] = {
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06, 0x0c, 0x9d, 0x10, 0x26,
         0x24, 0x81, 0x01, 0x40, 0x28, 0x01, 0xa0, 0x0a, 0x37, 0x06, 0x04, 0x1a,
         0x29, 0x00, 0xa1, 0x02, 0x68, 0x01, 0x20, 0x00, 0x01, 0x02, 0x27, 0x08,
         0x04, 0x1a, 0x29, 0x00, 0xa1, 0x02, 0x28, 0x01, 0x20, 0x00, 0x01, 0x02,
         0x27, 0x00, 0x20, 0x2a, 0x21, 0x00, 0xab, 0x0a, 0x00, 0x00, 0x4a, 0x30,
         0x18, 0x00, 0x0b, 0x82, 0x21, 0x04, 0x1b, 0x80, 0x21, 0x04, 0x27, 0x00,
         0x20, 0x4a, 0x21, 0x01, 0xab, 0x1a, 0x00, 0x00, 0x4a, 0x30, 0x18, 0x00,
         0x23, 0x02, 0x07, 0x02, 0x26, 0x81, 0xa4, 0x12, 0x33, 0x00, 0x07, 0x02,
         0x26, 0x81, 0xa4, 0x12, 0x25, 0x04, 0x15, 0x00, 0x24, 0x00, 0xa0, 0x22,
         0x00, 0xe0, 0x00, 0x55, 0x25, 0x08, 0x15, 0x00, 0x34, 0x40, 0xa0, 0x22,
         0x00, 0xe0, 0x04, 0x55, 0x0e, 0x00, 0x00, 0x00,
      };
      static const uint8_t rgba8_eot_3[] = {
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06, 0x0c, 0x9d, 0x10, 0x26,
         0x24, 0x81, 0x01, 0x40, 0x28, 0x01, 0xa0, 0x0a, 0x37, 0x06, 0x04, 0x1a,
         0x29, 0x00, 0xa1, 0x02, 0x68, 0x01, 0x20, 0x00, 0x01, 0x02, 0x27, 0x08,
         0x04, 0x1a, 0x29, 0x00, 0xa1, 0x02, 0x28, 0x01, 0x20, 0x00, 0x01, 0x02,
         0x27, 0x00, 0x20, 0x2a, 0x21, 0x00, 0xab, 0x0a, 0x00, 0x00, 0x4a, 0x30,
         0x18, 0x00, 0x0b, 0x82, 0x21, 0x04, 0x1b, 0x80, 0x21, 0x04, 0x27, 0x00,
         0x20, 0x4a, 0x21, 0x01, 0xab, 0x1a, 0x00, 0x00, 0x4a, 0x30, 0x18, 0x00,
         0x23, 0x82, 0x07, 0x02, 0x22, 0x81, 0xa4, 0x12, 0x33, 0x80, 0x07, 0x02,
         0x22, 0x81, 0xa4, 0x12, 0xa7, 0x00, 0x20, 0x6a, 0x21, 0x02, 0xab, 0x0a,
         0x00, 0x00, 0x4a, 0x30, 0x18, 0x00, 0xa3, 0x02, 0x07, 0x02, 0x26, 0x82,
         0xa4, 0x02, 0xb3, 0x00, 0x07, 0x02, 0x26, 0x82, 0xa4, 0x02, 0x25, 0x04,
         0x15, 0x00, 0x24, 0x00, 0xa0, 0x22, 0x00, 0xe0, 0x00, 0x55, 0x25, 0x08,
         0x15, 0x00, 0x34, 0x40, 0xa0, 0x22, 0x00, 0xe0, 0x04, 0x55, 0x25, 0x10,
         0x15, 0x00, 0xa4, 0x80, 0xa0, 0x22, 0x01, 0xe0, 0x00, 0x55, 0x0e, 0x00,
         0x00, 0x00,
      };
      static const uint8_t rgba8_eot_4[] = {
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06, 0x0c, 0x9d, 0x10, 0x26,
         0x24, 0x81, 0x01, 0x40, 0x28, 0x01, 0xa0, 0x0a, 0x37, 0x06, 0x04, 0x1a,
         0x29, 0x00, 0xa1, 0x02, 0x68, 0x01, 0x20, 0x00, 0x01, 0x02, 0x27, 0x08,
         0x04, 0x1a, 0x29, 0x00, 0xa1, 0x02, 0x28, 0x01, 0x20, 0x00, 0x01, 0x02,
         0x27, 0x00, 0x20, 0x2a, 0x21, 0x00, 0xab, 0x0a, 0x00, 0x00, 0x4a, 0x30,
         0x18, 0x00, 0x0b, 0x82, 0x21, 0x04, 0x1b, 0x80, 0x21, 0x04, 0x27, 0x00,
         0x20, 0x4a, 0x21, 0x01, 0xab, 0x1a, 0x00, 0x00, 0x4a, 0x30, 0x18, 0x00,
         0x23, 0x82, 0x07, 0x02, 0x22, 0x81, 0xa4, 0x12, 0x33, 0x80, 0x07, 0x02,
         0x22, 0x81, 0xa4, 0x12, 0xa7, 0x00, 0x20, 0x6a, 0x21, 0x02, 0xab, 0x0a,
         0x00, 0x00, 0x4a, 0x30, 0x18, 0x00, 0xa3, 0x82, 0x07, 0x02, 0x22, 0x82,
         0xa4, 0x02, 0xb3, 0x80, 0x07, 0x02, 0x22, 0x82, 0xa4, 0x02, 0xa7, 0x00,
         0x20, 0x8a, 0x21, 0x03, 0xab, 0x1a, 0x00, 0x00, 0x4a, 0x30, 0x18, 0x00,
         0xa3, 0x02, 0x07, 0x02, 0x26, 0x83, 0xa4, 0x12, 0xb3, 0x00, 0x07, 0x02,
         0x26, 0x83, 0xa4, 0x12, 0x25, 0x04, 0x15, 0x00, 0x24, 0x00, 0xa0, 0x22,
         0x00, 0xe0, 0x00, 0x55, 0x25, 0x08, 0x15, 0x00, 0x34, 0x40, 0xa0, 0x22,
         0x00, 0xe0, 0x04, 0x55, 0x25, 0x10, 0x15, 0x00, 0xa4, 0x80, 0xa0, 0x22,
         0x01, 0xe0, 0x00, 0x55, 0x25, 0x20, 0x15, 0x00, 0xb4, 0xc0, 0xa0, 0x22,
         0x01, 0xe0, 0x04, 0x55, 0x0e, 0x00, 0x00, 0x00,
      };
      static const uint8_t rgba8_eot_5[] = {
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06, 0x0c, 0x9d, 0x10, 0x26,
         0x24, 0x81, 0x01, 0x40, 0x28, 0x01, 0xa0, 0x0a, 0x37, 0x06, 0x04, 0x1a,
         0x29, 0x00, 0xa1, 0x02, 0x68, 0x01, 0x20, 0x00, 0x01, 0x02, 0x27, 0x08,
         0x04, 0x1a, 0x29, 0x00, 0xa1, 0x02, 0x28, 0x01, 0x20, 0x00, 0x01, 0x02,
         0x27, 0x00, 0x20, 0x2a, 0x21, 0x00, 0xab, 0x0a, 0x00, 0x00, 0x4a, 0x30,
         0x18, 0x00, 0x0b, 0x82, 0x21, 0x04, 0x1b, 0x80, 0x21, 0x04, 0x27, 0x00,
         0x20, 0x4a, 0x21, 0x01, 0xab, 0x1a, 0x00, 0x00, 0x4a, 0x30, 0x18, 0x00,
         0x23, 0x82, 0x07, 0x02, 0x22, 0x81, 0xa4, 0x12, 0x33, 0x80, 0x07, 0x02,
         0x22, 0x81, 0xa4, 0x12, 0xa7, 0x00, 0x20, 0x6a, 0x21, 0x02, 0xab, 0x0a,
         0x00, 0x00, 0x4a, 0x30, 0x18, 0x00, 0xa3, 0x82, 0x07, 0x02, 0x22, 0x82,
         0xa4, 0x02, 0xb3, 0x80, 0x07, 0x02, 0x22, 0x82, 0xa4, 0x02, 0xa7, 0x00,
         0x20, 0x8a, 0x21, 0x03, 0xab, 0x1a, 0x00, 0x00, 0x4a, 0x30, 0x18, 0x00,
         0xa3, 0x82, 0x07, 0x02, 0x22, 0x83, 0xa4, 0x12, 0xb3, 0x80, 0x07, 0x02,
         0x22, 0x83, 0xa4, 0x12, 0x27, 0x00, 0x20, 0xaa, 0x21, 0x04, 0xab, 0x2a,
         0x00, 0x00, 0x4a, 0x30, 0x18, 0x00, 0x23, 0x02, 0x07, 0x02, 0x26, 0x84,
         0xa4, 0x22, 0x33, 0x00, 0x07, 0x02, 0x26, 0x84, 0xa4, 0x22, 0x25, 0x04,
         0x15, 0x00, 0x24, 0x00, 0xa0, 0x22, 0x00, 0xe0, 0x00, 0x55, 0x25, 0x08,
         0x15, 0x00, 0x34, 0x40, 0xa0, 0x22, 0x00, 0xe0, 0x04, 0x55, 0x25, 0x10,
         0x15, 0x00, 0xa4, 0x80, 0xa0, 0x22, 0x01, 0xe0, 0x00, 0x55, 0x25, 0x20,
         0x15, 0x00, 0xb4, 0xc0, 0xa0, 0x22, 0x01, 0xe0, 0x04, 0x55, 0x25, 0x40,
         0x1d, 0x01, 0x24, 0x02, 0xa0, 0x22, 0x00, 0xe0, 0x00, 0x55, 0x0e, 0x00,
         0x00, 0x00,
      };
      static const uint8_t rgba8_eot_6[] = {
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06, 0x0c, 0x9d, 0x10, 0x26,
         0x24, 0x81, 0x01, 0x40, 0x28, 0x01, 0xa0, 0x0a, 0x37, 0x06, 0x04, 0x1a,
         0x29, 0x00, 0xa1, 0x02, 0x68, 0x01, 0x20, 0x00, 0x01, 0x02, 0x27, 0x08,
         0x04, 0x1a, 0x29, 0x00, 0xa1, 0x02, 0x28, 0x01, 0x20, 0x00, 0x01, 0x02,
         0x27, 0x00, 0x20, 0x2a, 0x21, 0x00, 0xab, 0x0a, 0x00, 0x00, 0x4a, 0x30,
         0x18, 0x00, 0x0b, 0x82, 0x21, 0x04, 0x1b, 0x80, 0x21, 0x04, 0x27, 0x00,
         0x20, 0x4a, 0x21, 0x01, 0xab, 0x1a, 0x00, 0x00, 0x4a, 0x30, 0x18, 0x00,
         0x23, 0x82, 0x07, 0x02, 0x22, 0x81, 0xa4, 0x12, 0x33, 0x80, 0x07, 0x02,
         0x22, 0x81, 0xa4, 0x12, 0xa7, 0x00, 0x20, 0x6a, 0x21, 0x02, 0xab, 0x0a,
         0x00, 0x00, 0x4a, 0x30, 0x18, 0x00, 0xa3, 0x82, 0x07, 0x02, 0x22, 0x82,
         0xa4, 0x02, 0xb3, 0x80, 0x07, 0x02, 0x22, 0x82, 0xa4, 0x02, 0xa7, 0x00,
         0x20, 0x8a, 0x21, 0x03, 0xab, 0x1a, 0x00, 0x00, 0x4a, 0x30, 0x18, 0x00,
         0xa3, 0x82, 0x07, 0x02, 0x22, 0x83, 0xa4, 0x12, 0xb3, 0x80, 0x07, 0x02,
         0x22, 0x83, 0xa4, 0x12, 0x27, 0x00, 0x20, 0xaa, 0x21, 0x04, 0xab, 0x2a,
         0x00, 0x00, 0x4a, 0x30, 0x18, 0x00, 0x23, 0x82, 0x07, 0x02, 0x22, 0x84,
         0xa4, 0x22, 0x33, 0x80, 0x07, 0x02, 0x22, 0x84, 0xa4, 0x22, 0x27, 0x00,
         0x20, 0xca, 0x21, 0x05, 0xab, 0x3a, 0x00, 0x00, 0x4a, 0x30, 0x18, 0x00,
         0x23, 0x02, 0x07, 0x02, 0x26, 0x85, 0xa4, 0x32, 0x33, 0x00, 0x07, 0x02,
         0x26, 0x85, 0xa4, 0x32, 0x25, 0x04, 0x15, 0x00, 0x24, 0x00, 0xa0, 0x22,
         0x00, 0xe0, 0x00, 0x55, 0x25, 0x08, 0x15, 0x00, 0x34, 0x40, 0xa0, 0x22,
         0x00, 0xe0, 0x04, 0x55, 0x25, 0x10, 0x15, 0x00, 0xa4, 0x80, 0xa0, 0x22,
         0x01, 0xe0, 0x00, 0x55, 0x25, 0x20, 0x15, 0x00, 0xb4, 0xc0, 0xa0, 0x22,
         0x01, 0xe0, 0x04, 0x55, 0x25, 0x40, 0x1d, 0x01, 0x24, 0x02, 0xa0, 0x22,
         0x00, 0xe0, 0x00, 0x55, 0x25, 0x00, 0x1d, 0x01, 0x34, 0x42, 0xa0, 0xa2,
         0x00, 0xe0, 0x04, 0x55, 0x0e, 0x00, 0x00, 0x00,
      };
      static const uint8_t rgba8_eot_7[] = {
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06, 0x0c, 0x9d, 0x10, 0x26,
         0x24, 0x81, 0x01, 0x40, 0x28, 0x01, 0xa0, 0x0a, 0x37, 0x06, 0x04, 0x1a,
         0x29, 0x00, 0xa1, 0x02, 0x68, 0x01, 0x20, 0x00, 0x01, 0x02, 0x27, 0x08,
         0x04, 0x1a, 0x29, 0x00, 0xa1, 0x02, 0x28, 0x01, 0x20, 0x00, 0x01, 0x02,
         0x27, 0x00, 0x20, 0x2a, 0x21, 0x00, 0xab, 0x0a, 0x00, 0x00, 0x4a, 0x30,
         0x18, 0x00, 0x0b, 0x82, 0x21, 0x04, 0x1b, 0x80, 0x21, 0x04, 0x25, 0x04,
         0x15, 0x00, 0x24, 0x00, 0xa0, 0x22, 0x00, 0xe0, 0x00, 0x55, 0x27, 0x00,
         0x20, 0x4a, 0x21, 0x01, 0xab, 0x1a, 0x00, 0x00, 0x4a, 0x30, 0x18, 0x00,
         0x23, 0x82, 0x07, 0x02, 0x22, 0x81, 0xa4, 0x12, 0x33, 0x80, 0x07, 0x02,
         0x22, 0x81, 0xa4, 0x12, 0xa7, 0x00, 0x20, 0x6a, 0x21, 0x02, 0xab, 0x0a,
         0x00, 0x00, 0x4a, 0x30, 0x18, 0x00, 0xa3, 0x82, 0x07, 0x02, 0x22, 0x82,
         0xa4, 0x02, 0xb3, 0x80, 0x07, 0x02, 0x22, 0x82, 0xa4, 0x02, 0xa7, 0x00,
         0x20, 0x8a, 0x21, 0x03, 0xab, 0x1a, 0x00, 0x00, 0x4a, 0x30, 0x18, 0x00,
         0xa3, 0x82, 0x07, 0x02, 0x22, 0x83, 0xa4, 0x12, 0xb3, 0x80, 0x07, 0x02,
         0x22, 0x83, 0xa4, 0x12, 0x27, 0x00, 0x20, 0xaa, 0x21, 0x04, 0xab, 0x2a,
         0x00, 0x00, 0x4a, 0x30, 0x18, 0x00, 0x23, 0x82, 0x07, 0x02, 0x22, 0x84,
         0xa4, 0x22, 0x33, 0x80, 0x07, 0x02, 0x22, 0x84, 0xa4, 0x22, 0x27, 0x00,
         0x20, 0xca, 0x21, 0x05, 0xab, 0x3a, 0x00, 0x00, 0x4a, 0x30, 0x18, 0x00,
         0x23, 0x82, 0x07, 0x02, 0x22, 0x85, 0xa4, 0x32, 0x33, 0x80, 0x07, 0x02,
         0x22, 0x85, 0xa4, 0x32, 0xa7, 0x00, 0x20, 0xea, 0x21, 0x06, 0xab, 0x2a,
         0x00, 0x00, 0x4a, 0x30, 0x18, 0x00, 0xa3, 0x02, 0x07, 0x02, 0x26, 0x86,
         0xa4, 0x22, 0xb3, 0x00, 0x07, 0x02, 0x26, 0x86, 0xa4, 0x22, 0x25, 0x08,
         0x15, 0x00, 0x34, 0x40, 0xa0, 0x22, 0x00, 0xe0, 0x04, 0x55, 0x25, 0x10,
         0x15, 0x00, 0xa4, 0x80, 0xa0, 0x22, 0x01, 0xe0, 0x00, 0x55, 0x25, 0x20,
         0x15, 0x00, 0xb4, 0xc0, 0xa0, 0x22, 0x01, 0xe0, 0x04, 0x55, 0x25, 0x40,
         0x1d, 0x01, 0x24, 0x02, 0xa0, 0x22, 0x00, 0xe0, 0x00, 0x55, 0x25, 0x00,
         0x1d, 0x01, 0x34, 0x42, 0xa0, 0xa2, 0x00, 0xe0, 0x04, 0x55, 0x25, 0x00,
         0x5d, 0x01, 0xa4, 0x82, 0xa0, 0x22, 0x01, 0xe0, 0x00, 0x55, 0x0e, 0x00,
         0x00, 0x00,
      };
      static const uint8_t rgba8_eot_8[] = {
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06, 0x0c, 0x9d, 0x10, 0x26,
         0x24, 0x81, 0x01, 0x40, 0x28, 0x01, 0xa0, 0x0a, 0x37, 0x06, 0x04, 0x1a,
         0x29, 0x00, 0xa1, 0x02, 0x68, 0x01, 0x20, 0x00, 0x01, 0x02, 0x27, 0x08,
         0x04, 0x1a, 0x29, 0x00, 0xa1, 0x02, 0x28, 0x01, 0x20, 0x00, 0x01, 0x02,
         0x27, 0x00, 0x20, 0x2a, 0x21, 0x00, 0xab, 0x0a, 0x00, 0x00, 0x4a, 0x30,
         0x18, 0x00, 0x0b, 0x82, 0x21, 0x04, 0x1b, 0x80, 0x21, 0x04, 0x27, 0x00,
         0x20, 0x4a, 0x21, 0x01, 0xab, 0x1a, 0x00, 0x00, 0x4a, 0x30, 0x18, 0x00,
         0x23, 0x82, 0x07, 0x02, 0x22, 0x81, 0xa4, 0x12, 0x33, 0x80, 0x07, 0x02,
         0x22, 0x81, 0xa4, 0x12, 0x25, 0x04, 0x15, 0x00, 0x24, 0x00, 0xa0, 0x22,
         0x00, 0xe0, 0x00, 0x55, 0x25, 0x08, 0x15, 0x00, 0x34, 0x40, 0xa0, 0x22,
         0x00, 0xe0, 0x04, 0x55, 0xa7, 0x00, 0x20, 0x6a, 0x21, 0x02, 0xab, 0x0a,
         0x00, 0x00, 0x4a, 0x30, 0x18, 0x00, 0xa3, 0x82, 0x07, 0x02, 0x22, 0x82,
         0xa4, 0x02, 0xb3, 0x80, 0x07, 0x02, 0x22, 0x82, 0xa4, 0x02, 0xa7, 0x00,
         0x20, 0x8a, 0x21, 0x03, 0xab, 0x1a, 0x00, 0x00, 0x4a, 0x30, 0x18, 0x00,
         0xa3, 0x82, 0x07, 0x02, 0x22, 0x83, 0xa4, 0x12, 0xb3, 0x80, 0x07, 0x02,
         0x22, 0x83, 0xa4, 0x12, 0x27, 0x00, 0x20, 0xaa, 0x21, 0x04, 0xab, 0x2a,
         0x00, 0x00, 0x4a, 0x30, 0x18, 0x00, 0x23, 0x82, 0x07, 0x02, 0x22, 0x84,
         0xa4, 0x22, 0x33, 0x80, 0x07, 0x02, 0x22, 0x84, 0xa4, 0x22, 0x27, 0x00,
         0x20, 0xca, 0x21, 0x05, 0xab, 0x3a, 0x00, 0x00, 0x4a, 0x30, 0x18, 0x00,
         0x23, 0x82, 0x07, 0x02, 0x22, 0x85, 0xa4, 0x32, 0x33, 0x80, 0x07, 0x02,
         0x22, 0x85, 0xa4, 0x32, 0xa7, 0x00, 0x20, 0xea, 0x21, 0x06, 0xab, 0x2a,
         0x00, 0x00, 0x4a, 0x30, 0x18, 0x00, 0xa3, 0x82, 0x07, 0x02, 0x22, 0x86,
         0xa4, 0x22, 0xb3, 0x80, 0x07, 0x02, 0x22, 0x86, 0xa4, 0x22, 0xa7, 0x00,
         0x20, 0x0a, 0x21, 0x87, 0xab, 0x3a, 0x00, 0x00, 0x4a, 0x30, 0x18, 0x00,
         0xa3, 0x02, 0x07, 0x02, 0x26, 0x87, 0xa4, 0x32, 0xb3, 0x00, 0x07, 0x02,
         0x26, 0x87, 0xa4, 0x32, 0x25, 0x10, 0x15, 0x00, 0xa4, 0x80, 0xa0, 0x22,
         0x01, 0xe0, 0x00, 0x55, 0x25, 0x20, 0x15, 0x00, 0xb4, 0xc0, 0xa0, 0x22,
         0x01, 0xe0, 0x04, 0x55, 0x25, 0x40, 0x1d, 0x01, 0x24, 0x02, 0xa0, 0x22,
         0x00, 0xe0, 0x00, 0x55, 0x25, 0x00, 0x1d, 0x01, 0x34, 0x42, 0xa0, 0xa2,
         0x00, 0xe0, 0x04, 0x55, 0x25, 0x00, 0x5d, 0x01, 0xa4, 0x82, 0xa0, 0x22,
         0x01, 0xe0, 0x00, 0x55, 0x2d, 0x00, 0x1d, 0x01, 0xb4, 0xc2, 0xa0, 0x22,
         0x01, 0xe0, 0x04, 0x55, 0x0e, 0x00, 0x00, 0x00,
      };
      static const struct {
         const uint8_t *code;
         size_t size;
         uint16_t stores[AGX_MAX_RTS];
      } programs[] = {
         [2] = {rgba8_eot_2, sizeof(rgba8_eot_2), {100, 112}},
         [3] = {rgba8_eot_3, sizeof(rgba8_eot_3), {130, 142, 154}},
         [4] = {rgba8_eot_4, sizeof(rgba8_eot_4), {160, 172, 184, 196}},
         [5] = {rgba8_eot_5, sizeof(rgba8_eot_5), {190, 202, 214, 226, 238}},
         [6] = {rgba8_eot_6, sizeof(rgba8_eot_6), {220, 232, 244, 256, 268, 280}},
         [7] = {rgba8_eot_7, sizeof(rgba8_eot_7), {70, 262, 274, 286, 298, 310, 322}},
         [8] = {rgba8_eot_8, sizeof(rgba8_eot_8), {100, 112, 304, 316, 328, 340, 352, 364}},
      };
      assert(multi_eot_count >= 2 && multi_eot_count < ARRAY_SIZE(programs));
      ctx->max_reg = MAX2(ctx->max_reg, 7);
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 8 * multi_eot_count);
      /* Six or more targets exceed agx3_append's 8-bit length. */
      const size_t size = programs[multi_eot_count].size;
      uint8_t program[sizeof(rgba8_eot_8)];
      memcpy(program, programs[multi_eot_count].code, size);
      for (unsigned rt = 0; rt < multi_eot_count; ++rt) {
         if (ctx->key->g17p_mrt_eot_selectors[rt] == 0x75)
            continue;
         /* Native427's R32-word programs change IMGWBLK field29 (3 -> 2).
          * Native464's complete three-target MS program additionally changes
          * field30 (1 -> 2) in precisely these stores. All19 instructions,
          * including argument/coordinate setup and waits, otherwise match.
          * Native1121 (run 1130): across all 60 mixed-format test programs of
          * the capture, the two- to eight-target programs differ only in each
          * target's IMGWBLK field29, a function of its lane width alone:
          * 3 for U8NORM-stored, 2 for 32-bit and 1 for 16-bit lanes. */
         const unsigned offset = programs[multi_eot_count].stores[rt];
         if (offset + 12 > size)
            return AGX3_PACK_ERROR;
         uint64_t fields[AGX3_MAX_FIELDS];
         const int32_t form =
            agx3_decode(&agx3c_table, program + offset, 12, AGX3_MN_imgwblk, fields);
         const uint8_t selector = ctx->key->g17p_mrt_eot_selectors[rt];
         /* Native1130 (run 1139): 8-bit integer lanes (selector 0x55) store
          * with f29 = 0 in the 39 workloads' end-of-tile programs, which are
          * otherwise these programs byte for byte. */
         const bool native_mixed_i8 = selector == 0x55 &&
                                      ctx->key->g18_native_mixed;
         if (form < 0 || fields[29] != 3 || fields[30] != 1 ||
             (selector != 0x35 && !native_mixed_i8 && selector != 0x15))
            return AGX3_PACK_ERROR;
         fields[29] = selector == 0x15 ? 1 : native_mixed_i8 ? 0 : 2;
         if (ctx->key->g17p_rt0_eot_multisample)
            fields[30] = 2;
         if (!agx3_compose(&agx3c_table, form, fields, program + offset))
            return AGX3_PACK_ERROR;
      }
      memcpy(util_dynarray_grow_bytes(emission, 1, size), program, size);
      return AGX3_PACK_EMITTED;
   }
   if (multi_eot_count) {
      static_assert(sizeof(agx3_g17p_quad_eot) == 208, "native quad EOT");
      ctx->max_reg = MAX2(ctx->max_reg, 7);
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 8 * multi_eot_count);
      if (ctx->key->g17p_mrt_eot_count) {
         /* The builtin for targets of any formats is the raw32 program with
          * each target's IMGWBLK selector replaced by its format's. A
          * multisample attachment changes three bytes per target, as the
          * one-target multisample builtin does: its PACK_IMGWIR wire-width
          * byte gains 2, and its IMGWBLK clears bit 7 of byte 8 and puts the
          * multisample dimension class 2 in the selector's low nibble. */
         static const struct {
            const uint8_t *text;
            uint16_t imgwblk[AGX_MAX_RTS];
            uint16_t imgwir_wire[AGX_MAX_RTS];
         } templates[] = {
            [2] = {agx3_g17p_dual_raw32_eot, {92, 104}, {43, 63}},
            [3] = {agx3_g17p_triple_raw32_eot, {124, 136, 148},
                   {43, 63, 95}},
            [4] = {agx3_g17p_quad_eot, {156, 168, 180, 192},
                   {43, 63, 95, 127}},
            [5] = {agx3_g17p_penta_raw32_eot, {188, 200, 212, 224, 236},
                   {43, 63, 95, 127, 159}},
            [6] = {agx3_g17p_hexa_raw32_eot, {220, 232, 244, 256, 268, 280},
                   {43, 63, 95, 127, 159, 191}},
            [7] = {agx3_g17p_hepta_raw32_eot,
                   {60, 264, 276, 288, 300, 312, 324},
                   {43, 75, 107, 139, 171, 203, 235}},
            [8] = {agx3_g17p_octa_raw32_eot,
                   {92, 104, 308, 320, 332, 344, 356, 368},
                   {43, 63, 119, 151, 183, 215, 247, 279}},
         };
         static_assert(sizeof(agx3_g17p_octa_raw32_eot) == 32 + 44 * 8,
                       "native octa EOT");
         const size_t size = 32 + 44 * multi_eot_count;
         const bool multisample = ctx->key->g17p_rt0_eot_multisample;
         uint8_t program[sizeof(agx3_g17p_octa_raw32_eot)];
         memcpy(program, templates[multi_eot_count].text, size);
         for (unsigned rt = 0; rt < multi_eot_count; ++rt) {
            const unsigned imgwblk = templates[multi_eot_count].imgwblk[rt];
            uint8_t selector = ctx->key->g17p_mrt_eot_selectors[rt];
            const bool store_samples = multisample &&
               !(ctx->key->g17p_mrt_eot_resolve_mask & BITFIELD_BIT(rt));
            if (store_samples) {
               program[templates[multi_eot_count].imgwir_wire[rt]] |= 0x02;
               program[imgwblk + 8] &= 0x7f;
               selector = (selector & 0xf0) | 0x02;
            }
            /* Byte 9 of the target's 12-byte IMGWBLK. */
            program[imgwblk + 9] = selector;
         }
         /* Six or more targets exceed agx3_append's 8-bit length. */
         memcpy(util_dynarray_grow_bytes(emission, 1, size), program, size);
         return AGX3_PACK_EMITTED;
      }
      if (!ctx->key->g17p_rt0_eot_i32) {
         return (multi_eot_count == 3
            ? agx3_append(emission, agx3_g17p_rgba8_3_eot,
                            sizeof(agx3_g17p_rgba8_3_eot))
            : agx3_append(emission, agx3_g17p_rgba8_4_eot,
                            sizeof(agx3_g17p_rgba8_4_eot))) ? AGX3_PACK_EMITTED : AGX3_PACK_ERROR;
      }
      if (multi_eot_count == 2)
         return (agx3_append(emission, agx3_g17p_dual_raw32_eot,
                            sizeof(agx3_g17p_dual_raw32_eot))) ? AGX3_PACK_EMITTED : AGX3_PACK_ERROR;
      if (multi_eot_count == 3)
         return (agx3_append(emission, agx3_g17p_triple_raw32_eot,
                            sizeof(agx3_g17p_triple_raw32_eot))) ? AGX3_PACK_EMITTED : AGX3_PACK_ERROR;
      return (agx3_append(emission, agx3_g17p_quad_eot, sizeof(agx3_g17p_quad_eot))) ? AGX3_PACK_EMITTED : AGX3_PACK_ERROR;
   }

   const bool r11g11b10_mrt_eot =
      agx3_is_g17p_dual_word_mrt_eot(ctx, enc, AGX_FORMAT_I32);
   if (r11g11b10_mrt_eot ||
       agx3_is_g17p_dual_word_mrt_eot(ctx, enc, AGX_FORMAT_U8NORM)) {
      /* applegpu_g17p builtin EOT generated from two
       * format3/layout2/components4 attachment keys. SHA256 of the native
       * image: b464eff17eb9adb2a6a68da4ea8a5b43ae6eee16edb28065c474ea2d51026f6c. */
      static const uint8_t g17p_rgba8_mrt_eot[] = {
         0x1c,0x9c,0x10,0x06,0x24,0xa8,0x14,0x06,
         0x0c,0x9d,0x10,0x26,0x24,0xa9,0x10,0x26,
         0x9f,0x10,0x54,0x02,0x03,0x04,0x14,0x00,
         0x50,0x26,0x0a,0x00,0x9f,0x20,0x54,0x00,
         0x03,0x00,0x10,0x00,0x50,0x26,0x0a,0x00,
         0x97,0x05,0x54,0x04,0x00,0x08,0x00,0x00,
         0xd0,0x40,0x10,0x00,0x0b,0x82,0x21,0x04,
         0x1b,0x80,0x21,0x04,0x97,0x05,0x54,0x0c,
         0x00,0x10,0x00,0x00,0xd0,0x44,0x10,0x00,
         0x4b,0x03,0x2f,0x00,0x42,0x00,0x00,0x04,
         0x00,0x00,0x5b,0x01,0x2f,0x00,0x42,0x00,
         0x00,0x04,0x00,0x00,0x57,0x10,0x54,0x00,
         0x00,0x00,0x00,0x80,0xa8,0x75,0x00,0x00,
         0x57,0x20,0x54,0x08,0x00,0x10,0x00,0x90,
         0xa8,0x75,0x00,0x00,0x0e,0x00,0x00,0x00,
      };
      static_assert(sizeof(g17p_rgba8_mrt_eot) == 120,
                    "RGBA8 MRT EOT native length");
      uint8_t g17p_r11g11b10_mrt_eot[sizeof(g17p_rgba8_mrt_eot)];
      ctx->max_reg = MAX2(ctx->max_reg, 7);
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 16);
      if (r11g11b10_mrt_eot) {
         /* The exact dual-target control/coordinate program is grounded by
          * the native RGBA8 pair above. Each attachment is one tile word;
          * packed R11G11B10 selects the independently grounded scalar-I32
          * imgwblk tail (0x35) for both stores. No other bytes encode the
          * logical format, which remains the PBE descriptor's responsibility. */
         memcpy(g17p_r11g11b10_mrt_eot, g17p_rgba8_mrt_eot,
                sizeof(g17p_rgba8_mrt_eot));
         g17p_r11g11b10_mrt_eot[101] = 0x35;
         g17p_r11g11b10_mrt_eot[113] = 0x35;
         return (agx3_append(emission, g17p_r11g11b10_mrt_eot,
                            sizeof(g17p_r11g11b10_mrt_eot))) ? AGX3_PACK_EMITTED : AGX3_PACK_ERROR;
      }

      return (agx3_append(emission, g17p_rgba8_mrt_eot,
                         sizeof(g17p_rgba8_mrt_eot))) ? AGX3_PACK_EMITTED : AGX3_PACK_ERROR;
   }

   if (agx3_is_g17p_wboit_mrt_eot(ctx, enc)) {
      /* Fresh applegpu_g17p builtin EOT for RGBA16F partition0 plus R8 in a
       * distinct partition. The text is invariant for every nonzero second
       * partition tested. SHA256 of these exact 238 bytes:
       * eef32d3c9b7bca4de2efba861662e32c34f49af5877dabce0ad9da5cd50d5173.
       *
       * Requesting partition0 for both targets instead produces the old
       * 120-byte text. That same-partition specialization stores RT0 but does
       * not select the separate RT1 tile partition used by WBOIT.
       * Metadata: common0=4, D=32; ESL pointer2 plus kind0 count3 at dest2;
       * qword bindpoint0x32 at dest0; kind5 attachment0/1 at dest0/2; mask3.
       */
      static const uint8_t g17p_wboit_mrt_eot[] = {
         0x1c,0x9c,0x10,0x06,0x24,0xa8,0x14,0x06,0x0c,0x9d,0x10,0x26,
         0x24,0xa9,0x10,0x26,0x9f,0x10,0x54,0x02,0x03,0x04,0x14,0x00,
         0x50,0x26,0x0a,0x00,0x9f,0x20,0x54,0x00,0x03,0x00,0x10,0x00,
         0x50,0x26,0x0a,0x00,0x97,0x05,0x54,0x04,0x00,0x08,0x00,0x00,
         0xd0,0x40,0x10,0x00,0x0b,0x82,0x21,0x04,0x1b,0x80,0x21,0x04,
         0x23,0x00,0x20,0x00,0x57,0x10,0x54,0x00,0x00,0x00,0x00,0x80,
         0xa8,0x15,0x00,0x00,0x06,0x04,0x24,0x99,0x15,0x06,0x00,0x01,
         0x00,0x00,0x0a,0x84,0x33,0x04,0x06,0x00,0x17,0x28,0x02,0x00,
         0x8f,0x04,0x54,0x22,0x0f,0x01,0x54,0x60,0x00,0x00,0x00,0x00,
         0x00,0x00,0x23,0x00,0x20,0x01,0x0f,0x05,0x54,0x1a,0x34,0x98,
         0x11,0x06,0x00,0x01,0x00,0x00,0x9f,0x01,0x54,0x05,0x03,0x02,
         0x14,0x00,0x15,0x04,0x0a,0x84,0x23,0x86,0x06,0x00,0x0f,0x28,
         0x02,0x00,0x32,0x04,0x35,0x06,0x26,0x81,0x0f,0x08,0x22,0x80,
         0x9f,0x01,0x54,0x04,0x02,0x08,0x18,0x00,0x17,0x05,0x8f,0x04,
         0x54,0x22,0x0f,0x00,0x54,0xcc,0xff,0xff,0xff,0xff,0xff,0x00,
         0x0f,0x06,0x04,0x02,0x00,0x00,0x0f,0x00,0x54,0x94,0xff,0xff,
         0xff,0xff,0xff,0x00,0x0f,0x06,0x04,0x02,0x00,0x00,0x97,0x05,
         0x54,0x0c,0x00,0x10,0x00,0x00,0xd0,0x40,0x10,0x00,0x4b,0x02,
         0x29,0x04,0x5b,0x00,0x29,0x04,0x57,0x10,0x54,0x08,0x00,0x10,
         0x00,0x80,0xa8,0x75,0x00,0x00,0x0e,0x00,0x00,0x00,
      };
      static_assert(sizeof(g17p_wboit_mrt_eot) == 238,
                    "WBOIT MRT EOT native length");
      /* Native metadata has common field0=4, D=32: four 32-bit GPRs and 16
       * staging halves. */
      ctx->max_reg = MAX2(ctx->max_reg, 7);
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 16);
      return (agx3_append(emission, g17p_wboit_mrt_eot,
                         sizeof(g17p_wboit_mrt_eot))) ? AGX3_PACK_EMITTED : AGX3_PACK_ERROR;
   }

   const enum agx3_g17p_rt0_eot_kind layered_eot =
      agx3_g17p_layered_eot_kind(ctx, enc);
   if (enc == AGX_ISA_ENCODING_AGX3C && agx_is_g18g(&ctx->key->dev) &&
       layered_eot != AGX3_G17P_RT0_EOT_NONE) {
      /* Original BTNF kind5 array and multisample-array builtins, swept
       * across eight formats and 1..4 components on G17G and G18G. Layer
       * comes from SR12; PACK_IMGWIR publishes layer followed by the XY
       * tile origin. The ordinary G18 reencoder converts this whole G17G
       * program to the independently witnessed 88-byte G18G program.
       * Only IMGWBLK lane class and dimension vary across the sweep. */
      uint8_t program[] = {
         0x1c,0x9c,0x10,0x06,0x34,0xa8,0x10,0x06,
         0x0c,0x9d,0x10,0x26,0x24,0xa9,0x14,0x26,
         0x24,0x82,0x01,0x10,0x28,0x02,0xa0,0x0a,
         0x27,0x06,0x04,0x0a,0x21,0x00,0xa1,0x02,
         0xa8,0x01,0x20,0x00,0x01,0x02,0x37,0x08,
         0x04,0x0a,0x21,0x00,0xa1,0x02,0x68,0x01,
         0x20,0x00,0x01,0x02,0x27,0x10,0x00,0x3a,
         0x2d,0x00,0xab,0x02,0x00,0x00,0x40,0x30,
         0x28,0x00,0x27,0x00,0x20,0x2a,0x21,0x00,
         0xab,0x0a,0x00,0x00,0x4a,0x40,0x18,0x00,
         0x25,0x04,0x15,0x00,0x24,0x00,0xa0,0x22,
         0x00,0xe0,0x00,0x45,0x0e,0x00,0x00,0x00,
      };
      static_assert(sizeof(program) == 96, "G17G layered EOT native length");
      static const uint8_t lane_class[] = {
         [AGX3_G17P_RT0_EOT_SELECTOR_75] = 0xe0,
         [AGX3_G17P_RT0_EOT_SELECTOR_15] = 0x20,
         [AGX3_G17P_RT0_EOT_SELECTOR_35] = 0x60,
         [AGX3_G17P_RT0_EOT_SELECTOR_55] = 0xa0,
         [AGX3_G17P_RT0_EOT_NORM16] = 0x20,
      };
      program[89] = lane_class[layered_eot];
      if (layered_eot == AGX3_G17P_RT0_EOT_NORM16)
         program[87] = 0x2a;
      if (ctx->key->g17p_rt0_eot_multisample)
         program[91] = 0x85;
      /* Native metadata: four GPR dwords, 16 staging bytes. */
      ctx->max_reg = 7;
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 8);
      return (agx3_append(emission, program, sizeof(program))) ? AGX3_PACK_EMITTED : AGX3_PACK_ERROR;
   }
   const enum agx3_g17p_rt0_eot_kind rt0_eot =
      agx3_g17p_rt0_eot_kind(ctx, enc);
   if (enc == AGX_ISA_ENCODING_AGX3C && agx_is_g18g(&ctx->key->dev) &&
       rt0_eot != AGX3_G17P_RT0_EOT_NONE) {
      /* Genuine applegpu_g17g BTNF kind5 programs. The same architecture
       * independent builtin request that produced the G17P tables produces
       * these when it is compiled for applegpu_g17g; see the adjacent probe
       * README for the reproducible generation and the binding ABI. The five
       * programs differ from one another only in the IMGWBLK format class,
       * exactly as the G17P ones do, and G17G spells that class in the two
       * bytes at offsets 69 and 71 rather than a single selector byte.
       * Replace the complete proven graph, never a lone IMGWBLK operation. */
      static const uint8_t g17g_rt0_eot[][78] = {{
         /* 0x75: U8NORM and SRGBA8, the RT0 baseline: native key0x10020300. */
         0x1c,0x9c,0x10,0x06,0x24,0xa8,0x14,0x06,
         0x0c,0x9d,0x10,0x26,0x24,0x81,0x01,0x40,
         0x28,0x01,0xa0,0x0a,0x27,0x06,0x04,0x0a,
         0x21,0x00,0xa1,0x02,0x68,0x01,0x20,0x00,
         0x01,0x02,0x37,0x08,0x04,0x0a,0x21,0x00,
         0xa1,0x02,0x28,0x01,0x20,0x00,0x01,0x02,
         0x27,0x00,0x20,0x2a,0x21,0x00,0xab,0x0a,
         0x00,0x00,0x4a,0x30,0x18,0x00,0x25,0x04,
         0x15,0x00,0x24,0x00,0xa0,0x22,0x00,0xe0,
         0x00,0x55,0x0e,0x00,0x00,0x00,
      }, {
         /* 0x15: I16 and F16 words, and the two-word U16NORM pair: key0x08020800. */
         0x1c,0x9c,0x10,0x06,0x24,0xa8,0x14,0x06,
         0x0c,0x9d,0x10,0x26,0x24,0x81,0x01,0x40,
         0x28,0x01,0xa0,0x0a,0x27,0x06,0x04,0x0a,
         0x21,0x00,0xa1,0x02,0x68,0x01,0x20,0x00,
         0x01,0x02,0x37,0x08,0x04,0x0a,0x21,0x00,
         0xa1,0x02,0x28,0x01,0x20,0x00,0x01,0x02,
         0x27,0x00,0x20,0x2a,0x21,0x00,0xab,0x0a,
         0x00,0x00,0x4a,0x30,0x18,0x00,0x25,0x04,
         0x15,0x00,0x24,0x00,0xa0,0x22,0x00,0x20,
         0x00,0x55,0x0e,0x00,0x00,0x00,
      }, {
         /* 0x35: I32 words and the three-lane R5G6B5 store: key0x0c020b00. */
         0x1c,0x9c,0x10,0x06,0x24,0xa8,0x14,0x06,
         0x0c,0x9d,0x10,0x26,0x24,0x81,0x01,0x40,
         0x28,0x01,0xa0,0x0a,0x27,0x06,0x04,0x0a,
         0x21,0x00,0xa1,0x02,0x68,0x01,0x20,0x00,
         0x01,0x02,0x37,0x08,0x04,0x0a,0x21,0x00,
         0xa1,0x02,0x28,0x01,0x20,0x00,0x01,0x02,
         0x27,0x00,0x20,0x2a,0x21,0x00,0xab,0x0a,
         0x00,0x00,0x4a,0x30,0x18,0x00,0x25,0x04,
         0x15,0x00,0x24,0x00,0xa0,0x22,0x00,0x60,
         0x00,0x55,0x0e,0x00,0x00,0x00,
      }, {
         /* 0x55: The one-word I8 integer keys: key0x04020400. */
         0x1c,0x9c,0x10,0x06,0x24,0xa8,0x14,0x06,
         0x0c,0x9d,0x10,0x26,0x24,0x81,0x01,0x40,
         0x28,0x01,0xa0,0x0a,0x27,0x06,0x04,0x0a,
         0x21,0x00,0xa1,0x02,0x68,0x01,0x20,0x00,
         0x01,0x02,0x37,0x08,0x04,0x0a,0x21,0x00,
         0xa1,0x02,0x28,0x01,0x20,0x00,0x01,0x02,
         0x27,0x00,0x20,0x2a,0x21,0x00,0xab,0x0a,
         0x00,0x00,0x4a,0x30,0x18,0x00,0x25,0x04,
         0x15,0x00,0x24,0x00,0xa0,0x22,0x00,0xa0,
         0x00,0x55,0x0e,0x00,0x00,0x00,
      }, {
         /* 0x95: The normalized-16 keys: key0x10020600. */
         0x1c,0x9c,0x10,0x06,0x24,0xa8,0x14,0x06,
         0x0c,0x9d,0x10,0x26,0x24,0x81,0x01,0x40,
         0x28,0x01,0xa0,0x0a,0x27,0x06,0x04,0x0a,
         0x21,0x00,0xa1,0x02,0x68,0x01,0x20,0x00,
         0x01,0x02,0x37,0x08,0x04,0x0a,0x21,0x00,
         0xa1,0x02,0x28,0x01,0x20,0x00,0x01,0x02,
         0x27,0x00,0x20,0x2a,0x21,0x00,0xab,0x0a,
         0x00,0x00,0x4a,0x30,0x18,0x00,0x25,0x04,
         0x15,0x00,0x24,0x00,0xa0,0x2a,0x00,0x20,
         0x00,0x55,0x0e,0x00,0x00,0x00,
      }};
      static_assert(sizeof(g17g_rt0_eot[0]) == 78, "G17G EOT native length");
      static_assert(ARRAY_SIZE(g17g_rt0_eot) ==
                       AGX3_G17P_RT0_EOT_NORM16 -
                          AGX3_G17P_RT0_EOT_SELECTOR_75 + 1,
                    "one G17G program per format class");
      /* Native common0=3 GPR dwords and fields38/42 D=16. */
      ctx->max_reg = MAX2(ctx->max_reg, 7);
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 8);
      const unsigned index = rt0_eot - AGX3_G17P_RT0_EOT_SELECTOR_75;
      assert(index < ARRAY_SIZE(g17g_rt0_eot) &&
             "the classifier refuses the G17P-only builtins on G17G");
      uint8_t program[sizeof(g17g_rt0_eot[0])];
      memcpy(program, g17g_rt0_eot[index], sizeof(program));
      if (ctx->key->g17p_rt0_eot_multisample) {
         /* Native G17G/G18G 2DMS EOT programs preserve the complete 2D
          * coordinate wire for all five format classes and 1..4 components.
          * Only IMGWBLK field30 changes from 1 to 2 (byte73 bits6:7).
          * G17P's PACK_IMGWIR change does not apply here. */
         program[73] = (program[73] & 0x3f) | (2u << 6);
      }
      return (agx3_append(emission, program, sizeof(program))) ? AGX3_PACK_EMITTED : AGX3_PACK_ERROR;
   }
   const enum agx3_g17p_rt0_eot_kind m5_eot =
      layered_eot != AGX3_G17P_RT0_EOT_NONE ? layered_eot : rt0_eot;
   if (enc == AGX_ISA_ENCODING_AGX3C && m5_eot != AGX3_G17P_RT0_EOT_NONE) {
      static const uint8_t m5_rt0_eot[78] = {
         0x1c,0x9c,0x10,0x06,0x24,0xa8,0x14,0x06,
         0x0c,0x9d,0x10,0x26,0x24,0x81,0x01,0x40,
         0x28,0x01,0xa0,0x0a,0x27,0x06,0x04,0x0a,
         0x21,0x00,0xa1,0x02,0x68,0x01,0x20,0x00,
         0x01,0x02,0x37,0x08,0x04,0x0a,0x21,0x00,
         0xa1,0x02,0x28,0x01,0x20,0x00,0x01,0x02,
         0x27,0x00,0x20,0x2a,0x21,0x00,0xab,0x0a,
         0x00,0x00,0x4a,0x30,0x18,0x00,0x25,0x04,
         0x15,0x00,0x24,0x00,0xa0,0x22,0x00,0xe0,
         0x00,0x55,0x0e,0x00,0x00,0x00,
      };
      uint8_t program[sizeof(m5_rt0_eot)];
      memcpy(program, m5_rt0_eot, sizeof(program));
      switch (m5_eot) {
      case AGX3_G17P_RT0_EOT_SELECTOR_15: program[71] = 0x20; break;
      case AGX3_G17P_RT0_EOT_SELECTOR_35: program[71] = 0x60; break;
      case AGX3_G17P_RT0_EOT_SELECTOR_55: program[71] = 0xa0; break;
      case AGX3_G17P_RT0_EOT_SELECTOR_75: break;
      case AGX3_G17P_RT0_EOT_NORM16:
         program[69] = 0x2a;
         program[71] = 0x20;
         break;
      default: return AGX3_PACK_ERROR;
      }
      if (ctx->key->g17p_rt0_eot_multisample)
         program[73] = (program[73] & ~0xc0u) | 0x80u;
      if (ctx->key->m5_rt0_eot_store_and_resolve) {
         if (layered_eot != AGX3_G17P_RT0_EOT_NONE ||
             m5_eot != AGX3_G17P_RT0_EOT_SELECTOR_75 ||
             !ctx->key->g17p_rt0_eot_multisample)
            return AGX3_PACK_ERROR;

         /* Native G17G paired IMGWBLK retains the one coordinate wire for
          * the MS store (slot 2), then consumes it in the 2D resolve (slot 1).
          * Keep the qualified uniform-PBE ABI: U0 source, U4 tile offset,
          * and U8 resolve PBE. Descriptor operands remain uniform/64-bit. */
         static const struct agx3_field_kv img[] = {
            {9, 5}, {10, 2}, {14, 2}, {15, 1}, {16, 86},
            {29, 3}, {32, 1}, {45, 3}, {46, 3},
            {59, 1}, {60, 1}, {73, 1},
         };
         static const struct agx3_field_kv sample_store[] = {
            {12, 2}, {17, 1}, {30, 2},
         };
         static const struct agx3_field_kv resolve_store[] = {
            {12, 1}, {13, 1}, {30, 1}, {39, 8},
         };
         uint8_t sample_bytes[16], resolve_bytes[16];
         uint8_t sample_len = 0, resolve_len = 0;
         if (!agx3_bind_compose(enc, AGX3_MN_imgwblk, 0, img, ARRAY_SIZE(img),
                                sample_store, ARRAY_SIZE(sample_store),
                                sample_bytes, &sample_len) ||
             !agx3_bind_compose(enc, AGX3_MN_imgwblk, 0, img, ARRAY_SIZE(img),
                                resolve_store, ARRAY_SIZE(resolve_store),
                                resolve_bytes, &resolve_len) ||
             sample_len != 12 || resolve_len != 12)
            return AGX3_PACK_ERROR;
         ctx->max_reg = MAX2(ctx->max_reg, 7);
         ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 8);
         return (agx3_append(emission, program, 62) &&
                agx3_append(emission, sample_bytes, sample_len) &&
                agx3_append(emission, resolve_bytes, resolve_len) &&
                agx3_append(emission, program + 74, 4)) ? AGX3_PACK_EMITTED : AGX3_PACK_ERROR;
      }
      uint8_t layer_sr[AGX3_SR_MAX_LENGTH], layer_pack[16];
      uint8_t layer_sr_len = 0, layer_pack_len = 0;
      if (layered_eot != AGX3_G17P_RT0_EOT_NONE) {
         /* G17G's array intrinsic publishes a 16-bit layer into a GPR
          * wire before the XY wire. Keep the proven builtin's uniform
          * PBE pointer (halves 0..3) and tile byte offset (halves 4..5).
          * Native llvm.agx2.get.layerid reads SR12/W16, including internal
          * tile programs; dispatched grid Z is not the tile's layer.
          * XY origin arithmetic has consumed slot 2 and GPR half 4. */
         const struct agx3_sr_desc layer = {
            .dest = 6, .width = AGX3_SR_W16, .sr = 12, .slot = 2,
         };
         static const struct agx3_field_kv layer_fields[] = {
            {2, 4}, {9, 2}, {10, 2}, {11, 1}, {16, 6},
            {23, 2}, {24, 1}, {27, 1}, {38, 1}, {51, 1},
            {52, 2}, {56, 3}, {57, 118}, {59, 1}, {74, 4},
         };
         if (!agx3_sr_encode(enc, &layer, layer_sr, &layer_sr_len) ||
             !agx3_bind_compose(enc, AGX3_MN_pack_imgwir, 0, layer_fields,
                                ARRAY_SIZE(layer_fields), NULL, 0,
                                layer_pack, &layer_pack_len))
            return AGX3_PACK_ERROR;

         /* Native 2D/array pairs isolate the layer operand: f44=4,
          * f51=2, f55=1 on XY PACK_IMGWIR, and f31=2 on IMGWBLK.
          * All descriptor, offset, staging and format fields stay intact. */
         program[59] = 0x40;
         program[60] = 0x1c;
         program[73] &= ~0x10u;
      }
      ctx->max_reg = layered_eot != AGX3_G17P_RT0_EOT_NONE
                        ? 7 : MAX2(ctx->max_reg, 7);
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 8);
      if (layered_eot != AGX3_G17P_RT0_EOT_NONE) {
         return (agx3_append(emission, program, 48) &&
                agx3_append(emission, layer_sr, layer_sr_len) &&
                agx3_append(emission, layer_pack, layer_pack_len) &&
                agx3_append(emission, program + 48, sizeof(program) - 48)) ? AGX3_PACK_EMITTED : AGX3_PACK_ERROR;
      }
      return (agx3_append(emission, program, sizeof(program))) ? AGX3_PACK_EMITTED : AGX3_PACK_ERROR;
   }
   if (layered_eot != AGX3_G17P_RT0_EOT_NONE) {
      /* The 2D-array end-of-tile builtin is one 84-byte program for every
       * format, differing only in the IMGWBLK lane-class selector at byte
       * 77; the 2D multisample-array one also clears its dimension nibble. */
      uint8_t program[] = {
         0x1c,0x9c,0x10,0x06,0x34,0xa8,0x10,0x06,
         0x0c,0x9d,0x10,0x26,0x24,0xa9,0x14,0x26,
         0x24,0xc3,0x10,0x46,
         0x9f,0x10,0x54,0x00,0x00,0x04,0x18,0x00,0x50,0x06,0x0b,0x00,
         0x9f,0x20,0x54,0x02,0x00,0x00,0x14,0x00,0x50,0x06,0x0b,0x00,
         /* Native array wire: layer first, followed by the XY tile origin. */
         0x97,0x45,0x54,0x04,0x00,0x08,0x10,0x00,0xf0,0xc0,0x00,0x00,
         0x97,0x05,0x54,0x06,0x00,0x08,0x00,0x00,0xd0,0x40,0x10,0x00,
         0x57,0x10,0x54,0x00,0x00,0x00,0x00,0x80,0x28,0x75,0x00,0x00,
         0x0e,0x00,0x00,0x00,
      };
      static_assert(sizeof(program) == 84, "layered EOT native length");
      static const uint8_t selectors[] = {
         [AGX3_G17P_RT0_EOT_SELECTOR_75] = 0x75,
         [AGX3_G17P_RT0_EOT_SELECTOR_15] = 0x15,
         [AGX3_G17P_RT0_EOT_SELECTOR_35] = 0x35,
         [AGX3_G17P_RT0_EOT_SELECTOR_55] = 0x55,
         [AGX3_G17P_RT0_EOT_NORM16] = 0x95,
      };
      program[77] = ctx->key->g17p_rt0_eot_multisample
                       ? selectors[layered_eot] & 0xf0
                       : selectors[layered_eot];
      /* The matched synthetic graph used registers that are absent from the
       * replacement. The native metadata requires four GPR dwords. */
      ctx->max_reg = 7;
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 8);
      return (agx3_append(emission, program, sizeof(program))) ? AGX3_PACK_EMITTED : AGX3_PACK_ERROR;
   }

   if (rt0_eot != AGX3_G17P_RT0_EOT_NONE) {
      static const uint8_t g17p_rt0_eot[][68] = {{
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
         0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
         0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
         0x50, 0x06, 0x0b, 0x00,
         0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
         0x50, 0x06, 0x0b, 0x00,
         0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
         0xd0, 0x40, 0x10, 0x00,
         0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
         0xa8, 0x75, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
      }, {
         /* Native RG16 integer keys0x08020900/0x08020800. */
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
         0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
         0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
         0x50, 0x06, 0x0b, 0x00,
         0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
         0x50, 0x06, 0x0b, 0x00,
         0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
         0xd0, 0x40, 0x10, 0x00,
         0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
         0xa8, 0x15, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
      }, {
         /* applegpu-nt builtin EOT keys 0x04021100 (R32 float) and
          * 0x08021100 (R32G32 float) are byte-identical. */
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
         0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
         0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
         0x50, 0x06, 0x0b, 0x00,
         0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
         0x50, 0x06, 0x0b, 0x00,
         0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
         0xd0, 0x40, 0x10, 0x00,
         0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
         0xa8, 0x35, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
      }, {
         /* Native keys0x08020500/0x08020400: one RG8 integer I8 word. */
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
         0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
         0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
         0x50, 0x06, 0x0b, 0x00,
         0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
         0x50, 0x06, 0x0b, 0x00,
         0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
         0xd0, 0x40, 0x10, 0x00,
         0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
         0xa8, 0x55, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
      }, {
         /* Native format6/layout2/components4: RGBA16 SNORM. */
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
         0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
         0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
         0x50, 0x06, 0x0b, 0x00,
         0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
         0x50, 0x06, 0x0b, 0x00,
         0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
         0xd0, 0x40, 0x10, 0x00,
         0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
         0xa8, 0x95, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
      }};
      /* Apple's builtin end-of-tile factory emits a second program for every
       * one of these five selectors when the attachment's texture type is
       * 2D multisample (dimension 4) instead of 2D (dimension 2). The two
       * forms differ in exactly three bytes: the PACK_IMGWIR wire width
       * (field 5, 4 -> 6) and the IMGWBLK dimension class (field 30,
       * 1 -> 2, which straddles bytes 60 and 61). Nothing else moves --
       * no extra instruction computes a sample index, because the block
       * store covers every sample of the tile.
       */
      static const uint8_t g17p_rt0_eot_ms[][68] = {{
         /* 0x75 -> 0x72: U8NORM/S8NORM/RGB10A2, the RT0 baseline. */
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
         0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
         0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
         0x50, 0x06, 0x0b, 0x00, 0x9f, 0x20, 0x54, 0x02,
         0x00, 0x00, 0x10, 0x00, 0x50, 0x06, 0x0b, 0x00,
         0x97, 0x05, 0x54, 0x06, 0x00, 0x08, 0x00, 0x00,
         0xd0, 0x40, 0x10, 0x00, 0x57, 0x10, 0x54, 0x00,
         0x00, 0x00, 0x00, 0x80, 0x28, 0x72, 0x00, 0x00,
         0x0e, 0x00, 0x00, 0x00,
      }, {
         /* 0x15 -> 0x12: I16/F16, including RGB565. */
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
         0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
         0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
         0x50, 0x06, 0x0b, 0x00, 0x9f, 0x20, 0x54, 0x02,
         0x00, 0x00, 0x10, 0x00, 0x50, 0x06, 0x0b, 0x00,
         0x97, 0x05, 0x54, 0x06, 0x00, 0x08, 0x00, 0x00,
         0xd0, 0x40, 0x10, 0x00, 0x57, 0x10, 0x54, 0x00,
         0x00, 0x00, 0x00, 0x80, 0x28, 0x12, 0x00, 0x00,
         0x0e, 0x00, 0x00, 0x00,
      }, {
         /* 0x35 -> 0x32: I32, including the four-word keys. */
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
         0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
         0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
         0x50, 0x06, 0x0b, 0x00, 0x9f, 0x20, 0x54, 0x02,
         0x00, 0x00, 0x10, 0x00, 0x50, 0x06, 0x0b, 0x00,
         0x97, 0x05, 0x54, 0x06, 0x00, 0x08, 0x00, 0x00,
         0xd0, 0x40, 0x10, 0x00, 0x57, 0x10, 0x54, 0x00,
         0x00, 0x00, 0x00, 0x80, 0x28, 0x32, 0x00, 0x00,
         0x0e, 0x00, 0x00, 0x00,
      }, {
         /* 0x55 -> 0x52: the one-word I8 integer keys. */
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
         0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
         0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
         0x50, 0x06, 0x0b, 0x00, 0x9f, 0x20, 0x54, 0x02,
         0x00, 0x00, 0x10, 0x00, 0x50, 0x06, 0x0b, 0x00,
         0x97, 0x05, 0x54, 0x06, 0x00, 0x08, 0x00, 0x00,
         0xd0, 0x40, 0x10, 0x00, 0x57, 0x10, 0x54, 0x00,
         0x00, 0x00, 0x00, 0x80, 0x28, 0x52, 0x00, 0x00,
         0x0e, 0x00, 0x00, 0x00,
      }, {
         /* 0x95 -> 0x92: the normalized-16 keys. */
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
         0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
         0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
         0x50, 0x06, 0x0b, 0x00, 0x9f, 0x20, 0x54, 0x02,
         0x00, 0x00, 0x10, 0x00, 0x50, 0x06, 0x0b, 0x00,
         0x97, 0x05, 0x54, 0x06, 0x00, 0x08, 0x00, 0x00,
         0xd0, 0x40, 0x10, 0x00, 0x57, 0x10, 0x54, 0x00,
         0x00, 0x00, 0x00, 0x80, 0x28, 0x92, 0x00, 0x00,
         0x0e, 0x00, 0x00, 0x00,
      }};
      static_assert(sizeof(g17p_rt0_eot_ms) == sizeof(g17p_rt0_eot),
                    "one multisample program per selector");

      /* Native metadata common0=3 means six GPR halves; retain the existing
       * conservative eight-half allocation. Fields38/42 are D=16, requiring
       * eight staging halves independently. Main68 pads to128: LDSHDR f1=1.
       * The multisample forms have the same length, register and staging
       * metadata, so the launch program is unchanged. */
      ctx->max_reg = MAX2(ctx->max_reg, 7);
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 8);
      const unsigned index = rt0_eot - AGX3_G17P_RT0_EOT_SELECTOR_75;
      const uint8_t *program = ctx->key->g17p_rt0_eot_multisample
                                  ? g17p_rt0_eot_ms[index]
                                  : g17p_rt0_eot[index];
      return (agx3_append(emission, program, sizeof(g17p_rt0_eot[index]))) ? AGX3_PACK_EMITTED : AGX3_PACK_ERROR;
   }

   return AGX3_PACK_UNHANDLED;
}

/* Keep refusal diagnostics optional and emission failure transactional. */
#define AGX3_PACK_FAIL() do { \
   if (agx_compiler_debug & AGX_DBG_SHADERS) \
      fprintf(stderr, "AGX3 pack failure at %s:%u\n", \
              agx3_basename(__FILE__), __LINE__); \
   goto fail; \
} while (0)

bool
agx3_pack_binary(agx_context *ctx, struct util_dynarray *emission)
{
   char *first_refusal = agx3_refusal_buffer(false);
   if (first_refusal)
      first_refusal[0] = 0;
   const enum agx_isa_encoding enc = ctx->key->dev.encoding;

   /*
    * G18G (AGX3D) is refused here, whole shader, before anything is emitted.
    * Its core encoders ground a first instruction catalogue through the
    * agx3d table (the vertex_id read, vertex fetch, float ALU, ldimm, uvsw,
    * flat and smooth varyings, fltnorm, pixwait/pixrel, lstcol and end; see
    * the agx3_*_core.c admissions), but this packer is still the agx3b/agx3c
    * one: its hand-written end-of-tile, background and array programs are
    * agx3c or agx3b bytes, its compact paths call the agx3c hand packers, its
    * waits assume agx3c's per-slot masks where agx3d's re-encoded forms name
    * one slot, and Apple's G18G vertex programs add an epilogue on every
    * float varying (bit_andc/iself32/bit_and/bit_or over three driver
    * uniforms) that nothing here emits. Until those are grounded no agx3d
    * byte may leave through this path.
    */
   if (enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C)
      return false;

   agx3_carrier_init(enc == AGX_ISA_ENCODING_AGX3C &&
                     ctx->key->dev.emit_agx3d);

   /* MS MRT reloads reuse texture destinations for scalar sample indices.
    * Their synchronous ALU consumers and ldimm wait preludes can retire
    * slots 1..6. Keep the allocator within that common range rather than
    * assigning a generation whose WAW wait cannot be emitted. */
   const bool g18_ms_background = agx3_g18_msaa4_tile_shader(ctx) &&
      !ctx->key->secondary && ctx->key->fs.ignore_tib_dependencies &&
      ctx->key->g17p_mrt_bg;
   const unsigned nslots =
      (enc == AGX_ISA_ENCODING_AGX3C && !g18_ms_background) ? 8 : 6;
   const unsigned initial_size = emission->size;
   ctx->agx3_staging_known = true;
   ctx->agx3_fltnorm_start = false;

   const enum agx3_pack_result eot =
      agx3_try_pack_prebaked_eot(ctx, emission, enc);
   if (eot != AGX3_PACK_UNHANDLED)
      return eot == AGX3_PACK_EMITTED;

   struct util_dynarray fixups;
   uint16_t outstanding = 0;
   uint16_t image_write_slots = 0;
   uint8_t next_uvs_slot = 1;
   unsigned texture_ops_left = 0;
   bool g18g_low_bank_reads = true;
   bool g18g_low_bank_ops = true;
   unsigned tile_loads_left = 0;
   unsigned tile_loads_seen = 0;
   unsigned tile_stores_left = 0;
   unsigned image_writes_left = 0;
   unsigned texture_samples_left = 0;
   /* A derivative or quad operation after the last sample-group member: see
    * helpers_outlive_group below. */
   bool g17p_quad_op_after_group = false;
   unsigned sample_group_members_left = 0;
   /* Whether a quad-derivative or quad-shuffle consumer follows the last
    * f156 sample-group member in linear order (see the group flags below). */
   bool quad_consumer_after_last_member = false;
   unsigned device_loads_left = 0;
   bool has_coverage_write = false;
   bool has_memory_write = false;
   bool has_ray_traversal = false;
   /* The last pixel release of a program-ending fragment part, when no
    * colour store, coverage update, acquisition or branch follows it. */
   const agx_instr *g17p_final_release = NULL;
   bool has_zs_emit = false, zs_sample_loop = false;
   /* Apple's G17G depth/stencil-only fragments end with pixrel 513: the
    * shader's final release carries the 0x200 end-of-sequence bit when no
    * colour store follows it (depth+colour fragments release 1 after depthf
    * and 524 after the store). Find that final release up front. G18G
    * only (agx3c keeps its own final-release rule). */
   const agx_instr *g17g_final_signal = NULL;
   bool g17g_store_after_final_signal = false;
   unsigned zs_staging_halfs = 0;
   bool g17p_fragment_helper_protocol =
      (enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
      ctx->stage == MESA_SHADER_FRAGMENT && ctx->nir &&
      (ctx->nir->info.fs.needs_coarse_quad_helper_invocations ||
       ctx->nir->info.fs.needs_full_quad_helper_invocations);
   bool uniform_half_used[512] = {false};
   for (unsigned h = 0; h < MIN2(ctx->key->reserved_preamble, 512); ++h)
      uniform_half_used[h] = true;
   if (ctx->out) {
      const unsigned base = ctx->out->rodata.base_uniform;
      for (unsigned h = 0; h < ctx->out->rodata.size_16 && base + h < 512; ++h)
         uniform_half_used[base + h] = true;
   }
   bool has_cube_sample = false;
   bool has_packed_array_layer = false;
   bool has_g18_ms_array = false;
   unsigned observed_max_reg = ctx->max_reg;
   unsigned count = 0;
   bool cprog_imports = false;

   assert(enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C);
   util_dynarray_init(&fixups, ctx);
   /* G18G keeps each compare beside the CND that consumes it: agx3d folds
    * flag slots above 1, which is only sound for adjacent pairs
    * (agx3_transcode.c). */
   const bool early_compare = enc == AGX_ISA_ENCODING_AGX3B ||
      (enc == AGX_ISA_ENCODING_AGX3C && !ctx->key->dev.emit_agx3d &&
       !ctx->is_preamble && ctx->stage == MESA_SHADER_COMPUTE);
   struct util_dynarray spans, compares;
   util_dynarray_init(&spans, ctx);
   util_dynarray_init(&compares, ctx);

   agx_foreach_block(ctx, scan_block) {
      agx_foreach_instr_in_block(scan_block, scan_I) {
         /* An imported/read half is not padding either. A scalar export
          * must not overwrite it merely because no shader store defines it. */
         agx_foreach_src(scan_I, s) {
            const agx_index src = scan_I->src[s];
            if (src.type == AGX_INDEX_UNIFORM) {
               for (unsigned h = 0; h < agx_index_size_16(src) &&
                                    src.value + h < 512; ++h)
                  uniform_half_used[src.value + h] = true;
            }
         }
         /* The compact bank arrangement is only for ordinary ALU around
          * plain image reads/writes. Other asynchronous, control-flow,
          * sample and staging users keep the conservative placement. */
         switch (scan_I->op) {
         case AGX_OPCODE_IMAGE_LOAD:
            g18g_low_bank_reads &=
               scan_I->nr_srcs == 6 && scan_I->dim == AGX_DIM_2D &&
               !scan_I->agx3_tex_coords_staged &&
               agx3_tex_imm(scan_I->src[1], 0) &&
               agx3_tex_imm(scan_I->src[2], 0);
            break;
         case AGX_OPCODE_IMAGE_WRITE:
         case AGX_OPCODE_GET_SR:
         case AGX_OPCODE_MOV_IMM:
         case AGX_OPCODE_BITOP:
         case AGX_OPCODE_IADD:
         case AGX_OPCODE_IMAD:
         case AGX_OPCODE_FADD:
         case AGX_OPCODE_FMUL:
         case AGX_OPCODE_SCOPED_BARRIER_AGX3:
         case AGX_OPCODE_STOP:
            break;
         default:
            g18g_low_bank_ops = false;
            break;
         }
         cprog_imports |= scan_I->op == AGX_OPCODE_LOAD_CPROG_AGX3;
         if (scan_I->op != AGX_OPCODE_STOP)
            count++;
         /* Apple's "last texture op" marker goes on the final smp of the
          * whole stream, fetch or sample alike. */
         if (scan_I->op == AGX_OPCODE_TEXTURE_LOAD ||
             scan_I->op == AGX_OPCODE_IMAGE_LOAD ||
             scan_I->op == AGX_OPCODE_TEXTURE_SAMPLE)
            texture_ops_left++;
         if (scan_I->op == AGX_OPCODE_LD_TILE)
            tile_loads_left++;
         if ((scan_I->op == AGX_OPCODE_TEXTURE_LOAD ||
              scan_I->op == AGX_OPCODE_IMAGE_LOAD) &&
             scan_I->dim == AGX_DIM_2D_MS_ARRAY) {
            has_packed_array_layer = true;
            has_g18_ms_array |= ctx->key->dev.emit_agx3d;
         }
         if (scan_I->op == AGX_OPCODE_TEXTURE_SAMPLE &&
             ((scan_I->dim == AGX_DIM_2D_ARRAY &&
               ((scan_I->offset && !scan_I->offset_xyz) ||
                (scan_I->shadow &&
                 scan_I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN))) ||
              agx3_is_staged_3d_bias_min(scan_I)))
            has_packed_array_layer = true;
         /* A G17G cube-array comparison normalizes its layer into a general
          * register before publishing it beside the face; see
          * agx3_tex_normalize_cube_array_layer. */
         if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
             scan_I->op == AGX_OPCODE_TEXTURE_SAMPLE &&
             scan_I->dim == AGX_DIM_CUBE_ARRAY &&
             (scan_I->shadow || agx3_g17g_cube_gather(scan_I)))
            has_packed_array_layer = true;
         if (scan_I->op == AGX_OPCODE_ST_TILE)
            tile_stores_left++;
         if (scan_I->op == AGX_OPCODE_SIGNAL_PIX)
            g17p_final_release = scan_I;
         else if (scan_I->op == AGX_OPCODE_WAIT_PIX ||
                  scan_I->op == AGX_OPCODE_SAMPLE_MASK_AGX3 ||
                  agx3_is_cf_op(scan_I))
            g17p_final_release = NULL;
         if (scan_I->op == AGX_OPCODE_SAMPLE_MASK_AGX3 ||
             scan_I->op == AGX_OPCODE_ZS_EMIT_AGX3 ||
             scan_I->op == AGX_OPCODE_ZS_EMIT)
            has_coverage_write = true;
         if (scan_I->op == AGX_OPCODE_ZS_EMIT ||
             scan_I->op == AGX_OPCODE_ZS_EMIT_AGX3)
            has_zs_emit = true;
         if (scan_I->op == AGX_OPCODE_SIGNAL_PIX) {
            g17g_final_signal = scan_I;
            g17g_store_after_final_signal = false;
         } else if (scan_I->op == AGX_OPCODE_ST_TILE && g17g_final_signal) {
            g17g_store_after_final_signal = true;
         }
         if (scan_I->op == AGX_OPCODE_ZS_EMIT_AGX3)
            zs_staging_halfs += (scan_I->zs & 1) ? 4 : 2;
         if (scan_I->op == AGX_OPCODE_ZS_EMIT_AGX3 && (scan_I->zs & 4))
            zs_sample_loop = true;
         if (scan_I->op == AGX_OPCODE_ATOMIC ||
             scan_I->op == AGX_OPCODE_DEVICE_STORE ||
             scan_I->op == AGX_OPCODE_IMAGE_WRITE)
            has_memory_write = true;
         if (scan_I->op == AGX_OPCODE_INTERSECT_RAY_GEN1 ||
             scan_I->op == AGX_OPCODE_CONTINUE_RAY_GEN1 ||
             scan_I->op == AGX_OPCODE_INTERSECT_RAY_M5 ||
             scan_I->op == AGX_OPCODE_CONTINUE_RAY_M5)
            has_ray_traversal = true;
         if (scan_I->op == AGX_OPCODE_DEVICE_LOAD)
            device_loads_left++;
         if (scan_I->op == AGX_OPCODE_IMAGE_WRITE)
            image_writes_left++;
         if (scan_I->op == AGX_OPCODE_TEXTURE_SAMPLE) {
            texture_samples_left++;
            /* Explicit gradients carry their own staged derivatives and sit
             * outside f156 sample groups. Native AUTO/GRAD/AUTO groups the
             * two AUTO samples across the gradient without marking it. */
            if (scan_I->lod_mode != AGX_LOD_MODE_LOD_GRAD &&
                scan_I->lod_mode != AGX_LOD_MODE_LOD_GRAD_MIN) {
               sample_group_members_left++;
               g17p_quad_op_after_group = false;
               quad_consumer_after_last_member = false;
            }
         }
         switch (scan_I->op) {
         case AGX_OPCODE_DFDX:
         case AGX_OPCODE_DFDY:
         case AGX_OPCODE_QUAD_REDUCE:
         case AGX_OPCODE_QUAD_PREFIX:
         case AGX_OPCODE_QUAD_SHUFFLE:
         case AGX_OPCODE_QUAD_SHUFFLE_XOR:
         case AGX_OPCODE_QUAD_SHUFFLE_UP:
         case AGX_OPCODE_QUAD_SHUFFLE_DOWN:
         case AGX_OPCODE_QUAD_BALLOT:
         case AGX_OPCODE_FCMP_QUAD_BALLOT:
         case AGX_OPCODE_ICMP_QUAD_BALLOT:
            quad_consumer_after_last_member = true;
            break;
         default:
            break;
         }
         if (sample_group_members_left && agx3_reads_quad_lanes(scan_I))
            g17p_quad_op_after_group = true;
         if (scan_I->op == AGX_OPCODE_TEXTURE_SAMPLE &&
             (scan_I->dim == AGX_DIM_CUBE ||
              scan_I->dim == AGX_DIM_CUBE_ARRAY)) {
            /* A register level or bias outlives CEM, so it cannot share the
             * fixed immediate-corpus scratch. A cube-array layer is prepacked
             * beside it. */
            const bool bias_min =
               scan_I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN;
            const bool register_lod = bias_min ||
               ((scan_I->lod_mode == AGX_LOD_MODE_LOD_MIN ||
                 scan_I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS) &&
                scan_I->src[1].type != AGX_INDEX_IMMEDIATE);
            has_cube_sample |= !agx3_cube_own_scratch(scan_I);
            /* So is a cube face packed beside a bias and minimum. */
            if (register_lod &&
                (scan_I->dim == AGX_DIM_CUBE_ARRAY || bias_min))
               has_packed_array_layer = true;
         }
         agx_foreach_reg_dest(scan_I, d) {
            observed_max_reg = MAX2(
               observed_max_reg,
               scan_I->dest[d].value +
                  agx_index_size_16(scan_I->dest[d]) - 1);
         }
         agx_foreach_reg_src(scan_I, s) {
            observed_max_reg = MAX2(
               observed_max_reg,
               scan_I->src[s].value + agx_index_size_16(scan_I->src[s]) - 1);
         }
         /* Native keeps the fragment-input start marker clear in every
          * sampled or derivative fragment in the corpus, including explicit
          * LOD samples.  This conservatively preserves helper-capable lanes;
          * plain varying-only fragments retain the marker below. */
         if (scan_I->op == AGX_OPCODE_TEXTURE_SAMPLE ||
             (scan_I->op == AGX_OPCODE_ITRCOORDS_AGX3 &&
              scan_I->interpolation == AGX_INTERPOLATION_SAMPLE) ||
             scan_I->op == AGX_OPCODE_DFDX ||
             scan_I->op == AGX_OPCODE_DFDY)
            g17p_fragment_helper_protocol = true;
         if ((scan_I->op == AGX_OPCODE_UNIFORM_FADD_AGX3 ||
              scan_I->op == AGX_OPCODE_UNIFORM_FMUL_AGX3 ||
              scan_I->op == AGX_OPCODE_UNIFORM_FFMA_AGX3) &&
             scan_I->imm <= ARRAY_SIZE(uniform_half_used) - 2) {
            uniform_half_used[scan_I->imm] = true;
            uniform_half_used[scan_I->imm + 1] = true;
         }
         if (scan_I->op == AGX_OPCODE_UNIFORM_STORE &&
             scan_I->nr_srcs == 2 &&
             scan_I->src[1].type == AGX_INDEX_IMMEDIATE) {
            const unsigned base = scan_I->src[1].value;
            const unsigned stride = agx_size_align_16(scan_I->src[0].size);
            for (unsigned c = 0; c < 4; ++c) {
               if (!(scan_I->mask & BITFIELD_BIT(c)))
                  continue;
               for (unsigned h = 0; h < stride; ++h) {
                  if (base + c * stride + h < ARRAY_SIZE(uniform_half_used))
                     uniform_half_used[base + c * stride + h] = true;
               }
            }
         }
      }
   }

   const uint16_t entry_wait = cprog_imports ? (1u << 5) : 0;
   if (cprog_imports) {
      if (!ctx->is_preamble || !ctx->agx3_cprog_inputs_lowered ||
          (enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C) ||
          !ctx->out || !ctx->out->agx3_preamble_inputs.count)
         AGX3_PACK_FAIL();
      outstanding = entry_wait;
      for (unsigned h = 0; h < 512; ++h)
         uniform_half_used[h] |=
            !!(ctx->out->agx3_preamble_inputs.halves[h / 64] &
               (UINT64_C(1) << (h % 64)));
   }

   /* Native G17P and G17G fragment witnesses with ordinary SMPs clear f156 on
    * every earlier group member and set it only on the final member, including
    * an all-explicit-LEVEL group. Explicit gradients are outside the group.
    * Other stages keep their existing per-op encodings. */
   const bool g17p_sample_group =
                                  (enc == AGX_ISA_ENCODING_AGX3B ||
                                   enc == AGX_ISA_ENCODING_AGX3C) &&
                                  ctx->stage == MESA_SHADER_FRAGMENT &&
                                  sample_group_members_left > 1;
   /* G17G fragment witnesses terminate even a lone immediate-level sample:
    * t.sample(s, uv, level(2.0f)) carries f156 exactly like the implicit
    * forms, while the compute explicit-level witness does not. Mark the last
    * group member in G18G fragment programs; the implicit and register
    * shapes already infer the terminator on their own. */
   const bool g17g_fragment_group = enc == AGX_ISA_ENCODING_AGX3C &&
                                    ctx->key->dev.emit_agx3d &&
                                    ctx->stage == MESA_SHADER_FRAGMENT;

   /* G17G sampled fragments use unmarked coefficient inputs, then acquire
    * 512 after coordinate/LOD publication and immediately before SMP. */
   if (enc == AGX_ISA_ENCODING_AGX3C &&
       ctx->stage == MESA_SHADER_FRAGMENT && texture_samples_left)
      g17p_fragment_helper_protocol = true;

   /* CEM axis 0 writes a magnitude/face pair, axes 1 and 2 one 32-bit
    * coordinate each; the implicit face word is reserved too, or axis 1
    * overwrites it. */
   uint16_t cem_scratch[3] = {0};
   if (has_cube_sample) {
      /* M5's RA reserves a low bank so unrelated values may use the full
       * register file; older callers retain the bank above their allocation. */
      const unsigned base = ctx->agx3_cem_low_scratch
                               ? AGX3C_CEM_LOW_SCRATCH_BASE
                               : (observed_max_reg + 2) & ~1u;
      if (base + AGX3C_CEM_LOW_SCRATCH_AXIS2_OFF > 190)
         AGX3_PACK_FAIL();
      cem_scratch[0] = base;
      cem_scratch[1] = base + AGX3C_CEM_LOW_SCRATCH_AXIS1_OFF;
      cem_scratch[2] = base + AGX3C_CEM_LOW_SCRATCH_AXIS2_OFF;
      uint16_t reusable[3];
      if (agx3_reusable_cube_scratch(ctx, observed_max_reg, reusable)) {
         memcpy(cem_scratch, reusable, sizeof(reusable));
         ctx->max_reg =
            MAX2(ctx->max_reg,
                 MAX3(reusable[0] + 3, reusable[1] + 1, reusable[2] + 1));
      } else {
         /* M5's low CEM bank may sit below the current allocation. */
         ctx->max_reg = MAX2(ctx->max_reg, base + AGX3C_CEM_LOW_SCRATCH_SIZE - 1);
      }
   }

   /* MS-array reads and runtime-offset array gathers merge a normalized
    * layer with another operand. Reserve one transient word above every
    * live RA operand and CEM scratch; successive samples reuse it. */
   uint16_t ms_array_scratch = 0;
   if (has_packed_array_layer) {
      /* Array staging follows both CEM finalizers, so their first axis word
       * is dead and can hold the packed layer/LOD on M5. */
      const unsigned base = ctx->agx3_cem_low_scratch
                               ? cem_scratch[1]
                               : (MAX2(observed_max_reg, ctx->max_reg) + 2) & ~1u;
      if (base > (enc == AGX_ISA_ENCODING_AGX3C ? 254 : 190) ||
          (has_g18_ms_array && base + 3 >= 128))
         AGX3_PACK_FAIL();
      ms_array_scratch = base;
      ctx->max_reg = MAX2(ctx->max_reg, base + (has_g18_ms_array ? 3 : 1));
   }

   /* Apple's sampled G17P fragment objects use a distinct constant program
    * even when the texture and sampler operands are compact/immediate and no
    * sampler heap is involved.  Publish the semantic instruction class for
    * the linker instead of trying to reconstruct it from resource metadata. */
   if (ctx->out) {
      ctx->out->g17p_sampled_fragment |=
         enc == AGX_ISA_ENCODING_AGX3B &&
         ctx->stage == MESA_SHADER_FRAGMENT && texture_samples_left != 0;
   }

   /* The valid G17P colour-only corpus has one class-12 wait/store bracket.
    * Multi-RT shaders instead select read/write classes by sorted tile
    * partition and release each partition in descending order. */
   const struct agx3_g17p_mrt_protocol g17p_mrt =
      agx3_g17p_mrt_protocol(ctx);
   ctx->agx3_g18_mixed_protocol = g17p_mrt.valid && g17p_mrt.native_mixed;
   const bool g17p_multi_background = agx3_g17p_multi_background_protocol(ctx);
   const unsigned g17p_msaa_background_targets =
      agx3_g17p_sample_store_protocol(ctx, false);
   const bool g17p_msaa_background_stores = g17p_msaa_background_targets > 0;
   const bool g18_ms_mrt_background =
      agx3_g18_ms_mrt_background_protocol(ctx);
   const bool sample_colour_stores =
      tile_stores_left > 1 && agx3_g17p_sample_store_protocol(ctx, true);
   const bool g17p_sample_colour_stores =
      !ctx->key->dev.emit_agx3d && sample_colour_stores;
   const bool g18_sample_colour_stores =
      agx3_g18_msaa4_tile_shader(ctx) && sample_colour_stores;
   const bool g17p_colour_protocol =
      ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       ctx->stage == MESA_SHADER_FRAGMENT &&
       !ctx->key->fs.ignore_tib_dependencies &&
       (tile_stores_left == 1 || g17p_sample_colour_stores) &&
       tile_loads_left == 0) || g17p_mrt.valid;

   /* Multiple colour stores need an attachment-specific pixel transaction.
    * Emitting individual tile operations without a proven partition graph
    * produces a syntactically valid shader that writes only partition zero. */
   if ((enc == AGX_ISA_ENCODING_AGX3B || ctx->key->dev.emit_agx3d ||
        (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->fs.nr_samples <= 1)) &&
       ctx->stage == MESA_SHADER_FRAGMENT &&
       tile_stores_left > 1 && !g17p_mrt.valid && !g17p_multi_background &&
       !g17p_msaa_background_stores && !g17p_sample_colour_stores &&
       !g18_sample_colour_stores && !g18_ms_mrt_background)
      AGX3_PACK_FAIL();
   /* Exact native RGBA8 alpha pipeline: one class-8 read wait/load followed
    * by one class-4 write wait/store, under the same class-12 entry/release.
    * Keep MRT, special coordinates, other formats and control flow closed.
    */
   bool g17p_blend_protocol =
      (enc == AGX_ISA_ENCODING_AGX3B ||
       (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->m5_uniform_blend &&
        !ctx->key->secondary && !ctx->key->no_stop &&
        ctx->key->fs.nr_samples == 1)) &&
      ctx->stage == MESA_SHADER_FRAGMENT &&
      !ctx->key->fs.ignore_tib_dependencies && tile_loads_left == 1 &&
      tile_stores_left == 1;
   if (g17p_blend_protocol) {
      unsigned blocks = 0, phase = 0;
      /* An inline-blend program has the main's
       * control flow before its blend. Apple's throughput fragment is that
       * shape (a loop, then LLDCOL..LSTCOL in the exit block), so admit
       * several blocks when every pixel and tile instruction is in the
       * final block, which runs once per invocation after all control flow
       * has reconverged. */
      const bool inline_blend = !ctx->key->secondary &&
         (agx_get_fs_structure_flags() & AGX_FS_STRUCTURE_INLINE_BLEND);
      agx_block *last_block =
         list_last_entry(&ctx->blocks, agx_block, link);
      agx_foreach_block(ctx, protocol_block) {
         if (++blocks != 1 && !inline_blend)
            g17p_blend_protocol = false;
         agx_foreach_instr_in_block(protocol_block, P) {
            if (inline_blend && protocol_block != last_block &&
                (P->op == AGX_OPCODE_WAIT_PIX ||
                 P->op == AGX_OPCODE_SIGNAL_PIX ||
                 P->op == AGX_OPCODE_LD_TILE || P->op == AGX_OPCODE_ST_TILE))
               g17p_blend_protocol = false;
            if (P->op == AGX_OPCODE_WAIT_PIX) {
               if (phase == 0 && P->writeout == 8)
                  phase = 1;
               else if (phase == 2 && P->writeout == 4)
                  phase = 3;
               else
                  g17p_blend_protocol = false;
            } else if (P->op == AGX_OPCODE_LD_TILE ||
                       P->op == AGX_OPCODE_ST_TILE) {
               const bool load = P->op == AGX_OPCODE_LD_TILE;
               if (phase != (load ? 1 : 3) ||
                   !((agx3_g17p_8norm_format(P->format) &&
                      (P->mask == 3 || P->mask == 15)) ||
                     ((P->format == AGX_FORMAT_RGB10A2 ||
                       P->format == AGX_FORMAT_RGB9E5) && P->mask == 1)) ||
                    P->pixel_offset || P->explicit_coords)
                  g17p_blend_protocol = false;
               phase = load ? 2 : 4;
            } else if (P->op == AGX_OPCODE_SIGNAL_PIX) {
               g17p_blend_protocol = false;
            }
         }
      }
      g17p_blend_protocol &= phase == 4;
   }
   if (ctx->key->m5_uniform_blend && !ctx->is_preamble && !g17p_blend_protocol)
      AGX3_PACK_FAIL();
   /* A fragment program ends with a final release: the colour store's class
    * 12 with 0x200, or, with no colour store, its late depth class 1 with
    * 0x200 (pixwait(1), depthf, release); a colour store after depth
    * releases class 1 plainly. */
   if (enc != AGX_ISA_ENCODING_AGX3B || ctx->stage != MESA_SHADER_FRAGMENT ||
       ctx->key->no_stop || ctx->key->fs.ignore_tib_dependencies ||
       tile_stores_left != 0)
      g17p_final_release = NULL;
   const bool g17p_pixel_protocol =
      g17p_colour_protocol || g17p_blend_protocol || has_coverage_write;
   const bool g17p_colourless_fragment =
      (enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
      ctx->stage == MESA_SHADER_FRAGMENT &&
      !ctx->key->fs.ignore_tib_dependencies && tile_stores_left == 0 &&
      !(ctx->nir && (ctx->nir->info.outputs_written &
                     BITFIELD64_RANGE(FRAG_RESULT_DATA0, AGX_MAX_RTS)));
   const agx_instr *g17p_deferred_write =
      g17p_pixel_protocol && !g17p_mrt.valid
         ? agx3_g17p_deferred_write_wait(ctx, g17p_blend_protocol)
         : NULL;
   const agx_instr *g17g_deferred_write =
      enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
      ctx->stage == MESA_SHADER_FRAGMENT &&
      !ctx->key->fs.ignore_tib_dependencies && !g17p_mrt.valid
         ? agx3_g17g_deferred_colour_wait(ctx) : NULL;
   /* The partition a non-MRT colour transaction belongs to selects its
    * classes, exactly as the MRT protocol selects a lone store's. */
   unsigned g17p_lone_pair = 0;
   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       ctx->stage == MESA_SHADER_FRAGMENT &&
       !g17p_mrt.valid && (g17p_colour_protocol || g17p_blend_protocol) &&
       !agx3_g17p_lone_colour_pair(ctx, &g17p_lone_pair))
      AGX3_PACK_FAIL();
   /* The fused colour link template is the class-12 form only. */
   agx3_g17p_colour_link_info(ctx, g17p_colour_protocol &&
                                  g17p_deferred_write != NULL &&
                                  g17p_lone_pair == 0);
   ctx->agx3_colour_link.colour_classes =
      g17p_mrt.valid ? g17p_mrt.classes
                     : (g17p_colour_protocol || g17p_blend_protocol
                           ? BITFIELD_BIT(g17p_lone_pair)
                           : 0);
   ctx->agx3_colour_link.straight_line = list_length(&ctx->blocks) == 1;
   agx_foreach_instr_global(ctx, entry_I) {
      if (agx3_is_cf_op(entry_I) ||
          entry_I->op == AGX_OPCODE_SAMPLE_MASK_AGX3 ||
          entry_I->op == AGX_OPCODE_ZS_EMIT_AGX3)
         ctx->agx3_colour_link.straight_line = false;
   }
   const bool fragment_ray_query = has_ray_traversal &&
      (enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
      ctx->stage == MESA_SHADER_FRAGMENT;
   if (fragment_ray_query && (ctx->key->fs.nr_samples > 1 ||
       (ctx->agx3_colour_link.colour_classes & ~1u)))
      goto fail;
   bool ray_query_acquired = false;
   unsigned g17p_link_packs_seen = 0;
   unsigned g17p_mrt_stores_seen = 0;
   bool g17p_mrt_acquired = false;
   bool g17p_colour_open = false;
   unsigned texture_samples_remaining = texture_samples_left;
   unsigned sample_group_members_remaining = sample_group_members_left;
   const agx_instr *g17p_deferred_zs_wait = NULL;
   /* Depth/stencil writes in the open class-1 transaction. */
   unsigned g17p_zs_writes = 0, g17p_zs_staging = 0;
   const bool g17p_position_protocol =
      agx3_is_g17p_float2_position_vs(ctx, enc);
   unsigned g17p_positions_staged = 0;
   const unsigned g17p_output_count = agx3_g17p_vertex_output_count(ctx, enc);
   uint8_t g17p_output_order[AGX3_MAX_VERTEX_OUTPUTS] = {0};
   /* The original six-output/single-source protocol has an exact per-COPY
    * retirement contract even when its stores are terminal. Other terminal
    * six-output programs need batching to preserve distinct values through
    * RA, just like wider programs. */
   const bool g17p_legacy_single_source =
      g17p_output_count == nslots && agx3_vertex_outputs_share_source(ctx);
   const bool g17p_terminal_outputs = g17p_output_count &&
      !g17p_legacy_single_source &&
      agx3_outputs_are_terminal(ctx, g17p_output_count, g17p_output_order);
   /* The staging path has two lifetime modes: terminal bundles recycle each
    * six-slot batch, while the six-output fallback and the late-producer
    * suffix retire each staging producer before a later pure producer may
    * reuse its GPR. */
   const bool g17p_firefox_dense_outputs =
      agx3_is_firefox_dense_output_suffix(ctx, g17p_output_count);
   const bool g17p_user_output_protocol = g17p_terminal_outputs ||
      g17p_output_count == 6 ||
      (g17p_output_count >= 5 &&
       (g17p_terminal_outputs || agx3_outputs_leave_staging_alone(ctx))) ||
      (ctx->key->g17p_sparse_vertex_outputs && g17p_output_count > 0 &&
       agx3_outputs_leave_staging_alone(ctx)) ||
      g17p_firefox_dense_outputs;
   unsigned g17p_outputs_staged = 0;
   /* User-output graphs outside the qualified scalar shapes must not
    * silently fall back to immediate, potentially out-of-order writeout.
    * An unlinked layout indexes its user outputs at run time instead: no
    * output can then join a batched bundle, so every output, including the
    * fixed point size and clip distances behind them, is staged and written
    * where it is stored. */
   if (enc == AGX_ISA_ENCODING_AGX3B && ctx->stage == MESA_SHADER_VERTEX) {
      bool runtime_output_index = false;
      agx_foreach_instr_global(ctx, I) {
         runtime_output_index |= agx_is_vertex_store(I) &&
                                 I->nr_srcs >= 1 &&
                                 I->src[0].type != AGX_INDEX_IMMEDIATE;
      }
      agx_foreach_block(ctx, block) {
         agx_foreach_instr_in_block(block, I) {
            if (agx_is_vertex_store(I) && I->nr_srcs >= 1 &&
                I->src[0].type == AGX_INDEX_IMMEDIATE &&
                I->src[0].value >= 4 && !g17p_user_output_protocol &&
                !runtime_output_index)
               AGX3_REFUSE(I);
         }
      }
   }
   /* Across 71 fully decoded own-shader AGX3b vertex programs, f79 marks the
    * final LDDEV with zero exceptions and the first LDDEV uses slot 6 with
    * zero exceptions. Later slots vary with scheduling, so only the two
    * invariant choices are generalized here. UVSW hint bytes likewise vary
    * with liveness and were splice-proven inert; the byte-exact triangle path
    * above retains its native hints, while general shaders keep Mesa's. */
   const bool g17p_vertex_load_policy =
      enc == AGX_ISA_ENCODING_AGX3B && ctx->stage == MESA_SHADER_VERTEX &&
      device_loads_left > 0;
   /* Native read_write, buffer-read-to-write and sample-to-write kernels
    * assign their sole texture
    * producer to slot 1 and imgwci waits on that slot. Select slot 1 from the
    * exact one-producer/one-write compute graph; the ordinary dependency pass
    * still derives the write's wait bit from the producer result registers. */
   const bool g17p_storage_image_protocol =
      enc == AGX_ISA_ENCODING_AGX3B && ctx->stage == MESA_SHADER_COMPUTE &&
      image_writes_left == 1 && texture_ops_left == 1;
   /* G18G image writes stage from bank 0 in a kernel that reads no texture
    * or image, above every read's staging block in a program that does, and
    * above the outputs' UVSW staging in a vertex shader that does not
    * (agx3_emit_storage_image_write_g18g). Every other staging user records
    * its extent in g18g_other_staging_end, and the finished program is
    * checked against the floor below. */
   /* The captured G18G post-FS load/store kernel keeps write operands in
    * halves 0..15 and plain-image read coordinates in halves 16 and above.
    * The captured four-image M6 kernel also needs only f22=12 at a 1024-thread
    * workgroup. Keep plain reads in their existing slot-1 block and reuse a
    * single low write bank, retiring its outstanding owner before each reuse.
    * Reserving the worst-case half-80 floor wastes staging for these shapes.
    * Validate the actual read and write extents below as well.
    * This is independent of workgroup dimensions and application identity. */
   const bool g18g_compact_image_banks =
      ctx->key->dev.emit_agx3d && ctx->stage == MESA_SHADER_COMPUTE &&
      g18g_low_bank_ops && g18g_low_bank_reads &&
      image_writes_left > 0 && texture_ops_left > 0;
   /* Texture reads occupy halves below 80. Vertex outputs share that
    * staging file, so their publications and UVSW consumers need a disjoint
    * region even when exports precede samples. Leave all sampler slot bases
    * unchanged: moving them instead exceeds the sampler coordinate limit. */
   const uint16_t g18g_output_staging_base =
      ctx->key->dev.emit_agx3d && ctx->stage == MESA_SHADER_VERTEX &&
      texture_ops_left != 0
         ? MAX2(AGX3_G18G_READ_STAGING_LIMIT,
                ctx->agx3_g18g_output_staging_floor)
         : 0;
   const uint16_t g18g_image_write_floor =
      !ctx->key->dev.emit_agx3d || image_writes_left == 0 ||
      g18g_compact_image_banks ? 0
      : g18g_output_staging_base != 0
         ? g18g_output_staging_base + AGX3_G18G_OUTPUT_STAGING_LIMIT
      : texture_ops_left != 0                ? AGX3_G18G_READ_STAGING_LIMIT
      : ctx->stage == MESA_SHADER_VERTEX     ? AGX3_G18G_OUTPUT_STAGING_LIMIT
                                             : 0;
   bool g18g_image_write_placed = false;
   uint16_t g18g_other_staging_end = 0;
   uint16_t g18g_read_staging_end = 0;
   /* A G18G image read in a shader that writes no image stages its
    * coordinates at bank 0, as native run112 kick 19's read-only
    * texture2d<uint, access::read> kernel does (copy 5:0, copy 5:2, smp);
    * halves 16/18 are G17P's read/write-image bank. */
   const bool g18g_image_read_bank0 =
      ctx->key->dev.emit_agx3d && image_writes_left == 0;

   /* Assign one scoreboard timeline over the whole linearized shader, rather
    * than restarting at each basic block.  Fallthrough blocks may consume an
    * async result from their predecessor.  Control flow after an async op is
    * rejected below until CFG-aware state merging is grounded. */
   struct agx3_slot_instr *sl =
      rzalloc_array(ctx, struct agx3_slot_instr, count ? count : 1);
   struct agx3_slot_result *res =
      rzalloc_array(ctx, struct agx3_slot_result, count ? count : 1);
   agx_instr **ordered = rzalloc_array(ctx, agx_instr *, count ? count : 1);
   uint8_t *native_coord_slots =
      rzalloc_array(ctx, uint8_t, count ? count : 1);
   uint32_t *native_coord_values =
      rzalloc_array(ctx, uint32_t, count ? count : 1);
   unsigned resource_capacity = 0, resource_use_capacity = 0;
   agx_foreach_instr_global(ctx, resource_scan) {
      resource_capacity += resource_scan->agx3_tex_coord_start ||
         agx_is_texture_publication(resource_scan);
      resource_use_capacity += resource_scan->agx3_tex_coords_staged ||
         agx_texture_resource(resource_scan);
   }
   struct agx3_resource_value *resources =
      rzalloc_array(ctx, struct agx3_resource_value,
                    resource_capacity ? resource_capacity : 1);
   struct agx3_resource_use *resource_uses =
      rzalloc_array(ctx, struct agx3_resource_use,
                    resource_use_capacity ? resource_use_capacity : 1);
   uint16_t *resource_bases =
      rzalloc_array(ctx, uint16_t, resource_capacity ? resource_capacity : 1);
   uint32_t *publication_values = rzalloc_array(ctx, uint32_t,
                                               resource_capacity + 1);
   uint32_t *resource_last_use = rzalloc_array(ctx, uint32_t,
                                               resource_capacity + 1);
   if (sl == NULL || res == NULL || ordered == NULL ||
       native_coord_slots == NULL || !native_coord_values ||
       !resources || !resource_uses ||
       !resource_bases || !publication_values || !resource_last_use)
      AGX3_PACK_FAIL();

   unsigned resource_count = 0, resource_use_count = 0;
   unsigned active_coord_resource = UINT32_MAX;

   uint8_t native_coord_preferred_slot = 0;
   agx_foreach_instr_global(ctx, stage_scan) {
      if (!stage_scan->agx3_tex_coords_staged)
         continue;
      const bool register_operands =
         stage_scan->nr_srcs == 6 &&
         (stage_scan->src[3].type == AGX_INDEX_REGISTER ||
          stage_scan->src[4].type == AGX_INDEX_REGISTER);
      native_coord_preferred_slot = register_operands ? 6 : 1;
      break;
   }

   unsigned k = 0;
   unsigned vertex_loads_scheduled = 0;
   const bool batch_device_loads =
      enc == AGX_ISA_ENCODING_AGX3C && !ctx->is_preamble &&
      ctx->stage == MESA_SHADER_COMPUTE && ctx->nir && !ctx->nir->info.internal;
   agx_foreach_block(ctx, slot_block) {
      agx_foreach_instr_in_block(slot_block, I) {
         if (I->op == AGX_OPCODE_STOP)
            continue;
         ordered[k] = I;
         struct agx3_slot_instr *e = &sl[k++];
         /* EXPORT is a zero-byte register-ABI declaration after RA, not a
          * physical read. Letting it consume an async dependency marks the
          * slot waited without emitting any wait, so a linked epilog reads
          * the result too early. Real consumers retain their own waits;
          * outstanding results are synchronized at the shader-part tail. */
         if (I->op == AGX_OPCODE_EXPORT)
            continue;
         if (agx_is_texture_publication(I) && I->agx3_tex_coords_staged) {
            if (I->op != AGX_OPCODE_PUBLISH_TEXTURE_SCALAR_AGX3 ||
                I->nr_srcs != 3 || !agx_is_null(I->src[2]) ||
                active_coord_resource == UINT32_MAX || !I->imm ||
                I->imm > resource_capacity || publication_values[I->imm])
               goto fail;
            publication_values[I->imm] = active_coord_resource + 1;
            active_coord_resource = UINT32_MAX;
            continue;
         }
         e->is_async = agx3_is_async(I) || I->agx3_tex_coord_start;
         /* Native G17G loads16/loads32 issue several independent 32-bit
          * device loads on each queue before the first consumer wait.
          * Keep coherent operations and other asynchronous engines separate. */
         if (batch_device_loads &&
             I->op == AGX_OPCODE_DEVICE_LOAD &&
             I->format == AGX_FORMAT_I32 && !I->coherent &&
             !I->agx3_tex_coord_start) {
            e->join_class = 1;
            e->join_limit = 8;
         }
         /* Native query-once +0x5fc reads the traversal completion byte on
          * slot8, waiting for the intersect/continue engine on that slot. */
         if (enc == AGX_ISA_ENCODING_AGX3C && I->scs_rt_completion)
            e->preferred_slot = 8;
         e->latches_srcs = agx_is_texture_publication(I);
         /* Release ordering includes reads whose results are used after the
          * fence. Register dependencies alone do not order those reads.
          * On M5, loads still in flight when an invocation stops at a
          * workgroup barrier are retired there as well, so none is
          * outstanding while the workgroup waits. */
         e->drain = (I->op == AGX_OPCODE_SCOPED_BARRIER_AGX3 &&
                     (I->imm == AGX3_BARRIER_SCOPE_DEVICE ||
                      I->imm == AGX3_BARRIER_SCOPE_DEVICE_FENCE ||
                      agx3_m5_rendezvous_barrier(ctx, enc, I->imm))) ||
                    I->op == AGX_OPCODE_RT_BARRIER_GEN1 ||
                    I->op == AGX_OPCODE_FENCE_UVB_GEN1;
         e->reissues_live_slot = I->op == AGX_OPCODE_LDCF;
         if (I->agx3_tex_coord_start)
            e->preferred_slot = native_coord_preferred_slot;
         /* Plain 2D and 3D register-LOD fetches have two independent
          * scoreboard generations: PACK_SMPIR/COPY publish staging on slot 2
          * and TEX returns on slot 1. Native array fetches instead use slot 1
          * for their combined layer/LOD publication and the following TEX,
          * so they have no independent transient slot. */
         if (enc == AGX_ISA_ENCODING_AGX3B &&
             (I->op == AGX_OPCODE_TEXTURE_LOAD ||
              I->op == AGX_OPCODE_IMAGE_LOAD) &&
             I->lod_mode == AGX_LOD_MODE_LOD_MIN && I->nr_srcs >= 2 &&
             agx3_tex_plain_reg(I->src[1]) &&
             I->src[1].size == AGX_SIZE_32 &&
             agx_channels(I->src[1]) == 1) {
            e->preferred_slot = 1;
            if (I->dim != AGX_DIM_2D_ARRAY)
               e->transient_slots = 1u << 1;
         }
         if ((enc == AGX_ISA_ENCODING_AGX3B &&
              I->op == AGX_OPCODE_IMAGE_LOAD) ||
              /* Keep compact reads in the established slot-1 block ending
               * at half 24. Reusing this slot retires its previous generation
               * through the normal dependency allocator. */
             (g18g_compact_image_banks && I->op == AGX_OPCODE_IMAGE_LOAD) ||
             (g17p_storage_image_protocol &&
              (I->op == AGX_OPCODE_TEXTURE_SAMPLE ||
               I->op == AGX_OPCODE_TEXTURE_LOAD)))
            e->preferred_slot = 1;
         if (enc == AGX_ISA_ENCODING_AGX3B &&
             I->op == AGX_OPCODE_ZS_EMIT_AGX3)
            e->preferred_slot = 1;
         /* A returned G17P device atomic is a two-stage operation in Apple's
          * native stream: atomicdev publishes the old value in TPS half 0 on
          * slot 1, then rld waits slot 1 and publishes the architectural GPR
          * result on slot 6. Model slot 1 as an internal generation and slot
          * 6 as the logical instruction's result dependency. */
         const bool g17p_returned_atomic =
            enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_ATOMIC &&
            I->nr_dests == 1 && !agx_is_null(I->dest[0]);
         if (g17p_returned_atomic) {
            e->preferred_slot = 6;
            e->transient_slots = 1u << 0;
         }
         /* Apple's G17P programmable-blend and framebuffer-fetch corpus
          * assigns LLDCol to slot 6, including scalar, pair, and RGBA loads.
          * Keep tile reads in that native hardware domain instead of applying
          * the generic round-robin starting point. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_LD_TILE)
            e->preferred_slot = 6;
         /* Native multisample reloads keep the texture result on slot 1 and
          * publish the packed sample/layer coordinate independently on slot
          * 2. This prevents the fetch from waiting on its own generation. */
         if (enc == AGX_ISA_ENCODING_AGX3B &&
             I->op == AGX_OPCODE_TEXTURE_LOAD &&
             (I->dim == AGX_DIM_2D_MS ||
              I->dim == AGX_DIM_2D_MS_ARRAY))
            e->preferred_slot = 1;
         if (g17p_vertex_load_policy && I->op == AGX_OPCODE_DEVICE_LOAD &&
             vertex_loads_scheduled++ == 0)
            e->preferred_slot = 6;
         /* G18 CEM finalization has a three-bit transaction selector and
          * cannot publish on slot 8. Native cube sampling uses slot 1;
          * choose it before dependency assignment so scratch, result and
          * slot-reuse waits all refer to the same representable generation.
          * Fixed sampler-bank retirement already serializes these samples. */
         if (ctx->key->dev.emit_agx3d &&
             I->op == AGX_OPCODE_TEXTURE_SAMPLE && !I->query_lod &&
             (I->dim == AGX_DIM_CUBE || I->dim == AGX_DIM_CUBE_ARRAY))
            e->preferred_slot = 1;
         if (I->agx3_tex_coords_staged)
            e->preferred_slot = 1;
         /* A cube sample whose descriptors are uniforms or immediates groups
          * CEM's face finalizers with its own result slot (f12), and that
          * field only carries slots 1..7: on AGX3C a round-robin slot 8 made
          * the whole shader unencodable. Register descriptors stage on slot
          * 6 instead and are unaffected. */
         if (enc == AGX_ISA_ENCODING_AGX3C && agx3_tex_cube_groups_result_slot(I))
            e->max_slot = 7;

         unsigned nd = 0, ns = 0;
         if (!I->agx3_tex_coord_stage) {
            for (unsigned d = 0; d < I->nr_dests; ++d) {
               agx_index dest = I->dest[d];
               /* Masked samples write a dense result, not SSA padding. */
               if (enc == AGX_ISA_ENCODING_AGX3C &&
                   I->op == AGX_OPCODE_TEXTURE_SAMPLE && !I->sparse &&
                   I->gather == AGX_GATHER_NONE && !I->query_lod &&
                   d == 0 && dest.size == AGX_SIZE_32 &&
                   I->mask &&
                   !(I->mask & ~0xfu)) {
                  const unsigned channels = util_bitcount(I->mask);
                  if (channels < agx_channels(dest))
                     dest.channels_m1 = channels - 1;
               }
               nd = agx3_expand_regs(dest, e->dst, nd);
            }
            /* G17P TPS reloads (RLD: constant-program input imports and
             * static spill reloads) into the two halves of one 32-bit
             * register must not be in flight together: one half is lost.
             * A CProg input word imported as two halves, high half first,
             * lost its high half; waiting for the first reload, or importing
             * the low half first, kept it. Claim the other half too, so the
             * reload waits for a write to it in flight and later writers of
             * it wait for the reload. */
            if (enc == AGX_ISA_ENCODING_AGX3B && I->nr_dests == 1 &&
                I->dest[0].size == AGX_SIZE_16 &&
                agx_channels(I->dest[0]) == 1 &&
                agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
                (I->op == AGX_OPCODE_LOAD_CPROG_AGX3 ||
                 (I->op == AGX_OPCODE_STACK_LOAD &&
                  !agx3_dynamic_stack_load(I)))) {
               if (nd >= AGX3_SLOT_MAX_REGS)
                  AGX3_PACK_FAIL();
               e->dst[nd++] = I->dest[0].value ^ 1;
            }
         }
         for (unsigned s = 0; s < I->nr_srcs; ++s) {
             if (I->agx3_tex_coords_staged && s == 0)
               continue;
            if (agx_texture_resource(I) && (s == 0 || (s == 5 && I->shadow)))
               continue;
            const unsigned first = ns;
            ns = agx3_expand_regs(I->src[s], e->src, ns);
            /* AGX3C SMP reads a register texture/sampler descriptor after
             * issue: redefining that register while the sample is in flight
             * gave wrong texels intermittently (vertex-stage comparisons
             * with runtime offsets, 13 of 40 runs on M5 vmfix6; 0 of 40
             * with the register treated as read late). Write-after-read
             * then waits for the sample's slot before the redefinition. */
            if (enc == AGX_ISA_ENCODING_AGX3C &&
                I->op == AGX_OPCODE_TEXTURE_SAMPLE && (s == 3 || s == 4) &&
                I->src[s].type == AGX_INDEX_REGISTER) {
               for (unsigned t = first; t < ns && t < 32; ++t)
                  e->late_src_mask |= UINT32_C(1) << t;
            }
         }
         if (I->op == AGX_OPCODE_LOAD_CPROG_AGX3) {
            if (ns >= AGX3_SLOT_MAX_REGS)
               AGX3_PACK_FAIL();
            e->src[ns++] = AGX3_SLOT_CPROG_INPUT_TOKEN;
         }
         if (nd > AGX3_SLOT_MAX_REGS || ns > AGX3_SLOT_MAX_REGS) {
            agx3_report_refusal(I, __FILE__, __LINE__);
            AGX3_PACK_FAIL();
         }
         if (I->agx3_tex_coord_start) {
            if (active_coord_resource != UINT32_MAX)
               AGX3_PACK_FAIL();
            active_coord_resource = resource_count++;
            /* Both scalar publications join one native completion event.
             * Represent their coordinate pair as a real resource value,
             * independent of the discarded GPR destinations.
             */
            resources[active_coord_resource] = (struct agx3_resource_value){
               .producer = k - 1, .halves = 4, .alignment = 4,
               .fixed_base = I->op == AGX_OPCODE_FFMA ? UINT16_MAX : 0,
            };
         }
         if (I->agx3_tex_coord_stage) {
            if (active_coord_resource == UINT32_MAX)
               goto fail;
            native_coord_values[k - 1] = active_coord_resource + 1;
         }
         if (I->agx3_tex_coords_staged) {
            if (active_coord_resource == UINT32_MAX)
               AGX3_PACK_FAIL();
            resource_uses[resource_use_count++] = (struct agx3_resource_use){
               .instruction = k - 1, .value = active_coord_resource,
               .first = 0, .count = 4, .through_completion = true,
            };
            active_coord_resource = UINT32_MAX;
         }
         if (agx_is_texture_publication(I)) {
            const bool scalar = I->op == AGX_OPCODE_PUBLISH_TEXTURE_SCALAR_AGX3;
            if (!I->imm || I->imm > resource_capacity ||
                publication_values[I->imm] || I->nr_srcs != (scalar ? 3 : 2))
               AGX3_PACK_FAIL();
            const unsigned value = resource_count++;
            publication_values[I->imm] = value + 1;
            resources[value] = (struct agx3_resource_value){
               .producer = k - 1,
               .halves = agx_is_null(I->src[scalar ? 2 : 1]) ? 4 : 6,
               .alignment = enc == AGX_ISA_ENCODING_AGX3C ? 8 : 4,
               .fixed_base = UINT16_MAX,
            };
         }
         if (agx_texture_resource(I)) {
            const unsigned id = agx_texture_resource(I);
            if (I->op != AGX_OPCODE_TEXTURE_SAMPLE || id > resource_capacity ||
                !publication_values[id])
               AGX3_PACK_FAIL();
            const unsigned value = publication_values[id] - 1;
            resource_uses[resource_use_count++] = (struct agx3_resource_use){
               .instruction = k - 1, .value = value,
               .count = agx_index_size_16(I->src[0]), .through_completion = true,
            };
            resource_last_use[value] = k - 1;
         }

         /* rld's asynchronous read of the hidden atomic-return words must
          * complete before a later stack access may reuse them. */
         if (g17p_returned_atomic) {
            const unsigned tps_half = ctx->agx3_atomic_return_tps_half;
            assert(nd + 2 <= AGX3_SLOT_MAX_REGS);
            assert(tps_half + 1 < AGX3_SLOT_TPS_HALF_TOKENS);
            e->dst[nd++] = AGX3_SLOT_TPS_TOKEN_BASE + tps_half;
            e->dst[nd++] = AGX3_SLOT_TPS_TOKEN_BASE + tps_half + 1;
         }

         /* CEM is a compound cube-coordinate pipeline, not a single opaque
          * sampler operation. Its three calculations write the dedicated
          * CEM scratch bank before SMP consumes the staged result.
          * Keep those writes owned by the sample's scoreboard generation so
          * a following cube pipeline cannot clobber them while the first SMP
          * is still in flight.  Apple's two-cube G17P fragment does the
          * equivalent by completing both CEM pipelines before issuing either
          * SMP, with separate staging banks for the two results.
          *
          * A four-channel f32 texture result expands to eight tokens, leaving
          * room for the four scratch words, including axis0's implicit face. */
         if ((enc == AGX_ISA_ENCODING_AGX3B ||
              enc == AGX_ISA_ENCODING_AGX3C) &&
             I->op == AGX_OPCODE_TEXTURE_SAMPLE &&
             (I->dim == AGX_DIM_CUBE ||
              I->dim == AGX_DIM_CUBE_ARRAY)) {
            const bool bias_min =
               I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN;
            const bool register_lod = bias_min ||
               ((I->lod_mode == AGX_LOD_MODE_LOD_MIN ||
                 I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS) &&
                I->src[1].type != AGX_INDEX_IMMEDIATE);
            uint16_t own_scratch[3];
            const uint16_t *scratch =
               agx3_cube_scratch(I, cem_scratch, own_scratch);
            assert(nd + 5 <= AGX3_SLOT_MAX_REGS);
            e->dst[nd++] = scratch[0];
            e->dst[nd++] = scratch[0] + 2;
            e->dst[nd++] = scratch[1];
            e->dst[nd++] = scratch[2];
            /* The cube-array layer prepacked beside a register LOD, or the
             * cube face packed beside a bias and minimum. */
            if (register_lod && (I->dim == AGX_DIM_CUBE_ARRAY || bias_min))
               e->dst[nd++] = ms_array_scratch;
         }
         /* A 2D-array comparison and a staged 3D bias pack their minimum
          * LOD into the same word, which SMP reads after its staging
          * completes. */
         if (enc == AGX_ISA_ENCODING_AGX3B &&
             I->op == AGX_OPCODE_TEXTURE_SAMPLE &&
             ((I->dim == AGX_DIM_2D_ARRAY && I->shadow &&
               I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN) ||
              agx3_is_staged_3d_bias_min(I))) {
            assert(nd < AGX3_SLOT_MAX_REGS);
            e->dst[nd++] = ms_array_scratch;
         }
         if (ctx->key->dev.emit_agx3d &&
             (I->op == AGX_OPCODE_TEXTURE_LOAD || I->op == AGX_OPCODE_IMAGE_LOAD) &&
             I->dim == AGX_DIM_2D_MS_ARRAY) {
            assert(nd + 2 <= AGX3_SLOT_MAX_REGS);
            e->dst[nd++] = ms_array_scratch;
            e->dst[nd++] = ms_array_scratch + 2;
         }
         e->nr_dsts = (uint8_t)nd;
         e->nr_srcs = (uint8_t)ns;
         if ((I->op == AGX_OPCODE_STACK_LOAD ||
              I->op == AGX_OPCODE_STACK_STORE) &&
             !agx3_append_static_spill_tokens(I, e, enc)) {
            /* Report like every other refusal.  Without this the whole
             * shader fails with no diagnostic at all, which is how a
             * scratch-lowered constant array once looked like a mystery
             * VK_ERROR_FEATURE_NOT_PRESENT from vkCreateComputePipelines. */
            agx3_report_refusal(I, __FILE__, __LINE__);
            AGX3_PACK_FAIL();
         }
      }
   }
   assert(k == count);

   /* A common descriptor diamond loads two pointers, dereferences both, then
    * selects between their values. The two value loads must occupy the native
    * slots-1-and-2 pair: an ISEL waiting on two other live slot generations
    * is not a grounded hardware form and intermittently consumes one result
    * early. Reuse slot 3 for the serialized pointer loads, whose generations
    * are retired by their respective dereferences, while the two value loads
    * remain overlapped on slots 2 and 1. */
   if (enc == AGX_ISA_ENCODING_AGX3B) {
      for (unsigned i = 0; i + 4 < count; ++i) {
         if (ordered[i]->op != AGX_OPCODE_DEVICE_LOAD ||
             ordered[i + 1]->op != AGX_OPCODE_DEVICE_LOAD ||
             ordered[i + 2]->op != AGX_OPCODE_DEVICE_LOAD ||
             ordered[i + 3]->op != AGX_OPCODE_DEVICE_LOAD ||
             ordered[i + 4]->op != AGX_OPCODE_ICMPSEL)
            continue;

         bool available = true;
         for (unsigned j = 0; j < 4; ++j) {
            available &= sl[i + j].preferred_slot == 0 &&
                         sl[i + j].transient_slots == 0;
         }
         if (!available ||
             !agx3_slot_instr_reads(&sl[i + 1], &sl[i]) ||
             !agx3_slot_instr_reads(&sl[i + 3], &sl[i + 2]) ||
             !agx3_slot_instr_reads(&sl[i + 4], &sl[i + 1]) ||
             !agx3_slot_instr_reads(&sl[i + 4], &sl[i + 3]))
            continue;

         sl[i].preferred_slot = 3;
         sl[i + 1].preferred_slot = 2;
         sl[i + 2].preferred_slot = 3;
         sl[i + 3].preferred_slot = 1;
      }
   }

   /* Keep adjacent async producers feeding ISEL overlapped on slots 2 and 1.
    * Emission retires that pair through a data-dependent ALU carrier before
    * ISEL, avoiding both a multi-slot ISEL wait and concurrent reuse of one
    * scoreboard slot. Producers with a compound internal slot contract
    * already have a preferred/transient assignment and remain outside this
    * rule. */
   if (enc == AGX_ISA_ENCODING_AGX3B) {
      for (unsigned i = 0; i + 2 < count; ++i) {
         if (!sl[i].is_async || !sl[i + 1].is_async ||
             ordered[i + 2]->op != AGX_OPCODE_ICMPSEL ||
             sl[i].preferred_slot || sl[i + 1].preferred_slot ||
             sl[i].transient_slots || sl[i + 1].transient_slots)
            continue;

         if (agx3_slot_instr_reads(&sl[i + 2], &sl[i]) &&
             agx3_slot_instr_reads(&sl[i + 2], &sl[i + 1])) {
            const bool device_pair = ordered[i]->op == AGX_OPCODE_DEVICE_LOAD &&
               ordered[i + 1]->op == AGX_OPCODE_DEVICE_LOAD;
            const bool texture_then_device =
               (ordered[i]->op == AGX_OPCODE_TEXTURE_SAMPLE ||
                ordered[i]->op == AGX_OPCODE_TEXTURE_LOAD ||
                ordered[i]->op == AGX_OPCODE_IMAGE_LOAD) &&
               ordered[i + 1]->op == AGX_OPCODE_DEVICE_LOAD;
            if (device_pair || texture_then_device) {
               sl[i].preferred_slot = 2;
               sl[i + 1].preferred_slot = 1;
            }
         }
      }
   }

   const bool gen1_mesh = enc == AGX_ISA_ENCODING_AGX3B &&
      ctx->key->dev.chip == AGX_CHIP_G17P &&
      ctx->stage == MESA_SHADER_MESH && !ctx->is_preamble;
   if (gen1_mesh) {
      bool exports_pending = false;
      for (unsigned i = 0; i < count; ++i) {
         if (ordered[i]->op == AGX_OPCODE_FENCE_UVB_GEN1)
            exports_pending = false;
         else if (ordered[i]->op == AGX_OPCODE_STORE_UVB_M5)
            exports_pending = true;
         else if (exports_pending && sl[i].is_async &&
                  !sl[i].preferred_slot && !sl[i].transient_slots &&
                  !sl[i].reissues_live_slot && !sl[i].join_class) {
            /* Unlinked mesh interfaces load UVS selectors from the root
             * table between vertex exports. Restrict every ordinary async
             * producer in this region, not just loads: a restricted load
             * changes the shared cursor and could otherwise move a later
             * system-value read onto an observed phase channel.
             * Slot 1 is outside both phase channels. Keep it an ordinary
             * allocation so reuse retires a live generation; retain normal
             * RAW/WAW waits and all emitter guards. Required compound,
             * preferred, reissuable and joined protocols stay unchanged.
             * This does not assign an implicit UVST producer or lifetime. */
            sl[i].max_slot = 1;
         }
      }
   }

   const struct agx3_slot_entry input_entry = {
      .token = AGX3_SLOT_CPROG_INPUT_TOKEN, .slot = 6,
   };
   if (active_coord_resource != UINT32_MAX ||
       (resource_count &&
        !agx3_resource_lower(resources, resource_count, resource_uses,
                              resource_use_count, sl, count,
                              AGX3_SLOT_RESOURCE_HALVES, resource_bases, sl)))
      AGX3_PACK_FAIL();
   if (!agx3_slot_assign_with_entry(sl, count, nslots, &input_entry,
                                    cprog_imports ? 1 : 0, res))
      AGX3_PACK_FAIL();

   /* Emission drains every live scoreboard generation before control flow,
    * but the linear slot assignment above cannot see that architectural
    * boundary. Remove waits belonging to those retired generations until a
    * later producer reuses the slot. Otherwise a post-branch ISEL may carry a
    * stale slot together with two real producers, escaping the native
    * slots-1-and-2 pair even though only that pair is live. */
   uint16_t cf_retired = 0, cf_live = entry_wait;
   for (unsigned i = 0; i < count; ++i) {
      res[i].wait_mask &= (uint16_t)~cf_retired;
      cf_live &= (uint16_t)~res[i].wait_mask;

      if (res[i].slot) {
         const uint16_t bit = (uint16_t)(1u << (res[i].slot - 1));
         cf_retired &= (uint16_t)~bit;
         cf_live |= bit;
      }

      const bool live_begin = enc == AGX_ISA_ENCODING_AGX3C &&
         ordered[i]->op == AGX_OPCODE_BEGIN_CF;
      if (agx3_is_cf_op(ordered[i]) && !live_begin) {
         cf_retired |= cf_live;
         cf_live = 0;
      }
   }

   uint8_t active_coord_slot = 0;
   unsigned staged_components = 0;
   for (unsigned i = 0; i < count; ++i) {
      const agx_instr *I = ordered[i];
      if (I->agx3_tex_coord_start) {
         if (active_coord_slot || !res[i].slot)
            AGX3_PACK_FAIL();
         active_coord_slot = res[i].slot;
      }
      if (I->agx3_tex_coord_stage) {
         if (!active_coord_slot)
            AGX3_PACK_FAIL();
         native_coord_slots[i] = active_coord_slot;
         staged_components |= BITFIELD_BIT(I->agx3_tex_coord_component);
      }
      if (I->agx3_tex_coords_staged) {
         if (!active_coord_slot || staged_components != 3)
            AGX3_PACK_FAIL();
         native_coord_slots[i] = active_coord_slot;
         active_coord_slot = 0;
         staged_components = 0;
      }
   }
   if (active_coord_slot)
      AGX3_PACK_FAIL();

   /* G17G rotates each staged plain-2D sample's operand block by scoreboard
    * slot (agx3_emit_texture_sample). A per-lane level or bias landing in a
    * rotated block is packed through Apple's register relay first: reserve
    * one transient word above every live RA operand and the other scratch,
    * reused by successive samples, only when some such sample left slot 1.
    * The relay is a seven-bit COPY source and PACK_SMPIR destination. */
   uint16_t g17g_lod_relay = 0;
   if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d) {
      bool relay_needed = false;
      for (unsigned i = 0; i < count; ++i)
         relay_needed |= agx3_g17g_staged_block(ordered[i], ctx->key->dev.emit_agx3d) &&
                         agx3_g17g_register_lod(ordered[i]) && res[i].slot > 1;
      if (relay_needed) {
         const unsigned base =
            (MAX2(observed_max_reg, ctx->max_reg) + 2) & ~1u;
         if (base >= 128)
            goto fail;
         g17g_lod_relay = base;
         ctx->max_reg = MAX2(ctx->max_reg, base + 1);
      }
   }

   const bool *tps_rereads = agx3_static_reload_rereads(ctx, ordered, count);
   if (!tps_rereads)
      AGX3_PACK_FAIL();

   k = 0;
   const bool interleaved_zs =
      agx3_g17p_interleaved_zs(ctx) && enc == AGX_ISA_ENCODING_AGX3B &&
      ctx->stage == MESA_SHADER_FRAGMENT && has_zs_emit && g17p_mrt.valid &&
      g17p_mrt.multisample_combined &&
      (g17p_mrt.combined_classes == 12 || g17p_mrt.combined_classes == 48 ||
       g17p_mrt.combined_classes == 60 || g17p_mrt.combined_classes == 192 ||
       g17p_mrt.combined_classes == 252);
   unsigned vertex_loads_seen = 0;
   unsigned loop_depth = 0;
   /* f63 on F32INTERP and f48 on LDCF are two encodings of the same native
    * fragment-input entry marker. Apple's mixed flat+smooth fragment marks
    * whichever family comes first and leaves the other family's first op
    * unmarked; restarting here corrupts the coefficient-fetch protocol. */
   bool fragment_input_started = false;
   struct agx3_staging_banks staging_banks = {0};
   /* Samples stage fixed halves 0..19. Per-sample DEPTHF exports reserve
    * successive four-half payloads (two for stencil-only), not just the
    * first sample's 0..3. Image writes must not claim any of those halves,
    * including payloads that later unrolled samples have yet to publish. */
   const uint16_t fixed_staging_floor =
      texture_samples_left ? 24
      : (ctx->stage == MESA_SHADER_FRAGMENT &&
         (has_coverage_write || has_zs_emit)) ? 8
                                              : 0;
   const uint16_t image_write_bank_floor =
      agx_is_g18g(&ctx->key->dev)
         ? MAX2(fixed_staging_floor, ALIGN_POT(zs_staging_halfs, 8))
         : ALIGN_POT(MAX2(fixed_staging_floor, zs_staging_halfs),
                     enc == AGX_ISA_ENCODING_AGX3C ? 16 : 8);
   if ((has_coverage_write ||
        ((has_memory_write || has_ray_traversal) && !ctx->key->secondary)) &&
       (enc == AGX_ISA_ENCODING_AGX3B ||
        (agx_is_g18g(&ctx->key->dev) && has_memory_write &&
         !ctx->key->secondary)) &&
       ctx->stage == MESA_SHADER_FRAGMENT &&
       !ctx->key->fs.ignore_tib_dependencies) {
      /* A discarding or side-effecting fragment enters the pixel
       * transaction before its first device load or memory side effect
       * (stores and atomics alike), then uses unmarked SR/RHW inputs; the
       * input protocol is not restarted after MOVMSK. This is also the
       * native G17G atomic_color path: acquire 512 before the atomic and use
       * an unmarked first RHW input. agx_fast_link removes the epilog's gate. */
      /* G18 memory writers retain this main-part entry; coverage-only
       * entry handling remains separate. */
      uint8_t entry[AGX3_PIXEL_MAX_LENGTH], entry_len;
      if (!agx3_pixel_colour_entry(enc, entry, &entry_len) ||
          !agx3_append(emission, entry, entry_len))
         AGX3_PACK_FAIL();
      ctx->agx3_colour_link.entry_emitted = true;
      ctx->agx3_colour_link.entry_offset = emission->size - entry_len - initial_size;
      g17p_colour_open = true;
      fragment_input_started = true;
   }

   if (interleaved_zs) {
      uint8_t bytes[AGX3_PIXEL_MAX_LENGTH], len;
      if (!agx3_pixel_wait(enc, g17p_mrt.combined_classes | 1, 0, bytes,
                           &len) ||
          !agx3_append(emission, bytes, len))
         goto fail;
      g17p_mrt_acquired = true;
   }

   /* Native G17G fragments enter their pixel transaction once, in uniform
    * control flow ahead of the first kill (native G17G compiler discard and
    * alpha-to-coverage fragments: pixwait 512, then each kill in its own
    * cndst). When the first coverage update is nested, open the entry before
    * the top-level IF that leads to it, so no path skips it; agx_fast_link
    * relies on this part's entry having run when it removes a later part's.
    */
   const agx_instr *m5_coverage_entry_before = NULL;
   if (enc == AGX_ISA_ENCODING_AGX3C && ctx->stage == MESA_SHADER_FRAGMENT &&
       has_coverage_write) {
      int depth = 0;
      const agx_instr *top_if = NULL;
      bool done = false;
      agx_foreach_block(ctx, scan_block) {
         agx_foreach_instr_in_block(scan_block, scan_I) {
            if (done)
               break;
            if (scan_I->op == AGX_OPCODE_IF_ICMP ||
                scan_I->op == AGX_OPCODE_IF_FCMP) {
               if (depth == 0)
                  top_if = scan_I;
               depth += scan_I->nest;
            } else if (scan_I->op == AGX_OPCODE_POP_EXEC) {
               depth = MAX2(depth - (int)scan_I->nest, 0);
            } else if (scan_I->op == AGX_OPCODE_WHILE_ICMP ||
                       scan_I->op == AGX_OPCODE_WHILE_FCMP ||
                       scan_I->op == AGX_OPCODE_BREAK_IF_ICMP ||
                       scan_I->op == AGX_OPCODE_BREAK_IF_FCMP) {
               done = true; /* loops keep the lazy entry */
            } else if (scan_I->op == AGX_OPCODE_SAMPLE_MASK_AGX3 ||
                       scan_I->op == AGX_OPCODE_ZS_EMIT_AGX3) {
               if (depth > 0)
                  m5_coverage_entry_before = top_if;
               done = true;
            }
         }
      }
   }

   /* Native G17G side-effecting fragments read their initial coefficients
    * without the implicit entry marker, then acquire before the conditional
    * side effect. Preserve the coverage read preceding that acquisition. */
   const bool m5_sidefx_entry = enc == AGX_ISA_ENCODING_AGX3C &&
      ctx->stage == MESA_SHADER_FRAGMENT && has_memory_write &&
      !ctx->key->secondary && !ctx->key->fs.ignore_tib_dependencies;
   fragment_input_started |= m5_sidefx_entry;

   /* Native1094 fbfetch (Apple M6, verified): every 2x/4x tile-reading
    * fragment (advanced multiply at 2x and 4x, softlight and the no-rate
    * fetch at 4x) opens its pixel transaction before its first tile read --
    * pixwait512 (073000050000), then pixwait8 (474000050000) and the first
    * lldcol -- and holds it to its single final pixrel524 (671000040000),
    * so every sample's read, blend and write is one ordered transaction;
    * its first interpolation keeps the input marker. Here the main part
    * reads the samples and the epilog writes them, so the main takes the
    * entry at its start and agx_fast_link drops the epilog's own gate.
    * Apple interleaves each sample's store with the next sample's load
    * under pixwait12; Mesa reads every sample, then writes every sample,
    * inside the same transaction. */
   if (!g17p_colour_open &&
       enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
       ctx->stage == MESA_SHADER_FRAGMENT && !ctx->key->secondary &&
       !ctx->key->fs.ignore_tib_dependencies &&
       (ctx->key->fs.nr_samples == 2 || ctx->key->fs.nr_samples == 4) &&
       tile_loads_left && !tile_stores_left) {
      uint8_t entry[AGX3_PIXEL_MAX_LENGTH], entry_len;
      if (!agx3_pixel_colour_entry(enc, entry, &entry_len) ||
          !agx3_append(emission, entry, entry_len))
         goto fail;
      ctx->agx3_colour_link.entry_emitted = true;
      ctx->agx3_colour_link.entry_offset = emission->size - entry_len - initial_size;
      g17p_colour_open = true;
   }


   if (g18_ms_mrt_background) {
      uint8_t acquire[AGX3_PIXEL_MAX_LENGTH], acquire_len = 0;
      /* Acquire before texture reads, as in the complete native reload.
       * That background body has no colour-entry or release instruction. */
      if (!agx3_pixel_wait(enc, 252, 0x7f, acquire, &acquire_len) ||
          !agx3_append(emission, acquire, acquire_len))
         goto fail;
   }

   /* The previous instruction's asynchronous producer of r0, which joins
    * agx3_r0_inflight once that instruction has been emitted. */
   uint16_t r0_issued = 0;

   bool mesh_allocated = false, mesh_indices_fenced = false;
   bool mesh_exports_pending = false;
   unsigned mesh_vertex_fences = 0;

   struct agx3_pack_state st = {
      .ctx = ctx,
      .emission = emission,
      .enc = enc,
      .initial_size = initial_size,
      .nslots = nslots,
      .sl = sl,
      .res = res,
      .ordered = ordered,
      .native_coord_slots = native_coord_slots,
      .native_coord_values = native_coord_values,
      .resources = resources,
      .resource_uses = resource_uses,
      .resource_bases = resource_bases,
      .publication_values = publication_values,
      .resource_last_use = resource_last_use,
      .tps_rereads = tps_rereads,
      .count = count,
      .resource_capacity = resource_capacity,
      .fixups = &fixups,
      .spans = &spans,
      .compares = &compares,
      .early_compare = early_compare,
      .outstanding = outstanding,
      .image_write_slots = image_write_slots,
      .staging_banks = staging_banks,
      .next_uvs_slot = next_uvs_slot,
      .fixed_staging_floor = fixed_staging_floor,
      .image_write_bank_floor = image_write_bank_floor,
      .loop_depth = loop_depth,
      .texture_ops_left = texture_ops_left,
      .image_writes_left = image_writes_left,
      .texture_samples_left = texture_samples_left,
      .texture_samples_remaining = texture_samples_remaining,
      .sample_group_members_left = sample_group_members_left,
      .sample_group_members_remaining = sample_group_members_remaining,
      .g17p_quad_op_after_group = g17p_quad_op_after_group,
      .quad_consumer_after_last_member = quad_consumer_after_last_member,
      .g17p_sample_group = g17p_sample_group,
      .g17g_fragment_group = g17g_fragment_group,
      .g17p_storage_image_protocol = g17p_storage_image_protocol,
      .g18g_compact_image_banks = g18g_compact_image_banks,
      .g18g_image_write_placed = g18g_image_write_placed,
      .g18g_image_read_bank0 = g18g_image_read_bank0,
      .g18g_output_staging_base = g18g_output_staging_base,
      .g18g_image_write_floor = g18g_image_write_floor,
      .g18g_other_staging_end = g18g_other_staging_end,
      .g18g_read_staging_end = g18g_read_staging_end,
      .cem_scratch = {cem_scratch[0], cem_scratch[1], cem_scratch[2]},
      .ms_array_scratch = ms_array_scratch,
      .g17g_lod_relay = g17g_lod_relay,
      .tile_loads_left = tile_loads_left,
      .tile_loads_seen = tile_loads_seen,
      .tile_stores_left = tile_stores_left,
      .has_coverage_write = has_coverage_write,
      .has_memory_write = has_memory_write,
      .has_ray_traversal = has_ray_traversal,
      .has_zs_emit = has_zs_emit,
      .zs_sample_loop = zs_sample_loop,
      .g17p_fragment_helper_protocol = g17p_fragment_helper_protocol,
      .fragment_input_started = fragment_input_started,
      .interleaved_zs = interleaved_zs,
      .g17p_multi_background = g17p_multi_background,
      .g17p_msaa_background_targets = g17p_msaa_background_targets,
      .g17p_msaa_background_stores = g17p_msaa_background_stores,
      .g18_ms_mrt_background = g18_ms_mrt_background,
      .g17p_sample_colour_stores = g17p_sample_colour_stores,
      .g18_sample_colour_stores = g18_sample_colour_stores,
      .g17p_colour_protocol = g17p_colour_protocol,
      .g17p_blend_protocol = g17p_blend_protocol,
      .g17p_pixel_protocol = g17p_pixel_protocol,
      .g17p_colourless_fragment = g17p_colourless_fragment,
      .fragment_ray_query = fragment_ray_query,
      .ray_query_acquired = ray_query_acquired,
      .g17p_mrt_acquired = g17p_mrt_acquired,
      .g17p_colour_open = g17p_colour_open,
      .m5_sidefx_entry = m5_sidefx_entry,
      .g17p_lone_pair = g17p_lone_pair,
      .g17p_link_packs_seen = g17p_link_packs_seen,
      .g17p_mrt_stores_seen = g17p_mrt_stores_seen,
      .g17p_zs_writes = g17p_zs_writes,
      .g17p_zs_staging = g17p_zs_staging,
      .g17p_final_release = g17p_final_release,
      .g17g_final_signal = g17g_final_signal,
      .g17g_store_after_final_signal = g17g_store_after_final_signal,
      .g17p_deferred_write = g17p_deferred_write,
      .g17g_deferred_write = g17g_deferred_write,
      .g17p_deferred_zs_wait = g17p_deferred_zs_wait,
      .m5_coverage_entry_before = m5_coverage_entry_before,
      .g17p_mrt = g17p_mrt,
      .device_loads_left = device_loads_left,
      .vertex_loads_seen = vertex_loads_seen,
      .g17p_position_protocol = g17p_position_protocol,
      .g17p_legacy_single_source = g17p_legacy_single_source,
      .g17p_terminal_outputs = g17p_terminal_outputs,
      .g17p_firefox_dense_outputs = g17p_firefox_dense_outputs,
      .g17p_user_output_protocol = g17p_user_output_protocol,
      .g17p_vertex_load_policy = g17p_vertex_load_policy,
      .g17p_positions_staged = g17p_positions_staged,
      .g17p_output_count = g17p_output_count,
      .g17p_outputs_staged = g17p_outputs_staged,
      .gen1_mesh = gen1_mesh,
      .mesh_allocated = mesh_allocated,
      .mesh_indices_fenced = mesh_indices_fenced,
      .mesh_exports_pending = mesh_exports_pending,
      .mesh_vertex_fences = mesh_vertex_fences,
      .cprog_imports = cprog_imports,
   };
   memcpy(st.g17p_output_order, g17p_output_order, sizeof(st.g17p_output_order));
   memcpy(st.uniform_half_used, uniform_half_used, sizeof(st.uniform_half_used));

   agx_foreach_block(ctx, block) {
      block->offset = emission->size;
      block->loop_exit_offset = -1;
      st.block = block;
      st.after_loop_backedge = false;
      st.after_loop_exit_test = false;
      st.fetch_group = (struct agx3_fetch_group){0};
      st.fixed_gradient_staging = 0;
      if (enc == AGX_ISA_ENCODING_AGX3C && ctx->stage == MESA_SHADER_FRAGMENT) {
         agx_foreach_instr_in_block(block, J) {
            if (J->op == AGX_OPCODE_TEXTURE_SAMPLE && !agx_texture_resource(J) &&
                J->lod_mode == AGX_LOD_MODE_LOD_GRAD) {
               st.fixed_gradient_staging = MAX2(st.fixed_gradient_staging,
                  (J->dim == AGX_DIM_3D ? AGX3_TEX_GRADIENT_BANK_DDY_Z
                                        : AGX3_TEX_GRADIENT_BANK_DDY_Y) + 2);
            }
         }
      }
      agx_foreach_instr_in_block(block, I) {
         block->last_offset = emission->size;

         if (I->op == AGX_OPCODE_STOP)
            continue;

         const unsigned instr_index = k;
         const uint8_t slot = res[k].slot;
         const uint16_t wait = res[k].wait_mask;
         k++;
         const off_t span_start = emission->size;
         if (early_compare) {
            const struct agx3_instr_span span = {
               .I = I, .block = block, .start = span_start,
               .wait = res[instr_index].wait_mask,
               .slot = res[instr_index].slot,
            };
            util_dynarray_append(&spans, span);
         }

         /* A wait retires its slots before this instruction issues its own. */
         st.outstanding &= (uint16_t)~wait;
         agx3_carrier_update_r0(r0_issued, wait);
         r0_issued = 0;
         if (agx3_carrier_is_1140() && slot) {
            for (unsigned d = 0; d < sl[instr_index].nr_dsts; ++d) {
               if (sl[instr_index].dst[d] < 2)
                  r0_issued = (uint16_t)(1u << (slot - 1));
            }
         }
         st.outstanding_here = st.outstanding;
         st.image_write_slots &= st.outstanding_here;
         if (slot)
            st.image_write_slots &= (uint16_t)~(1u << (slot - 1));
         if (slot)
            st.outstanding |= (uint16_t)(1u << (slot - 1));

         st.I = I;
         st.instr_index = instr_index;
         st.slot = slot;
         st.wait = wait;
         st.native_coord_slot = native_coord_slots[instr_index];
         st.span_start = span_start;

         if (st.gen1_mesh && st.mesh_exports_pending && slot &&
             (1u << (slot - 1)) == (st.mesh_indices_fenced ? 4u : 2u))
            AGX3_REFUSE(I);

         enum agx3_pack_result r;
         if ((r = agx3_pack_mesh_fence_instr(&st)) != AGX3_PACK_UNHANDLED) {
            if (r == AGX3_PACK_ERROR)
               goto fail;
            continue;
         }
         if ((r = agx3_pack_matrix_instr(&st)) != AGX3_PACK_UNHANDLED) {
            if (r == AGX3_PACK_ERROR)
               goto fail;
            continue;
         }
         if ((r = agx3_pack_cf_instr(&st)) != AGX3_PACK_UNHANDLED) {
            if (r == AGX3_PACK_ERROR)
               goto fail;
            continue;
         }
         if ((r = agx3_pack_tex_pre_instr(&st)) != AGX3_PACK_UNHANDLED) {
            if (r == AGX3_PACK_ERROR)
               goto fail;
            continue;
         }
         if ((r = agx3_pack_uniform_float_instr(&st)) != AGX3_PACK_UNHANDLED) {
            if (r == AGX3_PACK_ERROR)
               goto fail;
            continue;
         }
         if (I->op == AGX_OPCODE_EXPORT)
            continue;
         if (I->op == AGX_OPCODE_NO_VARYINGS) {
            if (ctx->stage != MESA_SHADER_VERTEX)
               AGX3_REFUSE(I);
            continue;
         }
         if ((r = agx3_pack_tex_pub_instr(&st)) != AGX3_PACK_UNHANDLED) {
            if (r == AGX3_PACK_ERROR)
               goto fail;
            continue;
         }
         if ((r = agx3_pack_vary_uvsw_instr(&st)) != AGX3_PACK_UNHANDLED) {
            if (r == AGX3_PACK_ERROR)
               goto fail;
            continue;
         }
         if ((r = agx3_pack_tex_instr(&st)) != AGX3_PACK_UNHANDLED) {
            if (r == AGX3_PACK_ERROR)
               goto fail;
            continue;
         }
         if ((r = agx3_pack_frag_pixel_instr(&st)) != AGX3_PACK_UNHANDLED) {
            if (r == AGX3_PACK_ERROR)
               goto fail;
            continue;
         }
         if ((r = agx3_pack_mem_instr(&st)) != AGX3_PACK_UNHANDLED) {
            if (r == AGX3_PACK_ERROR)
               goto fail;
            continue;
         }
         if ((r = agx3_pack_subgroup_instr(&st)) != AGX3_PACK_UNHANDLED) {
            if (r == AGX3_PACK_ERROR)
               goto fail;
            continue;
         }
         if (agx3_pack_alu_instr(&st) != AGX3_PACK_EMITTED)
            goto fail;
      }
   }

   /* Sync mutable state modified during dispatch back to locals for epilog. */
   outstanding = st.outstanding;
   staging_banks = st.staging_banks;
   ray_query_acquired = st.ray_query_acquired;
   g18g_image_write_placed = st.g18g_image_write_placed;
   g18g_other_staging_end = st.g18g_other_staging_end;
   g18g_read_staging_end = st.g18g_read_staging_end;
   mesh_allocated = st.mesh_allocated;
   mesh_indices_fenced = st.mesh_indices_fenced;
   mesh_vertex_fences = st.mesh_vertex_fences;

   if (enc == AGX_ISA_ENCODING_AGX3C && fragment_ray_query &&
       g17p_colourless_fragment && ray_query_acquired) {
      uint8_t release[AGX3_PIXEL_MAX_LENGTH], release_len;
      if (!agx3_pixel_release(enc, 4, false, 0, release, &release_len) ||
          !agx3_append(emission, release, release_len))
         AGX3_PACK_FAIL();
   }

   if (early_compare) {
      const unsigned early =
         agx3_apply_early_compares(ctx, emission, initial_size, &spans,
                                   &compares, &fixups);
      if (agx_compiler_debug & AGX_DBG_SHADERS)
         fprintf(stderr, "AGX3 early loop compares: %u\n", early);
   }

   /* Removing an already-open pixel entry is safe when both ends of each
    * relative branch move together. Control flow elsewhere in the part does
    * not require retaining a duplicate entry or padding its former bytes. */
   ctx->agx3_colour_link.entry_unspanned =
      ctx->agx3_colour_link.entry_emitted;
   const off_t entry_start = initial_size + ctx->agx3_colour_link.entry_offset;
   const off_t entry_end = entry_start + AGX3_PIXEL_MAX_LENGTH;
   util_dynarray_foreach(&fixups, struct agx3_branch_fixup, fixup) {
      const off_t target = agx3_fixup_target(fixup);
      const bool before = fixup->offset < entry_start && target < entry_start;
      const bool after = fixup->offset >= entry_end && target >= entry_end;
      ctx->agx3_colour_link.entry_unspanned &= before || after;
      if (!agx3_fixup_branch(emission, enc, *fixup)) {
         if (agx_compiler_debug & AGX_DBG_SHADERS)
            fprintf(stderr,
                    "AGX3 branch fixup refused: at=%lld target=%lld last=%lld "
                    "skip_end=%u loop_exit=%u bn=%u\n",
                    (long long)fixup->offset,
                    (long long)fixup->block->offset,
                    (long long)fixup->block->last_offset, fixup->skip_to_end,
                    fixup->loop_exit, fixup->is_bn);
         AGX3_PACK_FAIL();
      }
   }

   /* Texture reads must stay below the separately placed vertex outputs.
    * Validate emitted extents rather than assuming the reserved bound. */
   if (g18g_output_staging_base &&
       g18g_read_staging_end > g18g_output_staging_base) {
      /* The caller may retry with the outputs above the reads: tell it how
       * far they reached, and how much it must fit above them, the outputs
       * and, if the shader writes images, one image-write bank (the 16-half
       * bank_span of agx3_emit_storage_image_write). */
      ctx->agx3_g18g_read_staging_end = g18g_read_staging_end;
      ctx->agx3_g18g_output_staging_span =
         AGX3_G18G_OUTPUT_STAGING_LIMIT + (image_writes_left ? 16 : 0);
      char *first = agx3_refusal_buffer(true);
      if (first && !first[0])
         snprintf(first, AGX3_REFUSAL_SIZE,
                  "G18G vertex texture reads stage to half %u, past the "
                  "output staging at %u",
                  g18g_read_staging_end, g18g_output_staging_base);
      if (agx_compiler_debug & (AGX_DBG_SHADERS | AGX_DBG_REFUSE))
         fprintf(stderr, "G18G vertex texture reads stage to half %u, past "
                 "the output staging at %u\n",
                 g18g_read_staging_end, g18g_output_staging_base);
      goto fail;
   }

   /* Every G18G image write was staged from g18g_image_write_floor up.
    * Refuse the program if a read's staging block or an output's UVSW
    * staging reached that far after all, rather than let a write share
    * staging halves with them. */
   if (g18g_compact_image_banks) {
      if (!g18g_image_write_placed || staging_banks.end > 16 ||
          g18g_other_staging_end < 24 ||
          g18g_other_staging_end > AGX3_G18G_READ_STAGING_LIMIT)
         goto fail;
      /* Reserve the read's full eight-half block, matching the native
       * load/store program's f22=12 for read slot 1 and write bank 0. */
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs,
                                    g18g_other_staging_end);
   } else if (g18g_image_write_placed &&
              g18g_other_staging_end > g18g_image_write_floor) {
      if (agx_compiler_debug & (AGX_DBG_SHADERS | AGX_DBG_REFUSE))
         fprintf(stderr, "G18G imgwci refusal: other staging reaches half "
                 "%u, past the image-write banks at %u\n",
                 g18g_other_staging_end, g18g_image_write_floor);
      goto fail;
   }

   if (enc == AGX_ISA_ENCODING_AGX3B) {
      const unsigned hinted = agx3_apply_loop_head_hints(ctx, emission);
      if (agx_compiler_debug & AGX_DBG_SHADERS)
         fprintf(stderr, "AGX3 loop-head hints: %u\n", hinted);
   }

   /* Separately compiled parts do not share a scoreboard ledger. Apple's
    * monolithic sampled fragment waits for TEX on its first FLTNORM; our
    * epilog cannot know that upstream slot, so finish only the still-live
    * async work at this boundary. Constant-colour parts and parts with real
    * in-part consumers add no instructions. */
   if (ctx->key->no_stop && !ctx->is_preamble && outstanding != 0 &&
       !agx3_retire_slots_noop(emission, enc, outstanding))
      AGX3_PACK_FAIL();

   if (gen1_mesh && mesh_allocated &&
       (!mesh_indices_fenced ||
        (ctx->nir && ctx->nir->info.mesh.max_vertices_out &&
         mesh_vertex_fences + 1 !=
            DIV_ROUND_UP(ctx->nir->info.mesh.max_vertices_out, 32))))
      AGX3_PACK_FAIL();

   /* Native1061's attachmentless1x/2x/4x fragment is pixel entry, final
    * release and end even when its epilog has no stores. This terminal
    * secondary part has no emitted instructions or outstanding work. Mark
    * its entry so the linker can retain only the first gate across parts. */
   if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
       ctx->stage == MESA_SHADER_FRAGMENT && ctx->key->secondary &&
       !ctx->is_preamble && !ctx->key->no_stop &&
       !ctx->key->fs.ignore_tib_dependencies && emission->size == initial_size) {
      uint8_t pixel[AGX3_PIXEL_MAX_LENGTH], length = 0;
      if (!agx3_pixel_colour_entry(enc, pixel, &length) ||
          !agx3_append(emission, pixel, length))
         goto fail;
      ctx->agx3_colour_link.entry_emitted = true;
      ctx->agx3_colour_link.entry_offset = 0;
      if (!agx3_pixel_release(enc, 0, true, 0, pixel, &length) ||
          !agx3_append(emission, pixel, length))
         goto fail;
   }

   if (!ctx->key->no_stop || ctx->is_preamble) {
      /* Native G17G's complete empty fragment still opens and releases the
       * pixel transaction. An empty no-stop main remains empty so its linked
       * epilog owns this terminal protocol. No registers or staging are used. */
      if (enc == AGX_ISA_ENCODING_AGX3C &&
          ctx->stage == MESA_SHADER_FRAGMENT && !ctx->is_preamble &&
          count == 0 && emission->size == initial_size) {
         uint8_t bytes[AGX3_PIXEL_MAX_LENGTH], len;
         if (!agx3_pixel_colour_entry(enc, bytes, &len) ||
             !agx3_append(emission, bytes, len) ||
             !agx3_pixel_release(enc, 0, true, 0, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_PACK_FAIL();
      }
      const struct agx3_sel_instr end = {.op = AGX3_SEL_OP_END};
      if (!agx3_emit(emission, enc, &end))
         AGX3_PACK_FAIL();
   }

   util_dynarray_fini(&spans);
   util_dynarray_fini(&compares);
   util_dynarray_fini(&fixups);
   return true;

fail:
   /* Do not leave a valid-looking prefix behind.  The compiler entrypoint
    * propagates this failure and clears the shader part transactionally. */
   emission->size = initial_size;
   util_dynarray_fini(&spans);
   util_dynarray_fini(&compares);
   util_dynarray_fini(&fixups);
   return false;
}

#undef AGX3_PACK_FAIL

/* G18G vertex outputs share the staging file with texture reads and are
 * staged from AGX3_G18G_READ_STAGING_LIMIT. When agx3_pack_binary() reports
 * reads that reach past it, pack once more with the outputs (and an
 * image-write bank, if the shader writes images) right above the reads,
 * provided they still end within AGX3_STAGED_WRITE_LIMIT. The outputs start
 * at a multiple of 8 halves, a read block, like the default base. That costs
 * no range: the highest base that fits, 112, or 96 with an image-write bank,
 * is itself a multiple of 8. Nothing the refused attempt reserved carries
 * over to the second: neither its staging footprint nor the scratch
 * registers it raised max_reg for (the cube and array-layer temporaries, the
 * G17G LOD relay). */
bool
agx3_pack_program(agx_context *ctx, struct util_dynarray *emission)
{
   const unsigned staging_halfs = ctx->agx3_staging_halfs;
   const unsigned max_reg = ctx->max_reg;
   ctx->agx3_g18g_output_staging_floor = 0;
   ctx->agx3_g18g_read_staging_end = 0;
   if (agx3_pack_binary(ctx, emission))
      return true;

   const unsigned read_end = ctx->agx3_g18g_read_staging_end;
   const unsigned output_floor = ALIGN_POT(read_end, 8);
   if (!read_end || output_floor + ctx->agx3_g18g_output_staging_span >
                       AGX3_STAGED_WRITE_LIMIT)
      return false;

   ctx->agx3_staging_halfs = staging_halfs;
   ctx->max_reg = max_reg;
   ctx->agx3_g18g_read_staging_end = 0;
   ctx->agx3_g18g_output_staging_floor = output_floor;
   const bool packed = agx3_pack_binary(ctx, emission);
   if (agx_compiler_debug & (AGX_DBG_SHADERS | AGX_DBG_REFUSE))
      fprintf(stderr,
              "G18G vertex outputs staged from half %u above texture "
              "reads to half %u: %s\n",
              output_floor, read_end, packed ? "packed" : "refused");
   return packed;
}
