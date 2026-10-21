/*
 * Copyright 2021 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "util/lut.h"
#include "util/macros.h"
#include "agx_builder.h"
#include "agx3_select_core.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "agx_opcodes.h"

/* AGX peephole optimizer responsible for instruction combining. It operates in
 * a forward direction and a backward direction, in each case traversing in
 * source order. SSA means the forward pass satisfies the invariant:
 *
 *    Every def is visited before any of its uses.
 *
 * Dually, the backend pass satisfies the invariant:
 *
 *    Every use of a def is visited before the def.
 *
 * This means the forward pass can propagate modifiers forward, whereas the
 * backwards pass propagates modifiers backward. Consider an example:
 *
 *    1 = fabs 0
 *    2 = fround 1
 *    3 = fsat 1
 *
 * The forwards pass would propagate the fabs to the fround (since we can
 * lookup the fabs from the fround source and do the replacement). By contrast
 * the backwards pass would propagate the fsat back to the fround (since when
 * we see the fround we know it has only a single user, fsat).  Propagatable
 * instruction have natural directions (like pushforwards and pullbacks).
 *
 * We are careful to update the tracked state whenever we modify an instruction
 * to ensure the passes are linear-time and converge in a single iteration.
 *
 * Size conversions are worth special discussion. Consider the snippet:
 *
 *    2 = fadd 0, 1
 *    3 = f2f16 2
 *    4 = fround 3
 *
 * A priori, we can move the f2f16 in either direction. But it's not equal --
 * if we move it up to the fadd, we get FP16 for two instructions, whereas if
 * we push it into the fround, we effectively get FP32 for two instructions. So
 * f2f16 is backwards. Likewise, consider
 *
 *    2 = fadd 0, 1
 *    3 = f2f32 1
 *    4 = fround 3
 *
 * This time if we move f2f32 up to the fadd, we get FP32 for two, but if we
 * move it down to the fround, we get FP16 to too. So f2f32 is backwards.
 */

static bool
agx_is_fmov(agx_instr *def)
{
   return (def->op == AGX_OPCODE_FADD || def->op == AGX_OPCODE_HADD) &&
          agx_is_equiv(def->src[1], agx_negzero());
}

/* Compose floating-point modifiers with floating-point sources */

static agx_index
agx_compose_float_src(agx_index to, agx_index from)
{
   if (to.abs) {
      from.neg = false;
      from.abs = true;
   }

   from.neg ^= to.neg;

   return from;
}

/* AGX3's grounded fmin/fmax encoding requires the compared values to remain
 * tied to the selected values. Inlining an immediate into only one side (a
 * minifloat into the comparison, or raw bits that fit an integer immediate,
 * such as +0.0, into the selection) breaks that identity, and the resulting
 * general FSEL has no NaN-correct grounded form. Keep the shared SSA value
 * explicit on both sides through register allocation. */
static bool
agx_preserve_g17p_tied_fcmpsel(agx_context *ctx, agx_instr *I, unsigned src)
{
   return ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_FCMPSEL && src < 4 &&
          agx_is_equiv(I->src[src], I->src[src ^ 2]);
}

/*
 * A tied FCMPSEL whose condition is GTN or LTN is a minimum or a maximum, and
 * G17P encodes exactly that as a compact six-byte fsel whose second operand is
 * either a register pair or a single minifloat deposited into both the compare
 * and the select slot. The tie is what makes it a min/max, so an immediate has
 * to reach both slots or neither: folding it into just one leaves the packer a
 * general mixed-width select it has no encoding for, which is how max(x, 0.0)
 * -- normalize(), clamp(), saturate(), discard -- used to fail the shader
 * closed.  Sources 2 and 3 are not float sources in general (a select can
 * carry an integer), so the float interpretation is only taken here, where the
 * tie proves the operand is the compared float.
 *
 * Either pair can hold the constant: max(0.0, x) puts it in sources 0 and 2,
 * max(x, 0.0) in 1 and 3. Half-folding the first pair is worse than the
 * second, because agx_opt_compact_constants then sees the surviving reference
 * as the constant's only float use and narrows it to 16 bits, handing the
 * packer a 32-bit maximum against a 16-bit register.
 */
static bool
agx_g17p_tied_minmax(agx_context *ctx, const agx_instr *I)
{
   if (!(ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
         I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 && !I->invert_cond &&
         !I->saturate &&
         (I->fcond == AGX_FCOND_GTN || I->fcond == AGX_FCOND_LTN)))
      return false;

   /* agx_is_equiv() ignores modifiers, and a modifier on one half of a tie
    * would make the two operands different values. Require them identical. */
   for (unsigned i = 0; i < 2; ++i) {
      const agx_index a = I->src[i], b = I->src[i + 2];
      if (!agx_is_equiv(a, b) || a.size != b.size || a.abs != b.abs ||
          a.neg != b.neg || a.memory != b.memory)
         return false;
   }

   return true;
}

/*
 * (0 < x) ? x : 0 and (0 > x) ? x : 0 are exactly G17P's tied maximum
 * (x GTN 0) ? x : 0 and minimum (x LTN 0) ? x : 0: a NaN x compares false
 * either way, and so does a zero of either sign. The same holds for
 * (x < 0) ? x : 0 and (x > 0) ? x : 0. Fusion leaves the comparator zero as a
 * minifloat and the selected zero as a bit pattern, which are the same value.
 * Rewrite these clamps to the tied form, which has the compact native
 * encoding Apple uses for max(0.0, x).
 */
static void
agx_g17p_tie_zero_minmax(agx_context *ctx, agx_instr *I)
{
   if (!(ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
         I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 && !I->invert_cond &&
         !I->saturate &&
         (I->fcond == AGX_FCOND_LT || I->fcond == AGX_FCOND_GT)))
      return;

   for (unsigned s = 0; s < 4; ++s) {
      if (I->src[s].abs || I->src[s].neg)
         return;
   }

   const bool zero[4] = {
      agx_is_equiv(I->src[0], agx_immediate(0)),
      agx_is_equiv(I->src[1], agx_immediate(0)),
      agx_is_equiv(I->src[2], agx_immediate(0)),
      agx_is_equiv(I->src[3], agx_immediate(0)),
   };
   const bool crossed = zero[0] && zero[3] && !zero[1] &&
                        agx_is_equiv(I->src[1], I->src[2]);
   const bool tied = zero[1] && zero[3] && !zero[0] &&
                     agx_is_equiv(I->src[0], I->src[2]);
   if (!crossed && !tied)
      return;

   const agx_index x = crossed ? I->src[1] : I->src[0];
   if (x.type != AGX_INDEX_NORMAL || x.size != I->dest[0].size)
      return;

   const bool max = crossed ? I->fcond == AGX_FCOND_LT
                            : I->fcond == AGX_FCOND_GT;
   I->src[0] = I->src[2] = x;
   I->src[1] = I->src[3] = agx_immediate(0);
   I->fcond = max ? AGX_FCOND_GTN : AGX_FCOND_LTN;
}

static bool
agx_can_fold_half_uniform(const agx_instr *I, agx_instr **defs)
{
   if (I->nr_dests != 1 || I->dest[0].size != AGX_SIZE_32 ||
       agx_channels(I->dest[0]) != 1 || I->saturate)
      return false;

   unsigned half_uniforms = 0;
   bool full_gpr = false;
   agx_foreach_src(I, s) {
      agx_index src = I->src[s];
      agx_instr *def = src.type == AGX_INDEX_NORMAL ? defs[src.value] : NULL;
      if (def && agx_is_fmov(def) && !def->saturate && !src.abs && !src.neg &&
          !def->src[0].abs && !def->src[0].neg) {
         if (def->src[0].type == AGX_INDEX_UNIFORM ||
             def->src[0].type == AGX_INDEX_IMMEDIATE)
            src = def->src[0];
      }
      half_uniforms += src.type == AGX_INDEX_UNIFORM && src.size == AGX_SIZE_16;
      full_gpr |=
         (src.type == AGX_INDEX_NORMAL || src.type == AGX_INDEX_REGISTER) &&
         src.size == AGX_SIZE_32 && !src.memory && agx_channels(src) == 1;
   }
   return half_uniforms == 1 && full_gpr;
}

/*
 * AGX3B composes a 16-bit source on the f32 ALU slots: f43 on the f35
 * slot, f26 on f18 and f60 on the f32fma addend, grounded per the fp16
 * probe corpus. Whether folding an f2f32 move into consumer slot s
 * produces an encodable instruction depends on the op, the slot, the
 * other operands and the destination width. Only the witnessed
 * combinations are admitted:
 *
 *    f32add: src0-16 (w01/w21), src1-16 with a negate (w19), a single
 *            saturated 16-bit source (w25), or src0-16 under a narrow
 *            destination (w23). Both sources narrow is unwitnessed.
 *    f32mul: either source 16-bit, including both (w02/w07), and one
 *            16-bit source under a narrow destination (the existing
 *            fmul_mixed_half endpoint). No modifiers or saturate beside
 *            a narrow operand.
 *    f32fma: src0-16 (w08), src0+src1 16-bit (w03), addend-16 (w09),
 *            and a 16-bit multiplicand beside a minifloat multiplier
 *            (w22). A lone 16-bit src1, a 16-bit addend beside a narrow
 *            multiplicand, a narrow destination, or saturation beside a
 *            narrow operand are unwitnessed.
 *
 * A negated 16-bit source is grounded on f32add (f48+f43 or f31+f26)
 * and on the f32fma addend (f65+f60) only; no absolute modifier is
 * witnessed beside a 16-bit operand.
 *
 * Because the sources are visited in order, a candidate that follows a
 * source already folded to 16 bits sees the post-fold width; the
 * combination checks therefore only need the current instruction.
 */
static bool
agx3b_f32_src_admits_half(const agx_instr *ins, unsigned s,
                          const agx_index h)
{
   const agx_index to = ins->src[s];

   /* A plain scalar half register is the only source the witnessed
    * width slots take. */
   if ((h.type != AGX_INDEX_NORMAL && h.type != AGX_INDEX_REGISTER) ||
       h.memory || agx_channels(h) != 1)
      return false;

   /* No absolute modifier is witnessed beside a 16-bit operand. */
   if (h.abs || to.abs)
      return false;

   const bool narrow = ins->dest[0].size == AGX_SIZE_16;
   const bool neg = to.neg ^ h.neg;

   /* Every other operand must keep the instruction on a witnessed
    * route: an immediate or a uniform beside a 16-bit register exists
    * only as the f32fma immediate-multiplier form (w22). */
   for (unsigned i = 0; i < ins->nr_srcs; ++i) {
      if (i == s)
         continue;
      const agx_index o = ins->src[i];
      if (o.type == AGX_INDEX_NORMAL || o.type == AGX_INDEX_REGISTER)
         continue;
      if (ins->op == AGX_OPCODE_FFMA && s < 2 && i < 2 &&
          o.type == AGX_INDEX_IMMEDIATE && !o.memory)
         continue;
      return false;
   }

   switch (ins->op) {
   case AGX_OPCODE_FADD: {
      const agx_index other = ins->src[1 - s];
      if (other.size == AGX_SIZE_16)
         return false; /* two 16-bit add sources are unwitnessed */
      if (narrow)
         /* w23: a clean 16-bit f35-side source under a narrow dest. */
         return s == 0 && !neg && !other.neg && !other.abs &&
                other.size == AGX_SIZE_32 && !ins->saturate;
      if (ins->saturate)
         /* w25: the saturated 16-bit source composes only with no
          * other modifier present. */
         return !neg && !other.neg && !other.abs;
      return true;
   }
   case AGX_OPCODE_FMUL: {
      const agx_index other = ins->src[1 - s];
      if (ins->saturate || neg || other.neg || other.abs)
         return false;
      if (narrow)
         /* fmul_mixed_half: one 16-bit source against a 32-bit one. */
         return other.size == AGX_SIZE_32;
      return true;
   }
   case AGX_OPCODE_FFMA:
      if (ins->saturate || narrow)
         return false;
      if (s == 2) {
         /* w09: a 16-bit addend, including negated (f65+f60), only
          * beside two 32-bit register multiplicands. */
         return ins->src[0].size == AGX_SIZE_32 &&
                ins->src[1].size == AGX_SIZE_32;
      }
      if (ins->src[2].type != AGX_INDEX_NORMAL &&
          ins->src[2].type != AGX_INDEX_REGISTER)
         return false;
      if (ins->src[2].size != AGX_SIZE_32)
         return false; /* a 16-bit addend beside a narrow multiplier */
      if (s == 0)
         /* w08/w03: src0-16 with a 32 or 16-bit src1. */
         return !neg;
      /* s == 1: f26 is grounded beside f43 (w03) or a minifloat src0
       * (w22), never alone beside a 32-bit f35. */
      return !neg &&
             (ins->src[0].type == AGX_INDEX_IMMEDIATE ||
              ins->src[0].size == AGX_SIZE_16);
   default:
      return false;
   }
}

static void
agx_optimizer_fmov(agx_context *ctx, agx_instr **defs, agx_instr *ins)
{
   agx_foreach_ssa_src(ins, s) {
      agx_index src = ins->src[s];
      if (agx_preserve_g17p_tied_fcmpsel(ctx, ins, s))
         continue;
      agx_instr *def = defs[src.value];

      if (def == NULL)
         continue; /* happens for phis in loops */
      if (!agx_is_fmov(def))
         continue;
      if (def->saturate)
         continue;
      if (ins->op == AGX_OPCODE_FCMPSEL && s >= 2)
         continue;

      /* We can fold f2f32 into 32-bit instructions, but we can't fold f2f16
       * into 16-bit instructions, since the latter would implicitly promote to
       * a 32-bit instruction which is not exact.
       */
      assert(def->src[0].size == AGX_SIZE_32 ||
             def->src[0].size == AGX_SIZE_16);
      assert(src.size == AGX_SIZE_32 || src.size == AGX_SIZE_16);

      if (src.size == AGX_SIZE_16 && def->src[0].size == AGX_SIZE_32)
         continue;

      /* AGX3B composes a 16-bit register onto the f32 ALU source slots
       * (f43/f26/f60), so an f2f32 move folds into the witnessed
       * width-composed endpoints instead of materializing a widened
       * temporary. The uniform path stays as before; a register half is
       * admitted only on the grounded combinations. */
      if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          (ins->op == AGX_OPCODE_FADD || ins->op == AGX_OPCODE_FMUL ||
           ins->op == AGX_OPCODE_FFMA) &&
          src.size == AGX_SIZE_32 && def->src[0].size == AGX_SIZE_16 &&
          !(def->src[0].type == AGX_INDEX_UNIFORM && !def->src[0].memory &&
            !def->src[0].abs && !def->src[0].neg && !src.abs && !src.neg &&
            agx_can_fold_half_uniform(ins, defs)) &&
          !agx3b_f32_src_admits_half(ins, s, def->src[0]))
         continue;

      /* FLTNORM and RGB10A2 packing have only captured f32 sources. Keep
       * the explicit widening used by linked half-width fragment outputs
       * before normalized packing. */
      if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          (ins->op == AGX_OPCODE_FLTNORM_AGX3 ||
           ins->op == AGX_OPCODE_PACK_RGB10A2_AGX3) &&
          src.size == AGX_SIZE_32 &&
          def->src[0].size == AGX_SIZE_16)
         continue;

      /* G17P's grounded f32 multiply into a half destination takes plain
       * registers only. Keep |x| or -x as its own move in front of such a
       * multiply rather than inventing an unencodable modified form. */
      if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          ins->op == AGX_OPCODE_FMUL && ins->dest[0].size == AGX_SIZE_16 &&
          src.size == AGX_SIZE_32 && (def->src[0].abs || def->src[0].neg))
         continue;

      /* The half-width TRUNC/ROUNDEVEN forms are grounded for plain
       * registers only; the packer refuses -x or |x| there, which fails the
       * whole pipeline (vkQuake's ssao_evaluate: roundeven of a negated
       * half). Keep the modifier on its own move in front instead. */
      if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          (ins->op == AGX_OPCODE_TRUNC || ins->op == AGX_OPCODE_ROUNDEVEN) &&
          src.size == AGX_SIZE_16 && (def->src[0].abs || def->src[0].neg))
         continue;

      ins->src[s] = agx_compose_float_src(src, def->src[0]);
   }
}

static bool
image_write_source_can_be_immediate(agx_instr *I, unsigned s)
{
   bool block = I->op == AGX_OPCODE_BLOCK_IMAGE_STORE;
   assert(I->op == AGX_OPCODE_IMAGE_WRITE || block);

   /* LOD can always be immediate. Actually, it's just zero so far, we don't
    * support nonzero LOD for images yet.
    */
   if (s == 2 && !block)
      return true;

   /* If the "bindless" source (source 3) is an immediate, it means we don't
    * have a bindless image, instead we have a texture state index. We're
    * allowed to have immediate texture state registers (source 4). However,
    * we're not allowed to have immediate bindless offsets (also source 4).
    */
   unsigned base = block ? 0 : 3;
   bool is_texture_state = (I->src[base].type == AGX_INDEX_IMMEDIATE);
   if (s == (base + 1) && is_texture_state)
      return true;

   /* Otherwise, must be from a register */
   return false;
}

static void
agx_optimizer_inline_imm(agx_context *ctx, agx_instr **defs, agx_instr *I)
{
   agx_foreach_ssa_src(I, s) {
      agx_index src = I->src[s];
      if (I->op == AGX_OPCODE_INTERSECT_RAY_GEN1 ||
          I->op == AGX_OPCODE_SCS_STORE_AGX3 ||
          I->op == AGX_OPCODE_SCS_LOAD_REGISTER_INDEXED_AGX3 ||
          I->op == AGX_OPCODE_SCS_STORE_REGISTER_INDEXED_AGX3 ||
          I->op == AGX_OPCODE_SCS_LOAD_REGISTER_BASE_AGX3 ||
          I->op == AGX_OPCODE_SCS_STORE_REGISTER_BASE_AGX3 ||
          I->op == AGX_OPCODE_SCS_LOAD_UNIFORM_INDEXED_AGX3 ||
          I->op == AGX_OPCODE_SCS_STORE_UNIFORM_INDEXED_AGX3)
         continue;

      /* Either compared source of a tied minimum/maximum may be inlined,
       * because the fold below carries it into its partner as well and so
       * keeps the tie. */
      if (agx_preserve_g17p_tied_fcmpsel(ctx, I, s) &&
          !(s < 2 && agx_g17p_tied_minmax(ctx, I)))
         continue;
      if (src.neg)
         continue;
      if (I->op == AGX_OPCODE_ALLOCATE_UVB_M5 ||
          I->op == AGX_OPCODE_STORE_UVB_M5 ||
          I->op == AGX_OPCODE_STORE_OBJECT_CHILDCOUNT_M5)
         continue;

      /* G17P's native subtraction takes two registers. Instruction selection
       * deliberately materializes an immediate minuend for `c - x`; folding
       * it back here would leave immediate-plus-negated-register, whose iadd
       * modifier means narrow sign extension rather than arithmetic negate.
       */
      if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_IADD &&
          ((s == 0 && I->src[1].neg) || (s == 1 && I->src[0].neg)))
         continue;

      /* G17P has no iadd of two immediates. The ushr count copy
       * `iadd #0, count` must keep its register count. */
      if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_IADD && I->nr_srcs == 2 &&
          I->src[1 - s].type == AGX_INDEX_IMMEDIATE)
         continue;

      /* G17P's arithmetic shift right takes its shifted VALUE in a register.
       * Only the count (source 1) has an immediate form -- `sra` is what
       * `metal-nt -arch applegpu_g17p` emits for `(-7) >> x`, and it first
       * spends an LDIMM to put the -7 in a register (`1cf922fe1e0cff0f`
       * ldimm then `a7015408000400d8e200` sra reg,reg).
       * Folding the constant into source 0 leaves a shape with no encoding
       * at all -- agx3_select_core.c's AGX3_SEL_OP_ASR requires two plain
       * 32-bit registers -- and the packer then refuses the whole shader.
       * Keep the MOV so selection reproduces Apple's ldimm+sra pair. */
      if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_ASR && s == 0)
         continue;

      agx_instr *def = defs[src.value];
      if (!def || def->op != AGX_OPCODE_MOV_IMM)
         continue;

      uint8_t value = def->imm;
      uint16_t value_u16 = def->imm;

      /* G17P's sample names its explicit level and its implicit-LOD bias in
       * the same immediate operand, and the packer decodes that operand as a
       * minifloat.  agx_is_float_src() reports the whole instruction, which
       * is integer-typed, so name this one source explicitly rather than
       * folding a float bit pattern in as an integer. */
      bool float_src =
         agx_is_float_src(I, s) ||
         (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_TEXTURE_SAMPLE && s == 1 &&
          (I->lod_mode == AGX_LOD_MODE_LOD_MIN ||
           I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS));

      if (I->op == AGX_OPCODE_ST_TILE && s == 0)
         continue;
      /* G17P tile stores only encode the whole-pixel 0xff mask as an
       * immediate. Keep all other masks, including zero coverage, in the
       * 16-bit register already produced by instruction selection.
       */
      if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_ST_TILE && s == 1 && def->imm != 0xff)
         continue;
      /* Tile loads likewise: a per-sample load (multisample framebuffer
       * fetch) names its sample mask in a register, as Apple's multisample
       * LLDCol does; only the whole-pixel 0xff mask is an immediate. */
      if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_LD_TILE && s == 0 && def->imm != 0xff)
         continue;
      if (I->op == AGX_OPCODE_ZS_EMIT && s != 0)
         continue;
      /* G17P names an explicit level and an implicit-LOD bias in the same
       * 12-bit immediate operand, so both fold; every other sample source
       * has to stay a register. */
      if (I->op == AGX_OPCODE_TEXTURE_SAMPLE && s != 4 &&
          !(ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B && s == 1 &&
            (I->lod_mode == AGX_LOD_MODE_LOD_MIN ||
             I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS)))
         continue;
      if ((I->op == AGX_OPCODE_DEVICE_STORE ||
           I->op == AGX_OPCODE_LOCAL_STORE || I->op == AGX_OPCODE_ATOMIC ||
           I->op == AGX_OPCODE_LOCAL_ATOMIC) &&
          s != 2)
         continue;
      if (I->op == AGX_OPCODE_ST_VARY && s != 0)
         continue;
      if ((I->op == AGX_OPCODE_LOCAL_LOAD || I->op == AGX_OPCODE_DEVICE_LOAD ||
           I->op == AGX_OPCODE_STACK_STORE) &&
          s != 1)
         continue;
      if (I->op == AGX_OPCODE_SPLIT)
         continue;

      /*
       * AGX3's immediate-lane shuffle holds the lane in a seven-bit field,
       * so every lane a subgroup can name folds in.  Only an out-of-range
       * constant -- undefined behaviour, but not a reason to refuse a shader
       * -- has to keep its mov and go through the register-lane form.
       */
      if (I->op == AGX_OPCODE_SHUFFLE && s == 1 && ctx->key &&
          ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B && def->imm > 0x7f)
         continue;

      if ((I->op == AGX_OPCODE_IMAGE_WRITE ||
           I->op == AGX_OPCODE_BLOCK_IMAGE_STORE) &&
          !image_write_source_can_be_immediate(I, s))
         continue;

      if (float_src) {
         bool fp16 = (def->dest[0].size == AGX_SIZE_16);
         assert(fp16 || (def->dest[0].size == AGX_SIZE_32));

         float f = fp16 ? _mesa_half_to_float(def->imm) : uif(def->imm);
         if (!agx_minifloat_exact(f))
            continue;

         /* Only the witnessed narrow-destination immediate forms encode:
          * an all-immediate FADD folds to a move, a narrowing FADD names
          * -0.0 beside a 32-bit register, and FMUL scales a plain 32-bit
          * register by a minifloat; no half-destination FFMA takes an
          * immediate at all. Every other half-destination FADD/FMUL/FFMA
          * substitution reaches the packer unencodable and refuses the
          * whole shader (rtpipe-all, ftex-bias-4, ftex-offlod-3). Keep the
          * move in that case; other operations and destination widths are
          * unchanged. */
         if (I->nr_dests == 1 && I->dest[0].size == AGX_SIZE_16 &&
             (I->op == AGX_OPCODE_FADD || I->op == AGX_OPCODE_FMUL ||
              I->op == AGX_OPCODE_FFMA)) {
            const agx_index *other =
               I->nr_srcs == 2 ? &I->src[s ^ 1] : NULL;
            const bool narrow_imm_form =
               (I->op == AGX_OPCODE_FADD && other != NULL &&
                ((other->type == AGX_INDEX_IMMEDIATE && !other->memory &&
                  !other->abs && !other->neg) ||
                 (f == 0.0f && signbit(f) &&
                  other->type != AGX_INDEX_IMMEDIATE && !other->memory &&
                  !other->abs && (!other->neg || !I->saturate) &&
                  other->size == AGX_SIZE_32))) ||
               (I->op == AGX_OPCODE_FMUL && !I->saturate &&
                other != NULL && other->type != AGX_INDEX_IMMEDIATE &&
                !other->memory && !other->abs &&
                other->size == AGX_SIZE_32);
            if (!narrow_imm_form)
               continue;
         }

         /* Both halves of a tied minimum/maximum, or neither. */
         if (s < 2 && agx_g17p_tied_minmax(ctx, I))
            I->src[s + 2] = agx_immediate_f(f);

         I->src[s] = agx_immediate_f(f);
      } else if (value == def->imm) {
         I->src[s] = agx_immediate(value);
      } else if (value_u16 == def->imm && agx_allows_16bit_immediate(I)) {
         I->src[s] = agx_abs(agx_immediate(value_u16));
      } else if ((I->op == AGX_OPCODE_IADD || I->op == AGX_OPCODE_IMAD) &&
                 s == agx_negate_src_index(I)) {
         unsigned bits = agx_size_align_16(def->dest[0].size) * 16;
         uint64_t mask = BITFIELD64_MASK(bits);
         uint64_t negated = (-def->imm) & mask;
         value = negated;

         /* Try to negate the immediate */
         if (value == negated) {
            I->src[s] = agx_neg(agx_immediate(value));
         }
      }
   }
}

/*
 * Fuse not into and/or/xor. Specifically, acts on not and fuses:
 *
 *    not(and(x, y) -> nand(x, y)
 *    not(or(x, y) -> nor(x, y)
 *    not(xor(x, y) -> xnor(x, y)
 */
static bool
agx_optimizer_not(agx_context *ctx, agx_instr *I, agx_instr *use)
{
   /* Check for bit op and use of not op */
   if (I->op != AGX_OPCODE_BITOP || use->op != AGX_OPCODE_NOT)
      return false;

   /* G17P has native word-wide NAND/NOR/XNOR, but the half-width forms
    * still need the separate complement. Do not create an unencodable
    * truth table or disable the supported word-wide fusion. */
   const unsigned truth_table = I->truth_table ^ 0xf;
   if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       I->dest[0].size != AGX_SIZE_32 &&
       truth_table != AGX3_SEL_TT_AND && truth_table != AGX3_SEL_TT_OR &&
       truth_table != AGX3_SEL_TT_XOR && truth_table != AGX3_SEL_TT_NOT)
      return false;

   /* Remap operation to the appropriate one */
   I->truth_table ^= 0xF;
   I->dest[0] = use->dest[0];

   return true;
}

/* Whether an f32 producer may write a half destination directly on
 * AGX3B -- the witnessed f10 narrow forms only:
 *
 *    f32mul: two plain 32-bit registers (w14), or one 16-bit register
 *            against a 32-bit one (the fmul_mixed_half endpoint).
 *    f32add: two registers, no absolute modifier, at most one negated
 *            input (w04/w26); a 16-bit source only unmodified in the
 *            f35 slot (w23), and never a 16-bit f18 source.
 *    f32fma: three plain 32-bit registers (w05).
 *
 * Saturation beside a narrow destination is unwitnessed on all three.
 */
static bool
agx3b_f32_dest_admits_narrow(const agx_instr *I)
{
   if (I->saturate || I->nr_dests != 1)
      return false;

   switch (I->op) {
   case AGX_OPCODE_FMUL:
      if (I->nr_srcs != 2)
         return false;
      for (unsigned i = 0; i < 2; ++i)
         if (I->src[i].abs || I->src[i].neg || I->src[i].memory ||
             (I->src[i].type != AGX_INDEX_NORMAL &&
              I->src[i].type != AGX_INDEX_REGISTER))
            return false;
      if (I->src[0].size == AGX_SIZE_16 &&
          I->src[1].size == AGX_SIZE_16)
         return false;
      return I->src[0].size == AGX_SIZE_32 ||
             I->src[1].size == AGX_SIZE_32;
   case AGX_OPCODE_FADD:
      if (I->nr_srcs != 2)
         return false;
      for (unsigned i = 0; i < 2; ++i)
         if (I->src[i].abs || I->src[i].memory ||
             (I->src[i].type != AGX_INDEX_NORMAL &&
              I->src[i].type != AGX_INDEX_REGISTER))
            return false;
      if (I->src[0].neg && I->src[1].neg)
         return false;
      if (I->src[1].size == AGX_SIZE_16)
         return false;
      if (I->src[0].size == AGX_SIZE_16 &&
          (I->src[0].neg || I->src[1].neg))
         return false;
      return I->src[0].size != AGX_SIZE_16 ||
             I->src[1].size == AGX_SIZE_32;
   case AGX_OPCODE_FFMA:
      if (I->nr_srcs != 3)
         return false;
      for (unsigned i = 0; i < 3; ++i)
         if (I->src[i].size != AGX_SIZE_32 || I->src[i].abs ||
             I->src[i].neg || I->src[i].memory ||
             (I->src[i].type != AGX_INDEX_NORMAL &&
              I->src[i].type != AGX_INDEX_REGISTER))
            return false;
      return true;
   default:
      return false;
   }
}

static bool
agx_optimizer_fmov_rev(agx_context *ctx, agx_instr *I, agx_instr *use)
{
   if (!agx_is_fmov(use))
      return false;
   if (use->src[0].neg || use->src[0].abs)
      return false;

   /* G17P's native saturate is a standalone f32add.ep2 of the value and -0.
    * Pulling that modifier into its producer invents saturated forms for
    * operations such as fmul and transcendental functions. Keep the exact
    * Apple sequence intact for every producer. */
   if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       use->saturate)
      return false;

   /* We can fold f2f16 into 32-bit instructions, but we can't fold f2f32 into
    * 16-bit instructions, since the latter would implicitly promote to a 32-bit
    * instruction which is not exact.
    */
   assert(use->dest[0].size == AGX_SIZE_32 || use->dest[0].size == AGX_SIZE_16);
   assert(I->dest[0].size == AGX_SIZE_32 || I->dest[0].size == AGX_SIZE_16);

   if (I->dest[0].size == AGX_SIZE_16 && use->dest[0].size == AGX_SIZE_32)
      return false;

   /* AGX3B narrows an f32 result into a half destination on f32add
    * (w04/w23/w26), f32mul (w14 and the grounded mixed-half endpoint)
    * and f32fma (w05), all through the f10 destination width field.
    * Keep an explicit f32-to-f16 move after everything else so each
    * remaining producer rounds exactly once through the native
    * narrowing form. */
   if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       I->dest[0].size == AGX_SIZE_32 && use->dest[0].size == AGX_SIZE_16 &&
       !agx3b_f32_dest_admits_narrow(I))
      return false;

   /* saturate(saturate(x)) = saturate(x) */
   I->saturate |= use->saturate;
   I->dest[0] = use->dest[0];
   return true;
}

static bool
agx_icond_is_unsigned(enum agx_icond cond)
{
   switch (cond) {
   case AGX_ICOND_UEQ:
   case AGX_ICOND_ULT:
   case AGX_ICOND_UGT:
      return true;

   case AGX_ICOND_SEQ:
   case AGX_ICOND_SLT:
   case AGX_ICOND_SGT:
      return false;
   }

   UNREACHABLE("invalid condition");
}

static bool
agx_supports_zext(agx_instr *I, unsigned s)
{
   switch (I->op) {
   case AGX_OPCODE_IADD:
   case AGX_OPCODE_IMAD:
   case AGX_OPCODE_INTL:
   case AGX_OPCODE_FFS:
   case AGX_OPCODE_BITREV:
   case AGX_OPCODE_BFI:
   case AGX_OPCODE_BFEIL:
   case AGX_OPCODE_EXTR:
   case AGX_OPCODE_BITOP:
   case AGX_OPCODE_WHILE_ICMP:
   case AGX_OPCODE_IF_ICMP:
   case AGX_OPCODE_ELSE_ICMP:
   case AGX_OPCODE_BREAK_IF_ICMP:
   case AGX_OPCODE_ICMP_BALLOT:
   case AGX_OPCODE_ICMP_QUAD_BALLOT:
      return true;

   case AGX_OPCODE_ICMP:
   case AGX_OPCODE_ICMPSEL:
      /* Only the comparisons can be extended, not the selection. And we can
       * only zero-extend with unsigned comparison. Presumably the hardware
       * sign-extends with signed comparisons but we don't handle that yet.
       */
      return (s < 2) && agx_icond_is_unsigned(I->icond);

   default:
      return false;
   }
}

bool
agx_opt_collect_split(agx_context *ctx)
{
   agx_instr **defs = calloc(ctx->alloc, sizeof(*defs));
   agx_index *aliases = calloc(ctx->alloc, sizeof(*aliases));
   bool progress = false;
   if (!defs || !aliases)
      goto done;
   for (unsigned i = 0; i < ctx->alloc; ++i)
      aliases[i] = agx_null();
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_dest(I, d)
         defs[I->dest[d].value] = I;
   }
   agx_foreach_instr_global(ctx, I) {
      if (I->op != AGX_OPCODE_SPLIT || I->nr_srcs != 1 || I->saturate ||
          I->src[0].type != AGX_INDEX_NORMAL || I->src[0].memory ||
          I->src[0].abs || I->src[0].neg)
         continue;
      agx_instr *collect = defs[I->src[0].value];
      if (!collect || collect->op != AGX_OPCODE_COLLECT ||
          collect->nr_dests != 1 || collect->saturate ||
          collect->dest[0].memory ||
          collect->dest[0].size != I->src[0].size ||
          collect->nr_srcs != I->nr_dests ||
          agx_channels(collect->dest[0]) != collect->nr_srcs ||
          agx_channels(I->src[0]) != I->nr_dests)
         continue;
      bool hinted = false;
      agx_foreach_ssa_dest(I, d) {
         const agx_index dest = I->dest[d], src = collect->src[d];
         if (dest.size != I->src[0].size || dest.memory ||
             agx_channels(dest) != 1 || src.type != AGX_INDEX_NORMAL ||
             src.size != dest.size || src.memory || src.abs || src.neg ||
             agx_channels(src) != 1)
            continue;
         aliases[dest.value] = src;
         if (!hinted) {
            if (!ctx->collect_split_hints.data)
               util_dynarray_init(&ctx->collect_split_hints, ctx);
            agx_builder b = agx_init_builder(ctx, agx_before_instr(collect));
            agx_instr *hint = agx_alloc_instr(
               &b, AGX_OPCODE_COLLECT, 1, collect->nr_srcs);
            hint->dest[0] = collect->dest[0];
            agx_foreach_src(collect, s)
               hint->src[s] = collect->src[s];
            util_dynarray_append(&ctx->collect_split_hints, hint);
            hinted = true;
         }
      }
   }
   for (unsigned i = 0; i < ctx->alloc; ++i) {
      if (agx_is_null(aliases[i]))
         continue;
      agx_index root = aliases[i];
      unsigned steps = 0;
      while (!agx_is_null(aliases[root.value])) {
         if (++steps >= ctx->alloc)
            goto done;
         root = aliases[root.value];
      }
      unsigned value = i;
      while (!agx_is_null(aliases[value])) {
         unsigned next = aliases[value].value;
         aliases[value] = root;
         value = next;
      }
   }
   util_dynarray_foreach(&ctx->collect_split_hints, agx_instr *, hint) {
      agx_instr *I = *hint;
      agx_foreach_ssa_src(I, s) {
         agx_index src = I->src[s], alias = aliases[src.value];
         if (!agx_is_null(alias) && src.size == alias.size && !src.memory &&
             agx_channels(src) == 1)
            agx_replace_src(I, s, alias);
      }
   }
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_src(I, s) {
         const agx_index src = I->src[s], alias = aliases[src.value];
         if (agx_is_null(alias) || src.size != alias.size || src.memory ||
             agx_channels(src) != 1)
            continue;
         agx_replace_src(I, s, alias);
         progress = true;
      }
   }
done:
   free(aliases);
   free(defs);
   return progress;
}

static void
agx_optimizer_copyprop(agx_context *ctx, agx_instr **defs, agx_instr *I)
{
   agx_foreach_ssa_src(I, s) {
      agx_index src = I->src[s];
      agx_instr *def = defs[src.value];

      if (def == NULL)
         continue; /* happens for phis in loops */
      if (def->op != AGX_OPCODE_MOV)
         continue;

      /* At the moment, not all instructions support size conversions. Notably
       * RA pseudo instructions don't handle size conversions. This should be
       * refined in the future.
       */
      if (def->src[0].size != src.size &&
          !(def->src[0].size < src.size && agx_supports_zext(I, s)))
         continue;

      /* AGX2 can carry the zero-extension modifier on integer arithmetic and
       * compare-select, but the grounded G17P forms are width-matched.
       * Preserve the explicit 16-to-32 move so the AGX3 packer emits its
       * witnessed COPY-low plus CLR-high sequence first. ft1 belongs too: its
       * encoder admits a 32-bit register operand only, so the extension must
       * stay materialized. */
      if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          def->src[0].size == AGX_SIZE_16 && src.size == AGX_SIZE_32 &&
          (I->op == AGX_OPCODE_IADD || I->op == AGX_OPCODE_IMAD ||
           I->op == AGX_OPCODE_ICMP || I->op == AGX_OPCODE_ICMPSEL ||
           I->op == AGX_OPCODE_BFI || I->op == AGX_OPCODE_BFEIL ||
           (I->op == AGX_OPCODE_EXTR && s < 2) ||
           I->op == AGX_OPCODE_BITOP || I->op == AGX_OPCODE_BALLOT ||
           I->op == AGX_OPCODE_QUAD_BALLOT ||
           I->op == AGX_OPCODE_ICMP_BALLOT ||
           I->op == AGX_OPCODE_ICMP_QUAD_BALLOT ||
           I->op == AGX_OPCODE_FFS))
         continue;

      /* Optimize split(64-bit uniform) so we can get better copyprop of the
       * 32-bit uniform parts. This helps reduce moves with 64-bit uniforms.
       */
      if (I->op == AGX_OPCODE_SPLIT && def->src[0].type == AGX_INDEX_UNIFORM &&
          src.size == AGX_SIZE_64 && I->dest[0].size == AGX_SIZE_32) {

         assert(I->nr_dests == 2 && "decomposing a 64-bit scalar");
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));

         agx_index lo = def->src[0];
         lo.size = AGX_SIZE_32;

         agx_index hi = lo;
         hi.value += 2 /* half of 64-bits = 32-bits = 2 x 16-bits */;

         defs[I->dest[0].value] = agx_mov_to(&b, I->dest[0], lo);
         defs[I->dest[1].value] = agx_mov_to(&b, I->dest[1], hi);

         agx_remove_instruction(I);
         continue;
      }

      /* Immediate inlining happens elsewhere */
      if (def->src[0].type == AGX_INDEX_IMMEDIATE)
         continue;

      /* ALU instructions cannot take 64-bit. G17P SMP's texture/sampler index
       * operands can (agx_lower_uniform_sources
       * copies back any placement the native form cannot encode). */
      if (def->src[0].size == AGX_SIZE_64 &&
          !(I->op == AGX_OPCODE_DEVICE_LOAD && s == 0) &&
          !(I->op == AGX_OPCODE_DEVICE_STORE && s == 1) &&
          !(I->op == AGX_OPCODE_ATOMIC && s == 1) &&
          !(I->op == AGX_OPCODE_INTERSECT_RAY_GEN1 && s < 2) &&
          !(I->op == AGX_OPCODE_SCS_LOAD_UNIFORM_INDEXED_AGX3 && s == 0) &&
          !(I->op == AGX_OPCODE_SCS_STORE_UNIFORM_INDEXED_AGX3 && s == 1) &&
          !(I->op == AGX_OPCODE_TEXTURE_SAMPLE && (s == 3 || s == 4) &&
            ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
            def->src[0].type == AGX_INDEX_UNIFORM))
         continue;

      /* Forward a descriptor uniform into SMP only when the pair it forms is
       * a native one. Otherwise agx_lower_uniform_sources would copy it back
       * per instruction, where the shared SSA move serves every sample. */
      if (I->op == AGX_OPCODE_TEXTURE_SAMPLE && (s == 3 || s == 4) &&
          def->src[0].type == AGX_INDEX_UNIFORM) {
         agx_index pair[2] = {I->src[3], I->src[4]};
         pair[s - 3] = def->src[0];
         agx_index *other = &pair[4 - s];
         if (other->type == AGX_INDEX_NORMAL) {
            agx_instr *odef = defs[other->value];
            if (odef && odef->op == AGX_OPCODE_MOV &&
                odef->src[0].type == AGX_INDEX_UNIFORM &&
                odef->src[0].size == other->size)
               *other = odef->src[0];
         }
         if (agx3_tex_uniform_pair_unwitnessed(ctx, I, pair[0], pair[1]))
            continue;
      }

      agx_replace_src(I, s, def->src[0]);

      /* If we are zero-extending into an instruction that distinguishes sign
       * and zero extend, make sure we pick zero-extend.
       */
      if (def->src[0].size < src.size &&
          (I->op == AGX_OPCODE_IMAD || I->op == AGX_OPCODE_IADD)) {

         assert(agx_supports_zext(I, s));
         I->src[s].abs = true;
      }
   }
}

/*
 * Fuse conditions into if. Specifically, acts on if_icmp and fuses:
 *
 *    if_icmp(cmp(x, y, *), 0, ne/eq) -> if_cmp(x, y, *)
 */
static void
agx_optimizer_if_cmp(agx_instr **defs, agx_instr *I)
{
   /* Check for unfused if */
   if (!agx_is_equiv(I->src[1], agx_zero()) || I->icond != AGX_ICOND_UEQ ||
       I->src[0].type != AGX_INDEX_NORMAL)
      return;

   /* Check for condition */
   agx_instr *def = defs[I->src[0].value];
   if (def->op != AGX_OPCODE_ICMP && def->op != AGX_OPCODE_FCMP)
      return;

   /* Fuse */
   I->src[0] = def->src[0];
   I->src[1] = def->src[1];
   I->invert_cond = def->invert_cond ^ !I->invert_cond;

   if (def->op == AGX_OPCODE_ICMP) {
      I->op = AGX_OPCODE_IF_ICMP;
      I->icond = def->icond;
   } else {
      I->op = AGX_OPCODE_IF_FCMP;
      I->fcond = def->fcond;
   }
}

/*
 * Fuse invert into if. Acts on if_icmp and fuses:
 *
 *    if_icmp(xor(x, 1), 0, ne) -> if_cmp(x, 0, eq)
 */
static void
agx_optimizer_if_not(agx_instr **defs, agx_instr *I)
{
   /* Check for unfused if */
   if (!agx_is_equiv(I->src[1], agx_zero()) || I->icond != AGX_ICOND_UEQ ||
       I->src[0].type != AGX_INDEX_NORMAL)
      return;

   /* Check for invert */
   agx_instr *def = defs[I->src[0].value];
   if (def->op != AGX_OPCODE_BITOP ||
       !agx_is_equiv(def->src[1], agx_immediate(1)) ||
       def->truth_table != UTIL_LUT2(a ^ b))
      return;

   /* Fuse */
   I->src[0] = def->src[0];
   I->invert_cond = !I->invert_cond;
}

/*
 * Evaluate an integer compare of two immediates on AGX3, which has no compare
 * or select form without a register compare operand. Immediate inlining
 * leaves one behind for e.g. umax(undef, 1), a constant compare can feed a
 * select as a value, and fusing a compare of two constants into a select
 * produces one; robust cube image coordinates do all of these.
 */
bool
agx_eval_immediate_icond(agx_context *ctx, enum agx_icond icond, agx_index x,
                         agx_index y, bool *result)
{
   if (!ctx->key || ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       x.type != AGX_INDEX_IMMEDIATE || y.type != AGX_INDEX_IMMEDIATE ||
       x.neg || y.neg)
      return false;

   /* A 16-bit immediate with its top bit set is negative in a 16-bit signed
    * comparison but not in a 32-bit one, and the comparison width is no
    * longer known here. Below that bit, signed and unsigned orderings agree.
    */
   const bool is_signed = icond == AGX_ICOND_SEQ || icond == AGX_ICOND_SLT ||
                          icond == AGX_ICOND_SGT;
   if (is_signed && (x.value >= 0x8000 || y.value >= 0x8000))
      return false;

   switch (icond) {
   case AGX_ICOND_UEQ:
   case AGX_ICOND_SEQ:
      *result = x.value == y.value;
      return true;
   case AGX_ICOND_ULT:
   case AGX_ICOND_SLT:
      *result = x.value < y.value;
      return true;
   case AGX_ICOND_UGT:
   case AGX_ICOND_SGT:
      *result = x.value > y.value;
      return true;
   default:
      return false;
   }
}

static agx_instr *
agx_optimizer_fold_icmp(agx_context *ctx, agx_instr *I)
{
   bool cond;
   if (!agx_eval_immediate_icond(ctx, I->icond, I->src[0], I->src[1], &cond))
      return NULL;

   agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
   agx_instr *folded =
      agx_mov_imm_to(&b, I->dest[0], cond != I->invert_cond);
   agx_remove_instruction(I);
   return folded;
}

static agx_instr *
agx_optimizer_fold_icmpsel(agx_context *ctx, agx_instr *I)
{
   bool cond;
   if (!agx_eval_immediate_icond(ctx, I->icond, I->src[0], I->src[1], &cond))
      return NULL;

   const agx_index chosen = cond ? I->src[2] : I->src[3];
   if (chosen.neg ||
       !(chosen.type == AGX_INDEX_IMMEDIATE ||
         ((chosen.type == AGX_INDEX_NORMAL ||
           chosen.type == AGX_INDEX_REGISTER) &&
          !chosen.abs)))
      return NULL;

   agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
   agx_instr *folded = chosen.type == AGX_INDEX_IMMEDIATE
                          ? agx_mov_imm_to(&b, I->dest[0], chosen.value)
                          : agx_mov_to(&b, I->dest[0], chosen);
   agx_remove_instruction(I);
   return folded;
}

static bool
agx_same_select_value(agx_index a, agx_index b)
{
   return agx_is_equiv(a, b) && a.size == b.size && a.abs == b.abs &&
          a.neg == b.neg && a.memory == b.memory &&
          a.channels_m1 == b.channels_m1;
}

struct agx_select_domain {
   agx_instr **defs;
   agx_index variable;
   uint64_t cuts[32];
   unsigned count;
};

static bool agx_select_collect_value(struct agx_select_domain *d,
                                     agx_index value, unsigned depth);

static bool
agx_select_plain(agx_index value)
{
   return !value.abs && !value.neg && !value.memory &&
          agx_channels(value) == 1 && value.size <= AGX_SIZE_32;
}

static bool
agx_select_collect_condition(struct agx_select_domain *d, const agx_instr *I,
                             unsigned depth)
{
   if (!depth || !agx_select_plain(I->src[0]) || !agx_select_plain(I->src[1]) ||
       I->icond == 3 || I->icond == 7)
      return false;
   for (unsigned side = 0; side < 2; ++side) {
      if (!agx_same_select_value(I->src[side], d->variable))
         continue;
      const agx_index other = I->src[side ^ 1];
      if (other.type != AGX_INDEX_IMMEDIATE || d->count > 30)
         return false;
      d->cuts[d->count++] = other.value;
      d->cuts[d->count++] = (uint64_t)other.value + 1;
      return true;
   }
   return agx_select_collect_value(d, I->src[0], depth - 1) &&
          agx_select_collect_value(d, I->src[1], depth - 1);
}

static bool
agx_select_collect_value(struct agx_select_domain *d, agx_index value,
                         unsigned depth)
{
   if (!agx_select_plain(value))
      return false;
   if (value.type == AGX_INDEX_IMMEDIATE)
      return true;
   if (!depth || value.type != AGX_INDEX_NORMAL ||
       agx_same_select_value(value, d->variable))
      return false;
   const agx_instr *I = d->defs[value.value];
   if (!I || (I->op != AGX_OPCODE_ICMPSEL && I->op != AGX_OPCODE_ICMP) ||
       I->saturate || (I->op == AGX_OPCODE_ICMPSEL && I->invert_cond) ||
       value.size != I->dest[0].size || !agx_select_plain(I->dest[0]) ||
       !agx_select_collect_condition(d, I, depth - 1))
      return false;
   return I->op == AGX_OPCODE_ICMP ||
          (agx_select_collect_value(d, I->src[2], depth - 1) &&
           agx_select_collect_value(d, I->src[3], depth - 1));
}

static uint32_t agx_select_evaluate_value(const struct agx_select_domain *d,
                                          agx_index value, uint32_t input);

static bool
agx_select_evaluate_condition(const struct agx_select_domain *d,
                              const agx_instr *I, uint32_t input)
{
   uint32_t x = agx_select_evaluate_value(d, I->src[0], input);
   uint32_t y = agx_select_evaluate_value(d, I->src[1], input);
   const bool half =
      I->src[0].size == AGX_SIZE_16 && I->src[1].size == AGX_SIZE_16;
   if (half) {
      x &= 0xffff;
      y &= 0xffff;
   }
   const int32_t sx = half ? (int16_t)x : (int32_t)x;
   const int32_t sy = half ? (int16_t)y : (int32_t)y;
   switch (I->icond) {
   case AGX_ICOND_UEQ:
   case AGX_ICOND_SEQ:
      return x == y;
   case AGX_ICOND_ULT:
      return x < y;
   case AGX_ICOND_UGT:
      return x > y;
   case AGX_ICOND_SLT:
      return sx < sy;
   case AGX_ICOND_SGT:
      return sx > sy;
   default:
      UNREACHABLE("invalid select condition");
   }
}

static uint32_t
agx_select_evaluate_value(const struct agx_select_domain *d, agx_index value,
                          uint32_t input)
{
   if (value.type == AGX_INDEX_IMMEDIATE)
      return value.value;
   if (agx_same_select_value(value, d->variable))
      return input;
   const agx_instr *I = d->defs[value.value];
   const bool condition = agx_select_evaluate_condition(d, I, input);
   if (I->op == AGX_OPCODE_ICMP)
      return condition != I->invert_cond;
   return agx_select_evaluate_value(d, I->src[condition ? 2 : 3], input);
}

static int
agx_select_compare_cut(const void *a, const void *b)
{
   const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
   return (x > y) - (x < y);
}

static void
agx_optimizer_select_guard(agx_context *ctx, agx_instr **defs, agx_instr *I)
{
   if (!ctx->key || ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       I->op != AGX_OPCODE_ICMPSEL || I->saturate || I->invert_cond ||
       agx_channels(I->dest[0]) != 1)
      return;
   for (unsigned branch = 2; branch < 4; ++branch) {
      const agx_index value = I->src[branch];
      if (value.type != AGX_INDEX_NORMAL || !agx_select_plain(value) ||
          value.size != I->dest[0].size)
         continue;
      const agx_instr *inner = defs[value.value];
      if (!inner ||
          (inner->op != AGX_OPCODE_ICMPSEL && inner->op != AGX_OPCODE_ICMP) ||
          inner->saturate ||
          (inner->op == AGX_OPCODE_ICMPSEL && inner->invert_cond))
         continue;
      agx_instr boolean_select;
      agx_index boolean_sources[4];
      if (inner->op == AGX_OPCODE_ICMP) {
         boolean_select = *inner;
         boolean_sources[0] = inner->src[0];
         boolean_sources[1] = inner->src[1];
         boolean_sources[2] = agx_immediate(!inner->invert_cond);
         boolean_sources[3] = agx_immediate(inner->invert_cond);
         boolean_select.src = boolean_sources;
         boolean_select.nr_srcs = 4;
         boolean_select.op = AGX_OPCODE_ICMPSEL;
         boolean_select.invert_cond = false;
         inner = &boolean_select;
      }
      for (unsigned base = 2; base < 4; ++base) {
         if (!agx_same_select_value(inner->src[base], I->src[branch ^ 1]))
            continue;
         for (unsigned side = 0; side < 4; ++side) {
            const agx_index variable =
               side < 2 ? inner->src[side] : I->src[side - 2];
            if (variable.type != AGX_INDEX_NORMAL ||
                !agx_select_plain(variable))
               continue;
            const uint64_t limit = UINT64_C(1)
                                   << (variable.size == AGX_SIZE_16 ? 16 : 32);
            struct agx_select_domain d = {
               .defs = defs,
               .variable = variable,
               .cuts = {0, limit / 2, limit},
               .count = 3,
            };
            if (!agx_select_collect_condition(&d, I, 8) ||
                !agx_select_collect_condition(&d, inner, 8))
               continue;
            qsort(d.cuts, d.count, sizeof(d.cuts[0]), agx_select_compare_cut);
            bool active[32] = {false};
            for (unsigned c = 0; c + 1 < d.count; ++c) {
               active[c] = agx_select_evaluate_condition(&d, I, d.cuts[c]) ==
                              (branch == 2) &&
                           agx_select_evaluate_condition(
                              &d, inner, d.cuts[c]) == (base == 3);
            }
            bool any = false, all = true;
            for (unsigned c = 0; c + 1 < d.count; ++c) {
               if (d.cuts[c] == d.cuts[c + 1])
                  continue;
               any |= active[c];
               all &= active[c];
            }
            if (!any || all) {
               I->src[0] = I->src[1] = agx_zero();
               I->src[2] = I->src[3] = inner->src[all ? (base ^ 1) : base];
               I->icond = AGX_ICOND_UEQ;
               return;
            }
            const enum agx_icond conditions[] = {
               AGX_ICOND_UEQ, AGX_ICOND_ULT, AGX_ICOND_UGT,
               AGX_ICOND_SLT, AGX_ICOND_SGT,
            };
            for (unsigned op = 0; op < ARRAY_SIZE(conditions); ++op) {
               for (unsigned c = 0; c < d.count; ++c) {
                  for (unsigned delta = 0; delta < 2; ++delta) {
                     const uint64_t constant = d.cuts[c] - delta;
                     if (constant >= 0x10000)
                        continue;
                     agx_index sources[2] = {variable, agx_immediate(constant)};
                     agx_instr comparison = {
                        .src = sources,
                        .icond = conditions[op],
                     };
                     bool match = true, inverse = true;
                     for (unsigned cell = 0; cell + 1 < d.count; ++cell) {
                        if (d.cuts[cell] == d.cuts[cell + 1])
                           continue;
                        const bool first = agx_select_evaluate_condition(
                           &d, &comparison, d.cuts[cell]);
                        const bool last = agx_select_evaluate_condition(
                           &d, &comparison, d.cuts[cell + 1] - 1);
                        match &= first == active[cell] && last == active[cell];
                        inverse &=
                           first != active[cell] && last != active[cell];
                        if (!match && !inverse)
                           break;
                     }
                     if (!match && !inverse)
                        continue;
                     I->src[0] = sources[0];
                     I->src[1] = sources[1];
                     I->src[2] = inner->src[inverse ? base : (base ^ 1)];
                     I->src[3] = inner->src[inverse ? (base ^ 1) : base];
                     I->icond = comparison.icond;
                     return;
                  }
               }
            }
         }
      }
   }
}

/*
 * Fuse conditions into select. Specifically, acts on icmpsel and fuses:
 *
 *    icmpsel(cmp(x, y, *), 0, z, w, eq) -> cmpsel(x, y, w, z, *)
 *
 * Care must be taken to invert the condition by swapping cmpsel arguments.
 */
static void
agx_optimizer_cmpsel(agx_context *ctx, agx_instr **defs, agx_instr *I)
{
   /* Check for unfused select */
   if (!agx_is_equiv(I->src[1], agx_zero()) || I->icond != AGX_ICOND_UEQ ||
       I->src[0].type != AGX_INDEX_NORMAL)
      return;

   /* Check for condition */
   agx_instr *def = defs[I->src[0].value];
   if (!def)
      return;
   if (def->op == AGX_OPCODE_ICMPSEL && !def->saturate && !def->invert_cond &&
       agx_select_plain(I->src[0]) && I->src[0].size == def->dest[0].size &&
       agx_select_plain(def->src[2]) && agx_select_plain(def->src[3])) {
      bool yes, no;
      if (!agx_eval_immediate_icond(ctx, I->icond, def->src[2], I->src[1],
                                    &yes) ||
          !agx_eval_immediate_icond(ctx, I->icond, def->src[3], I->src[1], &no))
         return;
      const agx_index a = I->src[yes ? 2 : 3];
      const agx_index b = I->src[no ? 2 : 3];
      I->src[0] = yes == no ? agx_zero() : def->src[0];
      I->src[1] = yes == no ? agx_zero() : def->src[1];
      I->src[2] = a;
      I->src[3] = b;
      I->icond = yes == no ? AGX_ICOND_UEQ : def->icond;
      return;
   }
   if (def->op != AGX_OPCODE_ICMP && def->op != AGX_OPCODE_FCMP)
      return;

   /* Fuse */
   I->src[0] = def->src[0];
   I->src[1] = def->src[1];

   /* In the unfused select, the condition is inverted due to the form:
    *
    *    (cond == 0) ? x : y
    *
    * So we need to swap the arguments when fusing to become cond ? y : x. If
    * the condition was supposed to be inverted, we don't swap since it's
    * already inverted. cmpsel does not have an invert_cond bit to use.
    */
   if (!def->invert_cond) {
      agx_index temp = I->src[2];
      I->src[2] = I->src[3];
      I->src[3] = temp;
   }

   if (def->op == AGX_OPCODE_ICMP) {
      I->op = AGX_OPCODE_ICMPSEL;
      I->icond = def->icond;
   } else {
      I->op = AGX_OPCODE_FCMPSEL;
      I->fcond = def->fcond;
   }
}

/*
 * Fuse conditions into ballots:
 *
 *    ballot(cmp(x, y)) -> ballot_cmp(x, y)
 */
static void
agx_optimizer_ballot(agx_context *ctx, agx_instr **defs, agx_instr *I)
{
   /* The grounded G17P ballot form compares a full 32-bit register against
    * zero.  Common subgroup reduction lowering legitimately creates
    * ballot(true) to distinguish full from partial active subgroups, and the
    * immediate inliner runs before this fusion pass.  Materialize that exact
    * boolean here so lower_pseudo reaches the witnessed register ballot
    * instead of leaving an unencodable immediate predicate after RA. */
   if (I->op == AGX_OPCODE_BALLOT && ctx->key &&
      ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
      I->src[0].type == AGX_INDEX_IMMEDIATE) {
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_index predicate = agx_temp(ctx, AGX_SIZE_32);
      agx_mov_imm_to(&b, predicate, I->src[0].value);
      I->src[0] = predicate;
      return;
   }

   /* Likewise for a quad ballot, whose witnessed register form takes a 16-bit
    * predicate. */
   if (I->op == AGX_OPCODE_QUAD_BALLOT && ctx->key &&
      ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
      I->src[0].type == AGX_INDEX_IMMEDIATE) {
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_index predicate = agx_temp(ctx, AGX_SIZE_16);
      agx_mov_imm_to(&b, predicate, I->src[0].value);
      I->src[0] = predicate;
      return;
   }

   if (I->src[0].type != AGX_INDEX_NORMAL)
      return;

   agx_instr *def = defs[I->src[0].value];

   /* The same register-against-zero ballot is witnessed on G17P for a 16-bit
    * source only in its equal-to-zero form, while an unfused ballot counts
    * the lanes where its predicate is nonzero. Widen a half predicate so it
    * reaches the 32-bit form instead.
    */
   if (I->op == AGX_OPCODE_BALLOT && ctx->key &&
       ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       I->src[0].size == AGX_SIZE_16 &&
       !(def && (def->op == AGX_OPCODE_ICMP || def->op == AGX_OPCODE_FCMP))) {
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_index wide = agx_temp(ctx, AGX_SIZE_32);
      agx_mov_to(&b, wide, I->src[0]);
      I->src[0] = wide;
      return;
   }

   if (!def || (def->op != AGX_OPCODE_ICMP && def->op != AGX_OPCODE_FCMP))
      return;

   bool quad = I->op == AGX_OPCODE_QUAD_BALLOT;
   assert(quad || I->op == AGX_OPCODE_BALLOT);

   /* Replace with a fused instruction since the # of sources changes */
   agx_builder b = agx_init_builder(ctx, agx_before_instr(I));

   /* Every G17P SIMD ballot Apple's compiler emits compares one register
    * against a constant, so a ballot of two registers has no witnessed
    * encoding. Keep the comparison, with the opposite sense, as a boolean and
    * ballot the lanes where it is zero: the witnessed simd_broadcast_first()
    * form.
    */
   /* `x != N` for a nonzero N has no immediate form, and the register-zero
    * equality form it would otherwise take is measured wrong on G17P (see
    * agx3_emit_subgroup_ballot). Ballot the opposite compare instead, exactly
    * as for a register operand below. */
   const bool agx3_inverted_eq_imm =
      def->op == AGX_OPCODE_ICMP && def->invert_cond &&
      (def->icond == AGX_ICOND_UEQ || def->icond == AGX_ICOND_SEQ) &&
      def->src[1].type == AGX_INDEX_IMMEDIATE && def->src[1].value != 0;

   /* G17P's fused float ballot compares the register in f16 against a
    * second operand -- a minifloat immediate, or another register (the
    * simd_broadcast_first() allequal witness compares two registers). Take
    * the fused form whenever the second operand is register or immediate;
    * the packer refuses modified/memory operands, which keep the
    * opposite-compare fallback below. */
   /* Only EQ with a register second operand is byte-verified against
    * hardware output (the allequal witness); the immediate-operand and
    * non-EQ register forms are encoder-table derived and miscompile on
    * real G17P (sg-vote-3 all-lanes-equal bug). ASAHI_G17P_FBALLOT
    * re-enables the unwitnessed fused forms for probing. */
   const bool fused_fballot =
      !quad && ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
      def->op == AGX_OPCODE_FCMP &&
      (def->fcond == AGX_FCOND_EQ || def->fcond == AGX_FCOND_LT ||
       def->fcond == AGX_FCOND_GT || def->fcond == AGX_FCOND_GE ||
       def->fcond == AGX_FCOND_LE) &&
      ((def->fcond == AGX_FCOND_EQ &&
        def->src[0].type == AGX_INDEX_NORMAL &&
        def->src[1].type == AGX_INDEX_NORMAL) ||
       debug_get_bool_option("ASAHI_G17P_FBALLOT", false)) &&
      /* src0 may be an immediate: it swaps into the second-operand slot
       * with the reversed condition below. */
      ((def->src[0].type == AGX_INDEX_NORMAL &&
        (def->src[1].type == AGX_INDEX_NORMAL ||
         def->src[1].type == AGX_INDEX_IMMEDIATE)) ||
       (def->src[0].type == AGX_INDEX_IMMEDIATE &&
        def->src[1].type == AGX_INDEX_NORMAL)) &&
      !def->src[1].memory && !def->src[1].abs && !def->src[1].neg &&
      !def->src[1].kill && !def->src[0].memory && !def->src[0].abs &&
      !def->src[0].neg && !def->src[0].kill;

   if (!quad && ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       !fused_fballot &&
       (def->src[1].type != AGX_INDEX_IMMEDIATE || agx3_inverted_eq_imm)) {
      agx_index opposite = agx_temp(ctx, AGX_SIZE_16);

      if (def->op == AGX_OPCODE_ICMP) {
         agx_icmp_to(&b, opposite, def->src[0], def->src[1], def->icond,
                     !def->invert_cond);
      } else {
         agx_fcmp_to(&b, opposite, def->src[0], def->src[1], def->fcond,
                     !def->invert_cond);
      }

      agx_icmp_ballot_to(&b, I->dest[0], opposite, agx_zero(), AGX_ICOND_UEQ,
                         false);
      agx_remove_instruction(I);
      return;
   }

   /* A float compare with the immediate first exchanges its operands so the
    * register lands in the ballot's f16 source; the reversed condition keeps
    * the comparison's meaning. */
   agx_index bs0 = def->src[0], bs1 = def->src[1];
   enum agx_fcond fcond = def->fcond;
   if (def->op == AGX_OPCODE_FCMP &&
       def->src[0].type == AGX_INDEX_IMMEDIATE) {
      bs0 = def->src[1];
      bs1 = def->src[0];
      switch (fcond) {
      case AGX_FCOND_LT: fcond = AGX_FCOND_GT; break;
      case AGX_FCOND_GT: fcond = AGX_FCOND_LT; break;
      case AGX_FCOND_LE: fcond = AGX_FCOND_GE; break;
      case AGX_FCOND_GE: fcond = AGX_FCOND_LE; break;
      default: break;
      }
   }

   agx_instr *fused = agx_icmp_ballot_to(
      &b, I->dest[0], bs0, bs1, def->icond, def->invert_cond);

   if (def->op == AGX_OPCODE_ICMP) {
      fused->op = quad ? AGX_OPCODE_ICMP_QUAD_BALLOT : AGX_OPCODE_ICMP_BALLOT;
   } else {
      fused->op = quad ? AGX_OPCODE_FCMP_QUAD_BALLOT : AGX_OPCODE_FCMP_BALLOT;
      fused->fcond = fcond;
   }

   agx_remove_instruction(I);
}

/*
 * Fuse not srcs into bitop.
 */
static void
agx_optimizer_bitop(agx_context *ctx, agx_instr **defs, agx_instr *I)
{
   agx_foreach_ssa_src(I, s) {
      agx_index src = I->src[s];
      agx_instr *def = defs[src.value];

      /* If we find a not, select new operation and fuse */
      if (def->op == AGX_OPCODE_NOT) {
         const unsigned truth_table =
            util_lut2_invert_source(I->truth_table, s);
         agx_index other = I->src[1 - s];
         if (other.type == AGX_INDEX_NORMAL) {
            const agx_instr *other_def = defs[other.value];
            if (other_def && other_def->op == AGX_OPCODE_NOT &&
                other_def->src[0].size == AGX_SIZE_32)
               other = other_def->src[0];
         }
         /* Enable the witnessed uniform forms. Keep the existing restriction
          * for register-only expressions until their generated code is
          * profitable as well as their encoding being supported.
          */
         const bool native_uniform_word =
            I->dest[0].size == AGX_SIZE_32 &&
            I->src[0].size == AGX_SIZE_32 &&
            I->src[1].size == AGX_SIZE_32 &&
            def->src[0].size == AGX_SIZE_32 &&
            (def->src[0].type == AGX_INDEX_UNIFORM ||
             other.type == AGX_INDEX_UNIFORM);

         /* The two-register truth tables the AGX3B packer encodes as their
          * own instructions: bit_nor (0x1), bit_andc either side (0x2/0x4),
          * bit_nand (0x7), bit_xnor (0x9) and bit_orc either side (0xb/0xd).
          * They take register operands only, so require a 32-bit register on
          * both sides after the fuse. */
         const bool named_reg_word =
            I->dest[0].size == AGX_SIZE_32 &&
            I->src[0].size == AGX_SIZE_32 &&
            I->src[1].size == AGX_SIZE_32 &&
            def->src[0].size == AGX_SIZE_32 &&
            def->src[0].type == AGX_INDEX_NORMAL &&
            I->src[1 - s].type == AGX_INDEX_NORMAL &&
            (truth_table == 0x1 || truth_table == 0x2 ||
             truth_table == 0x4 || truth_table == 0x7 ||
             truth_table == 0x9 || truth_table == 0xb ||
             truth_table == 0xd);
         if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
             !native_uniform_word && !named_reg_word &&
             truth_table != AGX3_SEL_TT_AND &&
             truth_table != AGX3_SEL_TT_OR &&
             truth_table != AGX3_SEL_TT_XOR &&
             truth_table != AGX3_SEL_TT_NOT)
            continue;

         I->truth_table = truth_table;
         I->src[s] = def->src[0];
      }
   }
}

/*
 * Fuse sign-extends into addition-like instructions:
 */
static void
agx_optimizer_signext(agx_instr **defs, agx_instr *I)
{
   agx_foreach_ssa_src(I, s) {
      agx_index src = I->src[s];
      agx_instr *def = defs[src.value];

      if (def == NULL || def->op != AGX_OPCODE_SIGNEXT)
         continue;

      agx_replace_src(I, s, def->src[0]);
      assert(!I->src[s].abs && "sign-extended");
   }
}

/* ====== AGX3B peepholes ====== */

static bool
agx3_imm_value(agx_index x, uint64_t *out)
{
   if (x.type != AGX_INDEX_IMMEDIATE || x.abs || x.neg || x.kill)
      return false;
   *out = x.value;
   return true;
}

/* x << s with a zero base and a zero insert mask is the NIR ishl spelling on
 * this backend. Returns the 32-bit value and the immediate shift. */
static bool
agx3_match_shl(agx_instr *D, agx_index *value, unsigned *shift)
{
   uint64_t base, s;
   if (!D || D->op != AGX_OPCODE_BFI || D->nr_srcs != 3 || D->bfi_mask != 0 ||
       D->saturate || !agx3_imm_value(D->src[0], &base) || base != 0 ||
       !agx3_imm_value(D->src[2], &s) || s >= 32 ||
       D->src[1].type != AGX_INDEX_NORMAL ||
       D->src[1].size != AGX_SIZE_32 || D->src[1].abs || D->src[1].neg)
      return false;
   *value = D->src[1];
   *shift = s;
   return true;
}

static agx_index agx3_peel_copy(agx_instr **defs, agx_index x);

/* (v >> s) or the masked extract (v >> s) & ((1 << m) - 1): a bfeil with a
 * zero base and an immediate offset. */
static bool
agx3_match_srl(agx_instr *D, agx_index *value, unsigned *shift,
               unsigned *width)
{
   uint64_t base, s;
   if (!D || D->op != AGX_OPCODE_BFEIL || D->nr_srcs != 3 || D->saturate ||
       !agx3_imm_value(D->src[0], &base) || base != 0 ||
       !agx3_imm_value(D->src[2], &s) || s >= 32 ||
       D->src[1].type != AGX_INDEX_NORMAL ||
       D->src[1].size != AGX_SIZE_32 || D->src[1].abs || D->src[1].neg)
      return false;
   *value = D->src[1];
   *shift = s;
   *width = D->bfi_mask;
   return true;
}

/* iadd(0, x) and mov(x) are the plain 32-bit copies this backend emits when a
 * later instruction's operand slot needs a bare register (shift boundaries,
 * select sources). Chase through them when looking for a value's producer. */
static agx_index
agx3_peel_copy(agx_instr **defs, agx_index x)
{
   for (unsigned i = 0; i < 8 && x.type == AGX_INDEX_NORMAL; ++i) {
      agx_instr *D = defs[x.value];
      if (!D || D->nr_dests != 1 || D->saturate)
         break;
      if (D->op == AGX_OPCODE_MOV && D->nr_srcs == 1 &&
          D->src[0].type == AGX_INDEX_NORMAL &&
          D->src[0].size == x.size && !D->src[0].abs && !D->src[0].neg) {
         x = D->src[0];
         continue;
      }
      if (D->op == AGX_OPCODE_IADD && D->nr_srcs == 2 && D->shift == 0 &&
          D->src[0].type == AGX_INDEX_IMMEDIATE && D->src[0].value == 0 &&
          !D->src[0].abs && !D->src[0].neg &&
          D->src[1].type == AGX_INDEX_NORMAL &&
          D->src[1].size == x.size && !D->src[1].abs && !D->src[1].neg) {
         x = D->src[1];
         continue;
      }
      break;
   }
   return x;
}

/* Where a narrow field lives inside a packed 32-bit word. */
struct agx3_field {
   /* The packed word, or the 16-bit value itself when `direct`. */
   agx_index base;
   unsigned bitpos; /* bit offset of the field's low bit in `base` */
   unsigned width;  /* 8 or 16 */
   bool direct;     /* base is already the 16-bit narrow value */
};

/* If the 32-bit SSA value `x` is a zero- or sign-extension of a byte or
 * halfword lane of a packed 32-bit value, describe where the field sits so
 * the caller can feed the narrow value to a native narrow convert instead.
 * The narrow converter reads only the low 8/16 bits of a half register, so
 * no masking is needed at the new operand.
 */
static bool
agx3_match_field_extract(agx_instr **defs, agx_index x, bool is_signed,
                         struct agx3_field *out)
{
   agx_instr *D = defs[x.value];
   if (!D || D->nr_dests != 1 || D->saturate)
      return false;

   /* i2i32 / u2u32 of a half is already the narrow value. */
   if (is_signed && D->op == AGX_OPCODE_SIGNEXT &&
       D->src[0].type == AGX_INDEX_NORMAL &&
       D->src[0].size == AGX_SIZE_16 && !D->src[0].abs && !D->src[0].neg) {
      *out = (struct agx3_field){0};
      out->base = D->src[0];
      out->width = 16;
      out->direct = true;
      return true;
   }
   if (!is_signed && D->op == AGX_OPCODE_MOV &&
       D->src[0].type == AGX_INDEX_NORMAL &&
       D->src[0].size == AGX_SIZE_16 && !D->src[0].abs && !D->src[0].neg) {
      *out = (struct agx3_field){0};
      out->base = D->src[0];
      out->width = 16;
      out->direct = true;
      return true;
   }

   if (x.size != AGX_SIZE_32)
      return false;

   /* (v << s) >>a 24 sign-extends byte (24 - s) / 8; (v << 16) >>a 16 and
    * v >>a 16 sign-extend a halfword lane. These are how extract_i8/i16
    * arrive. */
   if (is_signed && D->op == AGX_OPCODE_ASR) {
      uint64_t count;
      if (!agx3_imm_value(D->src[1], &count) ||
          D->src[0].type != AGX_INDEX_NORMAL ||
          D->src[0].size != AGX_SIZE_32 || D->src[0].abs || D->src[0].neg)
         return false;
      agx_index v = agx3_peel_copy(defs, D->src[0]), w;
      unsigned s;
      agx_instr *E = defs[v.value];
      if (count == 24) {
         if (agx3_match_shl(E, &w, &s) && s <= 24 && (s % 8) == 0)
            *out = (struct agx3_field){.base = agx3_peel_copy(defs, w),
                                       .bitpos = 24 - s, .width = 8};
         else
            *out = (struct agx3_field){.base = v, .bitpos = 24, .width = 8};
         return true;
      }
      if (count == 16) {
         if (agx3_match_shl(E, &w, &s) && s == 16)
            *out = (struct agx3_field){.base = agx3_peel_copy(defs, w),
                                       .bitpos = 0, .width = 16};
         else
            *out = (struct agx3_field){.base = v, .bitpos = 16, .width = 16};
         return true;
      }
      return false;
   }

   if (is_signed)
      return false;

   /* Unsigned lanes: (v >> lsb) [& mask] produced by bfeil, with or without a
    * width mask. The convert reads only the low bits of the half it is
    * handed, so an unmasked srl is still an extract of its low lane. */
   agx_index v;
   unsigned lsb, width;
   if (agx3_match_srl(D, &v, &lsb, &width)) {
      v = agx3_peel_copy(defs, v);
      if ((lsb % 8) == 0 && lsb <= 24 &&
          (width == 0 || width == 8 || lsb == 0)) {
         /* masked byte extract, or a plain shift whose low byte is a lane */
         *out = (struct agx3_field){.base = v, .bitpos = lsb, .width = 8};
         return true;
      }
      if (width == 16 && (lsb == 0 || lsb == 16)) {
         *out = (struct agx3_field){.base = v, .bitpos = lsb, .width = 16};
         return true;
      }
      if (width == 0 && lsb == 16) {
         *out = (struct agx3_field){.base = v, .bitpos = 16, .width = 16};
         return true;
      }
      return false;
   }

   /* and(y, 0xff) / and(y, 0xffff), where y itself may be a shift. */
   if (D->op == AGX_OPCODE_BITOP && D->nr_srcs == 2 &&
       D->truth_table == AGX3_SEL_TT_AND &&
       D->dest[0].size == AGX_SIZE_32) {
      for (unsigned i = 0; i < 2; ++i) {
         agx_index a = D->src[i], y = D->src[i ^ 1];
         uint64_t mask;
         if (!agx3_imm_value(a, &mask) || y.type != AGX_INDEX_NORMAL ||
             y.size != AGX_SIZE_32 || y.abs || y.neg)
            continue;

         y = agx3_peel_copy(defs, y);
         agx_instr *E = defs[y.value];
         agx_index v2;
         unsigned lsb2, w2;
         if (mask == 0xff) {
            if (agx3_match_srl(E, &v2, &lsb2, &w2) && lsb2 <= 24 &&
                (lsb2 % 8) == 0)
               *out = (struct agx3_field){.base = agx3_peel_copy(defs, v2),
                                          .bitpos = lsb2, .width = 8};
            else
               *out = (struct agx3_field){.base = y, .bitpos = 0, .width = 8};
            return true;
         }
         if (mask == 0xffff) {
            if (agx3_match_srl(E, &v2, &lsb2, &w2) && lsb2 == 16)
               *out = (struct agx3_field){.base = agx3_peel_copy(defs, v2),
                                          .bitpos = 16, .width = 16};
            else
               *out = (struct agx3_field){.base = y, .bitpos = 0,
                                          .width = 16};
            return true;
         }
      }
   }
   return false;
}

/* u8/s8/u16/s16 -> f16/f32 conversions lower in NIR to a byte/halfword
 * extract followed by a 32-bit convert: `(v << 16) >>a 24` plus convert, or
 * `(v >> 8) & 0xff` plus convert. G17P's int2flt reads the low 8 or 16 bits
 * of a half register directly, so the field only has to reach the low bits
 * of a half -- a single logical shift plus the (free) subdivide, or nothing
 * at all when the value already lives in a half. */
static void
agx_optimizer_narrow_convert(agx_context *ctx, agx_instr **defs, agx_instr *I)
{
   if (!ctx->key || ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       I->nr_srcs != 2 || I->nr_dests != 1 || I->saturate ||
       I->round != AGX_ROUND_RTE ||
       I->src[0].type != AGX_INDEX_IMMEDIATE ||
       (I->src[0].value != AGX_CONVERT_S32_TO_F &&
        I->src[0].value != AGX_CONVERT_U32_TO_F) ||
       I->src[1].type != AGX_INDEX_NORMAL ||
       I->src[1].size != AGX_SIZE_32 || I->src[1].abs || I->src[1].neg)
      return;

   const bool is_signed = I->src[0].value == AGX_CONVERT_S32_TO_F;
   struct agx3_field f;
   if (!agx3_match_field_extract(defs, I->src[1], is_signed, &f))
      return;

   agx_index half = f.base;
   if (!f.direct) {
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_index w = f.base;
      if (f.bitpos) {
         w = agx_temp(ctx, AGX_SIZE_32);
         agx_bfeil_to(&b, w, agx_zero(), f.base, agx_immediate(f.bitpos), 0);
      }
      half = agx_temp(ctx, AGX_SIZE_16);
      agx_instr *split = agx_split(&b, 2, w);
      split->dest[0] = half;
      split->dest[1] = agx_temp(ctx, AGX_SIZE_16);
   }

   I->src[0] = agx_immediate(f.width == 8
                                ? (is_signed ? AGX_CONVERT_S8_TO_F
                                             : AGX_CONVERT_U8_TO_F)
                                : (is_signed ? AGX_CONVERT_S16_TO_F
                                             : AGX_CONVERT_U16_TO_F));
   I->src[1] = half;
}

/* iadd #0, x is how this backend materialises a 32-bit register copy for the
 * G17P shift/compare boundary forms. Fold copy-of-copy chains the helpers
 * leave behind: iadd(0, iadd(0, x)) and iadd(0, mov x) are just iadd(0, x). */
static void
agx_optimizer_iadd_copy(agx_context *ctx, agx_instr **defs, agx_instr *I)
{
   if (!ctx->key || ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       I->op != AGX_OPCODE_IADD || I->nr_srcs != 2 || I->nr_dests != 1 ||
       I->shift != 0 || I->saturate ||
       I->src[0].type != AGX_INDEX_IMMEDIATE || I->src[0].value != 0 ||
       I->src[0].abs || I->src[0].neg ||
       I->src[1].type != AGX_INDEX_NORMAL || I->src[1].abs ||
       I->src[1].neg || I->src[1].size != I->dest[0].size)
      return;

   agx_instr *D = defs[I->src[1].value];
   if (!D || D->nr_dests != 1 || D->saturate)
      return;

   if (D->op == AGX_OPCODE_IADD && D->nr_srcs == 2 && D->shift == 0 &&
       D->src[0].type == AGX_INDEX_IMMEDIATE && D->src[0].value == 0 &&
       !D->src[0].abs && !D->src[0].neg &&
       D->src[1].type == AGX_INDEX_NORMAL && !D->src[1].abs &&
       !D->src[1].neg && D->src[1].size == I->src[1].size) {
      I->src[1] = D->src[1];
   } else if (D->op == AGX_OPCODE_MOV && D->nr_srcs == 1 &&
              D->src[0].type == AGX_INDEX_NORMAL &&
              D->src[0].size == I->src[1].size && !D->src[0].abs &&
              !D->src[0].neg) {
      I->src[1] = D->src[0];
   }
}

/* Drop masks that a fixed left shift makes redundant:
 *
 *    bfi(0, and(x, M), s) == bfi(0, x, s)  when M covers [0, 32 - s)
 *    and(bfi(0, x, s), M) == bfi(0, x, s)  when M covers [s, 32)
 */
static void
agx_optimizer_pack_mask(agx_context *ctx, agx_instr **defs, agx_instr *I)
{
   if (!ctx->key || ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       I->nr_dests != 1 || I->dest[0].size != AGX_SIZE_32 || I->saturate)
      return;

   if (I->op == AGX_OPCODE_BFI && I->nr_srcs == 3 && I->bfi_mask == 0 &&
       I->src[1].type == AGX_INDEX_NORMAL &&
       I->src[2].type == AGX_INDEX_IMMEDIATE) {
      const unsigned s = I->src[2].value;
      if (s == 0 || s >= 32)
         return;
      agx_instr *D = defs[I->src[1].value];
      if (!D || D->op != AGX_OPCODE_BITOP || D->nr_srcs != 2 ||
          D->truth_table != AGX3_SEL_TT_AND ||
          D->dest[0].size != AGX_SIZE_32 || D->saturate)
         return;
      const uint64_t need = BITFIELD64_MASK(32 - s);
      for (unsigned i = 0; i < 2; ++i) {
         agx_index a = D->src[i], b = D->src[i ^ 1];
         uint64_t mask;
         if (!agx3_imm_value(a, &mask) || (mask & need) != need ||
             b.type != AGX_INDEX_NORMAL || b.size != AGX_SIZE_32 || b.abs ||
             b.neg)
            continue;
         I->src[1] = b;
         return;
      }
      return;
   }

   if (I->op == AGX_OPCODE_BITOP && I->nr_srcs == 2 &&
       I->truth_table == AGX3_SEL_TT_AND) {
      for (unsigned i = 0; i < 2; ++i) {
         agx_index a = I->src[i], b = I->src[i ^ 1];
         uint64_t mask;
         if (!agx3_imm_value(a, &mask) || b.type != AGX_INDEX_NORMAL ||
             b.size != AGX_SIZE_32 || b.abs || b.neg)
            continue;
         agx_instr *D = defs[b.value];
         if (!D || D->op != AGX_OPCODE_BFI || D->nr_srcs != 3 ||
             D->bfi_mask != 0 || D->saturate)
            continue;
         uint64_t base, sh;
         if (!agx3_imm_value(D->src[0], &base) || base != 0 ||
             !agx3_imm_value(D->src[2], &sh) || sh == 0 || sh >= 32)
            continue;
         const uint64_t keep = ~BITFIELD64_MASK(sh) & UINT32_C(0xffffffff);
         if ((mask & keep) != keep)
            continue;
         I->op = AGX_OPCODE_MOV;
         I->nr_srcs = 1;
         I->src[0] = b;
         return;
      }
   }
}

/* G17P's lddev/stdev add a sign-extended 16-bit byte offset on top of
 * base + (index << shift) in the register-index form too (lddev f76,
 * stdev f74). An index materialized as `iadd reg, #delta` therefore folds
 * into the access itself: the load/store keeps the register operand and
 * carries delta << scale in I->mem_offset. The offset is signed, so this
 * also covers index subtraction emitted as add of a wrapped immediate.
 */
static void
agx_optimizer_device_index_offset(agx_context *ctx, agx_instr **defs,
                                  agx_instr *I)
{
   if (!ctx->key || ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       (I->op != AGX_OPCODE_DEVICE_LOAD && I->op != AGX_OPCODE_DEVICE_STORE))
      return;

   const unsigned is = (I->op == AGX_OPCODE_DEVICE_STORE) ? 2 : 1;

   /* The index counts elements of `1 << total_shift` bytes; same
    * width table as agx3_pack_binary's agx3_fmt_to_width(). */
   unsigned w;
   switch (I->format) {
   case AGX_FORMAT_I8:
      w = 0;
      break;
   case AGX_FORMAT_I16:
   case AGX_FORMAT_F16:
      w = 1;
      break;
   case AGX_FORMAT_I32:
      w = 2;
      break;
   default:
      return; /* norm/packed formats have no AGX3 device-op witness */
   }
   const unsigned total_shift =
      I->shift + (I->memory_index_is_bytes ? 0 : w);
   if (total_shift > 15)
      return;

   /* Chase a chain of constant addends: iadd(iadd(reg, #a), #b) carries
    * (a + b) in the same field. Dead iadds are removed by DCE. */
   for (;;) {
      const agx_index idx = I->src[is];
      if (idx.type != AGX_INDEX_NORMAL || idx.memory || idx.neg ||
          idx.size != AGX_SIZE_32)
         return;

      agx_instr *D = defs[idx.value];
      if (!D || D->op != AGX_OPCODE_IADD || D->nr_srcs != 2 ||
          D->nr_dests != 1 || D->saturate || D->shift != 0 ||
          D->dest[0].size != AGX_SIZE_32)
         return;

      bool folded = false;
      for (unsigned i = 0; i < 2; ++i) {
         const agx_index imm = D->src[i];
         const agx_index reg = D->src[i ^ 1];
         if (reg.type != AGX_INDEX_NORMAL || reg.abs || reg.neg ||
             reg.memory || reg.size != AGX_SIZE_32)
            continue;

         /* The addend is either an inline immediate or a materialized
          * mov_imm (the shape a delta too large for the iadd field takes).
          * Interpret it signed so wrapped deltas (index - k) land in the
          * signed field as well. */
         int64_t delta;
         if (imm.type == AGX_INDEX_IMMEDIATE && !imm.abs && !imm.neg) {
            delta = (int32_t)imm.value;
         } else if (imm.type == AGX_INDEX_NORMAL && !imm.abs && !imm.neg &&
                    !imm.memory && imm.size == AGX_SIZE_32) {
            const agx_instr *M = defs[imm.value];
            if (!M || M->op != AGX_OPCODE_MOV_IMM || M->nr_dests != 1 ||
                M->saturate || M->dest[0].size != AGX_SIZE_32)
               continue;
            delta = (int32_t)M->imm;
         } else {
            continue;
         }

         const int64_t bytes = delta << total_shift;
         const int64_t merged = (int64_t)I->mem_offset + bytes;
         if (merged < -32768 || merged > 32767)
            continue;

         I->src[is] = reg;
         I->src[is].abs = idx.abs;
         I->src[is].kill = idx.kill;
         I->mem_offset = (int32_t)merged;
         folded = true;
         break;
      }
      if (!folded)
         return;
   }
}

/* and(x, 0x7fffffff) feeding a float compare is exactly fabs(x); likewise
 * xor(x, 0x80000000) is fneg(x). The G17P compare-select carries .abs and
 * .neg on the compared operands (agx3_fsel_desc's cmp{0,1}_abs/neg), so the
 * separate bitop is dead. Half compares use the corresponding 16-bit masks. */
static void
agx_optimizer_fcmp_mods(agx_context *ctx, agx_instr **defs, agx_instr *I)
{
   if (!ctx->key || ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       (I->op != AGX_OPCODE_FCMPSEL && I->op != AGX_OPCODE_FCMP) ||
       I->saturate || I->nr_srcs < 2)
      return;

   /* The .abs form is grounded for eq/lt/gt, and for ge/le with exactly one
    * absolute comparator; ltn/gtn compare select has no modifier form. */
   if (I->fcond != AGX_FCOND_EQ && I->fcond != AGX_FCOND_LT &&
       I->fcond != AGX_FCOND_GT && I->fcond != AGX_FCOND_GE &&
       I->fcond != AGX_FCOND_LE)
      return;

   const uint64_t sign = I->src[0].size == AGX_SIZE_16 ? 0x8000 : 0x80000000;
   const uint64_t notsign = sign - 1;

   for (unsigned s = 0; s < 2; ++s) {
      agx_index src = I->src[s];
      if (src.type != AGX_INDEX_NORMAL || src.abs || src.neg ||
          (src.size != AGX_SIZE_32 && src.size != AGX_SIZE_16))
         continue;
      agx_instr *D = defs[src.value];
      if (!D || D->op != AGX_OPCODE_BITOP || D->nr_srcs != 2 ||
          D->dest[0].size != src.size || D->saturate)
         continue;

      for (unsigned i = 0; i < 2; ++i) {
         agx_index a = D->src[i], b = D->src[i ^ 1];
         if (a.type != AGX_INDEX_IMMEDIATE || a.abs || a.neg ||
             b.type != AGX_INDEX_NORMAL || b.size != src.size || b.abs ||
             b.neg)
            continue;

         if (D->truth_table == AGX3_SEL_TT_AND && a.value == notsign) {
            /* ge/le take only one absolute comparator */
            if ((I->fcond == AGX_FCOND_GE || I->fcond == AGX_FCOND_LE) &&
                I->src[s ^ 1].abs)
               continue;
            I->src[s] = agx_abs(b);
            break;
         }
         if (D->truth_table == AGX3_SEL_TT_XOR && a.value == sign) {
            I->src[s] = agx_neg(b);
            break;
         }
      }
   }
}


void
agx_optimizer_forward(agx_context *ctx)
{
   agx_instr **defs = calloc(ctx->alloc, sizeof(*defs));

   agx_foreach_instr_global_safe(ctx, I) {
      struct agx_opcode_info info = agx_opcodes_info[I->op];

      agx_foreach_ssa_dest(I, d) {
         defs[I->dest[d].value] = I;
      }

      /* Optimize moves */
      agx_optimizer_copyprop(ctx, defs, I);

      /* Propagate fmov down */
      if (info.is_float || I->op == AGX_OPCODE_FCMPSEL ||
          I->op == AGX_OPCODE_FCMP)
         agx_optimizer_fmov(ctx, defs, I);

      /* Inline immediates if we can. TODO: systematic */
      if (I->op != AGX_OPCODE_COLLECT && I->op != AGX_OPCODE_IMAGE_LOAD &&
          I->op != AGX_OPCODE_TEXTURE_LOAD &&
          I->op != AGX_OPCODE_TEX_STATE_STORE &&
          I->op != AGX_OPCODE_SAMPLER_STATE_STORE &&
          I->op != AGX_OPCODE_UNIFORM_STORE && I->op != AGX_OPCODE_EXPORT)
         agx_optimizer_inline_imm(ctx, defs, I);

      /* Inlining a constant into fmov can leave two minifloat operands.
       * Fold the conversion exactly, including negative zero, before RA. */
      if (agx_is_fmov(I) && I->src[0].type == AGX_INDEX_IMMEDIATE &&
          !I->saturate) {
         float value = agx_minifloat_decode(I->src[0].value);
         if (I->src[0].abs)
            value = fabsf(value);
         if (I->src[0].neg)
            value = -value;
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_instr *mov = agx_mov_imm_to(&b, I->dest[0],
            I->dest[0].size == AGX_SIZE_16 ? _mesa_float_to_half(value) : fui(value));
         defs[I->dest[0].value] = mov;
         agx_remove_instruction(I);
         continue;
      }

      if (I->op == AGX_OPCODE_IF_ICMP) {
         agx_optimizer_if_not(defs, I);
         agx_optimizer_if_cmp(defs, I);
      } else if (I->op == AGX_OPCODE_ICMP) {
         agx_instr *folded = agx_optimizer_fold_icmp(ctx, I);
         if (folded)
            defs[folded->dest[0].value] = folded;
      } else if (I->op == AGX_OPCODE_ICMPSEL) {
         /* Fusing a compare of two inlined immediates yields a select the
          * fold below can resolve. */
         agx_optimizer_cmpsel(ctx, defs, I);
         agx_g17p_tie_zero_minmax(ctx, I);
         agx_optimizer_select_guard(ctx, defs, I);
         agx_instr *folded = agx_optimizer_fold_icmpsel(ctx, I);
         if (folded)
            defs[folded->dest[0].value] = folded;
      } else if (I->op == AGX_OPCODE_BALLOT ||
                 I->op == AGX_OPCODE_QUAD_BALLOT) {
         agx_optimizer_ballot(ctx, defs, I);
      } else if (I->op == AGX_OPCODE_BITOP) {
         agx_optimizer_bitop(ctx, defs, I);
         agx_optimizer_pack_mask(ctx, defs, I);
      } else if (I->op == AGX_OPCODE_BFI || I->op == AGX_OPCODE_BFEIL) {
         agx_optimizer_pack_mask(ctx, defs, I);
      } else if (I->op == AGX_OPCODE_IADD || I->op == AGX_OPCODE_IMAD) {
         agx_optimizer_signext(defs, I);
         agx_optimizer_iadd_copy(ctx, defs, I);
      } else if (I->op == AGX_OPCODE_CONVERT) {
         agx_optimizer_narrow_convert(ctx, defs, I);
      } else if (I->op == AGX_OPCODE_FCMPSEL || I->op == AGX_OPCODE_FCMP) {
         agx_optimizer_fcmp_mods(ctx, defs, I);
      } else if (I->op == AGX_OPCODE_DEVICE_LOAD ||
                 I->op == AGX_OPCODE_DEVICE_STORE) {
         agx_optimizer_device_index_offset(ctx, defs, I);
      }
   }

   free(defs);
}

static void
record_use(agx_instr **uses, BITSET_WORD *multiple, agx_instr *I, unsigned src)
{
   unsigned v = I->src[src].value;

   if (uses[v])
      BITSET_SET(multiple, v);
   else
      uses[v] = I;
}

void
agx_optimizer_backward(agx_context *ctx)
{
   agx_instr **uses = calloc(ctx->alloc, sizeof(*uses));
   BITSET_WORD *multiple = BITSET_CALLOC(ctx->alloc);

   agx_foreach_block_rev(ctx, block) {
      /* Phi sources are logically read at the end of predecessor, so process
       * our source in our successors' phis firsts. This ensures we set
       * `multiple` correctly with phi sources.
       */
      agx_foreach_successor(block, succ) {
         unsigned s = agx_predecessor_index(succ, block);

         agx_foreach_phi_in_block(succ, phi) {
            record_use(uses, multiple, phi, s);
         }
      }

      agx_foreach_instr_in_block_rev(block, I) {
         struct agx_opcode_info info = agx_opcodes_info[I->op];

         /* Skip phis, they're handled specially */
         if (I->op == AGX_OPCODE_PHI) {
            continue;
         }

         agx_foreach_ssa_src(I, s) {
            record_use(uses, multiple, I, s);
         }

         if (info.nr_dests != 1)
            continue;

         if (I->dest[0].type != AGX_INDEX_NORMAL)
            continue;

         agx_instr *use = uses[I->dest[0].value];

         if (!use || BITSET_TEST(multiple, I->dest[0].value))
            continue;

         if (agx_optimizer_not(ctx, I, use)) {
            agx_remove_instruction(use);
            continue;
         }

         /* Destination has a single use, try to propagate */
         if (info.is_float && agx_optimizer_fmov_rev(ctx, I, use)) {
            agx_remove_instruction(use);
            continue;
         }
      }
   }

   free(uses);
   free(multiple);
}
