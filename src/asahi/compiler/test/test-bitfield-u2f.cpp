#include <cstdint>
#include <cstring>
#include <functional>
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "agx_compile.h"
#include "agx_nir.h"
}

static float
as_float(uint32_t u)
{
   float f;
   memcpy(&f, &u, 4);
   return f;
}

static uint32_t
as_uint(float f)
{
   uint32_t u;
   memcpy(&u, &f, 4);
   return u;
}

/* Evaluate a 32-bit scalar expression over the input, as bits. */
static uint32_t
evaluate(nir_scalar v, nir_def *input, uint32_t x)
{
   v = nir_scalar_chase_movs(v);
   if (v.def == input)
      return x;
   if (nir_scalar_is_const(v))
      return nir_scalar_as_uint(v);
   if (!nir_scalar_is_alu(v)) {
      ADD_FAILURE() << "unexpected instruction";
      return 0;
   }
   auto src = [&](unsigned i) {
      return evaluate(nir_scalar_chase_alu_src(v, i), input, x);
   };
   switch (nir_scalar_alu_op(v)) {
   case nir_op_iand: return src(0) & src(1);
   case nir_op_ior: return src(0) | src(1);
   case nir_op_ushr: return src(0) >> (src(1) & 31);
   case nir_op_u2u32: return src(0) & 0xffff; /* from the 16-bit input */
   case nir_op_ubitfield_extract:
      return (src(0) >> (src(1) & 31)) & ((1u << src(2)) - 1);
   case nir_op_u2f32: return as_uint((float)src(0));
   case nir_op_fadd: return as_uint(as_float(src(0)) + as_float(src(1)));
   case nir_op_fmul: return as_uint(as_float(src(0)) * as_float(src(1)));
   default: ADD_FAILURE() << "unexpected opcode"; return 0;
   }
}

static unsigned
count(nir_shader *s, nir_op op)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            n += instr->type == nir_instr_type_alu &&
                 nir_instr_as_alu(instr)->op == op;
         }
      }
   }
   return n;
}

struct u2f_case {
   const char *name;
   std::function<nir_def *(nir_builder *, nir_def *)> build;
   std::function<float(uint32_t)> reference;
   bool lowered;
};

static void
check(const u2f_case &c, unsigned input_bits)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "%s", c.name);
   nir_def *input = agx_test_load_preamble(&b, 1, input_bits, 0);
   nir_def *keep = nir_mov(&b, c.build(&b, input));
   EXPECT_EQ(agx_nir_lower_bitfield_u2f(b.shader), c.lowered) << c.name;
   if (c.lowered) {
      EXPECT_EQ(count(b.shader, nir_op_u2f32), 0u) << c.name;
   }
   nir_validate_shader(b.shader, c.name);

   nir_scalar out = nir_scalar_chase_alu_src(nir_get_scalar(keep, 0), 0);
   auto test = [&](uint32_t x) {
      if (input_bits == 16)
         x &= 0xffff;
      EXPECT_EQ(evaluate(out, input, x), as_uint(c.reference(x)))
         << c.name << " x=" << std::hex << x;
   };
   for (uint32_t x = 0; x < 0x10000; ++x)
      test(x);
   uint32_t r = 0x98762341;
   for (unsigned i = 0; i < 20000; ++i) {
      r ^= r << 13; r ^= r >> 17; r ^= r << 5;
      test(r);
   }
   ralloc_free(b.shader);
}

TEST(BitfieldU2F, ExactForQ4Nibbles)
{
   const u2f_case cases[] = {
      {"ubfe+fadd folds", [](nir_builder *b, nir_def *x) {
          return nir_fadd_imm(b, nir_u2f32(b, nir_ubitfield_extract_imm(b, x, 4, 4)), -8.0);
       }, [](uint32_t x) { return (float)((x >> 4) & 15) - 8.0f; }, true},
      {"ubfe high field", [](nir_builder *b, nir_def *x) {
          return nir_fadd_imm(b, nir_u2f32(b, nir_ubitfield_extract_imm(b, x, 15, 8)), 0.5);
       }, [](uint32_t x) { return (float)((x >> 15) & 255) + 0.5f; }, true},
      {"ubfe without add", [](nir_builder *b, nir_def *x) {
          return nir_fmul_imm(b, nir_u2f32(b, nir_ubitfield_extract_imm(b, x, 8, 4)), 3.0);
       }, [](uint32_t x) { return (float)((x >> 8) & 15) * 3.0f; }, true},
      {"iand+fadd folds", [](nir_builder *b, nir_def *x) {
          return nir_fadd_imm(b, nir_u2f32(b, nir_iand_imm(b, x, 0xf)), -8.0);
       }, [](uint32_t x) { return (float)(x & 15) - 8.0f; }, true},
      {"iand without add", [](nir_builder *b, nir_def *x) {
          return nir_u2f32(b, nir_iand_imm(b, x, 0xf0));
       }, [](uint32_t x) { return (float)(x & 0xf0); }, true},
      {"wide mask stays", [](nir_builder *b, nir_def *x) {
          return nir_u2f32(b, nir_iand_imm(b, x, 0xffffff));
       }, [](uint32_t x) { return (float)(x & 0xffffff); }, false},
      {"field past the mantissa stays", [](nir_builder *b, nir_def *x) {
          return nir_u2f32(b, nir_ubitfield_extract_imm(b, x, 16, 8));
       }, [](uint32_t x) { return (float)((x >> 16) & 255); }, false},
      {"inexact constant does not fold", [](nir_builder *b, nir_def *x) {
          return nir_fadd_imm(b, nir_u2f32(b, nir_ubitfield_extract_imm(b, x, 0, 4)), 1e-10);
       }, [](uint32_t x) { return (float)(x & 15) + 1e-10f; }, true},
   };
   for (const u2f_case &c : cases)
      check(c, 32);
}

TEST(BitfieldU2F, ShiftOfAHalfIsAField)
{
   check({"ushr of u2u32", [](nir_builder *b, nir_def *x) {
             return nir_fadd_imm(b, nir_u2f32(b, nir_ushr_imm(b, nir_u2u32(b, x), 12)), -8.0);
          }, [](uint32_t x) { return (float)((x & 0xffff) >> 12) - 8.0f; }, true}, 16);
}

/* A unary source (here u2u32) is no field: it has no second source to read. */
TEST(BitfieldU2F, UnarySourceStays)
{
   check({"u2f32 of u2u32 stays", [](nir_builder *b, nir_def *x) {
             return nir_u2f32(b, nir_u2u32(b, x));
          }, [](uint32_t x) { return (float)(x & 0xffff); }, false}, 16);
}

/* The pass runs on every generation: a compute shader dequantizing Q4_0
 * nibbles, float((q >> 4 * i) & 15) - 8, compiles on each encoding. */
TEST(BitfieldU2F, CompilesOnEachTarget)
{
   const struct {
      agx_isa_encoding encoding;
      agx_chip chip;
      bool emit_agx3d;
   } targets[] = {
      {AGX_ISA_ENCODING_AGX2, AGX_CHIP_G13G, false},
      {AGX_ISA_ENCODING_AGX2, AGX_CHIP_G14G, false},
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G16S, false},
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G18G, true},
   };
   for (const auto &t : targets) {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                     &agx_nir_options, "q4");
      nir_def *addr = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *q = nir_load_global(&b, 1, 32, addr);
      nir_def *f[4];
      for (unsigned i = 0; i < 4; ++i) {
         f[i] = nir_fadd_imm(
            &b, nir_u2f32(&b, nir_ubitfield_extract_imm(&b, q, 4 * i, 4)),
            -8.0);
      }
      nir_store_global(&b, nir_vec(&b, f, 4), nir_iadd_imm(&b, addr, 16),
                       .align_mul = 16);
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;
      agx_shader_key key = {};
      key.dev.encoding = t.encoding;
      key.dev.chip = t.chip;
      key.dev.emit_agx3d = t.emit_agx3d;
      key.no_preamble = true;
      key.reserved_preamble = 16;
      agx_preprocess_nir(b.shader);
      agx_shader_part part = {};
      EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part))
         << "chip " << t.chip;
      free(part.binary);
      ralloc_free(b.shader);
   }
}
