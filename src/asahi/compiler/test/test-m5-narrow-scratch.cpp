/* SPDX-License-Identifier: MIT */
#include <inttypes.h>
#include <stdio.h>
#include <string>
#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
extern "C" {
#include "agx_compile.h"
#include "agx_debug.h"
#include "agx_nir.h"
}

/*
 * Private arrays of 8- and 16-bit scalars and vectors, larger than the
 * register-array threshold so preprocessing places them in scratch, written
 * and read through data-dependent indices: int8_t a[300], int16_t a[300],
 * uint16_t a[300], i8vec2 a[300] (two bytes at a register offset) and the rest
 * of {8, 16} bits x {1..4} components. The M5 Ultra must compile every one.
 */
enum scratch_shape {
   /* One store and one load at different dynamic indices, and one store at
    * a constant index. */
   SCRATCH_DYNAMIC,
   /* Every element stored at a constant index, then one dynamic read: the
    * shape of the subgroup broadcast tests' ops[] array. */
   SCRATCH_BROADCAST,
};

struct scratch_case {
   enum glsl_base_type base;
   unsigned components;
   unsigned length;
   enum scratch_shape shape;
};

struct scratch_result {
   bool compiled;
   unsigned size;
   uint64_t hash;
   std::string refusal;
};

static const char *
base_name(enum glsl_base_type base)
{
   switch (base) {
   case GLSL_TYPE_INT8:
      return "i8";
   case GLSL_TYPE_UINT8:
      return "u8";
   case GLSL_TYPE_INT16:
      return "i16";
   default:
      return "u16";
   }
}

static std::string
case_name(const scratch_case &c)
{
   return std::string(base_name(c.base)) + "x" + std::to_string(c.components) +
          "[" + std::to_string(c.length) + "]" +
          (c.shape == SCRATCH_BROADCAST ? " broadcast" : " dynamic");
}

static nir_def *
element_value(nir_builder *b, nir_def *word, const scratch_case &c,
              unsigned salt)
{
   const unsigned bits = glsl_base_type_get_bit_size(c.base);
   nir_def *channels[4];
   for (unsigned i = 0; i < c.components; ++i) {
      nir_def *x = nir_ushr_imm(b, word, 5 * i + 3 * salt);
      channels[i] = nir_u2uN(b, nir_iadd_imm(b, x, 17 * i + salt), bits);
   }
   return nir_vec(b, channels, c.components);
}

static nir_shader *
build_scratch_shader(const scratch_case &c)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "narrow scratch");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.internal = false;

   const struct glsl_type *element =
      glsl_vector_type(c.base, c.components);
   nir_variable *array = nir_local_variable_create(
      b.impl, glsl_array_type(element, c.length, 0), "a");

   nir_def *word = agx_test_load_preamble(&b, 1, 32, 4);
   nir_def *offsets = agx_test_load_preamble(&b, 2, 32, 8);
   nir_def *address = agx_test_load_preamble(&b, 1, 64, 16);
   nir_deref_instr *var = nir_build_deref_var(&b, array);
   nir_def *read;

   if (c.shape == SCRATCH_DYNAMIC) {
      nir_def *at = nir_iadd_imm(
         &b, nir_iand_imm(&b, nir_channel(&b, offsets, 0), 255), 1);
      nir_def *from = nir_iadd_imm(
         &b, nir_iand_imm(&b, nir_channel(&b, offsets, 1), 255), 2);
      nir_store_deref(&b, nir_build_deref_array_imm(&b, var, c.length - 1),
                      element_value(&b, word, c, 1),
                      nir_component_mask(c.components));
      nir_store_deref(&b, nir_build_deref_array(&b, var, at),
                      element_value(&b, word, c, 0),
                      nir_component_mask(c.components));
      read = nir_load_deref(&b, nir_build_deref_array(&b, var, from));
   } else {
      for (unsigned k = 0; k < c.length; ++k) {
         nir_store_deref(&b, nir_build_deref_array_imm(&b, var, k),
                         element_value(&b, nir_iadd_imm(&b, word, k), c, k & 3),
                         nir_component_mask(c.components));
      }
      nir_def *from = nir_umod_imm(&b, nir_channel(&b, offsets, 1), c.length);
      read = nir_load_deref(&b, nir_build_deref_array(&b, var, from));
   }

   const bool is_signed =
      c.base == GLSL_TYPE_INT8 || c.base == GLSL_TYPE_INT16;
   nir_def *wide = is_signed ? nir_i2i32(&b, read) : nir_u2u32(&b, read);
   nir_store_global(&b, wide, address, .align_mul = 4);
   return b.shader;
}

/* Preprocess as Honeykrisp does for each target (native AGX3 parts take the
 * G17P entry), then compile. */
static scratch_result
compile_scratch(const scratch_case &c, enum agx_isa_encoding encoding,
                enum agx_chip chip)
{
   glsl_type_singleton_init_or_ref();
   nir_shader *nir = build_scratch_shader(c);

   if (agx_chip_native_agx3(chip))
      agx_preprocess_nir_g17p(nir);
   else if (agx_chip_g17_m5(chip))
      agx_preprocess_nir_m5(nir);
   else
      agx_preprocess_nir(nir);

   agx_shader_key key = {};
   key.dev.encoding = encoding;
   key.dev.chip = chip;
   key.no_preamble = true;
   key.reserved_preamble = 32;
   key.has_scratch = true;
   agx_shader_part part = {};

   scratch_result r = {};
   r.compiled = agx_compile_shader_nir(nir, &key, &part);
   if (r.compiled) {
      r.size = part.info.binary_size;
      r.hash = UINT64_C(0xcbf29ce484222325);
      for (unsigned i = 0; i < r.size; ++i) {
         r.hash ^= ((const uint8_t *)part.binary)[i];
         r.hash *= UINT64_C(0x100000001b3);
      }
   } else {
      const char *why = agx_last_compile_refusal();
      const char *pack = agx3_last_pack_refusal();
      r.refusal = std::string(why ? why : "?") + " / " + (pack ? pack : "?");
   }
   free(part.binary);
   ralloc_free(nir);
   glsl_type_singleton_decref();
   return r;
}

static const enum glsl_base_type narrow_bases[] = {
   GLSL_TYPE_INT8,
   GLSL_TYPE_UINT8,
   GLSL_TYPE_INT16,
   GLSL_TYPE_UINT16,
};

TEST(M5NarrowScratch, DynamicArraysCompileOnM5Ultra)
{
   for (enum glsl_base_type base : narrow_bases) {
      for (unsigned components = 1; components <= 4; ++components) {
         const scratch_case c = {base, components, 300, SCRATCH_DYNAMIC};
         SCOPED_TRACE(case_name(c));
         scratch_result r =
            compile_scratch(c, AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17D);
         EXPECT_TRUE(r.compiled) << "refused: " << r.refusal;
         printf("m5-narrow-scratch: G17D %s %s %u\n", case_name(c).c_str(),
                r.compiled ? "compiled" : "refused", r.size);
      }
   }
}

/* The broadcast shape for every element type the subgroup tests place in
 * scratch (128 elements over 256 B). */
TEST(M5NarrowScratch, BroadcastArraysCompileOnM5Ultra)
{
   for (enum glsl_base_type base : narrow_bases) {
      const unsigned bits = glsl_base_type_get_bit_size(base);
      for (unsigned components = bits == 8 ? 3 : 2; components <= 4;
           ++components) {
         const scratch_case c = {base, components, 128, SCRATCH_BROADCAST};
         SCOPED_TRACE(case_name(c));
         scratch_result r =
            compile_scratch(c, AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17D);
         EXPECT_TRUE(r.compiled) << "refused: " << r.refusal;
         printf("m5-narrow-scratch: G17D %s %s %u\n", case_name(c).c_str(),
                r.compiled ? "compiled" : "refused", r.size);
      }
   }
}

/*
 * The same shaders on every other target. Each line names the binary each
 * compile produced (its size and FNV-1a hash) or that it was refused; a tree
 * that changes only the M5 Ultra prints the same lines as the tree before it.
 * Compiling twice gives the same result.
 */
TEST(M5NarrowScratch, OtherTargetsListed)
{
   static const struct {
      enum agx_chip chip;
      enum agx_isa_encoding encoding;
      const char *name;
   } targets[] = {
      {AGX_CHIP_G13G, AGX_ISA_ENCODING_AGX2, "G13G"},
      {AGX_CHIP_G13X, AGX_ISA_ENCODING_AGX2, "G13X"},
      {AGX_CHIP_G14G, AGX_ISA_ENCODING_AGX2, "G14G"},
      {AGX_CHIP_G16G, AGX_ISA_ENCODING_AGX3B, "G16G"},
      {AGX_CHIP_G17P, AGX_ISA_ENCODING_AGX3B, "G17P"},
      {AGX_CHIP_G17G, AGX_ISA_ENCODING_AGX3C, "G17G"},
      {AGX_CHIP_G17S, AGX_ISA_ENCODING_AGX3C, "G17S"},
   };
   for (const auto &t : targets) {
      for (enum glsl_base_type base : narrow_bases) {
         for (unsigned components = 1; components <= 4; ++components) {
            for (int shape = SCRATCH_DYNAMIC; shape <= SCRATCH_BROADCAST;
                 ++shape) {
               const scratch_case c = {
                  base, components, shape == SCRATCH_DYNAMIC ? 300u : 128u,
                  (enum scratch_shape)shape};
               SCOPED_TRACE(testing::Message() << t.name << " "
                                               << case_name(c));
               scratch_result a = compile_scratch(c, t.encoding, t.chip);
               scratch_result b = compile_scratch(c, t.encoding, t.chip);
               EXPECT_EQ(a.compiled, b.compiled);
               EXPECT_EQ(a.size, b.size);
               EXPECT_EQ(a.hash, b.hash);
               if (a.compiled)
                  printf("m5-narrow-scratch: %s %s %u %016" PRIx64 "\n",
                         t.name, case_name(c).c_str(), a.size, a.hash);
               else
                  printf("m5-narrow-scratch: %s %s refused\n", t.name,
                         case_name(c).c_str());
            }
         }
      }
   }
}
