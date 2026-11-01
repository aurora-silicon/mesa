/* SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
extern "C" {
#include "agx_compile.h"
#include "agx_debug.h"
#include "agx_nir.h"
}

/*
 * An AGX3C scratch access of an 8-bit vector moves its bytes packed, two to a
 * 16-bit register, as device and local byte vectors do. A private array of
 * i8vec3/i8vec4 that NIR places in scratch (subgroup broadcasts collected into
 * an array read back by a dynamic index) must compile: IR translation emitted
 * the packed stack load and store, which the validator then sized as one
 * register per byte and aborted the compile.
 */
static bool
compile_byte_scratch(unsigned components, enum agx_isa_encoding encoding,
                     enum agx_chip chip)
{
   glsl_type_singleton_init_or_ref();
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "byte scratch round trip");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->scratch_size = 64;

   nir_def *word = agx_test_load_preamble(&b, 1, 32, 4);
   nir_def *offsets = agx_test_load_preamble(&b, 2, 32, 8);
   nir_def *address = agx_test_load_preamble(&b, 1, 64, 16);
   nir_def *bytes =
      nir_trim_vector(&b, nir_unpack_32_4x8(&b, word), components);

   /* Different dynamic offsets, so the load cannot be forwarded from the
    * store and both reach the backend. */
   nir_def *at = nir_iand_imm(&b, nir_channel(&b, offsets, 0), 0x30);
   nir_def *from = nir_iand_imm(&b, nir_channel(&b, offsets, 1), 0x30);
   nir_store_scratch(&b, bytes, at, .align_mul = 4);
   nir_def *back = nir_load_scratch(&b, components, 8, from, .align_mul = 4);
   nir_store_global(&b, back, address, .align_mul = 4);

   agx_shader_key key = {};
   key.dev.encoding = encoding;
   key.dev.chip = chip;
   key.no_preamble = true;
   key.reserved_preamble = 32;
   key.has_scratch = true;
   /* An application shader: internal ones fold small scratch back into
    * variables before the backend. */
   b.shader->info.internal = false;
   agx_preprocess_nir(b.shader);
   agx_shader_part part = {};
   /* Debug builds validate after IR translation and exit on a bad width. */
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   if (!compiled) {
      const char *why = agx_last_compile_refusal();
      const char *pack = agx3_last_pack_refusal();
      ADD_FAILURE() << "refused: " << (why ? why : "?") << " / "
                    << (pack ? pack : "?");
   }
   free(part.binary);
   ralloc_free(b.shader);
   glsl_type_singleton_decref();
   return compiled;
}

/* Three and four bytes take the packed stack access. Two bytes become one
 * 16-bit word before the backend, a different access. */
TEST(AGX3PackedMemory, ByteVectorScratchCompilesOnM5)
{
   for (unsigned components = 3; components <= 4; ++components) {
      SCOPED_TRACE(components);
      EXPECT_TRUE(compile_byte_scratch(components, AGX_ISA_ENCODING_AGX3C,
                                       AGX_CHIP_G17D));
   }
}

TEST(AGX3PackedMemory, ByteVectorScratchCompilesOnAGX2)
{
   for (unsigned components = 2; components <= 4; ++components) {
      SCOPED_TRACE(components);
      EXPECT_TRUE(compile_byte_scratch(components, AGX_ISA_ENCODING_AGX2,
                                       AGX_CHIP_G14G));
   }
}

/*
 * Every packed memory access, whichever memory it addresses, carries its
 * masked elements back to back: two bytes to a 16-bit register, or whole words
 * of a 32-bit payload. Build each one at its packed width; agx_validate exits
 * the process on a width it does not accept.
 */
static agx_index
packed_data(agx_builder *b, enum agx_size size, unsigned registers)
{
   /* registers counts 16-bit registers; a 32-bit payload has half as many
    * channels. */
   agx_index data = agx_temp(b->shader, size);
   data.channels_m1 = registers / agx_size_align_16(size) - 1;
   return data;
}

static void
validate_packed(enum agx_opcode op, enum agx_format format,
                enum agx_size size, unsigned mask, unsigned registers)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17D;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->alloc = 64;

   agx_index data = packed_data(b, size, registers);
   agx_index address = agx_temp(b->shader, AGX_SIZE_64);
   agx_index index = agx_temp(b->shader, AGX_SIZE_32);
   agx_index local = agx_temp(b->shader, AGX_SIZE_16);
   agx_instr *I;
   switch (op) {
   case AGX_OPCODE_DEVICE_LOAD:
      I = agx_device_load_to(b, data, address, index, format, mask, 0, false);
      break;
   case AGX_OPCODE_DEVICE_STORE:
      I = agx_device_store(b, data, address, index, format, mask, 0, false);
      break;
   case AGX_OPCODE_LOCAL_LOAD:
      I = agx_local_load_to(b, data, agx_zero(), local, format, mask);
      break;
   case AGX_OPCODE_LOCAL_STORE:
      I = agx_local_store(b, data, agx_zero(), local, format, mask);
      break;
   case AGX_OPCODE_STACK_LOAD:
      I = agx_stack_load_to(b, data, index, format, mask);
      break;
   default:
      I = agx_stack_store(b, data, index, format, mask);
      break;
   }
   I->memory_is_packed = true;

   /* Every value read is defined first. */
   agx_builder pre = agx_init_builder(b->shader, agx_before_instr(I));
   agx_mov_imm_to(&pre, index, 0);
   agx_mov_imm_to(&pre, address, 0);
   agx_mov_imm_to(&pre, local, 0);
   if (op == AGX_OPCODE_DEVICE_STORE || op == AGX_OPCODE_LOCAL_STORE ||
       op == AGX_OPCODE_STACK_STORE) {
      if (registers == agx_size_align_16(size)) {
         agx_mov_imm_to(&pre, data, 0);
      } else {
         const unsigned n = registers / agx_size_align_16(size);
         agx_index parts[4];
         for (unsigned i = 0; i < n; ++i) {
            parts[i] = agx_temp(b->shader, size);
            agx_mov_imm_to(&pre, parts[i], i);
         }
         agx_instr *c = agx_collect_to(&pre, data, n);
         agx_foreach_src(c, s)
            c->src[s] = parts[s];
      }
   } else if (registers == agx_size_align_16(size)) {
      agx_unit_test(b, data);
   } else {
      const unsigned n = registers / agx_size_align_16(size);
      agx_instr *split = agx_split(b, n, data);
      agx_foreach_dest(split, d)
         split->dest[d] = agx_temp(b->shader, size);
   }

   agx_validate(b->shader, "packed memory widths");
   ralloc_free(mem);
}

TEST(AGX3PackedMemory, OneWidthForEveryPackedAccess)
{
   const enum agx_opcode ops[] = {
      AGX_OPCODE_DEVICE_LOAD, AGX_OPCODE_DEVICE_STORE, AGX_OPCODE_LOCAL_LOAD,
      AGX_OPCODE_LOCAL_STORE, AGX_OPCODE_STACK_LOAD,   AGX_OPCODE_STACK_STORE,
   };
   for (enum agx_opcode op : ops) {
      for (unsigned mask = 3; mask <= 15; mask = mask * 2 + 1) {
         SCOPED_TRACE(testing::Message() << agx_opcodes_info[op].name
                                         << " mask " << mask);
         /* Bytes, two to a 16-bit register. */
         const unsigned n = util_bitcount(mask);
         if (op != AGX_OPCODE_DEVICE_LOAD)
            validate_packed(op, AGX_FORMAT_I8, AGX_SIZE_16, mask,
                            DIV_ROUND_UP(n, 2));
         /* A 32-bit payload rounds up to whole words. */
         if (op == AGX_OPCODE_DEVICE_LOAD) {
            validate_packed(op, AGX_FORMAT_I8, AGX_SIZE_32, mask, 2);
            validate_packed(op, AGX_FORMAT_I16, AGX_SIZE_32, mask,
                            2 * DIV_ROUND_UP(2 * n, 4));
         }
      }
   }
}

/* The rule still refuses a packed access sized one register per byte, the
 * shape of the failure above. */
TEST(AGX3PackedMemoryDeathTest, BytePerRegisterStackStoreIsRefused)
{
   EXPECT_EXIT(validate_packed(AGX_OPCODE_STACK_STORE, AGX_FORMAT_I8,
                               AGX_SIZE_16, 7, 3),
               testing::ExitedWithCode(1), "Invalid vectors");
}
