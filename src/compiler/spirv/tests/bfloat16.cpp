/*
 * Copyright © 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "helpers.h"

/* Each module converts two vec4 of 32-bit floats to bfloat16 and stores them:
 *
 *    a = 0x3f80c000 (1.005859375), 0x3f818000 (a tie, odd kept bit),
 *        0xbf80c000 (-1.005859375), 0x7f800001 (a NaN with only low payload)
 *    b = 0x7f7fffff (the largest float), 0x3f808000 (a tie, even kept bit),
 *        0x00010000 (exact), 0xff800000 (-Inf)
 *
 * The inputs are constants, so constant folding yields the stored values.
 */
class BFloat16Convert : public spirv_test {
protected:
   void SetUp()
   {
      spirv_caps.BFloat16TypeKHR = true;
      spirv_caps.StorageBuffer16BitAccess = true;
   }

   unsigned count_alu(nir_op op)
   {
      unsigned count = 0;
      nir_foreach_block(block, nir_shader_get_entrypoint(shader)) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu &&
                nir_instr_as_alu(instr)->op == op)
               count++;
         }
      }
      return count;
   }

   void expect_stored(unsigned index, const uint16_t expected[4])
   {
      nir_intrinsic_instr *store =
         find_intrinsic(nir_intrinsic_store_deref, index);
      ASSERT_NE(store, nullptr);
      ASSERT_TRUE(nir_src_is_const(store->src[1]));
      for (unsigned i = 0; i < 4; i++) {
         EXPECT_EQ(nir_src_comp_as_uint(store->src[1], i), expected[i])
            << "store " << index << " component " << i;
      }
   }
};

TEST_F(BFloat16Convert, undecorated)
{
   /*
    ; Extra arguments for spirv-as --target-env vulkan1.3

                  OpCapability Shader
                  OpCapability StorageBuffer16BitAccess
                  OpCapability BFloat16TypeKHR
                  OpExtension "SPV_KHR_bfloat16"
                  OpMemoryModel Logical GLSL450
                  OpEntryPoint GLCompute %main "main" %out
                  OpExecutionMode %main LocalSize 1 1 1
                  OpDecorate %Out Block
                  OpMemberDecorate %Out 0 Offset 0
                  OpMemberDecorate %Out 1 Offset 8
                  OpDecorate %out DescriptorSet 0
                  OpDecorate %out Binding 0

          %void = OpTypeVoid
            %fn = OpTypeFunction %void
          %uint = OpTypeInt 32 0
        %v4uint = OpTypeVector %uint 4
         %float = OpTypeFloat 32
       %v4float = OpTypeVector %float 4
          %bf16 = OpTypeFloat 16 BFloat16KHR
        %v4bf16 = OpTypeVector %bf16 4
           %Out = OpTypeStruct %v4bf16 %v4bf16
       %ptr_Out = OpTypePointer StorageBuffer %Out
        %ptr_v4 = OpTypePointer StorageBuffer %v4bf16
        %uint_0 = OpConstant %uint 0
        %uint_1 = OpConstant %uint 1
          %a0 = OpConstant %uint 0x3f80c000
          %a1 = OpConstant %uint 0x3f818000
          %a2 = OpConstant %uint 0xbf80c000
          %a3 = OpConstant %uint 0x7f800001
          %b0 = OpConstant %uint 0x7f7fffff
          %b1 = OpConstant %uint 0x3f808000
          %b2 = OpConstant %uint 0x00010000
          %b3 = OpConstant %uint 0xff800000
        %bits_a = OpConstantComposite %v4uint %a0 %a1 %a2 %a3
        %bits_b = OpConstantComposite %v4uint %b0 %b1 %b2 %b3
           %out = OpVariable %ptr_Out StorageBuffer
          %main = OpFunction %void None %fn
         %entry = OpLabel
             %fa = OpBitcast %v4float %bits_a
             %fb = OpBitcast %v4float %bits_b
         %conv_a = OpFConvert %v4bf16 %fa
         %conv_b = OpFConvert %v4bf16 %fb
          %pa = OpAccessChain %ptr_v4 %out %uint_0
          %pb = OpAccessChain %ptr_v4 %out %uint_1
                  OpStore %pa %conv_a
                  OpStore %pb %conv_b
                  OpReturn
                  OpFunctionEnd
    */
   static const uint32_t words[] = {
      0x07230203, 0x00010600, 0x00070000, 0x00000021, 0x00000000, 0x00020011,
      0x00000001, 0x00020011, 0x00001151, 0x00020011, 0x000013fc, 0x0006000a,
      0x5f565053, 0x5f52484b, 0x6f6c6662, 0x36317461, 0x00000000, 0x0003000e,
      0x00000000, 0x00000001, 0x0006000f, 0x00000005, 0x00000001, 0x6e69616d,
      0x00000000, 0x00000002, 0x00060010, 0x00000001, 0x00000011, 0x00000001,
      0x00000001, 0x00000001, 0x00030047, 0x00000003, 0x00000002, 0x00050048,
      0x00000003, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x00000003,
      0x00000001, 0x00000023, 0x00000008, 0x00040047, 0x00000002, 0x00000022,
      0x00000000, 0x00040047, 0x00000002, 0x00000021, 0x00000000, 0x00020013,
      0x00000004, 0x00030021, 0x00000005, 0x00000004, 0x00040015, 0x00000006,
      0x00000020, 0x00000000, 0x00040017, 0x00000007, 0x00000006, 0x00000004,
      0x00030016, 0x00000008, 0x00000020, 0x00040017, 0x00000009, 0x00000008,
      0x00000004, 0x00040016, 0x0000000a, 0x00000010, 0x00000000, 0x00040017,
      0x0000000b, 0x0000000a, 0x00000004, 0x0004001e, 0x00000003, 0x0000000b,
      0x0000000b, 0x00040020, 0x0000000c, 0x0000000c, 0x00000003, 0x00040020,
      0x0000000d, 0x0000000c, 0x0000000b, 0x0004002b, 0x00000006, 0x0000000e,
      0x00000000, 0x0004002b, 0x00000006, 0x0000000f, 0x00000001, 0x0004002b,
      0x00000006, 0x00000010, 0x3f80c000, 0x0004002b, 0x00000006, 0x00000011,
      0x3f818000, 0x0004002b, 0x00000006, 0x00000012, 0xbf80c000, 0x0004002b,
      0x00000006, 0x00000013, 0x7f800001, 0x0004002b, 0x00000006, 0x00000014,
      0x7f7fffff, 0x0004002b, 0x00000006, 0x00000015, 0x3f808000, 0x0004002b,
      0x00000006, 0x00000016, 0x00010000, 0x0004002b, 0x00000006, 0x00000017,
      0xff800000, 0x0007002c, 0x00000007, 0x00000018, 0x00000010, 0x00000011,
      0x00000012, 0x00000013, 0x0007002c, 0x00000007, 0x00000019, 0x00000014,
      0x00000015, 0x00000016, 0x00000017, 0x0004003b, 0x0000000c, 0x00000002,
      0x0000000c, 0x00050036, 0x00000004, 0x00000001, 0x00000000, 0x00000005,
      0x000200f8, 0x0000001a, 0x0004007c, 0x00000009, 0x0000001b, 0x00000018,
      0x0004007c, 0x00000009, 0x0000001c, 0x00000019, 0x00040073, 0x0000000b,
      0x0000001d, 0x0000001b, 0x00040073, 0x0000000b, 0x0000001e, 0x0000001c,
      0x00050041, 0x0000000d, 0x0000001f, 0x00000002, 0x0000000e, 0x00050041,
      0x0000000d, 0x00000020, 0x00000002, 0x0000000f, 0x0003003e, 0x0000001f,
      0x0000001d, 0x0003003e, 0x00000020, 0x0000001e, 0x000100fd, 0x00010038,
   };

   get_nir(sizeof(words) / sizeof(words[0]), words);
   ASSERT_NE(shader, nullptr);

   /* Without a decoration the rounding is left to the backend. */
   EXPECT_EQ(count_alu(nir_op_f2bf), 2);

   NIR_PASS(_, shader, nir_opt_constant_folding);
   static const uint16_t a[4] = { 0x3f81, 0x3f82, 0xbf81, 0x7fc0 };
   static const uint16_t b[4] = { 0x7f80, 0x3f80, 0x0001, 0xff80 };
   expect_stored(0, a);
   expect_stored(1, b);
}

TEST_F(BFloat16Convert, rounding_mode_rtz)
{
   /*
    ; Extra arguments for spirv-as --target-env vulkan1.3

                  OpCapability Shader
                  OpCapability StorageBuffer16BitAccess
                  OpCapability BFloat16TypeKHR
                  OpExtension "SPV_KHR_bfloat16"
                  OpMemoryModel Logical GLSL450
                  OpEntryPoint GLCompute %main "main" %out
                  OpExecutionMode %main LocalSize 1 1 1
                  OpDecorate %Out Block
                  OpMemberDecorate %Out 0 Offset 0
                  OpMemberDecorate %Out 1 Offset 8
                  OpDecorate %out DescriptorSet 0
                  OpDecorate %out Binding 0
                  OpDecorate %conv_a FPRoundingMode RTZ
                  OpDecorate %conv_b FPRoundingMode RTZ
          %void = OpTypeVoid
            %fn = OpTypeFunction %void
          %uint = OpTypeInt 32 0
        %v4uint = OpTypeVector %uint 4
         %float = OpTypeFloat 32
       %v4float = OpTypeVector %float 4
          %bf16 = OpTypeFloat 16 BFloat16KHR
        %v4bf16 = OpTypeVector %bf16 4
           %Out = OpTypeStruct %v4bf16 %v4bf16
       %ptr_Out = OpTypePointer StorageBuffer %Out
        %ptr_v4 = OpTypePointer StorageBuffer %v4bf16
        %uint_0 = OpConstant %uint 0
        %uint_1 = OpConstant %uint 1
          %a0 = OpConstant %uint 0x3f80c000
          %a1 = OpConstant %uint 0x3f818000
          %a2 = OpConstant %uint 0xbf80c000
          %a3 = OpConstant %uint 0x7f800001
          %b0 = OpConstant %uint 0x7f7fffff
          %b1 = OpConstant %uint 0x3f808000
          %b2 = OpConstant %uint 0x00010000
          %b3 = OpConstant %uint 0xff800000
        %bits_a = OpConstantComposite %v4uint %a0 %a1 %a2 %a3
        %bits_b = OpConstantComposite %v4uint %b0 %b1 %b2 %b3
           %out = OpVariable %ptr_Out StorageBuffer
          %main = OpFunction %void None %fn
         %entry = OpLabel
             %fa = OpBitcast %v4float %bits_a
             %fb = OpBitcast %v4float %bits_b
         %conv_a = OpFConvert %v4bf16 %fa
         %conv_b = OpFConvert %v4bf16 %fb
          %pa = OpAccessChain %ptr_v4 %out %uint_0
          %pb = OpAccessChain %ptr_v4 %out %uint_1
                  OpStore %pa %conv_a
                  OpStore %pb %conv_b
                  OpReturn
                  OpFunctionEnd
    */
   static const uint32_t words[] = {
      0x07230203, 0x00010600, 0x00070000, 0x00000021, 0x00000000, 0x00020011,
      0x00000001, 0x00020011, 0x00001151, 0x00020011, 0x000013fc, 0x0006000a,
      0x5f565053, 0x5f52484b, 0x6f6c6662, 0x36317461, 0x00000000, 0x0003000e,
      0x00000000, 0x00000001, 0x0006000f, 0x00000005, 0x00000001, 0x6e69616d,
      0x00000000, 0x00000002, 0x00060010, 0x00000001, 0x00000011, 0x00000001,
      0x00000001, 0x00000001, 0x00030047, 0x00000003, 0x00000002, 0x00050048,
      0x00000003, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x00000003,
      0x00000001, 0x00000023, 0x00000008, 0x00040047, 0x00000002, 0x00000022,
      0x00000000, 0x00040047, 0x00000002, 0x00000021, 0x00000000, 0x00040047,
      0x00000004, 0x00000027, 0x00000001, 0x00040047, 0x00000005, 0x00000027,
      0x00000001, 0x00020013, 0x00000006, 0x00030021, 0x00000007, 0x00000006,
      0x00040015, 0x00000008, 0x00000020, 0x00000000, 0x00040017, 0x00000009,
      0x00000008, 0x00000004, 0x00030016, 0x0000000a, 0x00000020, 0x00040017,
      0x0000000b, 0x0000000a, 0x00000004, 0x00040016, 0x0000000c, 0x00000010,
      0x00000000, 0x00040017, 0x0000000d, 0x0000000c, 0x00000004, 0x0004001e,
      0x00000003, 0x0000000d, 0x0000000d, 0x00040020, 0x0000000e, 0x0000000c,
      0x00000003, 0x00040020, 0x0000000f, 0x0000000c, 0x0000000d, 0x0004002b,
      0x00000008, 0x00000010, 0x00000000, 0x0004002b, 0x00000008, 0x00000011,
      0x00000001, 0x0004002b, 0x00000008, 0x00000012, 0x3f80c000, 0x0004002b,
      0x00000008, 0x00000013, 0x3f818000, 0x0004002b, 0x00000008, 0x00000014,
      0xbf80c000, 0x0004002b, 0x00000008, 0x00000015, 0x7f800001, 0x0004002b,
      0x00000008, 0x00000016, 0x7f7fffff, 0x0004002b, 0x00000008, 0x00000017,
      0x3f808000, 0x0004002b, 0x00000008, 0x00000018, 0x00010000, 0x0004002b,
      0x00000008, 0x00000019, 0xff800000, 0x0007002c, 0x00000009, 0x0000001a,
      0x00000012, 0x00000013, 0x00000014, 0x00000015, 0x0007002c, 0x00000009,
      0x0000001b, 0x00000016, 0x00000017, 0x00000018, 0x00000019, 0x0004003b,
      0x0000000e, 0x00000002, 0x0000000c, 0x00050036, 0x00000006, 0x00000001,
      0x00000000, 0x00000007, 0x000200f8, 0x0000001c, 0x0004007c, 0x0000000b,
      0x0000001d, 0x0000001a, 0x0004007c, 0x0000000b, 0x0000001e, 0x0000001b,
      0x00040073, 0x0000000d, 0x00000004, 0x0000001d, 0x00040073, 0x0000000d,
      0x00000005, 0x0000001e, 0x00050041, 0x0000000f, 0x0000001f, 0x00000002,
      0x00000010, 0x00050041, 0x0000000f, 0x00000020, 0x00000002, 0x00000011,
      0x0003003e, 0x0000001f, 0x00000004, 0x0003003e, 0x00000020, 0x00000005,
      0x000100fd, 0x00010038,
   };

   get_nir(sizeof(words) / sizeof(words[0]), words);
   ASSERT_NE(shader, nullptr);

   EXPECT_EQ(count_alu(nir_op_f2bf), 0);

   NIR_PASS(_, shader, nir_opt_constant_folding);
   static const uint16_t a[4] = { 0x3f80, 0x3f81, 0xbf80, 0x7fc0 };
   static const uint16_t b[4] = { 0x7f7f, 0x3f80, 0x0001, 0xff80 };
   expect_stored(0, a);
   expect_stored(1, b);
}

TEST_F(BFloat16Convert, rounding_mode_rte)
{
   /*
    ; Extra arguments for spirv-as --target-env vulkan1.3

                  OpCapability Shader
                  OpCapability StorageBuffer16BitAccess
                  OpCapability BFloat16TypeKHR
                  OpExtension "SPV_KHR_bfloat16"
                  OpMemoryModel Logical GLSL450
                  OpEntryPoint GLCompute %main "main" %out
                  OpExecutionMode %main LocalSize 1 1 1
                  OpDecorate %Out Block
                  OpMemberDecorate %Out 0 Offset 0
                  OpMemberDecorate %Out 1 Offset 8
                  OpDecorate %out DescriptorSet 0
                  OpDecorate %out Binding 0
                  OpDecorate %conv_a FPRoundingMode RTE
                  OpDecorate %conv_b FPRoundingMode RTE
          %void = OpTypeVoid
            %fn = OpTypeFunction %void
          %uint = OpTypeInt 32 0
        %v4uint = OpTypeVector %uint 4
         %float = OpTypeFloat 32
       %v4float = OpTypeVector %float 4
          %bf16 = OpTypeFloat 16 BFloat16KHR
        %v4bf16 = OpTypeVector %bf16 4
           %Out = OpTypeStruct %v4bf16 %v4bf16
       %ptr_Out = OpTypePointer StorageBuffer %Out
        %ptr_v4 = OpTypePointer StorageBuffer %v4bf16
        %uint_0 = OpConstant %uint 0
        %uint_1 = OpConstant %uint 1
          %a0 = OpConstant %uint 0x3f80c000
          %a1 = OpConstant %uint 0x3f818000
          %a2 = OpConstant %uint 0xbf80c000
          %a3 = OpConstant %uint 0x7f800001
          %b0 = OpConstant %uint 0x7f7fffff
          %b1 = OpConstant %uint 0x3f808000
          %b2 = OpConstant %uint 0x00010000
          %b3 = OpConstant %uint 0xff800000
        %bits_a = OpConstantComposite %v4uint %a0 %a1 %a2 %a3
        %bits_b = OpConstantComposite %v4uint %b0 %b1 %b2 %b3
           %out = OpVariable %ptr_Out StorageBuffer
          %main = OpFunction %void None %fn
         %entry = OpLabel
             %fa = OpBitcast %v4float %bits_a
             %fb = OpBitcast %v4float %bits_b
         %conv_a = OpFConvert %v4bf16 %fa
         %conv_b = OpFConvert %v4bf16 %fb
          %pa = OpAccessChain %ptr_v4 %out %uint_0
          %pb = OpAccessChain %ptr_v4 %out %uint_1
                  OpStore %pa %conv_a
                  OpStore %pb %conv_b
                  OpReturn
                  OpFunctionEnd
    */
   static const uint32_t words[] = {
      0x07230203, 0x00010600, 0x00070000, 0x00000021, 0x00000000, 0x00020011,
      0x00000001, 0x00020011, 0x00001151, 0x00020011, 0x000013fc, 0x0006000a,
      0x5f565053, 0x5f52484b, 0x6f6c6662, 0x36317461, 0x00000000, 0x0003000e,
      0x00000000, 0x00000001, 0x0006000f, 0x00000005, 0x00000001, 0x6e69616d,
      0x00000000, 0x00000002, 0x00060010, 0x00000001, 0x00000011, 0x00000001,
      0x00000001, 0x00000001, 0x00030047, 0x00000003, 0x00000002, 0x00050048,
      0x00000003, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x00000003,
      0x00000001, 0x00000023, 0x00000008, 0x00040047, 0x00000002, 0x00000022,
      0x00000000, 0x00040047, 0x00000002, 0x00000021, 0x00000000, 0x00040047,
      0x00000004, 0x00000027, 0x00000000, 0x00040047, 0x00000005, 0x00000027,
      0x00000000, 0x00020013, 0x00000006, 0x00030021, 0x00000007, 0x00000006,
      0x00040015, 0x00000008, 0x00000020, 0x00000000, 0x00040017, 0x00000009,
      0x00000008, 0x00000004, 0x00030016, 0x0000000a, 0x00000020, 0x00040017,
      0x0000000b, 0x0000000a, 0x00000004, 0x00040016, 0x0000000c, 0x00000010,
      0x00000000, 0x00040017, 0x0000000d, 0x0000000c, 0x00000004, 0x0004001e,
      0x00000003, 0x0000000d, 0x0000000d, 0x00040020, 0x0000000e, 0x0000000c,
      0x00000003, 0x00040020, 0x0000000f, 0x0000000c, 0x0000000d, 0x0004002b,
      0x00000008, 0x00000010, 0x00000000, 0x0004002b, 0x00000008, 0x00000011,
      0x00000001, 0x0004002b, 0x00000008, 0x00000012, 0x3f80c000, 0x0004002b,
      0x00000008, 0x00000013, 0x3f818000, 0x0004002b, 0x00000008, 0x00000014,
      0xbf80c000, 0x0004002b, 0x00000008, 0x00000015, 0x7f800001, 0x0004002b,
      0x00000008, 0x00000016, 0x7f7fffff, 0x0004002b, 0x00000008, 0x00000017,
      0x3f808000, 0x0004002b, 0x00000008, 0x00000018, 0x00010000, 0x0004002b,
      0x00000008, 0x00000019, 0xff800000, 0x0007002c, 0x00000009, 0x0000001a,
      0x00000012, 0x00000013, 0x00000014, 0x00000015, 0x0007002c, 0x00000009,
      0x0000001b, 0x00000016, 0x00000017, 0x00000018, 0x00000019, 0x0004003b,
      0x0000000e, 0x00000002, 0x0000000c, 0x00050036, 0x00000006, 0x00000001,
      0x00000000, 0x00000007, 0x000200f8, 0x0000001c, 0x0004007c, 0x0000000b,
      0x0000001d, 0x0000001a, 0x0004007c, 0x0000000b, 0x0000001e, 0x0000001b,
      0x00040073, 0x0000000d, 0x00000004, 0x0000001d, 0x00040073, 0x0000000d,
      0x00000005, 0x0000001e, 0x00050041, 0x0000000f, 0x0000001f, 0x00000002,
      0x00000010, 0x00050041, 0x0000000f, 0x00000020, 0x00000002, 0x00000011,
      0x0003003e, 0x0000001f, 0x00000004, 0x0003003e, 0x00000020, 0x00000005,
      0x000100fd, 0x00010038,
   };

   get_nir(sizeof(words) / sizeof(words[0]), words);
   ASSERT_NE(shader, nullptr);

   EXPECT_EQ(count_alu(nir_op_f2bf), 0);

   NIR_PASS(_, shader, nir_opt_constant_folding);
   static const uint16_t a[4] = { 0x3f81, 0x3f82, 0xbf81, 0x7fc0 };
   static const uint16_t b[4] = { 0x7f80, 0x3f80, 0x0001, 0xff80 };
   expect_stored(0, a);
   expect_stored(1, b);
}
