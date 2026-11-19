/*
 * Copyright 2023 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_compile.h"
#include "agx_compiler.h"
#include "agx_opcodes.h"

/* Table describing the relationship between registers pressure and thread
 * count. Each entry describes a maximum number of registers and the associated
 * best-case thread count.
 *
 * Sorted in ascending order of maximum registers for easy lookup.
 */
static const struct agx_occupancy occupancies[] = {
   {104, 1024}, {112, 896}, {128, 832}, {136, 768}, {144, 704},
   {160, 640},  {184, 576}, {208, 512}, {232, 448}, {256, 384},
};

struct agx_occupancy
agx_occupancy_for_register_count(unsigned halfregs)
{
   for (unsigned i = 0; i < ARRAY_SIZE(occupancies); ++i) {
      unsigned max = occupancies[i].max_registers;
      assert((i == 0 || max > occupancies[i - 1].max_registers) && "ascending");

      if (halfregs <= max)
         return occupancies[i];
   }

   UNREACHABLE("Register count must be less than the maximum");
}

unsigned
agx_max_registers_for_occupancy(unsigned occupancy)
{
   unsigned max_regs = 0;

   for (unsigned i = 0; i < ARRAY_SIZE(occupancies); ++i) {
      if (occupancy <= occupancies[i].max_threads)
         max_regs = occupancies[i].max_registers;
      else
         break;
   }

   assert(max_regs > 0 && "Thread count must be less than the maximum");
   return max_regs;
}

unsigned
agx_register_file_halves(const struct agx_device_key *dev)
{
   switch (dev->encoding) {
   case AGX_ISA_ENCODING_AGX2:
      return AGX_NUM_REGS;
   case AGX_ISA_ENCODING_AGX3B:
      return AGX3B_NUM_REGS;
   case AGX_ISA_ENCODING_AGX3C:
      /* G18G compiles as AGX3C with emit_agx3d; both files end at r125. */
      return dev->emit_agx3d ? AGX3D_NUM_REGS : AGX3C_NUM_REGS;
   case AGX_ISA_ENCODING_AGX3D:
      return AGX3D_NUM_REGS;
   default:
      return 0;
   }
}

/*
 * A compute program that issues several texture reads spends much of its
 * time waiting on them, and more resident threads hide that wait. Capping its
 * registers trades scratch spills for those threads. The cap follows from two
 * properties of the device:
 *
 *  - the occupancy steps (occupancies[] above): how many threads a core keeps
 *    resident at each per-thread register budget, and
 *  - the per-thread register file (agx_register_file_halves()).
 *
 * The target is 5/8 of the most threads a core keeps resident (the first
 * step): 640 of 1024 threads, 20 of 32 SIMD groups. The cap is the largest
 * budget that still reaches that step, 160 halves with the current steps.
 * Tuned on G17C against the uncapped file and a 192-half cap.
 * A device gets no cap when the cap would not raise its resident threads
 * above what its whole file allows, or when its register file is unknown
 * (agx_register_file_halves() returns 0).
 *
 * AGX2 and AGX3C currently resolve to 160: their files (256 and 252 halves)
 * reach 384 threads uncapped, so the cap raises that to 640. AGX3B (192
 * halves) already reaches 512, too small a step, so it gets no cap. G18G gets
 * no cap, as it takes no scratch for compute. All encodings share one
 * occupancy table; a part whose steps differ needs its own table before this
 * derivation is right for it.
 */
unsigned
agx_sampled_compute_register_cap(const struct agx_device_key *dev)
{
   /* G18G launches compute without scratch (hk_dispatch_g17g_native()
    * refuses it), so the spills the cap trades for threads would refuse the
    * launch there.
    */
   if (dev->emit_agx3d || dev->encoding == AGX_ISA_ENCODING_AGX3D)
      return 0;

   const unsigned file = agx_register_file_halves(dev);
   if (!file)
      return 0;

   const unsigned target = occupancies[0].max_threads * 5 / 8;
   const unsigned cap = agx_max_registers_for_occupancy(target);

   /* The spills only pay for a large step in resident threads: half again
    * as many or more. 384 -> 640 threads gained on G17C; 512 -> 640 lost
    * 7-10% on G16G with the same sample workload.
    */
   const unsigned capped = agx_occupancy_for_register_count(cap).max_threads;
   const unsigned uncapped =
      agx_occupancy_for_register_count(file).max_threads;
   if (cap >= file || capped * 2 < uncapped * 3)
      return 0;

   return cap;
}

/* Crude cycle model for G13G */
enum alu_unit {
   NONE,
   SCIB,
   IC,
   F32,
   F16,
};

struct alu_timing {
   enum alu_unit unit;
   unsigned latency;
   unsigned tp;
};

/* clang-format off */
struct alu_timing op_timings[] = {
   [AGX_OPCODE_FFMA]          = { F32, 2, 1 },
   [AGX_OPCODE_FADD]          = { F32, 2, 1 },
   [AGX_OPCODE_FMUL]          = { F32, 2, 1 },
   /* Fused vertex stores perform the same ALU operation with a staging-file
    * destination. This classification also drives upstream register-cache
    * hints; it is not a measured latency for the subsequent UVSW. */
   [AGX_OPCODE_ST_VARY_MUL_AGX3] = { F32, 2, 1 },
   [AGX_OPCODE_ST_VARY_FMA_AGX3] = { F32, 2, 1 },

   [AGX_OPCODE_HFMA]          = { F16, 2, 1 },
   [AGX_OPCODE_HADD]          = { F16, 2, 1 },
   [AGX_OPCODE_HMUL]          = { F16, 2, 1 },

   [AGX_OPCODE_MOV_IMM]       = { SCIB, 1, 1 },
   [AGX_OPCODE_BITOP]         = { SCIB, 2, 1 }, /* tp might be 2 for 32-bit / no $? */
   [AGX_OPCODE_ICMPSEL]       = { SCIB, 2, 1 },
   [AGX_OPCODE_FCMPSEL]       = { SCIB, 2, 1 },
   [AGX_OPCODE_IADD]          = { SCIB, 2, 1 },

   [AGX_OPCODE_GET_SR]          = { SCIB, 2, 2 },
   [AGX_OPCODE_GET_SR_BARRIER]  = { SCIB, 2, 2 },
   [AGX_OPCODE_GET_SR_COVERAGE] = { SCIB, 2, 2 },

   [AGX_OPCODE_IMAD]          = { IC, 3, 2 },
   [AGX_OPCODE_BFI]           = { IC, 3, 2 },
   [AGX_OPCODE_BFEIL]         = { IC, 3, 2 },
   [AGX_OPCODE_EXTR]          = { IC, 3, 2 },
   [AGX_OPCODE_ASR]           = { IC, 3, 2 },
   [AGX_OPCODE_FLOOR]         = { IC, 3, 2 },
   [AGX_OPCODE_SIN_PT_1]      = { IC, 3, 2 },
   [AGX_OPCODE_SIN_PT_2]      = { IC, 5, 2 },
   [AGX_OPCODE_LOG2]          = { IC, 5, 2 },
   [AGX_OPCODE_EXP2]          = { IC, 5, 2 },
   [AGX_OPCODE_RCP]           = { IC, 5, 3 },
   [AGX_OPCODE_RSQRT]         = { IC, 6, 4 },
   [AGX_OPCODE_SRSQRT]        = { IC, 6, 4 },

   // XXX: check this
   [AGX_OPCODE_SIMD_PREFIX]      = { SCIB, 18, 18 },
   [AGX_OPCODE_SIMD_REDUCE]      = { SCIB, 24, 24 },
   [AGX_OPCODE_QUAD_PREFIX]      = { SCIB, 18, 18 },
   [AGX_OPCODE_QUAD_REDUCE]      = { SCIB, 24, 24 },
   [AGX_OPCODE_SHUFFLE]          = { SCIB, 5, 2   },
   [AGX_OPCODE_QUAD_SHUFFLE]     = { SCIB, 5, 2   },

   [AGX_OPCODE_ICMP_BALLOT]      = { SCIB, 5, 2   },
   [AGX_OPCODE_FCMP_BALLOT]      = { SCIB, 5, 2   },
   [AGX_OPCODE_ICMP_QUAD_BALLOT] = { SCIB, 4, 2   },
   [AGX_OPCODE_FCMP_QUAD_BALLOT] = { SCIB, 4, 2   },
};
/* clang-format on */

static struct alu_timing
alu_timing(const agx_instr *I)
{
   return I->op < ARRAY_SIZE(op_timings) ? op_timings[I->op]
                                         : (struct alu_timing){0};
}

bool
agx_is_alu(const agx_instr *I)
{
   return alu_timing(I).unit != NONE;
}

/*
 * TODO: Model non-ALU instructions, latency, register cache, 64-bit, etc.
 */
struct agx_cycle_estimate
agx_estimate_cycles(agx_context *ctx)
{
   struct agx_cycle_estimate est = {0};

   agx_foreach_instr_global(ctx, I) {
      struct alu_timing alu = alu_timing(I);

      if (alu.unit == IC) {
         est.ic += alu.tp * 2;

         /* In addition to the IC work, IC ops appear to be dispatched along
          * with SCIB.
          */
         est.f_scib += 1;
      } else if (alu.unit) {
         est.f_scib += alu.tp;
      } else {
         /* TODO */
      }
   }

   /* IC and F/SCIB run in parallel across warps */
   est.alu = MAX2(est.ic, est.f_scib);
   return est;
}
