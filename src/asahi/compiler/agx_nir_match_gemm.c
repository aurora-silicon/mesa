/* SPDX-License-Identifier: MIT */
#include "agx_nir_match_gemm.h"
#include "agx_nir.h"

#include "compiler/nir/nir_builder.h"

#define TILE_BYTES (4 * AGX_GEMM_TILE_WORDS)
#define TILE_MATH (AGX_GEMM_WIDTH * AGX_GEMM_ROWS_PER_LANE)

static bool
fence(const nir_intrinsic_instr *intr)
{
   return intr->intrinsic == nir_intrinsic_barrier &&
          nir_intrinsic_execution_scope(intr) == SCOPE_WORKGROUP &&
          (nir_intrinsic_memory_scope(intr) == SCOPE_DEVICE ||
           nir_intrinsic_memory_scope(intr) == SCOPE_WORKGROUP) &&
          nir_intrinsic_memory_modes(intr) == nir_var_mem_shared &&
          nir_intrinsic_memory_semantics(intr) == NIR_MEMORY_ACQ_REL;
}

static nir_loop *
innermost_loop(nir_block *block)
{
   for (nir_cf_node *node = block->cf_node.parent; node; node = node->parent) {
      if (node->type == nir_cf_node_loop)
         return nir_cf_node_as_loop(node);
   }
   return NULL;
}

/* Model only known, bounded local index expressions. Reject unsupported
 * system values, uniform/buffer loads, phis and unproven arithmetic.
 */
static bool
local_uint_rec(nir_scalar scalar, unsigned x, unsigned y, unsigned depth,
               uint32_t *value, unsigned *budget)
{
   if (!depth || !*budget || !scalar.def || scalar.def->bit_size > 32)
      return false;
   --*budget;

   if (nir_scalar_is_const(scalar)) {
      *value = nir_scalar_as_uint(scalar);
      return true;
   }

   if (nir_scalar_is_intrinsic(scalar)) {
      if (nir_scalar_as_intrinsic(scalar)->intrinsic !=
            nir_intrinsic_load_local_invocation_id || scalar.comp > 2)
         return false;
      const unsigned xyz[] = { x, y, 0 };
      *value = xyz[scalar.comp];
      return true;
   }

   if (!nir_scalar_is_alu(scalar))
      return false;

   const nir_op op = nir_scalar_alu_op(scalar);
   switch (op) {
   case nir_op_mov:
   case nir_op_u2u16:
   case nir_op_u2u32:
   case nir_op_iadd:
   case nir_op_imul:
   case nir_op_ishl:
   case nir_op_iand:
   case nir_op_ior:
   case nir_op_imadshl_agx:
      break;
   default:
      return false;
   }

   uint32_t lhs, rhs = 0;
   if (!local_uint_rec(nir_scalar_chase_alu_src(scalar, 0), x, y, depth - 1, &lhs,
                       budget))
      return false;
   if (op != nir_op_mov && op != nir_op_u2u16 && op != nir_op_u2u32 &&
       !local_uint_rec(nir_scalar_chase_alu_src(scalar, 1), x, y, depth - 1, &rhs,
                       budget))
      return false;

   uint64_t wide;
   switch (op) {
   case nir_op_mov:
   case nir_op_u2u16:
   case nir_op_u2u32:
      wide = lhs;
      break;
   case nir_op_iadd:
      wide = (uint64_t)lhs + rhs;
      break;
   case nir_op_imul:
      wide = (uint64_t)lhs * rhs;
      break;
   case nir_op_ishl:
      if (rhs >= scalar.def->bit_size)
         return false;
      wide = (uint64_t)lhs << rhs;
      break;
   case nir_op_iand:
      wide = lhs & rhs;
      break;
   case nir_op_ior:
      wide = lhs | rhs;
      break;
   case nir_op_imadshl_agx: {
      uint32_t add, shift;
      if (!local_uint_rec(nir_scalar_chase_alu_src(scalar, 2), x, y, depth - 1, &add,
                          budget) ||
          !local_uint_rec(nir_scalar_chase_alu_src(scalar, 3), x, y, depth - 1, &shift,
                          budget) ||
          shift >= scalar.def->bit_size ||
          (uint64_t)lhs * rhs > UINT32_MAX ||
          ((uint64_t)add << shift) > UINT32_MAX)
         return false;
      wide = (uint64_t)lhs * rhs + ((uint64_t)add << shift);
      break;
   }
   default:
      return false;
   }

   if (wide > BITFIELD64_MASK(scalar.def->bit_size))
      return false;
   *value = wide;
   return true;
}

/* Index expressions are a few dozen nodes; the budget bounds the walk of a
 * shared subexpression reached along many paths. */
static bool
local_uint(nir_scalar scalar, unsigned x, unsigned y, unsigned depth,
           uint32_t *value)
{
   unsigned budget = 512;
   return local_uint_rec(scalar, x, y, depth, value, &budget);
}

static bool
shared_word(nir_intrinsic_instr *intr, unsigned comp, unsigned x, unsigned y,
            unsigned *address)
{
   const bool is_load = intr->intrinsic == nir_intrinsic_load_shared;
   if (!is_load && intr->intrinsic != nir_intrinsic_store_shared)
      return false;

   nir_def *data = is_load ? &intr->def : intr->src[0].ssa;
   nir_def *offset = intr->src[is_load ? 0 : 1].ssa;
   const int base = nir_intrinsic_base(intr);
   const unsigned allowed_access = ACCESS_CAN_SPECULATE | ACCESS_CAN_REORDER;
   if (data->bit_size != 32 ||
       (is_load ? (data->num_components < 1 || data->num_components > 4 ||
                   comp >= data->num_components)
                : (data->num_components != 1 || comp != 0 ||
                   nir_intrinsic_write_mask(intr) != 1)) ||
       offset->bit_size != 16 || offset->num_components != 1 ||
       (nir_intrinsic_access(intr) & ~allowed_access) != 0 ||
       nir_intrinsic_align_mul(intr) < 4 ||
       nir_intrinsic_align_offset(intr) % 4 || base < 0 || base % 4)
      return false;

   uint32_t index;
   if (!local_uint(nir_scalar_resolved(offset, 0), x, y, 12, &index))
      return false;
   *address = (unsigned)base + 4 * (index + comp);
   return true;
}

struct operands {
   nir_intrinsic_instr *a;
   nir_intrinsic_instr *b;
   unsigned a_comp;
   unsigned b_comp;
   unsigned a0;
   unsigned b0;
};

static bool
fma_operands(nir_alu_instr *math, nir_instr *begin, nir_instr *end,
             struct operands *out)
{
   nir_scalar def = nir_get_scalar(&math->def, 0);
   nir_scalar lhs = nir_scalar_chase_alu_src(def, 0);
   nir_scalar rhs = nir_scalar_chase_alu_src(def, 1);
   if (!nir_scalar_is_intrinsic(lhs) || !nir_scalar_is_intrinsic(rhs))
      return false;
   nir_intrinsic_instr *left = nir_scalar_as_intrinsic(lhs);
   nir_intrinsic_instr *right = nir_scalar_as_intrinsic(rhs);
   if (left->intrinsic != nir_intrinsic_load_shared ||
       right->intrinsic != nir_intrinsic_load_shared)
      return false;

   /* Match in the convergent fence region and establish the axis roles from
    * actual addresses. Either multiplicand order is legal.
    */
   unsigned present = 0;
   for (nir_instr *instr = nir_instr_next(begin); instr != end; instr = nir_instr_next(instr)) {
      present += instr == &left->instr || instr == &right->instr;
      if (instr == &math->instr)
         break;
   }
   if (present != 2)
      return false;

   unsigned l0, r0, lx, rx;
   if (!shared_word(left, lhs.comp, 0, 0, &l0) ||
       !shared_word(right, rhs.comp, 0, 0, &r0) ||
       !shared_word(left, lhs.comp, 1, 0, &lx) ||
       !shared_word(right, rhs.comp, 1, 0, &rx))
      return false;
   if (lx == l0 && rx == r0 + 4) {
      *out = (struct operands){ left, right, lhs.comp, rhs.comp, l0, r0 };
      return true;
   }
   if (rx == r0 && lx == l0 + 4) {
      *out = (struct operands){ right, left, rhs.comp, lhs.comp, r0, l0 };
      return true;
   }
   return false;
}

static bool
collect_phase(nir_shader *shader, struct agx_gemm_match *m)
{
   const unsigned W = AGX_GEMM_WIDTH;
   nir_alu_instr *math[TILE_MATH];
   struct operands args[TILE_MATH];
   unsigned count = 0;
   m->a_bytes = UINT_MAX;
   m->b_bytes = UINT_MAX;

   for (nir_instr *ins = nir_instr_next(m->ready); ins != m->consumed;
        ins = nir_instr_next(ins)) {
      if (ins->type == nir_instr_type_load_const)
         continue;
      if (ins->type == nir_instr_type_intrinsic) {
         unsigned address;
         if (nir_instr_as_intrinsic(ins)->intrinsic != nir_intrinsic_load_shared ||
             !shared_word(nir_instr_as_intrinsic(ins), 0, 0, 0, &address))
            return false;
         continue;
      }
      if (ins->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *alu = nir_instr_as_alu(ins);
      if (alu->op != nir_op_ffma)
         continue;
      if (count == TILE_MATH || alu->def.bit_size != 32 ||
          alu->def.num_components != 1 || alu->fp_math_ctrl != nir_fp_fast_math ||
          !fma_operands(alu, m->ready, m->consumed, &args[count]))
         return false;
      m->a_bytes = MIN2(m->a_bytes, args[count].a0);
      m->b_bytes = MIN2(m->b_bytes, args[count].b0);
      math[count++] = alu;
   }

   if (count != TILE_MATH || ((m->a_bytes | m->b_bytes) & 7) ||
       m->a_bytes > 4 * (UINT16_MAX + 1u) - TILE_BYTES ||
       m->b_bytes > 4 * (UINT16_MAX + 1u) - TILE_BYTES ||
       m->a_bytes > shader->info.shared_size - TILE_BYTES ||
       m->b_bytes > shader->info.shared_size - TILE_BYTES ||
       (m->a_bytes < m->b_bytes ? m->b_bytes - m->a_bytes :
                                m->a_bytes - m->b_bytes) < TILE_BYTES)
      return false;

   for (unsigned i = 0; i < count; ++i) {
      const struct operands *f = &args[i];
      const unsigned b = (f->b0 - m->b_bytes) / 4;
      const unsigned a = (f->a0 - m->a_bytes) / 4;
      const unsigned k = b / W;
      const unsigned row = a / W;
      if (b % W || k >= W || a % W != k || row >= AGX_GEMM_ROWS_PER_LANE ||
          m->fma[k][row])
         return false;
      m->fma[k][row] = math[i];

      for (unsigned y = 0; y < AGX_GEMM_SUBGROUPS; ++y) {
         for (unsigned x = 0; x < W; ++x) {
            unsigned actual_a, actual_b;
            if (!shared_word(f->a, f->a_comp, x, y, &actual_a) ||
                actual_a != m->a_bytes + 4 * ((8 * y + row) * W + k) ||
                !shared_word(f->b, f->b_comp, x, y, &actual_b) ||
                actual_b != m->b_bytes + 4 * (k * W + x))
               return false;
         }
      }
   }
   return true;
}

static bool
phi_has_zero_and_end(nir_phi_instr *phi, nir_def *result)
{
   bool has_zero = false;
   bool has_result = false;
   unsigned count = 0;
   nir_foreach_phi_src(src, phi) {
      if (src->src.ssa == result)
         has_result = true;
      else if (nir_scalar_is_zero(nir_get_scalar(src->src.ssa, 0)))
         has_zero = true;
      ++count;
   }
   return has_zero && has_result && count == 2;
}

/* The value after the loop must be the merge of the uniform if that holds
 * only the loop's branch, as loop rotation leaves it: the chain's last value
 * from the branch with the loop, +0.0 from the other (the loop did not run).
 * The lowered chain's value is in fragment layout, spread over the lanes; a
 * merge of any other if could take +0.0 in some lanes only. The if must be
 * at the top level of the function: the epilogue's workgroup barrier at its
 * merge then runs in every invocation, also when the loop did not run (an
 * enclosing if could be uniform per subgroup but not per workgroup). */
static bool
guarded_exit(nir_loop *loop, nir_phi_instr *phi, nir_def *last)
{
   nir_cf_node *parent = loop->cf_node.parent;
   if (!parent || parent->type != nir_cf_node_if)
      return false;
   nir_if *nif = nir_cf_node_as_if(parent);
   if (parent->parent != &nir_cf_node_get_function(parent)->cf_node)
      return false;
   nir_cf_node *next = nir_cf_node_next(parent);
   if (nir_src_is_divergent(&nif->condition) || !next ||
       next->type != nir_cf_node_block || phi->instr.block != nir_cf_node_as_block(next))
      return false;

   bool in_then = false;
   foreach_list_typed(nir_cf_node, node, node, &nif->then_list)
      in_then |= node == &loop->cf_node;
   nir_block *loop_pred = in_then ? nir_if_last_then_block(nif)
                                  : nir_if_last_else_block(nif);
   unsigned count = 0;
   nir_foreach_phi_src(src, phi) {
      if (src->pred == loop_pred ? src->src.ssa != last
                                 : !nir_scalar_is_zero(nir_get_scalar(src->src.ssa, 0)))
         return false;
      ++count;
   }
   return count == 2;
}

static bool
only_user(nir_def *def, nir_instr *expected)
{
   unsigned count = 0;
   nir_foreach_use_including_if(src, def) {
      if (nir_src_is_if(src) || nir_src_use_instr(src) != expected)
         return false;
      ++count;
   }
   return count == 1;
}

static bool
recurrences(struct agx_gemm_match *m)
{
   nir_block *header = nir_loop_first_block(m->loop);
   for (unsigned row = 0; row < AGX_GEMM_ROWS_PER_LANE; ++row) {
      nir_alu_instr *seed = m->fma[0][row];
      nir_alu_instr *last = m->fma[AGX_GEMM_WIDTH - 1][row];
      nir_scalar initial =
         nir_scalar_chase_alu_src(nir_get_scalar(&seed->def, 0), 2);
      if (initial.comp != 0 || !nir_scalar_is_phi(initial))
         return false;

      nir_phi_instr *start = nir_scalar_as_phi(initial);
      if (start->instr.block != header || !phi_has_zero_and_end(start, &last->def) ||
          !only_user(&start->def, &seed->instr))
         return false;
      m->initial[row] = start;

      for (unsigned k = 1; k < AGX_GEMM_WIDTH; ++k) {
         nir_alu_instr *alu = m->fma[k][row];
         nir_alu_instr *prev = m->fma[k - 1][row];
         nir_scalar old = nir_scalar_chase_alu_src(nir_get_scalar(&alu->def, 0), 2);
         if (!nir_scalar_equal(old, nir_get_scalar(&prev->def, 0)) ||
             !only_user(&prev->def, &alu->instr))
            return false;
      }

      unsigned count = 0;
      nir_foreach_use_including_if(src, &last->def) {
         if (nir_src_is_if(src) || nir_src_use_instr(src)->type != nir_instr_type_phi)
            return false;
         nir_phi_instr *phi = nir_instr_as_phi(nir_src_use_instr(src));
         if (phi != start) {
            if (phi->instr.block->divergent || innermost_loop(phi->instr.block) ||
                !guarded_exit(m->loop, phi, &last->def))
               return false;
            m->exit[row] = phi;
         }
         ++count;
      }
      if (count != 2 || !m->exit[row] ||
          m->exit[row]->instr.block != m->exit[0]->instr.block)
         return false;
   }
   return true;
}

enum visit_state {
   UNSEEN,
   ACTIVE,
   DONE,
};

struct writes {
   enum visit_state state;
   uint16_t slots;
};

static bool
producer_slot(const struct agx_gemm_match *m, nir_intrinsic_instr *store,
              uint16_t *slot)
{
   unsigned first;
   if (!shared_word(store, 0, 0, 0, &first))
      return false;

   const bool a = first >= m->a_bytes && first - m->a_bytes < TILE_BYTES;
   const unsigned base = a ? m->a_bytes : m->b_bytes;
   if (first < base || (first - base) % (4 * AGX_GEMM_WIDTH))
      return false;

   const unsigned row = (first - base) / (4 * AGX_GEMM_WIDTH);
   if (row >= AGX_GEMM_ROWS_PER_LANE)
      return false;

   for (unsigned y = 0; y < AGX_GEMM_SUBGROUPS; ++y) {
      for (unsigned x = 0; x < AGX_GEMM_WIDTH; ++x) {
         unsigned address;
         const unsigned i = (8 * y + row) * AGX_GEMM_WIDTH + x;
         if (!shared_word(store, 0, x, y, &address) || address != base + 4 * i)
            return false;
      }
   }
   *slot = 1u << (row + (a ? 0 : AGX_GEMM_ROWS_PER_LANE));
   return true;
}

/*
 * Intersect writes at joins: a slot must be initialized on all incoming
 * paths, including predicated zero stores in the edge producer.
 */
static bool
producer_writes(const struct agx_gemm_match *m, nir_block *block,
                struct writes *blocks, unsigned depth)
{
   struct writes *record = &blocks[block->index];
   if (record->state == DONE)
      return true;
   if (!depth || record->state == ACTIVE || innermost_loop(block) != m->loop)
      return false;
   record->state = ACTIVE;

   uint16_t slots = 0;
   if (block != nir_loop_first_block(m->loop)) {
      if (!nir_block_num_preds(block))
         return false;
      slots = UINT16_MAX;
      nir_foreach_pred(pred, block) {
         if (!producer_writes(m, pred, blocks, depth - 1))
            return false;
         slots &= blocks[pred->index].slots;
      }
   }

   nir_foreach_instr(ins, block) {
      if (ins == m->ready)
         break;
      if (ins->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
      if (intr->intrinsic == nir_intrinsic_barrier)
         return false;
      if (nir_is_shared_access(intr)) {
         uint16_t slot;
         if (intr->intrinsic != nir_intrinsic_store_shared ||
             !producer_slot(m, intr, &slot))
            return false;
         slots |= slot;
      }
   }
   *record = (struct writes){ DONE, slots };
   return true;
}

static bool
validate_visibility(struct agx_gemm_match *m)
{
   nir_function_impl *impl = m->ready->block->impl;
   struct writes *blocks = calloc(impl->num_blocks, sizeof(*blocks));
   if (!blocks)
      return false;
   bool match = producer_writes(m, m->ready->block, blocks, impl->num_blocks) &&
                blocks[m->ready->block->index].slots == UINT16_MAX;

   /* Reject surviving observers of the scratch tile and nonuniform exits.
    * All shared uses must belong to the traversed producers or the phase.
    */
   unsigned breaks = 0;
   nir_foreach_block(block, impl) {
      bool in_phase = false;
      bool passed_phase = false;
      nir_foreach_instr(ins, block) {
         if (ins == m->ready) {
            in_phase = true;
         } else if (ins == m->consumed) {
            in_phase = false;
            passed_phase = true;
         } else if (ins->type == nir_instr_type_jump) {
            nir_jump_type jump = nir_instr_as_jump(ins)->type;
            if (innermost_loop(block) == m->loop) {
               match &= !block->divergent && jump == nir_jump_break &&
                        nir_block_dominates(m->ready->block, block);
               ++breaks;
            } else if (jump == nir_jump_return || jump == nir_jump_halt) {
               match &= nir_block_dominates(m->exit[0]->instr.block, block);
            }
         } else if (ins->type == nir_instr_type_call ||
                    ins->type == nir_instr_type_cmat_call ||
                    (ins->type == nir_instr_type_deref &&
                     (nir_instr_as_deref(ins)->modes & nir_var_mem_shared))) {
            match = false;
         } else if (ins->type == nir_instr_type_intrinsic &&
                    nir_is_shared_access(nir_instr_as_intrinsic(ins))) {
            match &= in_phase || (!passed_phase && blocks[block->index].state == DONE);
         }
      }
   }
   free(blocks);
   return match && breaks == 1;
}


static bool
is_local_id_comp(nir_scalar s, unsigned comp)
{
   s = nir_scalar_chase_movs(s);
   if (nir_scalar_is_alu(s) &&
       (nir_scalar_alu_op(s) == nir_op_u2u32 ||
        nir_scalar_alu_op(s) == nir_op_u2u16))
      s = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
   return nir_scalar_is_intrinsic(s) &&
          nir_scalar_as_intrinsic(s)->intrinsic ==
             nir_intrinsic_load_local_invocation_id &&
          s.comp == comp;
}

static bool
is_lane_x_linear(nir_scalar s)
{
   s = nir_scalar_chase_movs(s);
   if (nir_scalar_is_alu(s) &&
       (nir_scalar_alu_op(s) == nir_op_u2u32 ||
        nir_scalar_alu_op(s) == nir_op_u2u16))
      s = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
   if (!nir_scalar_is_intrinsic(s) || s.comp != 0)
      return false;
   nir_intrinsic_op intr = nir_scalar_as_intrinsic(s)->intrinsic;
   return intr == nir_intrinsic_load_local_invocation_id ||
          intr == nir_intrinsic_load_global_invocation_id;
}

/* Match col(x) = uniform_base + local_id.x (including via global_id.x,
 * which equals workgroup_base_x + local_id.x). Then for any target lane
 * column c, col(c) = (col_orig - local_id.x) + c. */
static bool
match_add_local_x(nir_scalar s, nir_def **col_orig)
{
   s = nir_scalar_chase_movs(s);
   if (s.comp != 0 || s.def->bit_size != 32)
      return false;
   if (is_lane_x_linear(s)) {
      *col_orig = s.def;
      return true;
   }
   if (!nir_scalar_is_alu(s))
      return false;
   nir_op op = nir_scalar_alu_op(s);
   if (op == nir_op_imadshl_agx) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      nir_scalar c = nir_scalar_chase_alu_src(s, 2);
      nir_scalar sh = nir_scalar_chase_alu_src(s, 3);
      if (nir_scalar_is_const(sh) && nir_scalar_as_uint(sh) == 0 &&
          is_lane_x_linear(c) && !a.def->divergent && !b.def->divergent) {
         *col_orig = s.def;
         return true;
      }
      return false;
   }
   if (op == nir_op_iadd) {
      for (unsigned i = 0; i < 2; ++i) {
         nir_scalar a = nir_scalar_chase_alu_src(s, i);
         nir_scalar b = nir_scalar_chase_alu_src(s, 1 - i);
         if (is_lane_x_linear(a) && !b.def->divergent) {
            *col_orig = s.def;
            return true;
         }
      }
   }
   return false;
}

static bool
is_ly_shl3(nir_scalar s)
{
   s = nir_scalar_chase_movs(s);
   if (!nir_scalar_is_alu(s))
      return false;
   nir_op op = nir_scalar_alu_op(s);
   if (op == nir_op_ishl) {
      return is_local_id_comp(nir_scalar_chase_alu_src(s, 0), 1) &&
             nir_scalar_is_const(nir_scalar_chase_alu_src(s, 1)) &&
             nir_scalar_as_uint(nir_scalar_chase_alu_src(s, 1)) == 3;
   }
   if (op == nir_op_imul) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      return (is_local_id_comp(a, 1) && nir_scalar_is_const(b) && nir_scalar_as_uint(b) == 8) ||
             (is_local_id_comp(b, 1) && nir_scalar_is_const(a) && nir_scalar_as_uint(a) == 8);
   }
   if (op == nir_op_imadshl_agx) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      nir_scalar c = nir_scalar_chase_alu_src(s, 2);
      nir_scalar sh = nir_scalar_chase_alu_src(s, 3);
      return nir_scalar_is_const(sh) && nir_scalar_as_uint(sh) == 3 &&
             is_local_id_comp(c, 1) &&
             ((nir_scalar_is_const(a) && nir_scalar_as_uint(a) == 0) ||
              (nir_scalar_is_const(b) && nir_scalar_as_uint(b) == 0));
   }
   return false;
}

static bool
is_lane_y_linear(nir_scalar s)
{
   s = nir_scalar_chase_movs(s);
   if (s.def->bit_size != 32)
      return false;
   if (nir_scalar_is_intrinsic(s)) {
      nir_intrinsic_op op = nir_scalar_as_intrinsic(s)->intrinsic;
      return (op == nir_intrinsic_load_local_invocation_id ||
              op == nir_intrinsic_load_global_invocation_id) &&
             s.comp == 1;
   }
   if (nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_iadd) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      return (is_lane_y_linear(a) && !b.def->divergent) ||
             (is_lane_y_linear(b) && !a.def->divergent);
   }
   return false;
}

static bool
match_add_local_y8(nir_scalar s, nir_def **row_base_out)
{
   s = nir_scalar_chase_movs(s);
   if (s.comp != 0 || s.def->bit_size != 32 || !s.def->divergent)
      return false;
   if (is_ly_shl3(s)) {
      *row_base_out = s.def;
      return true;
   }
   if (!nir_scalar_is_alu(s))
      return false;
   nir_op op = nir_scalar_alu_op(s);
   if (op == nir_op_ishl) {
      if (is_lane_y_linear(nir_scalar_chase_alu_src(s, 0)) &&
          nir_scalar_is_const(nir_scalar_chase_alu_src(s, 1)) &&
          nir_scalar_as_uint(nir_scalar_chase_alu_src(s, 1)) == 3) {
         *row_base_out = s.def;
         return true;
      }
      return false;
   }
   if (op == nir_op_imadshl_agx) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      nir_scalar c = nir_scalar_chase_alu_src(s, 2);
      nir_scalar sh = nir_scalar_chase_alu_src(s, 3);
      if (nir_scalar_is_const(sh) && nir_scalar_as_uint(sh) == 3 &&
          is_lane_y_linear(c) && !a.def->divergent && !b.def->divergent) {
         *row_base_out = s.def;
         return true;
      }
      return false;
   }
   if (op == nir_op_iadd) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      nir_def *dummy = NULL;
      if ((match_add_local_y8(a, &dummy) && !b.def->divergent) ||
          (match_add_local_y8(b, &dummy) && !a.def->divergent)) {
         *row_base_out = s.def;
         return true;
      }
   }
   return false;
}

static bool
match_k_row0(nir_scalar s, nir_def **k_base)
{
   s = nir_scalar_chase_movs(s);
   if (!nir_scalar_is_alu(s))
      return false;
   nir_op op = nir_scalar_alu_op(s);
   if (op == nir_op_imadshl_agx) {
      /* imadshl_agx(k_base, 1, local_id.y, 3) => k_base + (local_id.y << 3) */
      nir_scalar s0 = nir_scalar_chase_alu_src(s, 0);
      nir_scalar s1 = nir_scalar_chase_alu_src(s, 1);
      nir_scalar s2 = nir_scalar_chase_alu_src(s, 2);
      nir_scalar s3 = nir_scalar_chase_alu_src(s, 3);
      if (nir_scalar_is_const(s3) && nir_scalar_as_uint(s3) == 3 &&
          is_local_id_comp(s2, 1)) {
         if (nir_scalar_is_const(s1) && nir_scalar_as_uint(s1) == 1 &&
             s0.comp == 0 && s0.def->bit_size == 32 && !s0.def->divergent) {
            *k_base = s0.def;
            return true;
         }
         if (nir_scalar_is_const(s0) && nir_scalar_as_uint(s0) == 1 &&
             s1.comp == 0 && s1.def->bit_size == 32 && !s1.def->divergent) {
            *k_base = s1.def;
            return true;
         }
      }
      return false;
   }
   if (op == nir_op_iadd) {
      for (unsigned i = 0; i < 2; ++i) {
         nir_scalar a = nir_scalar_chase_alu_src(s, i);
         nir_scalar b = nir_scalar_chase_alu_src(s, 1 - i);
         if (is_ly_shl3(a) && b.comp == 0 && b.def->bit_size == 32 &&
             !b.def->divergent) {
            *k_base = b.def;
            return true;
         }
      }
   }
   return false;
}

static bool
match_linear_b_idx(nir_scalar idx, nir_scalar *row_out, nir_scalar *stride_out,
                   nir_scalar *col_out)
{
   idx = nir_scalar_chase_movs(idx);
   if (!nir_scalar_is_alu(idx))
      return false;
   nir_op op = nir_scalar_alu_op(idx);
   if (op == nir_op_imadshl_agx) {
      nir_scalar sh = nir_scalar_chase_alu_src(idx, 3);
      if (!nir_scalar_is_const(sh) || nir_scalar_as_uint(sh) != 0)
         return false;
      nir_scalar a = nir_scalar_chase_alu_src(idx, 0);
      nir_scalar b = nir_scalar_chase_alu_src(idx, 1);
      nir_scalar c = nir_scalar_chase_alu_src(idx, 2);
      if (b.def->bit_size == 32 && !b.def->divergent) {
         *row_out = a;
         *stride_out = b;
         *col_out = c;
         return true;
      }
      if (a.def->bit_size == 32 && !a.def->divergent) {
         *row_out = b;
         *stride_out = a;
         *col_out = c;
         return true;
      }
      return false;
   }
   if (op == nir_op_iadd) {
      for (unsigned i = 0; i < 2; ++i) {
         nir_scalar m = nir_scalar_chase_alu_src(idx, i);
         nir_scalar c = nir_scalar_chase_alu_src(idx, 1 - i);
         if (nir_scalar_is_alu(m) && nir_scalar_alu_op(m) == nir_op_imul) {
            nir_scalar a = nir_scalar_chase_alu_src(m, 0);
            nir_scalar b = nir_scalar_chase_alu_src(m, 1);
            if (b.def->bit_size == 32 && !b.def->divergent) {
               *row_out = a;
               *stride_out = b;
               *col_out = c;
               return true;
            }
            if (a.def->bit_size == 32 && !a.def->divergent) {
               *row_out = b;
               *stride_out = a;
               *col_out = c;
               return true;
            }
         }
      }
   }
   return false;
}

/* Recognize when all 8 B-tile producer stores unconditionally copy a linear
 * 32x32 slice from a read-only device buffer in the ready block, with no
 * other B shared stores or global writes anywhere in the loop. */

static bool
same_preamble_or_def(nir_scalar a, nir_scalar b)
{
   a = nir_scalar_chase_movs(a);
   b = nir_scalar_chase_movs(b);
   if (nir_scalar_equal(a, b))
      return true;
   if (nir_scalar_is_intrinsic(a) && nir_scalar_is_intrinsic(b)) {
      nir_intrinsic_instr *ia = nir_scalar_as_intrinsic(a);
      nir_intrinsic_instr *ib = nir_scalar_as_intrinsic(b);
      if (ia->intrinsic == nir_intrinsic_load_preamble &&
          ib->intrinsic == nir_intrinsic_load_preamble &&
          nir_intrinsic_preamble_class(ia) == nir_intrinsic_preamble_class(ib) &&
          ia->def.bit_size == ib->def.bit_size &&
          (nir_intrinsic_base(ia) + a.comp * (ia->def.bit_size / 16)) ==
          (nir_intrinsic_base(ib) + b.comp * (ib->def.bit_size / 16)))
         return true;
   }
   return false;
}

static bool
find_preamble_store(nir_shader *shader, unsigned base,
                    unsigned pclass, nir_scalar *out)
{
   nir_function_impl *pre = nir_shader_get_preamble(shader);
   if (!pre)
      return false;
   nir_intrinsic_instr *found = NULL;
   nir_foreach_block(block, pre) {
      nir_foreach_instr(ins, block) {
         if (ins->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
         if (intr->intrinsic != nir_intrinsic_store_preamble ||
             nir_intrinsic_preamble_class(intr) != pclass ||
             nir_intrinsic_base(intr) != base)
            continue;
         if (found)
            return false;
         found = intr;
      }
   }
   if (!found || found->src[0].ssa->bit_size != 32 ||
       found->src[0].ssa->num_components != 1)
      return false;
   *out = nir_scalar_resolved(found->src[0].ssa, 0);
   return true;
}

/* Evaluate s as an exact small multiple k * stride (chasing through preamble). */
static bool
eval_stride_multiple(nir_shader *shader, nir_scalar s, nir_scalar stride,
                     unsigned depth, uint32_t *k_out)
{
   if (!depth)
      return false;
   s = nir_scalar_chase_movs(s);
   if (same_preamble_or_def(s, stride)) {
      *k_out = 1;
      return true;
   }
   if (nir_scalar_is_intrinsic(stride)) {
      nir_intrinsic_instr *sintr = nir_scalar_as_intrinsic(stride);
      if (sintr->intrinsic == nir_intrinsic_load_preamble &&
          sintr->def.bit_size == 32 && sintr->def.num_components == 1) {
         nir_scalar stride_pre;
         if (find_preamble_store(shader, nir_intrinsic_base(sintr),
                                 nir_intrinsic_preamble_class(sintr),
                                 &stride_pre) &&
             same_preamble_or_def(s, stride_pre)) {
            *k_out = 1;
            return true;
         }
      }
   }
   if (nir_scalar_is_const(s)) {
      if (nir_scalar_as_uint(s) == 0) {
         *k_out = 0;
         return true;
      }
      return false;
   }
   if (nir_scalar_is_intrinsic(s)) {
      nir_intrinsic_instr *intr = nir_scalar_as_intrinsic(s);
      if (intr->intrinsic == nir_intrinsic_load_preamble &&
          intr->def.bit_size == 32 && intr->def.num_components == 1) {
         nir_scalar pre_val;
         if (find_preamble_store(shader, nir_intrinsic_base(intr),
                                 nir_intrinsic_preamble_class(intr), &pre_val))
            return eval_stride_multiple(shader, pre_val, stride, depth - 1, k_out);
      }
      return false;
   }
   if (!nir_scalar_is_alu(s))
      return false;
   nir_op op = nir_scalar_alu_op(s);
   if (op == nir_op_imul) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      uint32_t k0;
      if (nir_scalar_is_const(a) &&
          eval_stride_multiple(shader, b, stride, depth - 1, &k0)) {
         *k_out = (uint32_t)nir_scalar_as_uint(a) * k0;
         return true;
      }
      if (nir_scalar_is_const(b) &&
          eval_stride_multiple(shader, a, stride, depth - 1, &k0)) {
         *k_out = (uint32_t)nir_scalar_as_uint(b) * k0;
         return true;
      }
      return false;
   }
   if (op == nir_op_ishl) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      uint32_t k0;
      if (nir_scalar_is_const(b) && nir_scalar_as_uint(b) < 32 &&
          eval_stride_multiple(shader, a, stride, depth - 1, &k0)) {
         *k_out = k0 << nir_scalar_as_uint(b);
         return true;
      }
      return false;
   }
   if (op == nir_op_iadd) {
      uint32_t k0, k1;
      if (eval_stride_multiple(shader, nir_scalar_chase_alu_src(s, 0), stride, depth - 1, &k0) &&
          eval_stride_multiple(shader, nir_scalar_chase_alu_src(s, 1), stride, depth - 1, &k1)) {
         *k_out = k0 + k1;
         return true;
      }
      return false;
   }
   if (op == nir_op_imadshl_agx) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      nir_scalar c = nir_scalar_chase_alu_src(s, 2);
      nir_scalar sh = nir_scalar_chase_alu_src(s, 3);
      if (!nir_scalar_is_const(sh) || nir_scalar_as_uint(sh) >= 32)
         return false;
      uint32_t k1 = 0, k2 = 0;
      if ((nir_scalar_is_const(a) && nir_scalar_as_uint(a) == 0) ||
          (nir_scalar_is_const(b) && nir_scalar_as_uint(b) == 0)) {
         k1 = 0;
      } else if (nir_scalar_is_const(a) &&
                 eval_stride_multiple(shader, b, stride, depth - 1, &k1)) {
         k1 *= (uint32_t)nir_scalar_as_uint(a);
      } else if (nir_scalar_is_const(b) &&
                 eval_stride_multiple(shader, a, stride, depth - 1, &k1)) {
         k1 *= (uint32_t)nir_scalar_as_uint(b);
      } else {
         return false;
      }
      if (!eval_stride_multiple(shader, c, stride, depth - 1, &k2))
         return false;
      *k_out = k1 + (k2 << nir_scalar_as_uint(sh));
      return true;
   }
   return false;
}

static bool
same_preamble_or_def_64(nir_def *a, nir_def *b)
{
   if (a == b)
      return true;
   if (nir_def_is_intrinsic(a) && nir_def_is_intrinsic(b)) {
      nir_intrinsic_instr *ia = nir_def_as_intrinsic(a);
      nir_intrinsic_instr *ib = nir_def_as_intrinsic(b);
      if (ia->intrinsic == nir_intrinsic_load_preamble &&
          ib->intrinsic == nir_intrinsic_load_preamble &&
          nir_intrinsic_base(ia) == nir_intrinsic_base(ib) &&
          nir_intrinsic_preamble_class(ia) == nir_intrinsic_preamble_class(ib) &&
          a->bit_size == b->bit_size && a->num_components == b->num_components)
         return true;
   }
   if (nir_def_is_alu(a) && nir_def_is_alu(b)) {
      nir_alu_instr *aa = nir_def_as_alu(a);
      nir_alu_instr *ab = nir_def_as_alu(b);
      if (aa->op == nir_op_pack_64_2x32_split &&
          ab->op == nir_op_pack_64_2x32_split &&
          same_preamble_or_def(nir_scalar_chase_alu_src(nir_get_scalar(a, 0), 0),
                               nir_scalar_chase_alu_src(nir_get_scalar(b, 0), 0)) &&
          same_preamble_or_def(nir_scalar_chase_alu_src(nir_get_scalar(a, 0), 1),
                               nir_scalar_chase_alu_src(nir_get_scalar(b, 0), 1)))
         return true;
   }
   return false;
}

static bool
can_materialize_uniform_def(nir_def *def, nir_block *target)
{
   if (!def || def->divergent)
      return false;
   if (nir_block_dominates(nir_def_block(def), target))
      return true;
   nir_instr *ins = nir_def_instr(def);
   if (ins->type == nir_instr_type_load_const)
      return true;
   if (ins->type == nir_instr_type_intrinsic &&
       nir_instr_as_intrinsic(ins)->intrinsic == nir_intrinsic_load_preamble)
      return true;
   if (ins->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(ins);
      if (alu->op == nir_op_pack_64_2x32_split) {
         return can_materialize_uniform_def(alu->src[0].src.ssa, target) &&
                can_materialize_uniform_def(alu->src[1].src.ssa, target);
      }
   }
   return false;
}

static nir_def *
materialize_uniform_def(nir_builder *b, nir_def *def, nir_block *target)
{
   if (nir_block_dominates(nir_def_block(def), target))
      return def;
   nir_instr *ins = nir_def_instr(def);
   if (ins->type == nir_instr_type_alu &&
       nir_instr_as_alu(ins)->op == nir_op_pack_64_2x32_split) {
      nir_scalar lo_s = nir_scalar_chase_alu_src(nir_get_scalar(def, 0), 0);
      nir_scalar hi_s = nir_scalar_chase_alu_src(nir_get_scalar(def, 0), 1);
      nir_def *lo = materialize_uniform_def(b, lo_s.def, target);
      nir_def *hi = materialize_uniform_def(b, hi_s.def, target);
      if (lo->num_components > 1)
         lo = nir_channel(b, lo, lo_s.comp);
      if (hi->num_components > 1)
         hi = nir_channel(b, hi, hi_s.comp);
      return nir_pack_64_2x32_split(b, lo, hi);
   }
   nir_instr *clone = nir_instr_clone(b->shader, ins);
   nir_builder_instr_insert(b, clone);
   return nir_instr_def(clone);
}

static nir_def *
materialize_uniform_scalar(nir_builder *b, nir_scalar s, nir_block *target)
{
   s = nir_scalar_chase_movs(s);
   nir_def *d = materialize_uniform_def(b, s.def, target);
   return d->num_components == 1 ? d : nir_channel(b, d, s.comp);
}

static bool
match_const_loop_step_phi(nir_loop *loop, nir_scalar s, uint32_t expected_step)
{
   if (!loop)
      return false;
   s = nir_scalar_chase_movs(s);
   if (s.comp != 0 || !nir_def_is_phi(s.def))
      return false;
   nir_phi_instr *phi = nir_def_as_phi(s.def);
   if (phi->instr.block != nir_loop_first_block(loop) ||
       exec_list_length(&phi->srcs) != 2)
      return false;
   nir_block *latch = nir_loop_last_block(loop);
   nir_def *init_val = NULL, *back_val = NULL;
   nir_foreach_phi_src(ps, phi) {
      if (ps->pred == latch)
         back_val = ps->src.ssa;
      else
         init_val = ps->src.ssa;
   }
   if (!init_val || !back_val)
      return false;
   nir_scalar init_s = nir_scalar_resolved(init_val, 0);
   if (!nir_scalar_is_const(init_s) || nir_scalar_as_uint(init_s) != 0)
      return false;
   nir_scalar back_s = nir_scalar_resolved(back_val, 0);
   if (!nir_scalar_is_alu(back_s) || nir_scalar_alu_op(back_s) != nir_op_iadd)
      return false;
   nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(back_s, 0));
   nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(back_s, 1));
   return (nir_scalar_equal(a, s) && nir_scalar_is_const(b) &&
           nir_scalar_as_uint(b) == expected_step) ||
          (nir_scalar_equal(b, s) && nir_scalar_is_const(a) &&
           nir_scalar_as_uint(a) == expected_step);
}

static bool
is_loop_k_stride32(nir_loop *loop, nir_scalar s)
{
   s = nir_scalar_chase_movs(s);
   if (match_const_loop_step_phi(loop, s, 32))
      return true;
   if (!nir_scalar_is_alu(s))
      return false;
   nir_op op = nir_scalar_alu_op(s);
   if (op == nir_op_ishl) {
      nir_scalar sh = nir_scalar_chase_alu_src(s, 1);
      return nir_scalar_is_const(sh) && nir_scalar_as_uint(sh) == 5 &&
             match_const_loop_step_phi(loop, nir_scalar_chase_alu_src(s, 0), 1);
   }
   if (op == nir_op_imul) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      return (nir_scalar_is_const(a) && nir_scalar_as_uint(a) == 32 &&
              match_const_loop_step_phi(loop, b, 1)) ||
             (nir_scalar_is_const(b) && nir_scalar_as_uint(b) == 32 &&
              match_const_loop_step_phi(loop, a, 1));
   }
   if (op == nir_op_imadshl_agx) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      nir_scalar c = nir_scalar_chase_alu_src(s, 2);
      nir_scalar sh = nir_scalar_chase_alu_src(s, 3);
      return nir_scalar_is_const(sh) && nir_scalar_as_uint(sh) == 5 &&
             ((nir_scalar_is_const(a) && nir_scalar_as_uint(a) == 0) ||
              (nir_scalar_is_const(b) && nir_scalar_as_uint(b) == 0)) &&
             match_const_loop_step_phi(loop, c, 1);
   }
   return false;
}

static bool
same_loop_k_base(nir_loop *loop, nir_scalar a, nir_scalar b)
{
   if (same_preamble_or_def(a, b))
      return true;
   return is_loop_k_stride32(loop, a) && is_loop_k_stride32(loop, b);
}

static bool
match_add_const_loop(nir_loop *loop, nir_scalar s, nir_scalar base, uint32_t c)
{
   s = nir_scalar_chase_movs(s);
   base = nir_scalar_chase_movs(base);
   if (c == 0)
      return same_loop_k_base(loop, s, base);
   if (!nir_scalar_is_alu(s))
      return false;
   nir_op op = nir_scalar_alu_op(s);
   if (op == nir_op_iadd) {
      nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
      nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
      return (same_loop_k_base(loop, a, base) && nir_scalar_is_const(b) &&
              nir_scalar_as_uint(b) == c) ||
             (same_loop_k_base(loop, b, base) && nir_scalar_is_const(a) &&
              nir_scalar_as_uint(a) == c);
   }
   if (op == nir_op_imadshl_agx && loop && is_loop_k_stride32(loop, base)) {
      nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
      nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
      nir_scalar d = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 2));
      nir_scalar sh = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 3));
      if (nir_scalar_is_const(sh) && nir_scalar_as_uint(sh) == 0 &&
          nir_scalar_is_const(d) && nir_scalar_as_uint(d) == c) {
         if ((match_const_loop_step_phi(loop, a, 1) &&
              nir_scalar_is_const(b) && nir_scalar_as_uint(b) == 32) ||
             (match_const_loop_step_phi(loop, b, 1) &&
              nir_scalar_is_const(a) && nir_scalar_as_uint(a) == 32))
            return true;
      }
   }
   return false;
}

static bool
match_add_const(nir_scalar s, nir_scalar base, uint32_t c)
{
   return match_add_const_loop(NULL, s, base, c);
}

static bool
match_k_row_r(nir_loop *loop, nir_scalar s, nir_scalar k_base_s, unsigned r)
{
   s = nir_scalar_chase_movs(s);
   if (r == 0) {
      nir_def *kb = NULL;
      if (match_k_row0(s, &kb) &&
          same_loop_k_base(loop, nir_scalar_resolved(kb, 0), k_base_s))
         return true;
      if (nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_imadshl_agx &&
          loop && is_loop_k_stride32(loop, k_base_s)) {
         nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
         nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
         nir_scalar c = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 2));
         nir_scalar sh = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 3));
         if (nir_scalar_is_const(sh) && nir_scalar_as_uint(sh) == 0 &&
             is_ly_shl3(c) &&
             ((match_const_loop_step_phi(loop, a, 1) &&
               nir_scalar_is_const(b) && nir_scalar_as_uint(b) == 32) ||
              (match_const_loop_step_phi(loop, b, 1) &&
               nir_scalar_is_const(a) && nir_scalar_as_uint(a) == 32)))
            return true;
      }
      return false;
   }
   if (!nir_scalar_is_alu(s) || nir_scalar_alu_op(s) != nir_op_iadd)
      return false;
   nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
   nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
   /* (k_base + 8*ly) + r */
   if (nir_scalar_is_const(b) && nir_scalar_as_uint(b) == r &&
       match_k_row_r(loop, a, k_base_s, 0))
      return true;
   if (nir_scalar_is_const(a) && nir_scalar_as_uint(a) == r &&
       match_k_row_r(loop, b, k_base_s, 0))
      return true;
   /* (8*ly) + (k_base + r) */
   if (is_ly_shl3(a) && match_add_const_loop(loop, b, k_base_s, r))
      return true;
   if (is_ly_shl3(b) && match_add_const_loop(loop, a, k_base_s, r))
      return true;
   /* k_base + (8*ly + r) */
   if (same_loop_k_base(loop, a, k_base_s) && nir_scalar_is_alu(b) &&
       nir_scalar_alu_op(b) == nir_op_iadd) {
      nir_scalar b0 = nir_scalar_chase_alu_src(b, 0);
      nir_scalar b1 = nir_scalar_chase_alu_src(b, 1);
      if ((is_ly_shl3(b0) && nir_scalar_is_const(b1) && nir_scalar_as_uint(b1) == r) ||
          (is_ly_shl3(b1) && nir_scalar_is_const(b0) && nir_scalar_as_uint(b0) == r))
         return true;
   }
   if (same_loop_k_base(loop, b, k_base_s) && nir_scalar_is_alu(a) &&
       nir_scalar_alu_op(a) == nir_op_iadd) {
      nir_scalar a0 = nir_scalar_chase_alu_src(a, 0);
      nir_scalar a1 = nir_scalar_chase_alu_src(a, 1);
      if ((is_ly_shl3(a0) && nir_scalar_is_const(a1) && nir_scalar_as_uint(a1) == r) ||
          (is_ly_shl3(a1) && nir_scalar_is_const(a0) && nir_scalar_as_uint(a0) == r))
         return true;
   }
   return false;
}

static bool
match_and_pair(nir_scalar cond, nir_scalar *lhs, nir_scalar *rhs)
{
   cond = nir_scalar_chase_movs(cond);
   if (!nir_scalar_is_alu(cond))
      return false;
   nir_op op = nir_scalar_alu_op(cond);
   if (op == nir_op_iand) {
      *lhs = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 0));
      *rhs = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 1));
      return true;
   }
   if (op == nir_op_bcsel) {
      nir_scalar s0 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 0));
      nir_scalar s1 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 1));
      nir_scalar s2 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 2));
      if (nir_scalar_is_const(s2) && !nir_scalar_as_bool(s2)) {
         *lhs = s0;
         *rhs = s1;
         return true;
      }
   }
   return false;
}

static bool
match_ilt_uniform(nir_scalar cmp, nir_scalar *val_out, nir_scalar *lim_out)
{
   cmp = nir_scalar_chase_movs(cmp);
   if (!nir_scalar_is_alu(cmp) || nir_scalar_alu_op(cmp) != nir_op_ilt)
      return false;
   nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cmp, 0));
   nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cmp, 1));
   if (a.def->bit_size != 32 || b.def->bit_size != 32 || b.def->divergent)
      return false;
   *val_out = a;
   *lim_out = b;
   return true;
}

static bool
unwrap_byte_shift2(nir_scalar s, nir_scalar *elem_out)
{
   s = nir_scalar_chase_movs(s);
   if (!nir_scalar_is_alu(s))
      return false;
   nir_op op = nir_scalar_alu_op(s);
   if (op == nir_op_ishl) {
      nir_scalar sh = nir_scalar_chase_alu_src(s, 1);
      if (nir_scalar_is_const(sh) && nir_scalar_as_uint(sh) == 2) {
         *elem_out = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
         return true;
      }
      return false;
   }
   if (op == nir_op_imadshl_agx) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      nir_scalar c = nir_scalar_chase_alu_src(s, 2);
      nir_scalar sh = nir_scalar_chase_alu_src(s, 3);
      if (nir_scalar_is_const(sh) && nir_scalar_as_uint(sh) == 2 &&
          ((nir_scalar_is_const(a) && nir_scalar_as_uint(a) == 0) ||
           (nir_scalar_is_const(b) && nir_scalar_as_uint(b) == 0))) {
         *elem_out = nir_scalar_chase_movs(c);
         return true;
      }
   }
   return false;
}

static bool
block_inside_loop(const nir_block *block, const nir_loop *target)
{
   for (const nir_cf_node *n = block->cf_node.parent; n; n = n->parent) {
      if (n == &target->cf_node)
         return true;
   }
   return false;
}

static bool
agx_writes_external_memory(const nir_intrinsic_instr *intr)
{
   if (nir_intrinsic_writes_external_memory(intr))
      return true;
   switch (intr->intrinsic) {
   case nir_intrinsic_store_agx:
   case nir_intrinsic_global_atomic_agx:
   case nir_intrinsic_global_atomic_swap_agx:
   case nir_intrinsic_image_store_block_agx:
   case nir_intrinsic_bindless_image_store_block_agx:
   case nir_intrinsic_image_deref_store_block_agx:
      return true;
   default:
      return false;
   }
}

static bool
is_external_memory_access(const nir_intrinsic_instr *intr)
{
   if (intr->intrinsic == nir_intrinsic_barrier)
      return true;
   if (agx_writes_external_memory(intr))
      return true;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_agx:
   case nir_intrinsic_load_constant_agx:
   case nir_intrinsic_load_global:
   case nir_intrinsic_load_global_constant:
   case nir_intrinsic_load_global_predicated_agx:
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_load_ubo:
   case nir_intrinsic_image_load:
   case nir_intrinsic_bindless_image_load:
      return true;
   default:
      return false;
   }
}

static bool
def_only_feeds_fma_b(const nir_def *def,
                     nir_alu_instr *const fma[AGX_GEMM_WIDTH][AGX_GEMM_ROWS_PER_LANE],
                     const uint8_t b_src_idx[AGX_GEMM_WIDTH][AGX_GEMM_ROWS_PER_LANE],
                     unsigned depth)
{
   if (depth > 4)
      return false;
   nir_foreach_use_including_if(us, def) {
      if (nir_src_is_if(us))
         return false;
      nir_instr *u = nir_src_use_instr(us);
      if (u->type == nir_instr_type_alu) {
         nir_alu_instr *alu = nir_instr_as_alu(u);
         if (nir_op_is_vec_or_mov(alu->op)) {
            if (!def_only_feeds_fma_b(&alu->def, fma, b_src_idx, depth + 1))
               return false;
            continue;
         }
      }
      bool matched = false;
      for (unsigned k = 0; k < AGX_GEMM_WIDTH && !matched; ++k) {
         for (unsigned row = 0; row < AGX_GEMM_ROWS_PER_LANE; ++row) {
            if (u == &fma[k][row]->instr &&
                us == &fma[k][row]->src[b_src_idx[k][row]].src) {
               matched = true;
               break;
            }
         }
      }
      if (!matched)
         return false;
   }
   return true;
}

static bool
b_shared_only_read_by_fma(nir_function_impl *impl,
                          const struct agx_gemm_match *m)
{
   nir_intrinsic_instr *a_loads[TILE_MATH];
   nir_intrinsic_instr *b_loads[TILE_MATH];
   unsigned na = 0, nb = 0;
   uint8_t b_src_idx[AGX_GEMM_WIDTH][AGX_GEMM_ROWS_PER_LANE];

   for (unsigned k = 0; k < AGX_GEMM_WIDTH; ++k) {
      for (unsigned row = 0; row < AGX_GEMM_ROWS_PER_LANE; ++row) {
         struct operands ops;
         if (!fma_operands(m->fma[k][row], m->ready, m->consumed, &ops))
            return false;
         nir_scalar def = nir_get_scalar(&m->fma[k][row]->def, 0);
         nir_scalar s0 = nir_scalar_chase_alu_src(def, 0);
         nir_scalar s1 = nir_scalar_chase_alu_src(def, 1);
         if (nir_scalar_is_intrinsic(s0) && nir_scalar_as_intrinsic(s0) == ops.b)
            b_src_idx[k][row] = 0;
         else if (nir_scalar_is_intrinsic(s1) &&
                  nir_scalar_as_intrinsic(s1) == ops.b)
            b_src_idx[k][row] = 1;
         else
            return false;
         a_loads[na++] = ops.a;
         b_loads[nb++] = ops.b;
      }
   }

   nir_foreach_block(block, impl) {
      nir_foreach_instr(ins, block) {
         if (ins->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
         if (intr->intrinsic != nir_intrinsic_load_shared)
            continue;
         bool in_a = false, in_b = false;
         for (unsigned i = 0; i < na; ++i)
            if (a_loads[i] == intr)
               in_a = true;
         for (unsigned i = 0; i < nb; ++i)
            if (b_loads[i] == intr)
               in_b = true;
         if (in_a && !in_b) {
            /* A-tile loads were verified by collect_phase() to lie strictly in
             * [a_bytes, a_bytes + TILE_BYTES), disjoint from B. */
            continue;
         }
         if (!in_b)
            return false;
         if (!def_only_feeds_fma_b(&intr->def, m->fma, b_src_idx, 0))
            return false;
      }
   }
   return true;
}

static void
try_match_b_direct(nir_shader *shader, struct agx_gemm_match *m)
{
   m->b_direct = false;
   nir_function_impl *impl = m->ready->block->impl;

   /* Direct-B drops the B producer stores and may shrink shared_size to
    * b_bytes, so every load_shared that can overlap [b_bytes, b_bytes + 4 KiB)
    * must be a matched FMA B operand with no other uses. */
   if (!b_shared_only_read_by_fma(impl, m))
      return;

   nir_intrinsic_instr *stores[AGX_GEMM_ROWS_PER_LANE] = {0};

   /* Reject if any block anywhere inside m->loop (including nested loops)
    * writes external memory, or writes b_bytes outside m->ready->block before
    * m->ready. Unguarded Direct-B only: guarded B tiles keep shared staging
    * so the K-loop stays branch-free and compact on AGX3. */
   nir_foreach_block(block, impl) {
      if (!block_inside_loop(block, m->loop))
         continue;
      bool in_ready_prefix = (block == m->ready->block);
      nir_foreach_instr(ins, block) {
         if (ins == m->ready)
            in_ready_prefix = false;
         if (ins->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
         if (agx_writes_external_memory(intr))
            return;
         if (intr->intrinsic == nir_intrinsic_store_shared) {
            uint16_t slot;
            if (!producer_slot(m, intr, &slot))
               return;
            if (slot < (1u << AGX_GEMM_ROWS_PER_LANE))
               continue;
            unsigned row = ffs(slot >> AGX_GEMM_ROWS_PER_LANE) - 1;
            if (row >= AGX_GEMM_ROWS_PER_LANE || !in_ready_prefix || stores[row])
               return;
            stores[row] = intr;
         }
      }
   }

   nir_def *base = NULL, *k_base = NULL, *col_orig = NULL;
   nir_scalar stride = {0};
   nir_scalar row0_scalar = {0}, idx0_scalar = {0};
   enum gl_access_qualifier access = 0;
   bool is_const = false;

   for (unsigned r = 0; r < AGX_GEMM_ROWS_PER_LANE; ++r) {
      if (!stores[r])
         return;
      nir_scalar val = nir_scalar_resolved(stores[r]->src[0].ssa, 0);
      if (!nir_scalar_is_intrinsic(val))
         return;
      nir_intrinsic_instr *ld = nir_scalar_as_intrinsic(val);
      if (ld->intrinsic != nir_intrinsic_load_agx &&
          ld->intrinsic != nir_intrinsic_load_constant_agx)
         return;
      enum gl_access_qualifier ld_access = nir_intrinsic_access(ld);
      if (ld->def.num_components != 1 || ld->def.bit_size != 32 ||
          nir_intrinsic_format(ld) != PIPE_FORMAT_R32_UINT ||
          nir_intrinsic_base(ld) != 0 ||
          (ld_access & (ACCESS_VOLATILE | ACCESS_COHERENT)) ||
          (ld->intrinsic == nir_intrinsic_load_agx &&
           !(ld_access & ACCESS_NON_WRITEABLE)) ||
          ld->src[0].ssa->divergent)
         return;

      nir_def *r_base = ld->src[0].ssa;
      nir_scalar ld_idx = nir_scalar_resolved(ld->src[1].ssa, 0);

      if (r == 0) {
         nir_scalar row_s, col_s, row_stride;
         if (!match_linear_b_idx(ld_idx, &row_s, &row_stride, &col_s))
            return;
         nir_def *row_col_orig = NULL;
         if (!match_add_local_x(col_s, &row_col_orig))
            return;
         if (!match_k_row0(row_s, &k_base))
            return;
         base = r_base;
         stride = row_stride;
         col_orig = row_col_orig;
         row0_scalar = row_s;
         idx0_scalar = ld_idx;
         access = ld_access;
         is_const = ld->intrinsic == nir_intrinsic_load_constant_agx;
      } else {
         if (is_const != (ld->intrinsic == nir_intrinsic_load_constant_agx) ||
             access != ld_access ||
             !same_preamble_or_def_64(r_base, base))
            return;
         bool ok = false;
         nir_scalar is = nir_scalar_chase_movs(ld_idx);
         if (nir_scalar_is_alu(is) && nir_scalar_alu_op(is) == nir_op_iadd) {
            nir_scalar a = nir_scalar_chase_alu_src(is, 0);
            nir_scalar b = nir_scalar_chase_alu_src(is, 1);
            uint32_t mult = 0;
            if (nir_scalar_equal(nir_scalar_chase_movs(a), idx0_scalar) &&
                eval_stride_multiple(shader, b, stride, 6, &mult) &&
                mult == r)
               ok = true;
            else if (nir_scalar_equal(nir_scalar_chase_movs(b), idx0_scalar) &&
                     eval_stride_multiple(shader, a, stride, 6, &mult) &&
                     mult == r)
               ok = true;
         } else if (nir_scalar_is_alu(is) &&
                    nir_scalar_alu_op(is) == nir_op_imadshl_agx) {
            nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(is, 0));
            nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(is, 1));
            nir_scalar cc = nir_scalar_chase_movs(nir_scalar_chase_alu_src(is, 2));
            nir_scalar sh = nir_scalar_chase_movs(nir_scalar_chase_alu_src(is, 3));
            if (nir_scalar_is_const(sh) && nir_scalar_as_uint(sh) < 32) {
               uint32_t shift = (uint32_t)nir_scalar_as_uint(sh);
               uint32_t mult = 0;
               if (((nir_scalar_equal(a, idx0_scalar) && nir_scalar_is_const(b) &&
                     nir_scalar_as_uint(b) == 1) ||
                    (nir_scalar_equal(b, idx0_scalar) && nir_scalar_is_const(a) &&
                     nir_scalar_as_uint(a) == 1)) &&
                   eval_stride_multiple(shader, cc, stride, 6, &mult) &&
                   (mult << shift) == r) {
                  ok = true;
               }
               if (!ok && shift == 0 && nir_scalar_equal(cc, idx0_scalar)) {
                  if (nir_scalar_is_const(a) &&
                      eval_stride_multiple(shader, b, stride, 6, &mult) &&
                      (uint32_t)nir_scalar_as_uint(a) * mult == r)
                     ok = true;
                  else if (nir_scalar_is_const(b) &&
                           eval_stride_multiple(shader, a, stride, 6, &mult) &&
                           (uint32_t)nir_scalar_as_uint(b) * mult == r)
                     ok = true;
               }
            }
         }
         if (!ok) {
            nir_scalar row_s, col_s, row_stride;
            nir_def *row_col_orig = NULL;
            if (match_linear_b_idx(ld_idx, &row_s, &row_stride, &col_s) &&
                same_preamble_or_def(row_stride, stride) &&
                match_add_local_x(col_s, &row_col_orig) &&
                row_col_orig == col_orig &&
                (match_add_const(row_s, row0_scalar, r) ||
                 match_k_row_r(m->loop, row_s, nir_get_scalar(k_base, 0), r))) {
               ok = true;
            }
         }
         if (!ok)
            return;
      }
   }

   if (!can_materialize_uniform_def(base, m->ready->block) ||
       !can_materialize_uniform_def(stride.def, m->ready->block) ||
       !nir_block_dominates(nir_def_block(k_base), m->ready->block) ||
       !nir_block_dominates(nir_def_block(col_orig), m->ready->block))
      return;

   m->b_direct = true;
   m->b_constant = is_const;
   m->b_access = access;
   m->b_base = base;
   m->b_stride = stride;
   m->b_k_base = k_base;
   m->b_col_orig = col_orig;
   memcpy(m->b_stores, stores, sizeof(stores));
}

struct c_store_desc {
   nir_intrinsic_instr *store;
   nir_block *outer_block;
   nir_instr *insert_before;
   bool guarded;
   enum agx_gemm_bias_kind bias_kind;
   enum gl_access_qualifier store_access;
   enum gl_access_qualifier bias_access;
   bool bias_constant;
   nir_def *c_base;
   nir_def *bias_base;
   nir_scalar c_idx;
   nir_scalar row_s;
   nir_scalar stride_s;
   nir_def *row_base;
   nir_def *col_orig;
   nir_scalar m_limit;
   nir_scalar n_limit;
   bool has_clamp;
   bool clamp_agx;
   uint32_t clamp_lo_bits;
   uint32_t clamp_hi_bits;
   unsigned clamp_max_fp_ctrl;
   unsigned clamp_min_fp_ctrl;
   nir_if *row_if;
   nir_intrinsic_instr *bias_load;
};

static bool
single_use_src(nir_def *def, nir_src **out_src)
{
   nir_src *found = NULL;
   nir_foreach_use_including_if(us, def) {
      if (nir_src_is_if(us) || found)
         return false;
      found = us;
   }
   if (!found)
      return false;
   *out_src = found;
   return true;
}

static bool
match_c_store_use(nir_shader *shader, nir_src *use_src, unsigned r,
                  nir_block *exit_block, struct c_store_desc *out)
{
   if (nir_src_is_if(use_src))
      return false;
   nir_src *cur_src = use_src;
   nir_instr *user = nir_src_use_instr(cur_src);
   nir_intrinsic_instr *st = NULL;
   enum agx_gemm_bias_kind bias_kind = AGX_GEMM_BIAS_NONE;
   nir_scalar bias_s = {0};
   bool has_clamp = false;
   bool clamp_agx = false;
   uint32_t clamp_lo_bits = 0, clamp_hi_bits = 0;
   unsigned clamp_max_fp_ctrl = 0, clamp_min_fp_ctrl = 0;
   nir_if *row_if = NULL;
   nir_intrinsic_instr *bias_load = NULL;

   /* Optional bias addition: fadd(exit[r], bias). */
   if (user->type == nir_instr_type_alu &&
       nir_instr_as_alu(user)->op == nir_op_fadd) {
      nir_alu_instr *alu = nir_instr_as_alu(user);
      if (alu->def.num_components != 1 || alu->def.bit_size != 32 ||
          alu->fp_math_ctrl != 0)
         return false;
      unsigned other = (&alu->src[0].src == cur_src) ? 1 : 0;
      bias_s = nir_scalar_chase_movs(nir_scalar_chase_alu_src(
         nir_get_scalar(&alu->def, 0), other));
      if (!single_use_src(&alu->def, &cur_src))
         return false;
      user = nir_src_use_instr(cur_src);
   }

   /* Optional pointwise clamp: fmin(fmax(val, lo), hi) or fmin_agx(fmax_agx(...)). */
   if (user->type == nir_instr_type_alu &&
       (nir_instr_as_alu(user)->op == nir_op_fmax ||
        nir_instr_as_alu(user)->op == nir_op_fmax_agx)) {
      nir_alu_instr *max_alu = nir_instr_as_alu(user);
      const bool is_agx = max_alu->op == nir_op_fmax_agx;
      if (max_alu->def.num_components != 1 || max_alu->def.bit_size != 32)
         return false;
      unsigned lo_idx = (&max_alu->src[0].src == cur_src) ? 1 : 0;
      nir_scalar lo_s = nir_scalar_chase_movs(nir_scalar_chase_alu_src(
         nir_get_scalar(&max_alu->def, 0), lo_idx));
      if (!nir_scalar_is_const(lo_s))
         return false;
      nir_src *min_src = NULL;
      if (!single_use_src(&max_alu->def, &min_src))
         return false;
      nir_instr *min_ins = nir_src_use_instr(min_src);
      if (min_ins->type != nir_instr_type_alu ||
          nir_instr_as_alu(min_ins)->op != (is_agx ? nir_op_fmin_agx : nir_op_fmin))
         return false;
      nir_alu_instr *min_alu = nir_instr_as_alu(min_ins);
      if (min_alu->def.num_components != 1 || min_alu->def.bit_size != 32)
         return false;
      unsigned hi_idx = (&min_alu->src[0].src == min_src) ? 1 : 0;
      nir_scalar hi_s = nir_scalar_chase_movs(nir_scalar_chase_alu_src(
         nir_get_scalar(&min_alu->def, 0), hi_idx));
      if (!nir_scalar_is_const(hi_s))
         return false;
      has_clamp = true;
      clamp_agx = is_agx;
      clamp_lo_bits = (uint32_t)nir_scalar_as_uint(lo_s);
      clamp_hi_bits = (uint32_t)nir_scalar_as_uint(hi_s);
      clamp_max_fp_ctrl = max_alu->fp_math_ctrl;
      clamp_min_fp_ctrl = min_alu->fp_math_ctrl;
      if (!single_use_src(&min_alu->def, &cur_src))
         return false;
      user = nir_src_use_instr(cur_src);
   }

   if (user->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(user);
   if (intr->intrinsic != nir_intrinsic_store_agx || cur_src != &intr->src[0])
      return false;
   st = intr;

   if (st->num_components != 1 || st->src[0].ssa->bit_size != 32 ||
       nir_intrinsic_format(st) != PIPE_FORMAT_R32_UINT ||
       nir_intrinsic_base(st) != 0 ||
       (nir_intrinsic_access(st) & (ACCESS_VOLATILE | ACCESS_COHERENT)) ||
       st->src[1].ssa->divergent)
      return false;

   nir_block *st_block = st->instr.block;
   if (!nir_block_dominates(exit_block, st_block))
      return false;

   bool guarded = false;
   nir_block *outer_block = st_block;
   nir_instr *insert_before = &st->instr;
   nir_scalar m_limit = {0}, n_limit = {0};
   nir_scalar guard_row = {0};
   nir_def *guard_col = NULL;

   if (st_block->divergent) {
      /* Guarded store inside single-block then arm of if (row_r < M && col < N). */
      nir_cf_node *parent = st_block->cf_node.parent;
      if (!parent || parent->type != nir_cf_node_if)
         return false;
      nir_if *nif = nir_cf_node_as_if(parent);
      if (!exec_list_is_singular(&nif->then_list) ||
          nir_if_first_then_block(nif) != st_block ||
          !exec_list_is_singular(&nif->else_list) ||
          !exec_list_is_empty(&nir_if_first_else_block(nif)->instr_list))
         return false;
      outer_block = nir_cf_node_as_block(nir_cf_node_prev(&nif->cf_node));
      if (outer_block->divergent || !nir_block_dominates(exit_block, outer_block))
         return false;
      row_if = nif;
      /* Ensure no other stores/barriers/atomics in st_block. */
      nir_foreach_instr(ins, st_block) {
         if (ins == &st->instr)
            continue;
         if (ins->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *i = nir_instr_as_intrinsic(ins);
            if (i->intrinsic == nir_intrinsic_store_shared ||
                i->intrinsic == nir_intrinsic_barrier ||
                agx_writes_external_memory(i))
               return false;
         }
      }
      nir_scalar p0, p1;
      if (!match_and_pair(nir_scalar_resolved(nif->condition.ssa, 0), &p0, &p1))
         return false;
      nir_scalar v0, l0, v1, l1;
      if (!match_ilt_uniform(p0, &v0, &l0) || !match_ilt_uniform(p1, &v1, &l1))
         return false;
      nir_def *dummy = NULL;
      if (match_add_local_x(v0, &dummy)) {
         guard_col = dummy; n_limit = l0;
         guard_row = v1; m_limit = l1;
      } else if (match_add_local_x(v1, &dummy)) {
         guard_col = dummy; n_limit = l1;
         guard_row = v0; m_limit = l0;
      } else {
         return false;
      }
      guarded = true;
      insert_before = NULL;
   }

   nir_scalar c_idx = nir_scalar_resolved(st->src[2].ssa, 0);
   nir_scalar row_s = {0}, stride_s = {0}, col_s = {0};
   nir_def *row_base = NULL, *col_orig = NULL;

   if (r == 0 || guarded) {
      if (!match_linear_b_idx(c_idx, &row_s, &stride_s, &col_s))
         return false;
      if (!match_add_local_x(col_s, &col_orig))
         return false;
      if (r == 0) {
         if (!match_add_local_y8(row_s, &row_base))
            return false;
      } else {
         /* guarded r > 0: row_s = row_base + r */
         nir_scalar rs = nir_scalar_chase_movs(row_s);
         if (!nir_scalar_is_alu(rs) || nir_scalar_alu_op(rs) != nir_op_iadd)
            return false;
         nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(rs, 0));
         nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(rs, 1));
         if (nir_scalar_is_const(b) && nir_scalar_as_uint(b) == r &&
             match_add_local_y8(a, &row_base)) {
         } else if (nir_scalar_is_const(a) && nir_scalar_as_uint(a) == r &&
                    match_add_local_y8(b, &row_base)) {
         } else {
            return false;
         }
      }
      if (guarded) {
         if (guard_col != col_orig || !nir_scalar_equal(guard_row, nir_scalar_chase_movs(row_s)))
            return false;
      }
   }

   nir_def *bias_base = NULL;
   enum gl_access_qualifier bias_access = 0;
   bool bias_constant = false;

   if (bias_s.def) {
      if (!nir_scalar_is_intrinsic(bias_s))
         return false;
      nir_intrinsic_instr *bld = nir_scalar_as_intrinsic(bias_s);
      if (bld->intrinsic != nir_intrinsic_load_agx &&
          bld->intrinsic != nir_intrinsic_load_constant_agx)
         return false;
      if (bld->def.bit_size != 32 ||
          nir_intrinsic_format(bld) != PIPE_FORMAT_R32_UINT ||
          (nir_intrinsic_access(bld) & (ACCESS_VOLATILE | ACCESS_COHERENT)) ||
          bld->src[0].ssa->divergent)
         return false;
      uint32_t b_base = nir_intrinsic_base(bld);
      if (b_base & 0xffffu)
         return false;
      uint32_t byte_off = b_base >> AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT;
      if (byte_off % 4 != 0)
         return false;
      uint32_t elem_off = byte_off / 4 + bias_s.comp;

      nir_scalar bidx = nir_scalar_resolved(bld->src[1].ssa, 0);
      nir_def *b_col = NULL, *b_row = NULL;
      if (elem_off == 0 && match_add_local_x(bidx, &b_col)) {
         bias_kind = AGX_GEMM_BIAS_COL;
         col_orig = col_orig ? col_orig : b_col;
         if (b_col != col_orig)
            return false;
      } else if (elem_off == r && match_add_local_y8(bidx, &b_row)) {
         bias_kind = AGX_GEMM_BIAS_ROW;
         row_base = row_base ? row_base : b_row;
         if (b_row != row_base)
            return false;
      } else if (elem_off == 0 && r > 0 && nir_scalar_is_alu(bidx) &&
                 nir_scalar_alu_op(bidx) == nir_op_iadd) {
         nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(bidx, 0));
         nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(bidx, 1));
         if ((nir_scalar_is_const(b) && nir_scalar_as_uint(b) == r &&
              match_add_local_y8(a, &b_row)) ||
             (nir_scalar_is_const(a) && nir_scalar_as_uint(a) == r &&
              match_add_local_y8(b, &b_row))) {
            bias_kind = AGX_GEMM_BIAS_ROW;
            row_base = row_base ? row_base : b_row;
            if (b_row != row_base)
               return false;
         } else {
            return false;
         }
      } else {
         return false;
      }
      bias_base = bld->src[0].ssa;
      bias_access = nir_intrinsic_access(bld);
      bias_constant = bld->intrinsic == nir_intrinsic_load_constant_agx;
      bias_load = bld;
   }

   *out = (struct c_store_desc){
      .store = st,
      .outer_block = outer_block,
      .insert_before = insert_before,
      .guarded = guarded,
      .bias_kind = bias_kind,
      .store_access = nir_intrinsic_access(st),
      .bias_access = bias_access,
      .bias_constant = bias_constant,
      .c_base = st->src[1].ssa,
      .bias_base = bias_base,
      .c_idx = c_idx,
      .row_s = row_s,
      .stride_s = stride_s,
      .row_base = row_base,
      .col_orig = col_orig,
      .m_limit = m_limit,
      .n_limit = n_limit,
      .has_clamp = has_clamp,
      .clamp_agx = clamp_agx,
      .clamp_lo_bits = clamp_lo_bits,
      .clamp_hi_bits = clamp_hi_bits,
      .clamp_max_fp_ctrl = clamp_max_fp_ctrl,
      .clamp_min_fp_ctrl = clamp_min_fp_ctrl,
      .row_if = row_if,
      .bias_load = bias_load,
   };
   return true;
}

static bool
same_cf_parent(nir_block *a, nir_block *b)
{
   return a->cf_node.parent == b->cf_node.parent;
}

static void
try_match_c_direct(nir_shader *shader, struct agx_gemm_match *m)
{
   m->c_direct = false;
   m->c_group_count = 0;

   nir_block *exit_block = m->exit[0]->instr.block;
   if (exit_block->divergent)
      return;

   /* If the K-loop can be skipped (e.g. K == 0), every exit phi's non-loop
    * incoming value must be the exact same constant across all 8 rows so
    * direct fragment layout and per-row layout hold identical values. */
   bool have_outside_const = false;
   uint32_t outside_const = 0;
   for (unsigned r = 0; r < AGX_GEMM_ROWS_PER_LANE; ++r) {
      if (m->exit[r]->instr.block != exit_block)
         return;
      nir_foreach_phi_src(ps, m->exit[r]) {
         if (ps->src.ssa == &m->fma[AGX_GEMM_WIDTH - 1][r]->def)
            continue;
         nir_scalar s = nir_scalar_resolved(ps->src.ssa, 0);
         if (!nir_scalar_is_const(s))
            return;
         uint32_t v = (uint32_t)nir_scalar_as_uint(s);
         if (!have_outside_const) {
            have_outside_const = true;
            outside_const = v;
         } else if (v != outside_const) {
            return;
         }
      }
   }

   /* Collect all uses of exit[0] to seed candidate store groups. */
   struct c_store_desc seeds[AGX_GEMM_MAX_C_GROUPS];
   unsigned ngroups = 0;
   nir_foreach_use_including_if(us, &m->exit[0]->def) {
      if (ngroups >= AGX_GEMM_MAX_C_GROUPS)
         return;
      if (!match_c_store_use(shader, us, 0, exit_block, &seeds[ngroups]))
         return;
      ngroups++;
   }
   if (ngroups == 0)
      return;

   struct agx_gemm_c_group groups[AGX_GEMM_MAX_C_GROUPS] = {0};
   nir_scalar idx0[AGX_GEMM_MAX_C_GROUPS];
   for (unsigned g = 0; g < ngroups; ++g) {
      groups[g] = (struct agx_gemm_c_group){
         .insert_block = seeds[g].outer_block,
         .insert_before = seeds[g].insert_before,
         .guarded = seeds[g].guarded,
         .bias_kind = seeds[g].bias_kind,
         .store_access = seeds[g].store_access,
         .bias_access = seeds[g].bias_access,
         .bias_constant = seeds[g].bias_constant,
         .c_base = seeds[g].c_base,
         .c_stride = seeds[g].stride_s,
         .bias_base = seeds[g].bias_base,
         .row_base = seeds[g].row_base,
         .col_orig = seeds[g].col_orig,
         .m_limit = seeds[g].m_limit,
         .n_limit = seeds[g].n_limit,
         .has_clamp = seeds[g].has_clamp,
         .clamp_agx = seeds[g].clamp_agx,
         .clamp_lo_bits = seeds[g].clamp_lo_bits,
         .clamp_hi_bits = seeds[g].clamp_hi_bits,
         .clamp_max_fp_ctrl = seeds[g].clamp_max_fp_ctrl,
         .clamp_min_fp_ctrl = seeds[g].clamp_min_fp_ctrl,
      };
      groups[g].stores[0] = seeds[g].store;
      idx0[g] = seeds[g].c_idx;
   }
   nir_if *row_ifs[AGX_GEMM_MAX_C_GROUPS][AGX_GEMM_ROWS_PER_LANE] = {{0}};
   nir_intrinsic_instr *bias_loads[AGX_GEMM_MAX_C_GROUPS][AGX_GEMM_ROWS_PER_LANE] = {{0}};
   for (unsigned g = 0; g < ngroups; ++g) {
      row_ifs[g][0] = seeds[g].row_if;
      bias_loads[g][0] = seeds[g].bias_load;
   }

   /* Match every single use of exit[1..7] to one of the ngroups groups. */
   for (unsigned r = 1; r < AGX_GEMM_ROWS_PER_LANE; ++r) {
      unsigned matched_cnt = 0;
      nir_foreach_use_including_if(us, &m->exit[r]->def) {
         struct c_store_desc d;
         if (!match_c_store_use(shader, us, r, exit_block, &d))
            return;

         int found_g = -1;
         for (unsigned g = 0; g < ngroups; ++g) {
            if (groups[g].stores[r])
               continue;
            if (groups[g].guarded != d.guarded ||
                groups[g].bias_kind != d.bias_kind ||
                groups[g].store_access != d.store_access ||
                groups[g].has_clamp != d.has_clamp ||
                (d.has_clamp &&
                 (groups[g].clamp_agx != d.clamp_agx ||
                  groups[g].clamp_lo_bits != d.clamp_lo_bits ||
                  groups[g].clamp_hi_bits != d.clamp_hi_bits ||
                  groups[g].clamp_max_fp_ctrl != d.clamp_max_fp_ctrl ||
                  groups[g].clamp_min_fp_ctrl != d.clamp_min_fp_ctrl)) ||
                !same_preamble_or_def_64(groups[g].c_base, d.c_base))
               continue;
            if (d.guarded) {
               if (!same_cf_parent(groups[g].insert_block, d.outer_block) ||
                   d.row_base != groups[g].row_base ||
                   d.col_orig != groups[g].col_orig ||
                   !same_preamble_or_def(d.stride_s, groups[g].c_stride) ||
                   !same_preamble_or_def(d.m_limit, groups[g].m_limit) ||
                   !same_preamble_or_def(d.n_limit, groups[g].n_limit))
                  continue;
            } else {
               if (d.outer_block != groups[g].insert_block)
                  continue;
               /* Verify c_idx == idx0[g] + r * c_stride OR linear row_base+r. */
               bool ok = false;
               nir_scalar is = nir_scalar_chase_movs(d.c_idx);
               if (nir_scalar_is_alu(is) && nir_scalar_alu_op(is) == nir_op_iadd) {
                  nir_scalar a = nir_scalar_chase_alu_src(is, 0);
                  nir_scalar b = nir_scalar_chase_alu_src(is, 1);
                  uint32_t mult = 0;
                  if (nir_scalar_equal(nir_scalar_chase_movs(a), idx0[g]) &&
                      eval_stride_multiple(shader, b, groups[g].c_stride, 6, &mult) &&
                      mult == r)
                     ok = true;
                  else if (nir_scalar_equal(nir_scalar_chase_movs(b), idx0[g]) &&
                           eval_stride_multiple(shader, a, groups[g].c_stride, 6, &mult) &&
                           mult == r)
                     ok = true;
               } else if (nir_scalar_is_alu(is) &&
                          nir_scalar_alu_op(is) == nir_op_imadshl_agx) {
                  nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(is, 0));
                  nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(is, 1));
                  nir_scalar c = nir_scalar_chase_movs(nir_scalar_chase_alu_src(is, 2));
                  nir_scalar sh = nir_scalar_chase_movs(nir_scalar_chase_alu_src(is, 3));
                  if (nir_scalar_is_const(sh) && nir_scalar_as_uint(sh) < 32) {
                     uint32_t shift = (uint32_t)nir_scalar_as_uint(sh);
                     uint32_t mult = 0;
                     /* imadshl_agx(idx0, 1, stride, sh) == idx0 + (stride << sh) */
                     if (((nir_scalar_equal(a, idx0[g]) && nir_scalar_is_const(b) &&
                           nir_scalar_as_uint(b) == 1) ||
                          (nir_scalar_equal(b, idx0[g]) && nir_scalar_is_const(a) &&
                           nir_scalar_as_uint(a) == 1)) &&
                         eval_stride_multiple(shader, c, groups[g].c_stride, 6, &mult) &&
                         (mult << shift) == r) {
                        ok = true;
                     }
                     /* imadshl_agx(k, stride, idx0, 0) == k * stride + idx0 */
                     if (!ok && shift == 0 && nir_scalar_equal(c, idx0[g])) {
                        if (nir_scalar_is_const(a) &&
                            eval_stride_multiple(shader, b, groups[g].c_stride, 6, &mult) &&
                            (uint32_t)nir_scalar_as_uint(a) * mult == r)
                           ok = true;
                        else if (nir_scalar_is_const(b) &&
                                 eval_stride_multiple(shader, a, groups[g].c_stride, 6, &mult) &&
                                 (uint32_t)nir_scalar_as_uint(b) * mult == r)
                           ok = true;
                     }
                  }
               }
               if (!ok) {
                  nir_scalar rs, ss, cs;
                  nir_def *co = NULL;
                  if (match_linear_b_idx(d.c_idx, &rs, &ss, &cs) &&
                      same_preamble_or_def(ss, groups[g].c_stride) &&
                      match_add_local_x(cs, &co) && co == groups[g].col_orig &&
                      match_add_const(rs, nir_scalar_resolved(groups[g].row_base, 0), r))
                     ok = true;
               }
               if (!ok)
                  continue;
            }

            if (groups[g].bias_kind != AGX_GEMM_BIAS_NONE) {
               if (groups[g].bias_access != d.bias_access ||
                   groups[g].bias_constant != d.bias_constant ||
                   !same_preamble_or_def_64(groups[g].bias_base, d.bias_base))
                  continue;
               if (groups[g].bias_kind == AGX_GEMM_BIAS_ROW &&
                   d.row_base && d.row_base != groups[g].row_base)
                  continue;
               if (groups[g].bias_kind == AGX_GEMM_BIAS_COL &&
                   d.col_orig && d.col_orig != groups[g].col_orig)
                  continue;
            }

            found_g = (int)g;
            break;
         }
         if (found_g < 0)
            return;
         groups[found_g].stores[r] = d.store;
         row_ifs[found_g][r] = d.row_if;
         bias_loads[found_g][r] = d.bias_load;
         matched_cnt++;
      }
      if (matched_cnt != ngroups)
         return;
   }

   /* Final dominance/materialization & interference checks per group. */
   for (unsigned g = 0; g < ngroups; ++g) {
      nir_block *ib = groups[g].insert_block;
      if (!nir_block_dominates(nir_def_block(groups[g].row_base), ib) ||
          !nir_block_dominates(nir_def_block(groups[g].col_orig), ib) ||
          !can_materialize_uniform_def(groups[g].c_base, ib) ||
          !can_materialize_uniform_def(groups[g].c_stride.def, ib))
         return;
      if (groups[g].bias_kind != AGX_GEMM_BIAS_NONE &&
          !can_materialize_uniform_def(groups[g].bias_base, ib))
         return;
      if (groups[g].guarded &&
          (!can_materialize_uniform_def(groups[g].m_limit.def, ib) ||
           !can_materialize_uniform_def(groups[g].n_limit.def, ib)))
         return;
      if (!groups[g].guarded) {
         bool seen_any = false;
         unsigned seen_cnt = 0;
         nir_foreach_instr(ins, ib) {
            bool is_ours = false;
            for (unsigned r = 0; r < AGX_GEMM_ROWS_PER_LANE; ++r) {
               if (ins == &groups[g].stores[r]->instr) {
                  is_ours = true;
                  seen_any = true;
                  seen_cnt++;
                  break;
               }
            }
            if (is_ours || !seen_any || seen_cnt == AGX_GEMM_ROWS_PER_LANE)
               continue;
            if (ins->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *i = nir_instr_as_intrinsic(ins);
               bool is_bias = false;
               for (unsigned r = 0; r < AGX_GEMM_ROWS_PER_LANE; ++r) {
                  if (i == bias_loads[g][r]) {
                     is_bias = true;
                     break;
                  }
               }
               if (!is_bias &&
                   (i->intrinsic == nir_intrinsic_store_shared ||
                    is_external_memory_access(i)))
                  return;
            }
         }
         if (seen_cnt != AGX_GEMM_ROWS_PER_LANE)
            return;
      } else {
         /* Require the 8 guarded row ifs to form a contiguous run in their
          * parent CF list with only pure (side-effect/jump/memory-free) blocks
          * between them, and set insert_block to the block immediately before
          * the first row if in CF order. */
         nir_cf_node *parent = row_ifs[g][0]->cf_node.parent;
         struct exec_list *cf_list = NULL;
         if (parent->type == nir_cf_node_function)
            cf_list = &nir_cf_node_as_function(parent)->body;
         else if (parent->type == nir_cf_node_if) {
            nir_if *pif = nir_cf_node_as_if(parent);
            /* Determine whether row_ifs[g][0] is in then_list or else_list. */
            bool in_then = false;
            foreach_list_typed(nir_cf_node, cn, node, &pif->then_list) {
               if (cn == &row_ifs[g][0]->cf_node) {
                  in_then = true;
                  break;
               }
            }
            cf_list = in_then ? &pif->then_list : &pif->else_list;
         } else {
            return;
         }

         unsigned seen = 0;
         nir_block *first_prev = NULL;
         foreach_list_typed(nir_cf_node, cn, node, cf_list) {
            bool is_row_if = false;
            for (unsigned r = 0; r < AGX_GEMM_ROWS_PER_LANE; ++r) {
               if (cn == &row_ifs[g][r]->cf_node) {
                  is_row_if = true;
                  break;
               }
            }
            if (is_row_if) {
               if (seen == 0) {
                  nir_cf_node *prev = nir_cf_node_prev(cn);
                  if (!prev || prev->type != nir_cf_node_block)
                     return;
                  first_prev = nir_cf_node_as_block(prev);
               }
               seen++;
               continue;
            }
            if (seen == 0 || seen == AGX_GEMM_ROWS_PER_LANE)
               continue;
            if (cn->type != nir_cf_node_block)
               return;
            nir_block *mid = nir_cf_node_as_block(cn);
            if (nir_block_ends_in_jump(mid))
               return;
            nir_foreach_instr(ins, mid) {
               if (ins->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *i = nir_instr_as_intrinsic(ins);
               bool is_bias = false;
               for (unsigned r = 0; r < AGX_GEMM_ROWS_PER_LANE; ++r) {
                  if (i == bias_loads[g][r]) {
                     is_bias = true;
                     break;
                  }
               }
               if (!is_bias &&
                   (i->intrinsic == nir_intrinsic_store_shared ||
                    is_external_memory_access(i)))
                  return;
            }
         }
         if (seen != AGX_GEMM_ROWS_PER_LANE || !first_prev)
            return;
         groups[g].insert_block = first_prev;
      }
   }

   /* Direct-C changes which invocation writes each C element and drops the
    * C staging barrier. Reject if the shader has any other external-memory
    * write anywhere (e.g. a pre-loop zero-init or second store), or any other
    * external-memory access (read/write/barrier) in or after exit_block
    * outside our matched stores and matched bias loads. */
   nir_function_impl *impl = exit_block->impl;
   nir_block *preheader = nir_loop_predecessor_block(m->loop);
   nir_foreach_block(block, impl) {
      bool before_or_in_loop =
         block_inside_loop(block, m->loop) ||
         nir_block_dominates(block, preheader);
      nir_foreach_instr(ins, block) {
         if (ins->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
         bool is_matched_store = false;
         bool is_matched_bias = false;
         for (unsigned g = 0; g < ngroups; ++g) {
            for (unsigned r = 0; r < AGX_GEMM_ROWS_PER_LANE; ++r) {
               if (intr == groups[g].stores[r])
                  is_matched_store = true;
               if (intr == bias_loads[g][r])
                  is_matched_bias = true;
            }
         }
         if (is_matched_store || is_matched_bias)
            continue;
         if (agx_writes_external_memory(intr))
            return;
         if (!before_or_in_loop && is_external_memory_access(intr))
            return;
      }
   }

   m->c_direct = true;
   m->c_group_count = ngroups;
   memcpy(m->c_groups, groups, ngroups * sizeof(groups[0]));
}

bool
agx_nir_match_gemm(nir_shader *shader, struct agx_gemm_match *result)
{
   if (shader->info.stage != MESA_SHADER_COMPUTE || !shader->info.io_lowered ||
       shader->info.workgroup_size_variable || !shader->info.shared_element_index_agx ||
       shader->info.workgroup_size[0] != AGX_GEMM_WIDTH ||
       shader->info.workgroup_size[1] != AGX_GEMM_SUBGROUPS ||
       shader->info.workgroup_size[2] != 1 ||
       shader->info.min_subgroup_size != AGX_GEMM_WIDTH ||
       shader->info.max_subgroup_size != AGX_GEMM_WIDTH ||
       shader->info.shared_size < 2 * TILE_BYTES ||
       shader->info.float_controls_execution_mode)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_metadata_require(impl, nir_metadata_block_index | nir_metadata_divergence |
                              nir_metadata_dominance);
   nir_foreach_block(block, impl) {
      if (block->divergent)
         continue;
      nir_loop *loop = innermost_loop(block);
      if (!loop || nir_loop_has_continue_construct(loop) ||
          innermost_loop(nir_loop_predecessor_block(loop)))
         continue;

      nir_instr *previous = NULL;
      nir_foreach_instr(ins, block) {
         if (ins->type != nir_instr_type_intrinsic || !fence(nir_instr_as_intrinsic(ins)))
            continue;
         if (previous &&
             nir_intrinsic_memory_scope(nir_instr_as_intrinsic(previous)) ==
                nir_intrinsic_memory_scope(nir_instr_as_intrinsic(ins))) {
            struct agx_gemm_match match = { .loop = loop, .ready = previous, .consumed = ins };
            if (collect_phase(shader, &match) && recurrences(&match) &&
                validate_visibility(&match)) {
               try_match_b_direct(shader, &match);
               try_match_c_direct(shader, &match);
               *result = match;
               return true;
            }
         }
         previous = ins;
      }
   }
   return false;
}

static nir_def *
shared_index(nir_builder *b, nir_def *row, nir_def *col, unsigned byte_base)
{
   nir_def *word = nir_iadd(b, nir_ishl_imm(b, row, 5), col);
   return nir_u2u16(b, nir_iadd_imm(b, word, byte_base / 4));
}

static void
replace_phi_source(nir_phi_instr *phi, nir_def *old, nir_def *value)
{
   nir_foreach_phi_src(src, phi) {
      if (src->src.ssa == old) {
         nir_src_rewrite(&src->src, value);
         return;
      }
   }
   assert(!"validated phi is missing its source");
}

struct fragment_position {
   nir_def *row;
   nir_def *subgroup_row;
   nir_def *col;
};

static struct fragment_position
fragment_position(nir_builder *b)
{
   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *offset = nir_ishl_imm(b, nir_load_subgroup_id(b), 3);
   nir_def *row =
      nir_ior(b, nir_iand_imm(b, nir_ushr_imm(b, lane, 1), 3),
                 nir_iand_imm(b, nir_ushr_imm(b, lane, 2), 4));
   nir_def *col =
      nir_ior(b, nir_ishl_imm(b, nir_iand_imm(b, lane, 1), 1),
                 nir_iand_imm(b, nir_ushr_imm(b, lane, 1), 4));
   return (struct fragment_position){ nir_iadd(b, offset, row), row, col };
}

static nir_def *
emit_mul_uniform(nir_builder *b, nir_scalar a, nir_scalar c)
{
   a = nir_scalar_chase_movs(a);
   c = nir_scalar_chase_movs(c);
   if (nir_scalar_is_const(c)) {
      uint32_t k = (uint32_t)nir_scalar_as_uint(c);
      if (util_is_power_of_two_nonzero(k))
         return nir_ishl_imm(b, nir_channel(b, a.def, a.comp), ffs(k) - 1);
   }
   if (nir_scalar_is_const(a)) {
      uint32_t k = (uint32_t)nir_scalar_as_uint(a);
      if (util_is_power_of_two_nonzero(k))
         return nir_ishl_imm(b, nir_channel(b, c.def, c.comp), ffs(k) - 1);
   }
   return nir_imul(b, nir_channel(b, a.def, a.comp),
                   nir_channel(b, c.def, c.comp));
}

static nir_def *
emit_lane_x_base(nir_builder *b, nir_scalar s)
{
   s = nir_scalar_chase_movs(s);
   if (nir_scalar_is_alu(s) &&
       (nir_scalar_alu_op(s) == nir_op_u2u32 ||
        nir_scalar_alu_op(s) == nir_op_u2u16))
      s = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
   assert(nir_scalar_is_intrinsic(s));
   if (nir_scalar_as_intrinsic(s)->intrinsic == nir_intrinsic_load_global_invocation_id)
      return nir_ishl_imm(b, nir_channel(b, nir_load_workgroup_id(b), 0), 5);
   return nir_imm_int(b, 0);
}

static nir_def *
emit_tile_base_x(nir_builder *b, nir_def *col_orig)
{
   nir_scalar s = nir_scalar_resolved(col_orig, 0);
   if (is_lane_x_linear(s))
      return emit_lane_x_base(b, s);
   if (nir_scalar_is_alu(s)) {
      nir_op op = nir_scalar_alu_op(s);
      if (op == nir_op_iadd) {
         nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
         nir_scalar c = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
         if (is_lane_x_linear(a) && !c.def->divergent)
            return nir_iadd(b, emit_lane_x_base(b, a), nir_channel(b, c.def, c.comp));
         if (is_lane_x_linear(c) && !a.def->divergent)
            return nir_iadd(b, emit_lane_x_base(b, c), nir_channel(b, a.def, a.comp));
      } else if (op == nir_op_imadshl_agx) {
         nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
         nir_scalar c = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
         nir_scalar d = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 2));
         nir_scalar sh = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 3));
         if (nir_scalar_is_const(sh) && nir_scalar_as_uint(sh) == 0 &&
             is_lane_x_linear(d) && !a.def->divergent && !c.def->divergent)
            return nir_iadd(b, emit_mul_uniform(b, a, c), emit_lane_x_base(b, d));
      }
   }
   UNREACHABLE("match_add_local_x guarantees uniform tile X base");
}

static nir_def *
emit_lane_y_base_x8(nir_builder *b, nir_scalar s)
{
   s = nir_scalar_chase_movs(s);
   if (nir_scalar_is_intrinsic(s)) {
      if (nir_scalar_as_intrinsic(s)->intrinsic == nir_intrinsic_load_global_invocation_id)
         return nir_ishl_imm(b, nir_channel(b, nir_load_workgroup_id(b), 1), 5);
      return nir_imm_int(b, 0);
   }
   assert(nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_iadd);
   nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
   nir_scalar c = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
   if (is_lane_y_linear(a) && !c.def->divergent)
      return nir_iadd(b, emit_lane_y_base_x8(b, a),
                      nir_ishl_imm(b, nir_channel(b, c.def, c.comp), 3));
   assert(is_lane_y_linear(c) && !a.def->divergent);
   return nir_iadd(b, emit_lane_y_base_x8(b, c),
                   nir_ishl_imm(b, nir_channel(b, a.def, a.comp), 3));
}

static nir_def *
emit_tile_base_y_scalar(nir_builder *b, nir_scalar s, unsigned depth)
{
   s = nir_scalar_chase_movs(s);
   if (is_ly_shl3(s))
      return nir_imm_int(b, 0);
   if (depth && nir_scalar_is_alu(s)) {
      nir_op op = nir_scalar_alu_op(s);
      if (op == nir_op_ishl) {
         nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
         nir_scalar sh = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
         if (nir_scalar_is_const(sh) && nir_scalar_as_uint(sh) == 3 &&
             is_lane_y_linear(a))
            return emit_lane_y_base_x8(b, a);
      } else if (op == nir_op_imadshl_agx) {
         nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
         nir_scalar c = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
         nir_scalar d = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 2));
         nir_scalar sh = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 3));
         if (nir_scalar_is_const(sh) && nir_scalar_as_uint(sh) == 3 &&
             is_lane_y_linear(d) && !a.def->divergent && !c.def->divergent) {
            nir_def *ybase = emit_lane_y_base_x8(b, d);
            if ((nir_scalar_is_const(a) && nir_scalar_as_uint(a) == 0) ||
                (nir_scalar_is_const(c) && nir_scalar_as_uint(c) == 0))
               return ybase;
            return nir_iadd(b, emit_mul_uniform(b, a, c), ybase);
         }
      } else if (op == nir_op_iadd) {
         nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
         nir_scalar c = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
         if (is_ly_shl3(a) && !c.def->divergent)
            return nir_channel(b, c.def, c.comp);
         if (is_ly_shl3(c) && !a.def->divergent)
            return nir_channel(b, a.def, a.comp);
         nir_def *dummy = NULL;
         if (match_add_local_y8(a, &dummy) && !c.def->divergent)
            return nir_iadd(b, emit_tile_base_y_scalar(b, a, depth - 1),
                            nir_channel(b, c.def, c.comp));
         if (match_add_local_y8(c, &dummy) && !a.def->divergent)
            return nir_iadd(b, emit_tile_base_y_scalar(b, c, depth - 1),
                            nir_channel(b, a.def, a.comp));
      }
   }
   UNREACHABLE("match_add_local_y8 guarantees uniform tile Y base");
}

static nir_def *
emit_tile_interior_cond(nir_builder *b, nir_def *base, nir_def *limit)
{
   nir_def *ge32 = nir_ige_imm(b, limit, AGX_GEMM_WIDTH);
   nir_def *max_base = nir_iadd_imm(b, limit, -(int)AGX_GEMM_WIDTH);
   return nir_iand(b, ge32, nir_ige(b, max_base, base));
}

static nir_def *
emit_b_load(nir_builder *b, const struct agx_gemm_match *m, unsigned comps,
            nir_def *base, nir_def *idx, uint32_t mem_base)
{
   return m->b_constant
      ? nir_load_constant_agx(b, comps, 32, base, idx,
                              .access = m->b_access, .base = mem_base,
                              .format = PIPE_FORMAT_R32_UINT,
                              .sign_extend = false)
      : nir_load_agx(b, comps, 32, base, idx,
                     .access = m->b_access, .base = mem_base,
                     .format = PIPE_FORMAT_R32_UINT,
                     .sign_extend = false);
}

static nir_def *
emit_bias_load(nir_builder *b, const struct agx_gemm_c_group *cg, unsigned comps,
               nir_def *base, nir_def *idx, uint32_t mem_base)
{
   return cg->bias_constant
      ? nir_load_constant_agx(b, comps, 32, base, idx,
                              .access = cg->bias_access, .base = mem_base,
                              .format = PIPE_FORMAT_R32_UINT,
                              .sign_extend = false)
      : nir_load_agx(b, comps, 32, base, idx,
                     .access = cg->bias_access, .base = mem_base,
                     .format = PIPE_FORMAT_R32_UINT,
                     .sign_extend = false);
}

static void
emit_c_direct_fast(nir_builder *b, const struct agx_gemm_match *m,
                   const struct agx_gemm_c_group *cg,
                   nir_def *c_base, nir_def *c_stride, nir_def *bias_base,
                   nir_def *c_row, nir_def *c_col)
{
   nir_def *c_idx = nir_imadshl_agx(b, c_row, c_stride, c_col, nir_imm_int(b, 0));
   nir_def *rb = cg->bias_kind == AGX_GEMM_BIAS_ROW
      ? emit_bias_load(b, cg, 1, bias_base, c_row, 0)
      : NULL;
   for (unsigned tile = 0; tile < AGX_GEMM_ROWS_PER_LANE / 2; ++tile) {
      const uint32_t mem_base =
         (uint32_t)(uint16_t)(32 * tile) << AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT;
      nir_def *v0 = &m->exit[2 * tile]->def;
      nir_def *v1 = &m->exit[2 * tile + 1]->def;
      if (cg->bias_kind == AGX_GEMM_BIAS_ROW) {
         v0 = nir_fadd(b, v0, rb);
         v1 = nir_fadd(b, v1, rb);
      } else if (cg->bias_kind == AGX_GEMM_BIAS_COL) {
         nir_def *cb = emit_bias_load(b, cg, 2, bias_base, c_col, mem_base);
         v0 = nir_fadd(b, v0, nir_channel(b, cb, 0));
         v1 = nir_fadd(b, v1, nir_channel(b, cb, 1));
      }
      if (cg->has_clamp) {
         const unsigned saved_ctrl = b->fp_math_ctrl;
         nir_def *lo = nir_imm_int(b, cg->clamp_lo_bits);
         nir_def *hi = nir_imm_int(b, cg->clamp_hi_bits);
         b->fp_math_ctrl = cg->clamp_max_fp_ctrl;
         v0 = cg->clamp_agx ? nir_fmax_agx(b, v0, lo) : nir_fmax(b, v0, lo);
         v1 = cg->clamp_agx ? nir_fmax_agx(b, v1, lo) : nir_fmax(b, v1, lo);
         b->fp_math_ctrl = cg->clamp_min_fp_ctrl;
         v0 = cg->clamp_agx ? nir_fmin_agx(b, v0, hi) : nir_fmin(b, v0, hi);
         v1 = cg->clamp_agx ? nir_fmin_agx(b, v1, hi) : nir_fmin(b, v1, hi);
         b->fp_math_ctrl = saved_ctrl;
      }
      nir_def *val = nir_vec2(b, v0, v1);
      nir_store_agx(b, val, c_base, c_idx,
                    .access = cg->store_access, .base = mem_base,
                    .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);
   }
}

bool
agx_nir_lower_gemm(nir_shader *shader, const struct agx_gemm_match *m)
{
   nir_function_impl *impl = m->ready->block->impl;
   nir_builder b = nir_builder_create(impl);

   /* The fragment position and the shared indices depend only on the lane
    * and the subgroup: compute them once, before the K loop. */
   nir_block *preheader = nir_loop_predecessor_block(m->loop);
   b.cursor = nir_after_block_before_jump(preheader);
   const struct fragment_position input = fragment_position(&b);
   /* Use the AGX3 byte immediate for the small contiguous K and N
    * offsets. The per-lane register index still counts scalar words.
    */
   nir_def *left_index = shared_index(&b, input.row, input.col, m->a_bytes);
   nir_def *right_indices[AGX_GEMM_WIDTH / 8] = {0};
   if (!m->b_direct) {
      for (unsigned k = 0; k < AGX_GEMM_WIDTH; k += 8) {
         nir_def *k_row = nir_iadd_imm(&b, input.subgroup_row, k);
         right_indices[k / 8] = shared_index(&b, k_row, input.col, m->b_bytes);
      }
   }

   nir_def *b_col_base = NULL, *b_col = NULL;
   nir_def *b_stride_val = NULL, *b_stride8 = NULL;
   nir_def *b_base_val = NULL;
   if (m->b_direct) {
      bool hoist_b =
         nir_block_dominates(nir_def_block(m->b_col_orig), preheader) &&
         can_materialize_uniform_def(m->b_stride.def, preheader) &&
         can_materialize_uniform_def(m->b_base, preheader);
      if (hoist_b) {
         b.cursor = nir_after_block_before_jump(preheader);
         b_col_base = emit_tile_base_x(&b, m->b_col_orig);
         b_col = nir_iadd(&b, b_col_base, input.col);
         b_stride_val = materialize_uniform_scalar(&b, m->b_stride, preheader);
         b_stride8 = nir_ishl_imm(&b, b_stride_val, 3);
         b_base_val = materialize_uniform_def(&b, m->b_base, preheader);
      }
   }

   b.cursor = nir_after_instr(m->ready);
   nir_def *acc[AGX_GEMM_ROWS_PER_LANE / 2];
   for (unsigned tile = 0; tile < ARRAY_SIZE(acc); ++tile) {
      acc[tile] = nir_vec2(&b, &m->initial[2 * tile]->def,
                               &m->initial[2 * tile + 1]->def);
   }

   if (m->b_direct) {
      if (!b_col) {
         b_col_base = emit_tile_base_x(&b, m->b_col_orig);
         b_col = nir_iadd(&b, b_col_base, input.col);
         b_stride_val = materialize_uniform_scalar(&b, m->b_stride, m->ready->block);
         b_stride8 = nir_ishl_imm(&b, b_stride_val, 3);
         b_base_val = materialize_uniform_def(&b, m->b_base, m->ready->block);
      }
      nir_def *b_row0 = nir_iadd(&b, m->b_k_base, input.subgroup_row);
      nir_def *b_idx =
         nir_imadshl_agx(&b, b_row0, b_stride_val, b_col, nir_imm_int(&b, 0));

      for (unsigned k = 0; k < AGX_GEMM_WIDTH; k += 8) {
         nir_def *left = nir_load_shared(&b, 2, 32, left_index,
                                         .base = 4 * k, .align_mul = 8);
         if (k > 0)
            b_idx = nir_iadd(&b, b_idx, b_stride8);
         for (unsigned tile = 0; tile < ARRAY_SIZE(acc); ++tile) {
            const uint32_t mem_base =
               (uint32_t)(uint16_t)(32 * tile) << AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT;
            nir_def *right = emit_b_load(&b, m, 2, b_base_val, b_idx, mem_base);
            acc[tile] = nir_cmat_muladd_agx(&b, left, right, acc[tile],
               .src_type = nir_type_float32, .src_type2 = nir_type_float32);
         }
      }

      for (unsigned r = 0; r < AGX_GEMM_ROWS_PER_LANE; ++r)
         nir_instr_remove(&m->b_stores[r]->instr);
      if (m->b_bytes + TILE_BYTES == shader->info.shared_size &&
          m->a_bytes + TILE_BYTES <= m->b_bytes)
         shader->info.shared_size = m->b_bytes;
   } else {
      for (unsigned k = 0; k < AGX_GEMM_WIDTH; k += 8) {
         nir_def *left = nir_load_shared(&b, 2, 32, left_index,
                                         .base = 4 * k, .align_mul = 8);
         nir_def *right_index = right_indices[k / 8];
         for (unsigned tile = 0; tile < ARRAY_SIZE(acc); ++tile) {
            nir_def *right = nir_load_shared(&b, 2, 32, right_index,
                                              .base = 32 * tile, .align_mul = 8);
            acc[tile] = nir_cmat_muladd_agx(&b, left, right, acc[tile],
               .src_type = nir_type_float32, .src_type2 = nir_type_float32);
         }
      }
   }

   for (unsigned row = 0; row < AGX_GEMM_ROWS_PER_LANE; ++row) {
      nir_def *old = &m->fma[AGX_GEMM_WIDTH - 1][row]->def;
      nir_def *part = nir_channel(&b, acc[row / 2], row % 2);
      replace_phi_source(m->initial[row], old, part);
      replace_phi_source(m->exit[row], old, part);
   }

   b.cursor = nir_after_phis(m->exit[0]->instr.block);
   const struct fragment_position output = fragment_position(&b);

   if (m->c_direct) {
      for (unsigned g = 0; g < m->c_group_count; ++g) {
         const struct agx_gemm_c_group *cg = &m->c_groups[g];
         if (cg->guarded)
            b.cursor = nir_after_block_before_jump(cg->insert_block);
         else
            b.cursor = nir_before_instr(cg->insert_before);

         nir_def *c_base =
            materialize_uniform_def(&b, cg->c_base, cg->insert_block);
         nir_def *c_stride =
            materialize_uniform_scalar(&b, cg->c_stride, cg->insert_block);
         nir_def *bias_base = cg->bias_kind != AGX_GEMM_BIAS_NONE
            ? materialize_uniform_def(&b, cg->bias_base, cg->insert_block)
            : NULL;
         nir_def *tile_row_base =
            emit_tile_base_y_scalar(&b, nir_scalar_resolved(cg->row_base, 0), 4);
         nir_def *tile_col_base = emit_tile_base_x(&b, cg->col_orig);
         nir_def *c_row = nir_iadd(&b, tile_row_base, output.row);
         nir_def *c_col = nir_iadd(&b, tile_col_base, output.col);

         if (!cg->guarded) {
            emit_c_direct_fast(&b, m, cg, c_base, c_stride, bias_base,
                               c_row, c_col);
         } else {
            nir_def *m_lim =
               materialize_uniform_scalar(&b, cg->m_limit, cg->insert_block);
            nir_def *n_lim =
               materialize_uniform_scalar(&b, cg->n_limit, cg->insert_block);
            nir_def *c_interior = nir_iand(&b,
               emit_tile_interior_cond(&b, tile_row_base, m_lim),
               emit_tile_interior_cond(&b, tile_col_base, n_lim));

            nir_if *nif = nir_push_if(&b, c_interior);
            emit_c_direct_fast(&b, m, cg, c_base, c_stride, bias_base,
                               c_row, c_col);
            nir_push_else(&b, nif);
            {
               nir_if *row_if = nir_push_if(&b, nir_ilt(&b, c_row, m_lim));
               nir_def *rb = cg->bias_kind == AGX_GEMM_BIAS_ROW
                  ? emit_bias_load(&b, cg, 1, bias_base, c_row, 0)
                  : NULL;
               nir_def *c_idx0 =
                  nir_imadshl_agx(&b, c_row, c_stride, c_col, nir_imm_int(&b, 0));
               for (unsigned tile = 0; tile < AGX_GEMM_ROWS_PER_LANE / 2; ++tile) {
                  for (unsigned lc = 0; lc < 2; ++lc) {
                     unsigned off = 8 * tile + lc;
                     nir_def *col_k = nir_iadd_imm(&b, c_col, off);
                     nir_if *col_if = nir_push_if(&b, nir_ilt(&b, col_k, n_lim));
                     nir_def *v = &m->exit[2 * tile + lc]->def;
                     if (cg->bias_kind == AGX_GEMM_BIAS_ROW) {
                        v = nir_fadd(&b, v, rb);
                     } else if (cg->bias_kind == AGX_GEMM_BIAS_COL) {
                        nir_def *cb = emit_bias_load(&b, cg, 1, bias_base, col_k, 0);
                        v = nir_fadd(&b, v, cb);
                     }
                     if (cg->has_clamp) {
                        const unsigned saved_ctrl = b.fp_math_ctrl;
                        nir_def *lo = nir_imm_int(&b, cg->clamp_lo_bits);
                        nir_def *hi = nir_imm_int(&b, cg->clamp_hi_bits);
                        b.fp_math_ctrl = cg->clamp_max_fp_ctrl;
                        v = cg->clamp_agx ? nir_fmax_agx(&b, v, lo) : nir_fmax(&b, v, lo);
                        b.fp_math_ctrl = cg->clamp_min_fp_ctrl;
                        v = cg->clamp_agx ? nir_fmin_agx(&b, v, hi) : nir_fmin(&b, v, hi);
                        b.fp_math_ctrl = saved_ctrl;
                     }
                     nir_store_agx(&b, v, c_base, nir_iadd_imm(&b, c_idx0, off),
                                   .access = cg->store_access, .base = 0,
                                   .format = PIPE_FORMAT_R32_UINT,
                                   .sign_extend = false);
                     nir_pop_if(&b, col_if);
                  }
               }
               nir_pop_if(&b, row_if);
            }
            nir_pop_if(&b, nif);
         }

         for (unsigned r = 0; r < AGX_GEMM_ROWS_PER_LANE; ++r)
            nir_instr_remove(&cg->stores[r]->instr);
      }
   } else {
      /* The input barrier at the end of the last K iteration completed all
       * uses of A. Store the subgroup-owned C fragments into its allocation,
       * then resume the unchanged per-invocation tail and bias epilogue.
       */
      nir_def *staged = shared_index(&b, output.row, output.col, m->a_bytes);
      for (unsigned tile = 0; tile < ARRAY_SIZE(acc); ++tile) {
         nir_def *part = nir_vec2(&b, &m->exit[2 * tile]->def,
                                     &m->exit[2 * tile + 1]->def);
         nir_store_shared(&b, part, staged, .base = 32 * tile,
                           .write_mask = 3, .align_mul = 8);
      }
      nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                  .memory_scope = SCOPE_WORKGROUP, .memory_modes = nir_var_mem_shared,
                  .memory_semantics = NIR_MEMORY_ACQ_REL);
      nir_def *local = nir_load_local_invocation_id(&b);
      nir_def *column = nir_channel(&b, local, 0);
      nir_def *row_base = nir_ishl_imm(&b, nir_channel(&b, local, 1), 3);
      for (unsigned row = 0; row < AGX_GEMM_ROWS_PER_LANE; ++row) {
         nir_def *offset = shared_index(&b, nir_iadd_imm(&b, row_base, row),
                                        column, m->a_bytes);
         nir_def *result = nir_load_shared(&b, 1, 32, offset, .align_mul = 4);
         nir_def_rewrite_uses_after(&m->exit[row]->def, result);
      }
   }

   nir_progress(true, impl,
                m->c_direct ? nir_metadata_none
                                              : nir_metadata_control_flow);
   nir_opt_dce(shader);
   if (m->c_direct) {
      nir_opt_dead_cf(shader);
      nir_opt_dce(shader);
   }
   nir_divergence_analysis(shader);
   return true;
}
