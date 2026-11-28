/* SPDX-License-Identifier: GPL-2.0-only */

#include <gtest/gtest.h>

#include "util/u_idalloc.h"

class idalloc_sparse : public testing::TestWithParam<bool> {
protected:
   struct util_idalloc_sparse allocator;

   void SetUp() override { util_idalloc_sparse_init(&allocator); }
   void TearDown() override { util_idalloc_sparse_fini(&allocator); }

   unsigned alloc()
   {
      return GetParam() ? util_idalloc_sparse_alloc_range(&allocator, 1)
                        : util_idalloc_sparse_alloc(&allocator);
   }

   void check_rollover(bool reserve_tail)
   {
      const unsigned limit = UTIL_IDALLOC_MAX_IDS_PER_SEGMENT(&allocator);
      const unsigned max_elements =
         UTIL_IDALLOC_MAX_ELEMS_PER_SEGMENT(&allocator);

      for (unsigned i = 0; i < limit - reserve_tail; i++)
         ASSERT_EQ(alloc(), i);
      if (reserve_tail)
         util_idalloc_sparse_reserve(&allocator, limit - 1);

      ASSERT_EQ(allocator.segment[0].num_elements, max_elements);

      /* A stale search hint must not grow a full segment's bitmap. */
      const unsigned first = alloc();
      const unsigned second = alloc();
      EXPECT_EQ(first, limit);
      EXPECT_EQ(second, limit + 1);
      EXPECT_EQ(allocator.segment[0].num_elements, max_elements);
      EXPECT_EQ(allocator.segment[0].num_set_elements, max_elements);

      util_idalloc_sparse_free(&allocator, first);
      util_idalloc_sparse_free(&allocator, second);
      EXPECT_EQ(allocator.segment[1].num_set_elements, 0u);

      /* Freeing the old segment's tail makes it eligible for reuse. */
      util_idalloc_sparse_free(&allocator, limit - 1);
      EXPECT_EQ(alloc(), limit - 1);
      EXPECT_EQ(alloc(), limit);
      EXPECT_EQ(allocator.segment[0].num_elements, max_elements);
   }
};

TEST_P(idalloc_sparse, allocated_tail)
{
   check_rollover(false);
}

TEST_P(idalloc_sparse, reserved_tail)
{
   check_rollover(true);
}

INSTANTIATE_TEST_SUITE_P(single_id, idalloc_sparse, testing::Bool());
