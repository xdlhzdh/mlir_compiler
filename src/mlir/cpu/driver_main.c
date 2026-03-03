#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} memref2d_f32;

extern memref2d_f32 forward(
    float *a_alloc, float *a_aligned, int64_t a_offset, int64_t a_size0,
    int64_t a_size1, int64_t a_stride0, int64_t a_stride1, float *b_alloc,
    float *b_aligned, int64_t b_offset, int64_t b_size0, int64_t b_size1,
    int64_t b_stride0, int64_t b_stride1);

static int nearly_equal(float a, float b) { return fabsf(a - b) < 1e-4f; }

int main(void) {
  // A = I, so A*B should equal B.
  float a[16] = {
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
  };
  float b[16] = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  };

  memref2d_f32 out = forward(a, a, 0, 4, 4, 4, 1, b, b, 0, 4, 4, 4, 1);

  int ok = 1;
  for (int i = 0; i < 16; ++i) {
    if (!nearly_equal(out.aligned[i], b[i])) {
      ok = 0;
      break;
    }
  }

  printf("first row: %.1f %.1f %.1f %.1f\n", out.aligned[0], out.aligned[1],
         out.aligned[2], out.aligned[3]);
  printf("riscv_out:");
  for (int i = 0; i < 16; ++i) {
    printf(" %.6f", out.aligned[i]);
  }
  printf("\n");
  printf("matmul check: %s\n", ok ? "PASS" : "FAIL");

  free(out.allocated);
  return ok ? 0 : 1;
}
