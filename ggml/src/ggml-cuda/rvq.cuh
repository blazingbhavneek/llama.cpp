#pragma once

#include "common.cuh"

void ggml_cuda_mul_mat_id_rvq(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
