#include "rvq.cuh"
#include "ggml-impl.h"

#include <climits>

static constexpr int rvq_dim = 8;
static constexpr int rvq_entries = 512;
static constexpr int rvq_tile_codes = 8;
static constexpr int rvq_rows_per_thread = 2;

static __device__ bool rvq_group_contains(int32_t id, int32_t offset, int64_t experts) {
    return id >= offset && int64_t(id) - offset < experts;
}

static __global__ void rvq_project_book(
        const float * x, const float * book, float * lookup, const int32_t * ids,
        int64_t codes_per_row, int64_t x_slots, int64_t x_slot_stride, int64_t x_token_stride,
        int64_t slots, int64_t id_token_stride, int64_t experts, int32_t expert_offset) {
    const int64_t group = blockIdx.x;
    const int64_t vector = group / codes_per_row;
    const int64_t token = vector / x_slots;
    const int64_t x_slot = vector % x_slots;
    const int32_t * token_ids = ids + token * id_token_stride;
    // Ignore lookups for activations that no selected expert in this group uses.
    bool used = false;
    for (int64_t slot = x_slot; slot < slots; slot += x_slots) {
        used |= rvq_group_contains(token_ids[slot], expert_offset, experts);
    }
    if (!used) {
        return;
    }

    const int64_t col = group % codes_per_row;
    const float * a = x + token * x_token_stride + x_slot * x_slot_stride + col * rvq_dim;
    for (int entry = threadIdx.x; entry < rvq_entries; entry += blockDim.x) {
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < rvq_dim; ++k) {
            sum += a[k] * book[entry * rvq_dim + k];
        }
        lookup[group * rvq_entries + entry] = sum;
    }
}

static __global__ void rvq_mul_mat_id(
        const uint16_t * codes, const float * lookup, const int32_t * ids, const half * gains, float * dst,
        int64_t codes_per_row, int64_t rows, int64_t experts, int64_t slots, int64_t x_slots,
        int64_t id_token_stride, int32_t expert_offset) {
    const int warp_size = ggml_cuda_get_physical_warp_size();
    const int lane = threadIdx.x % warp_size;
    const int64_t row = int64_t(blockIdx.x) * (blockDim.x / warp_size) + threadIdx.x / warp_size;
    if (row >= rows) {
        return;
    }

    const int64_t token = blockIdx.y / slots;
    const int64_t slot = blockIdx.y % slots;
    const int64_t expert = int64_t(ids[token * id_token_stride + slot]) - expert_offset;
    float sum = 0.0f;
    if (expert >= 0 && expert < experts) {
        const int64_t expert_row = expert * rows + row;
        const uint16_t * w = codes + expert_row * codes_per_row;
        const float * lut = lookup + (token * x_slots + slot % x_slots) * codes_per_row * rvq_entries;
        for (int64_t col = lane; col < codes_per_row; col += warp_size) {
            const uint16_t code = w[col];
            const float * table = lut + col * rvq_entries;
            sum += table[code >> 8] + table[256 + (code & 255)];
        }
        sum = warp_reduce_sum(sum) * __half2float(gains[expert_row]);
    }
    if (lane == 0) {
        dst[blockIdx.y * rows + row] = sum;
    }
}

static __global__ void rvq_mul_mat_id_tiled(
        const uint16_t * codes, const float * lookup, const int32_t * ids, float * partials,
        int codes_per_row, int rows, int experts, int slots, int x_slots,
        int64_t id_token_stride, int32_t expert_offset) {
    const int token = blockIdx.y / slots;
    const int slot = blockIdx.y % slots;
    const int32_t id = ids[token * id_token_stride + slot];
    if (!rvq_group_contains(id, expert_offset, experts)) {
        return;
    }

    const int start = blockIdx.z * rvq_tile_codes;
    const float * lut = lookup + (int64_t(token * x_slots + slot % x_slots) * codes_per_row + start) * rvq_entries;

    // Stage this slice once per block. Threads reuse it across independent weight rows.
    __shared__ __align__(16) float tables[rvq_tile_codes * rvq_entries];
    for (int i = threadIdx.x; i < rvq_tile_codes * rvq_entries / 4; i += blockDim.x) {
        reinterpret_cast<float4 *>(tables)[i] = reinterpret_cast<const float4 *>(lut)[i];
    }
    __syncthreads();

    const int expert = id - expert_offset;
    for (int r = 0; r < rvq_rows_per_thread; ++r) {
        const int row = blockIdx.x * blockDim.x * rvq_rows_per_thread + r * blockDim.x + threadIdx.x;
        if (row >= rows) {
            break;
        }

        const uint16_t * w = codes + (int64_t(expert) * rows + row) * codes_per_row + start;
        const uint4 packed = *reinterpret_cast<const uint4 *>(w);
        const uint32_t words[rvq_tile_codes / 2] = { packed.x, packed.y, packed.z, packed.w };
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < rvq_tile_codes / 2; ++j) {
            const float * table = tables + 2*j * rvq_entries;
            sum += table[(words[j] >> 8) & 255] + table[256 + (words[j] & 255)];
            table += rvq_entries;
            sum += table[words[j] >> 24] + table[256 + ((words[j] >> 16) & 255)];
        }
        partials[(int64_t(blockIdx.y) * gridDim.z + blockIdx.z) * rows + row] = sum;
    }
}

static __global__ void rvq_mul_mat_id_merge(
        const float * partials, const int32_t * ids, const half * gains, float * dst,
        int rows, int experts, int slots, int tiles, int64_t id_token_stride, int32_t expert_offset) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const int token = blockIdx.y / slots;
    const int slot = blockIdx.y % slots;
    const int32_t id = ids[token * id_token_stride + slot];
    float sum = 0.0f;
    if (rvq_group_contains(id, expert_offset, experts)) {
        const float * part = partials + int64_t(blockIdx.y) * tiles * rows + row;
        for (int tile = 0; tile < tiles; ++tile) {
            sum += part[int64_t(tile) * rows];
        }
        sum *= __half2float(gains[int64_t(id - expert_offset) * rows + row]);
    }
    dst[int64_t(blockIdx.y) * rows + row] = sum;
}

void ggml_cuda_mul_mat_id_rvq(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * codes = dst->src[0];
    const ggml_tensor * x     = dst->src[1];
    const ggml_tensor * ids   = dst->src[2];
    const ggml_tensor * gains = dst->src[3];
    const ggml_tensor * book  = dst->src[4];
    const ggml_tensor * shared_lookup = dst->src[5];

    const int64_t codes_per_row = codes->ne[0] / rvq_dim;
    const int64_t rows = dst->ne[0];
    const int64_t slots = dst->ne[1];
    const int64_t x_slots = x->ne[1];
    const int64_t floats_per_token = x_slots * codes_per_row * rvq_entries;
    const int64_t tiles = codes_per_row / rvq_tile_codes;
    const bool tiled = codes_per_row % rvq_tile_codes == 0 && tiles <= 65535 &&
                       rows <= INT_MAX && codes->ne[2] <= INT_MAX;
    const int64_t partials_per_token = tiled ? slots * rows * tiles : 0;
    const int64_t scratch_per_token = partials_per_token + (shared_lookup ? 0 : floats_per_token);
    const int64_t scratch_budget = 48 * 1024 * 1024 / sizeof(float);
    const int64_t chunk = std::min({dst->ne[2], int64_t(65535) / slots,
                                  std::max(int64_t(1), scratch_budget / std::max(int64_t(1), scratch_per_token))});
    const int32_t expert_offset = ggml_get_op_params_i32(dst, 4);

    ggml_cuda_pool_alloc<float> lookup(ctx.pool());
    if (!shared_lookup) {
        lookup.alloc(chunk * floats_per_token);
    }
    ggml_cuda_pool_alloc<float> partials(ctx.pool());
    if (tiled) {
        partials.alloc(chunk * partials_per_token);
    }
    constexpr int threads = 128;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    cudaStream_t stream = ctx.stream();
    for (int64_t token = 0; token < dst->ne[2]; token += chunk) {
        const int64_t count = std::min(chunk, dst->ne[2] - token);
        const int64_t vectors = count * x_slots;
        const float * activation = (const float *) ((const char *) x->data + token * x->nb[2]);
        const int32_t * token_ids = (const int32_t *) ((const char *) ids->data + token * ids->nb[1]);
        const int64_t id_stride = ids->nb[1] / sizeof(int32_t);
        const float * tables = shared_lookup ? (const float *) shared_lookup->data + token * floats_per_token : lookup.get();
        if (!shared_lookup) {
            rvq_project_book<<<vectors * codes_per_row, threads, 0, stream>>>(
                    activation, (const float *) book->data, lookup.get(), token_ids,
                    codes_per_row, x_slots, x->nb[1] / sizeof(float), x->nb[2] / sizeof(float),
                    slots, id_stride, codes->ne[2], expert_offset);
        }

        float * output = (float *) dst->data + token * slots * rows;
        if (tiled) {
            constexpr int block_rows = threads * rvq_rows_per_thread;
            const dim3 grid((rows + block_rows - 1) / block_rows, count * slots, tiles);
            rvq_mul_mat_id_tiled<<<grid, threads, 0, stream>>>(
                    (const uint16_t *) codes->data, tables, token_ids, partials.get(), codes_per_row, rows,
                    codes->ne[2], slots, x_slots, id_stride, expert_offset);
            rvq_mul_mat_id_merge<<<dim3((rows + threads - 1) / threads, count * slots), threads, 0, stream>>>(
                    partials.get(), token_ids, (const half *) gains->data, output,
                    rows, codes->ne[2], slots, tiles, id_stride, expert_offset);
        } else {
            const int rows_per_block = threads / warp_size;
            const dim3 grid((rows + rows_per_block - 1) / rows_per_block, count * slots);
            rvq_mul_mat_id<<<grid, threads, 0, stream>>>(
                    (const uint16_t *) codes->data, tables, token_ids, (const half *) gains->data,
                    output, codes_per_row, rows, codes->ne[2], slots, x_slots, id_stride, expert_offset);
        }
    }
}
