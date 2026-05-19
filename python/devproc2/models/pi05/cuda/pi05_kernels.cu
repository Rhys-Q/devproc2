#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

static __device__ __forceinline__ float f32_from_bits_i64(long long bits) {
    unsigned int lo = static_cast<unsigned int>(bits & 0xffffffffull);
    return __uint_as_float(lo);
}

static __device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

static __device__ __forceinline__ float block_sum(float v) {
    __shared__ float shared[8];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();
    int num_warps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (wid == 0) v = warp_sum(v);
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = v;
    __syncthreads();
    return shared[0];
}

static __device__ __forceinline__ float warp_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
    }
    return v;
}

static __device__ __forceinline__ float block_max(float v) {
    __shared__ float shared[8];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();
    int num_warps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (wid == 0) v = warp_max(v);
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = v;
    __syncthreads();
    return shared[0];
}

static __device__ __forceinline__ float gelu_tanh_approx(float x) {
    return x / (1.0f + __expf(-1.5957691216057308f * x * (1.0f + 0.044715f * x * x)));
}

static __device__ __forceinline__ float gelu_erf_approx(float x) {
    float t = __tanhf(0.7978845608f * (x + 0.044715f * x * x * x));
    return 0.5f * x * (1.0f + t);
}

static __device__ __forceinline__ __nv_fp8_e4m3 to_fp8_e4m3(float x, float scale) {
    float inv = 1.0f / fmaxf(scale, 1.0e-12f);
    float y = fminf(fmaxf(x * inv, -448.0f), 448.0f);
    return __nv_fp8_e4m3(y);
}

extern "C" __global__ void pi05_image_u8_to_bf16_norm(
    const uint8_t* __restrict__ image_u8,
    long long n,
    __nv_bfloat16* __restrict__ out_bf16) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = static_cast<float>(image_u8[i]) * (1.0f / 127.5f) - 1.0f;
    out_bf16[i] = __float2bfloat16(x);
}

extern "C" __global__ void pi05_cast_f32_to_bf16(
    const float* __restrict__ x,
    long long n,
    __nv_bfloat16* __restrict__ out_bf16) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out_bf16[i] = __float2bfloat16(x[i]);
}

extern "C" __global__ void pi05_patch_im2col_bf16(
    const __nv_bfloat16* __restrict__ image,
    long long num_views,
    __nv_bfloat16* __restrict__ patches) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = num_views * 256ll * 588ll;
    if (idx >= total) return;

    long long patch_idx = idx / 588ll;
    long long feat_idx = idx - patch_idx * 588ll;
    long long view = patch_idx / 256ll;
    long long local_patch = patch_idx - view * 256ll;
    long long ph = local_patch / 16ll;
    long long pw = local_patch - ph * 16ll;
    long long pxh = feat_idx / 42ll;
    long long rem = feat_idx - pxh * 42ll;
    long long pxw = rem / 3ll;
    long long c = rem - pxw * 3ll;
    long long row = ph * 14ll + pxh;
    long long col = pw * 14ll + pxw;
    long long src = view * (224ll * 224ll * 3ll) + row * (224ll * 3ll) + col * 3ll + c;
    patches[idx] = image[src];
}

extern "C" __global__ void pi05_embedding_gather_bf16(
    const int32_t* __restrict__ token_ids,
    const __nv_bfloat16* __restrict__ embedding,
    long long num_tokens,
    long long hidden,
    __nv_bfloat16* __restrict__ out) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = num_tokens * hidden;
    if (i >= total) return;
    long long t = i / hidden;
    long long h = i - t * hidden;
    int32_t tok = token_ids[t];
    float scale = sqrtf(static_cast<float>(hidden));
    out[i] = __float2bfloat16(__bfloat162float(embedding[static_cast<long long>(tok) * hidden + h]) * scale);
}

extern "C" __global__ void pi05_rope_qwen3_bf16(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    long long S,
    long long NH,
    long long HD) {
    long long half_hd = HD >> 1;
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total_pairs = S * NH * half_hd;
    if (idx >= total_pairs) return;

    long long d = idx % half_hd;
    long long rem = idx / half_hd;
    long long h = rem % NH;
    long long s = rem / NH;
    long long base = s * NH * HD + h * HD;
    long long pair = 2 * d;
    long long rope_idx = s * HD + d;

    float c = __bfloat162float(cos_table[rope_idx]);
    float si = __bfloat162float(sin_table[rope_idx]);
    float lo = __bfloat162float(x[base + pair]);
    float hi = __bfloat162float(x[base + pair + 1]);
    x[base + pair] = __float2bfloat16(lo * c - hi * si);
    x[base + pair + 1] = __float2bfloat16(hi * c + lo * si);
}

extern "C" __global__ void pi05_qkv_split_bf16(
    const __nv_bfloat16* __restrict__ qkv,
    long long rows,
    long long q_dim,
    long long k_dim,
    long long v_dim,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long q_total = rows * q_dim;
    long long k_total = rows * k_dim;
    long long v_total = rows * v_dim;
    long long row_stride = q_dim + k_dim + v_dim;
    if (idx < q_total) {
        long long r = idx / q_dim;
        long long c = idx - r * q_dim;
        q[idx] = qkv[r * row_stride + c];
    }
    if (idx < k_total) {
        long long r = idx / k_dim;
        long long c = idx - r * k_dim;
        k[idx] = qkv[r * row_stride + q_dim + c];
    }
    if (idx < v_total) {
        long long r = idx / v_dim;
        long long c = idx - r * v_dim;
        v[idx] = qkv[r * row_stride + q_dim + k_dim + c];
    }
}

extern "C" __global__ void pi05_qkv_bias_split_bf16(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ bias,
    long long rows,
    long long q_dim,
    long long k_dim,
    long long v_dim,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long row_stride = q_dim + k_dim + v_dim;
    long long total = rows * row_stride;
    if (idx >= total) return;
    long long r = idx / row_stride;
    long long c = idx - r * row_stride;
    __nv_bfloat16 val = __float2bfloat16(
        __bfloat162float(qkv[idx]) + __bfloat162float(bias[c]));
    if (c < q_dim) {
        q[r * q_dim + c] = val;
    } else if (c < q_dim + k_dim) {
        c -= q_dim;
        k[r * k_dim + c] = val;
    } else {
        c -= q_dim + k_dim;
        v[r * v_dim + c] = val;
    }
}

extern "C" __global__ void pi05_qkv_split_rope_bf16(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ rope_interleaved,
    long long rows,
    long long q_dim,
    long long k_dim,
    long long v_dim,
    long long head_dim,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long half = head_dim >> 1;
    long long q_heads = q_dim / head_dim;
    long long k_heads = k_dim / head_dim;
    long long q_pairs = rows * q_heads * half;
    long long k_pairs = rows * k_heads * half;
    long long v_total = rows * v_dim;
    long long row_stride = q_dim + k_dim + v_dim;

    if (idx < q_pairs) {
        long long d = idx % half;
        long long t = idx / half;
        long long h = t % q_heads;
        long long r = t / q_heads;
        long long pair = 2 * d;
        long long src_base = r * row_stride + h * head_dim + pair;
        long long dst_base = r * q_dim + h * head_dim + pair;
        float c = __bfloat162float(rope_interleaved[r * head_dim + pair]);
        float s = __bfloat162float(rope_interleaved[r * head_dim + pair + 1]);
        float lo = __bfloat162float(qkv[src_base]);
        float hi = __bfloat162float(qkv[src_base + 1]);
        q[dst_base] = __float2bfloat16(lo * c - hi * s);
        q[dst_base + 1] = __float2bfloat16(hi * c + lo * s);
    }
    if (idx < k_pairs) {
        long long d = idx % half;
        long long t = idx / half;
        long long h = t % k_heads;
        long long r = t / k_heads;
        long long pair = 2 * d;
        long long src_base = r * row_stride + q_dim + h * head_dim + pair;
        long long dst_base = r * k_dim + h * head_dim + pair;
        float c = __bfloat162float(rope_interleaved[r * head_dim + pair]);
        float s = __bfloat162float(rope_interleaved[r * head_dim + pair + 1]);
        float lo = __bfloat162float(qkv[src_base]);
        float hi = __bfloat162float(qkv[src_base + 1]);
        k[dst_base] = __float2bfloat16(lo * c - hi * s);
        k[dst_base + 1] = __float2bfloat16(hi * c + lo * s);
    }
    if (idx < v_total) {
        long long r = idx / v_dim;
        long long c = idx - r * v_dim;
        v[idx] = qkv[r * row_stride + q_dim + k_dim + c];
    }
}

extern "C" __global__ void pi05_qkv_split_rope_cache_bf16(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ rope_interleaved,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    long long layer,
    long long rows,
    long long cache_rows,
    long long q_dim,
    long long k_dim,
    long long v_dim,
    long long head_dim,
    __nv_bfloat16* __restrict__ q) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long half = head_dim >> 1;
    long long q_heads = q_dim / head_dim;
    long long k_heads = k_dim / head_dim;
    long long q_pairs = rows * q_heads * half;
    long long k_pairs = rows * k_heads * half;
    long long v_total = rows * v_dim;
    long long row_stride = q_dim + k_dim + v_dim;
    long long cache_layer_base = layer * cache_rows * k_dim;

    if (idx < q_pairs) {
        long long d = idx % half;
        long long t = idx / half;
        long long h = t % q_heads;
        long long r = t / q_heads;
        long long pair = 2 * d;
        long long src_base = r * row_stride + h * head_dim + pair;
        long long dst_base = r * q_dim + h * head_dim + pair;
        float c = __bfloat162float(rope_interleaved[r * head_dim + pair]);
        float s = __bfloat162float(rope_interleaved[r * head_dim + pair + 1]);
        float lo = __bfloat162float(qkv[src_base]);
        float hi = __bfloat162float(qkv[src_base + 1]);
        q[dst_base] = __float2bfloat16(lo * c - hi * s);
        q[dst_base + 1] = __float2bfloat16(hi * c + lo * s);
    }
    if (idx < k_pairs) {
        long long d = idx % half;
        long long t = idx / half;
        long long h = t % k_heads;
        long long r = t / k_heads;
        long long pair = 2 * d;
        long long src_base = r * row_stride + q_dim + h * head_dim + pair;
        long long dst_base = cache_layer_base + r * k_dim + h * head_dim + pair;
        float c = __bfloat162float(rope_interleaved[r * head_dim + pair]);
        float s = __bfloat162float(rope_interleaved[r * head_dim + pair + 1]);
        float lo = __bfloat162float(qkv[src_base]);
        float hi = __bfloat162float(qkv[src_base + 1]);
        k_cache[dst_base] = __float2bfloat16(lo * c - hi * s);
        k_cache[dst_base + 1] = __float2bfloat16(hi * c + lo * s);
    }
    if (idx < v_total) {
        long long r = idx / v_dim;
        long long c = idx - r * v_dim;
        long long dst = cache_layer_base + r * v_dim + c;
        v_cache[dst] = qkv[r * row_stride + q_dim + k_dim + c];
    }
}

extern "C" __global__ void pi05_qkv_split_rope_concat_bf16(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ rope_interleaved,
    const __nv_bfloat16* __restrict__ prefix_k,
    const __nv_bfloat16* __restrict__ prefix_v,
    long long prefix_rows,
    long long suffix_rows,
    long long q_dim,
    long long k_dim,
    long long v_dim,
    long long head_dim,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ out_k,
    __nv_bfloat16* __restrict__ out_v) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long half = head_dim >> 1;
    long long q_heads = q_dim / head_dim;
    long long k_heads = k_dim / head_dim;
    long long q_pairs = suffix_rows * q_heads * half;
    long long k_pairs = suffix_rows * k_heads * half;
    long long suffix_v_total = suffix_rows * v_dim;
    long long prefix_total = prefix_rows * k_dim;
    long long row_stride = q_dim + k_dim + v_dim;

    if (idx < q_pairs) {
        long long d = idx % half;
        long long t = idx / half;
        long long h = t % q_heads;
        long long r = t / q_heads;
        long long pair = 2 * d;
        long long src_base = r * row_stride + h * head_dim + pair;
        long long dst_base = r * q_dim + h * head_dim + pair;
        float c = __bfloat162float(rope_interleaved[r * head_dim + pair]);
        float s = __bfloat162float(rope_interleaved[r * head_dim + pair + 1]);
        float lo = __bfloat162float(qkv[src_base]);
        float hi = __bfloat162float(qkv[src_base + 1]);
        q[dst_base] = __float2bfloat16(lo * c - hi * s);
        q[dst_base + 1] = __float2bfloat16(hi * c + lo * s);
    }
    if (idx < prefix_total) {
        out_k[idx] = prefix_k[idx];
        out_v[idx] = prefix_v[idx];
    }
    if (idx < k_pairs) {
        long long d = idx % half;
        long long t = idx / half;
        long long h = t % k_heads;
        long long r = t / k_heads;
        long long pair = 2 * d;
        long long src_base = r * row_stride + q_dim + h * head_dim + pair;
        long long dst_base = prefix_total + r * k_dim + h * head_dim + pair;
        float c = __bfloat162float(rope_interleaved[r * head_dim + pair]);
        float s = __bfloat162float(rope_interleaved[r * head_dim + pair + 1]);
        float lo = __bfloat162float(qkv[src_base]);
        float hi = __bfloat162float(qkv[src_base + 1]);
        out_k[dst_base] = __float2bfloat16(lo * c - hi * s);
        out_k[dst_base + 1] = __float2bfloat16(hi * c + lo * s);
    }
    if (idx < suffix_v_total) {
        long long r = idx / v_dim;
        long long c = idx - r * v_dim;
        out_v[prefix_total + idx] = qkv[r * row_stride + q_dim + k_dim + c];
    }
}

extern "C" __global__ void pi05_kv_concat_bf16(
    const __nv_bfloat16* __restrict__ prefix_k,
    const __nv_bfloat16* __restrict__ prefix_v,
    const __nv_bfloat16* __restrict__ suffix_k,
    const __nv_bfloat16* __restrict__ suffix_v,
    long long prefix_rows,
    long long suffix_rows,
    long long num_kv_heads,
    long long head_dim,
    __nv_bfloat16* __restrict__ out_k,
    __nv_bfloat16* __restrict__ out_v) {
    long long elems_per_row = num_kv_heads * head_dim;
    long long prefix_total = prefix_rows * elems_per_row;
    long long suffix_total = suffix_rows * elems_per_row;
    long long total = prefix_total + suffix_total;
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    if (idx < prefix_total) {
        out_k[idx] = prefix_k[idx];
        out_v[idx] = prefix_v[idx];
    } else {
        long long suffix_idx = idx - prefix_total;
        out_k[idx] = suffix_k[suffix_idx];
        out_v[idx] = suffix_v[suffix_idx];
    }
}

extern "C" __global__ void pi05_copy_kv_cache_layer_bf16(
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    long long layer,
    long long rows,
    long long kv_dim) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = rows * kv_dim;
    if (idx >= total) return;
    long long dst = layer * total + idx;
    k_cache[dst] = k[idx];
    v_cache[dst] = v[idx];
}

extern "C" __global__ void pi05_layer_norm_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    long long rows,
    long long cols,
    long long eps_bits,
    __nv_bfloat16* __restrict__ out) {
    long long r = blockIdx.x;
    if (r >= rows) return;
    float eps = f32_from_bits_i64(eps_bits);
    const __nv_bfloat16* row = x + r * cols;
    float sum = 0.0f;
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        sum += __bfloat162float(row[c]);
    }
    float mean = block_sum(sum) / static_cast<float>(cols);
    float var = 0.0f;
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        float d = __bfloat162float(row[c]) - mean;
        var += d * d;
    }
    float inv_std = rsqrtf(block_sum(var) / static_cast<float>(cols) + eps);
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = (__bfloat162float(row[c]) - mean) * inv_std;
        v = v * __bfloat162float(weight[c]) + __bfloat162float(bias[c]);
        out[r * cols + c] = __float2bfloat16(v);
    }
}

extern "C" __global__ void pi05_layer_norm_to_fp8_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    const float* __restrict__ scale,
    long long rows,
    long long cols,
    long long eps_bits,
    __nv_fp8_e4m3* __restrict__ out_fp8) {
    long long r = blockIdx.x;
    if (r >= rows) return;
    float eps = f32_from_bits_i64(eps_bits);
    const __nv_bfloat16* row = x + r * cols;
    float sum = 0.0f;
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        sum += __bfloat162float(row[c]);
    }
    float mean = block_sum(sum) / static_cast<float>(cols);
    float var = 0.0f;
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        float d = __bfloat162float(row[c]) - mean;
        var += d * d;
    }
    float inv_std = rsqrtf(block_sum(var) / static_cast<float>(cols) + eps);
    float s = *scale;
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = (__bfloat162float(row[c]) - mean) * inv_std;
        v = v * __bfloat162float(weight[c]) + __bfloat162float(bias[c]);
        v = __bfloat162float(__float2bfloat16(v));
        out_fp8[r * cols + c] = to_fp8_e4m3(v, s);
    }
}

extern "C" __global__ void pi05_rms_norm_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    long long rows,
    long long cols,
    long long eps_bits,
    __nv_bfloat16* __restrict__ out) {
    long long r = blockIdx.x;
    if (r >= rows) return;
    float eps = f32_from_bits_i64(eps_bits);
    const __nv_bfloat16* row = x + r * cols;
    float ssq = 0.0f;
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(row[c]);
        ssq += v * v;
    }
    float rstd = rsqrtf(block_sum(ssq) / static_cast<float>(cols) + eps);
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(row[c]) * rstd * __bfloat162float(weight[c]);
        out[r * cols + c] = __float2bfloat16(v);
    }
}

extern "C" __global__ void pi05_rms_norm_unit_bf16(
    const __nv_bfloat16* __restrict__ x,
    long long rows,
    long long cols,
    long long eps_bits,
    __nv_bfloat16* __restrict__ out) {
    long long r = blockIdx.x;
    if (r >= rows) return;
    float eps = f32_from_bits_i64(eps_bits);
    const __nv_bfloat16* row = x + r * cols;
    float ssq = 0.0f;
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(row[c]);
        ssq += v * v;
    }
    float rstd = rsqrtf(block_sum(ssq) / static_cast<float>(cols) + eps);
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(row[c]) * rstd;
        out[r * cols + c] = __float2bfloat16(v);
    }
}

extern "C" __global__ void pi05_rms_norm_unit_to_fp8_bf16(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ scale,
    long long rows,
    long long cols,
    long long eps_bits,
    __nv_fp8_e4m3* __restrict__ out_fp8) {
    long long r = blockIdx.x;
    if (r >= rows) return;
    float eps = f32_from_bits_i64(eps_bits);
    const __nv_bfloat16* row = x + r * cols;
    float ssq = 0.0f;
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(row[c]);
        ssq += v * v;
    }
    float rstd = rsqrtf(block_sum(ssq) / static_cast<float>(cols) + eps);
    float s = *scale;
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(row[c]) * rstd;
        v = __bfloat162float(__float2bfloat16(v));
        out_fp8[r * cols + c] = to_fp8_e4m3(v, s);
    }
}

extern "C" __global__ void pi05_rms_norm_to_fp8_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ scale,
    long long rows,
    long long cols,
    long long eps_bits,
    __nv_fp8_e4m3* __restrict__ out_fp8) {
    long long r = blockIdx.x;
    if (r >= rows) return;
    float eps = f32_from_bits_i64(eps_bits);
    const __nv_bfloat16* row = x + r * cols;
    float ssq = 0.0f;
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(row[c]);
        ssq += v * v;
    }
    float sum = block_sum(ssq);
    float rstd = rsqrtf(sum / static_cast<float>(cols) + eps);
    float s = *scale;
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        float w = __bfloat162float(weight[c]);
        float v = __bfloat162float(row[c]) * rstd * w;
        out_fp8[r * cols + c] = to_fp8_e4m3(v, s);
    }
}

extern "C" __global__ void pi05_residual_rms_norm_to_fp8_bf16(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ addend,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ scale,
    long long rows,
    long long cols,
    long long eps_bits,
    __nv_fp8_e4m3* __restrict__ out_fp8) {
    long long r = blockIdx.x;
    if (r >= rows) return;
    float eps = f32_from_bits_i64(eps_bits);
    __nv_bfloat16* row = residual + r * cols;
    const __nv_bfloat16* add = addend + r * cols;
    float ssq = 0.0f;
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(row[c]) + __bfloat162float(add[c]);
        row[c] = __float2bfloat16(v);
        ssq += v * v;
    }
    float sum = block_sum(ssq);
    float rstd = rsqrtf(sum / static_cast<float>(cols) + eps);
    float s = *scale;
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        float w = __bfloat162float(weight[c]);
        float v = __bfloat162float(row[c]) * rstd * w;
        out_fp8[r * cols + c] = to_fp8_e4m3(v, s);
    }
}

extern "C" __global__ void pi05_bias_residual_bf16(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    long long rows,
    long long cols) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = rows * cols;
    if (idx >= total) return;
    long long c = idx % cols;
    float v = __bfloat162float(residual[idx])
            + __bfloat162float(x[idx])
            + __bfloat162float(bias[c]);
    residual[idx] = __float2bfloat16(v);
}

extern "C" __global__ void pi05_bias_add_bf16(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    long long rows,
    long long cols) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = rows * cols;
    if (idx >= total) return;
    long long c = idx - (idx / cols) * cols;
    float v = __bfloat162float(x[idx]) + __bfloat162float(bias[c]);
    x[idx] = __float2bfloat16(v);
}

extern "C" __global__ void pi05_position_add_bf16(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ position,
    long long rows,
    long long positions,
    long long cols) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = rows * cols;
    if (idx >= total) return;
    long long c = idx - (idx / cols) * cols;
    long long r = idx / cols;
    long long p = r - (r / positions) * positions;
    float v = __bfloat162float(x[idx])
            + __bfloat162float(position[p * cols + c]);
    x[idx] = __float2bfloat16(v);
}

extern "C" __global__ void pi05_prefix_concat_bf16(
    const __nv_bfloat16* __restrict__ image_embs,
    const __nv_bfloat16* __restrict__ lang_embs,
    long long image_rows,
    long long lang_rows,
    long long hidden,
    __nv_bfloat16* __restrict__ out) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = (image_rows + lang_rows) * hidden;
    if (idx >= total) return;
    long long row = idx / hidden;
    long long col = idx - row * hidden;
    if (row < image_rows) {
        out[idx] = image_embs[row * hidden + col];
    } else {
        out[idx] = lang_embs[(row - image_rows) * hidden + col];
    }
}

extern "C" __global__ void pi05_residual_add_bf16(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ x,
    long long n) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    residual[idx] = __float2bfloat16(
        __bfloat162float(residual[idx]) + __bfloat162float(x[idx]));
}

extern "C" __global__ void pi05_gate_mul_residual_bf16(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gate,
    long long n) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = __bfloat162float(residual[idx])
            + __bfloat162float(x[idx]) * __bfloat162float(gate[idx]);
    residual[idx] = __float2bfloat16(v);
}

extern "C" __global__ void pi05_geglu_to_fp8_bf16(
    const __nv_bfloat16* __restrict__ merged_gate_up,
    const float* __restrict__ scale,
    long long rows,
    long long hidden,
    __nv_fp8_e4m3* __restrict__ out_fp8) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = rows * hidden;
    if (i >= total) return;
    long long r = i / hidden;
    long long h = i - r * hidden;
    const __nv_bfloat16* row = merged_gate_up + r * (2 * hidden);
    float gate = __bfloat162float(row[h]);
    float up = __bfloat162float(row[hidden + h]);
    out_fp8[i] = to_fp8_e4m3(gelu_tanh_approx(gate) * up, *scale);
}

extern "C" __global__ void pi05_geglu_bf16(
    const __nv_bfloat16* __restrict__ merged_gate_up,
    long long rows,
    long long hidden,
    __nv_bfloat16* __restrict__ out) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = rows * hidden;
    if (i >= total) return;
    long long r = i / hidden;
    long long h = i - r * hidden;
    const __nv_bfloat16* row = merged_gate_up + r * (2 * hidden);
    float gate = __bfloat162float(row[h]);
    float up = __bfloat162float(row[hidden + h]);
    out[i] = __float2bfloat16(gelu_tanh_approx(gate) * up);
}

extern "C" __global__ void pi05_gelu_inplace_bf16(
    __nv_bfloat16* __restrict__ x,
    long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = __float2bfloat16(gelu_erf_approx(__bfloat162float(x[i])));
}

extern "C" __global__ void pi05_bias_gelu_to_fp8_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    const float* __restrict__ scale,
    long long rows,
    long long cols,
    __nv_fp8_e4m3* __restrict__ out_fp8) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = rows * cols;
    if (idx >= total) return;
    long long c = idx - (idx / cols) * cols;
    float v = __bfloat162float(x[idx]) + __bfloat162float(bias[c]);
    v = __bfloat162float(__float2bfloat16(v));
    v = __bfloat162float(__float2bfloat16(gelu_erf_approx(v)));
    out_fp8[idx] = to_fp8_e4m3(v, *scale);
}

extern "C" __global__ void pi05_ada_rms_norm_style_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ style,
    long long rows,
    long long cols,
    long long eps_bits,
    __nv_bfloat16* __restrict__ out,
    __nv_bfloat16* __restrict__ gate_out) {
    long long r = blockIdx.x;
    if (r >= rows) return;
    float eps = f32_from_bits_i64(eps_bits);
    const __nv_bfloat16* row = x + r * cols;
    const __nv_bfloat16* style_row = style + r * 3 * cols;
    float ssq = 0.0f;
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(row[c]);
        ssq += v * v;
    }
    float rstd = rsqrtf(block_sum(ssq) / static_cast<float>(cols) + eps);
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        float normed = __bfloat162float(row[c]) * rstd * __bfloat162float(weight[c]);
        float scale = __bfloat162float(style_row[c]);
        float shift = __bfloat162float(style_row[cols + c]);
        out[r * cols + c] = __float2bfloat16(normed * (1.0f + scale) + shift);
        gate_out[r * cols + c] = style_row[2 * cols + c];
    }
}

extern "C" __global__ void pi05_ada_rms_norm_style_to_fp8_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ style,
    const float* __restrict__ scale,
    long long rows,
    long long cols,
    long long eps_bits,
    __nv_fp8_e4m3* __restrict__ out_fp8,
    __nv_bfloat16* __restrict__ gate_out) {
    long long r = blockIdx.x;
    if (r >= rows) return;
    float eps = f32_from_bits_i64(eps_bits);
    const __nv_bfloat16* row = x + r * cols;
    const __nv_bfloat16* style_row = style + r * 3 * cols;
    float ssq = 0.0f;
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(row[c]);
        ssq += v * v;
    }
    float rstd = rsqrtf(block_sum(ssq) / static_cast<float>(cols) + eps);
    float s = *scale;
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        float normed = __bfloat162float(row[c]) * rstd * __bfloat162float(weight[c]);
        float style_scale = __bfloat162float(style_row[c]);
        float shift = __bfloat162float(style_row[cols + c]);
        out_fp8[r * cols + c] = to_fp8_e4m3(normed * (1.0f + style_scale) + shift, s);
        gate_out[r * cols + c] = style_row[2 * cols + c];
    }
}

extern "C" __global__ void pi05_gate_residual_ada_norm_to_fp8_bf16(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ style,
    const float* __restrict__ scale,
    long long rows,
    long long cols,
    long long eps_bits,
    __nv_fp8_e4m3* __restrict__ out_fp8,
    __nv_bfloat16* __restrict__ gate_out) {
    long long r = blockIdx.x;
    if (r >= rows) return;
    float eps = f32_from_bits_i64(eps_bits);
    __nv_bfloat16* res_row = residual + r * cols;
    const __nv_bfloat16* x_row = x + r * cols;
    const __nv_bfloat16* gate_row = gate + r * cols;
    const __nv_bfloat16* style_row = style + r * 3 * cols;
    float ssq = 0.0f;
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(res_row[c])
                + __bfloat162float(x_row[c]) * __bfloat162float(gate_row[c]);
        res_row[c] = __float2bfloat16(v);
        ssq += v * v;
    }
    float rstd = rsqrtf(block_sum(ssq) / static_cast<float>(cols) + eps);
    float s = *scale;
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        float normed = __bfloat162float(res_row[c]) * rstd * __bfloat162float(weight[c]);
        float style_scale = __bfloat162float(style_row[c]);
        float shift = __bfloat162float(style_row[cols + c]);
        out_fp8[r * cols + c] = to_fp8_e4m3(normed * (1.0f + style_scale) + shift, s);
        gate_out[r * cols + c] = style_row[2 * cols + c];
    }
}

extern "C" __global__ void pi05_quantize_fp8_static_bf16(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ scale,
    long long n,
    __nv_fp8_e4m3* __restrict__ out_fp8) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out_fp8[i] = to_fp8_e4m3(__bfloat162float(x[i]), *scale);
}

extern "C" __global__ void pi05_quantize_fp8_dynamic_bf16(
    const __nv_bfloat16* __restrict__ x,
    long long n,
    __nv_fp8_e4m3* __restrict__ out_fp8,
    float* __restrict__ scale_out) {
    float local_max = 0.0f;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(__bfloat162float(x[i])));
    }
    float amax = block_max(local_max);
    float scale = fmaxf(amax / 448.0f, 1.0e-12f);
    if (threadIdx.x == 0) {
        *scale_out = scale;
    }
    __syncthreads();
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        out_fp8[i] = to_fp8_e4m3(__bfloat162float(x[i]), scale);
    }
}

extern "C" __global__ void pi05_reduce_amax_bf16(
    const __nv_bfloat16* __restrict__ x,
    long long n,
    float* __restrict__ partial_amax) {
    float local_max = 0.0f;
    long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += stride) {
        local_max = fmaxf(local_max, fabsf(__bfloat162float(x[i])));
    }
    partial_amax[blockIdx.x] = block_max(local_max);
}

extern "C" __global__ void pi05_amax_to_scale(
    const float* __restrict__ partial_amax,
    long long n,
    float* __restrict__ scale_out) {
    float local_max = 0.0f;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, partial_amax[i]);
    }
    float amax = block_max(local_max);
    if (threadIdx.x == 0) {
        *scale_out = fmaxf(amax / 448.0f, 1.0e-12f);
    }
}

extern "C" __global__ void pi05_attention_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    long long rows_q,
    long long rows_k,
    long long num_q_heads,
    long long num_kv_heads,
    long long head_dim,
    long long scale_bits,
    __nv_bfloat16* __restrict__ out) {
    extern __shared__ float scores[];
    long long q_row = blockIdx.x;
    long long q_head = blockIdx.y;
    if (q_row >= rows_q || q_head >= num_q_heads) return;

    long long group = num_q_heads / num_kv_heads;
    long long kv_head = q_head / group;
    float scale = f32_from_bits_i64(scale_bits);
    const __nv_bfloat16* q_vec = q + (q_row * num_q_heads + q_head) * head_dim;

    for (long long kk = 0; kk < rows_k; ++kk) {
        const __nv_bfloat16* k_vec = k + (kk * num_kv_heads + kv_head) * head_dim;
        float partial = 0.0f;
        for (long long d = threadIdx.x; d < head_dim; d += blockDim.x) {
            partial += __bfloat162float(q_vec[d]) * __bfloat162float(k_vec[d]);
        }
        float score = block_sum(partial) * scale;
        if (threadIdx.x == 0) scores[kk] = score;
        __syncthreads();
    }

    float local_max = -INFINITY;
    for (long long kk = threadIdx.x; kk < rows_k; kk += blockDim.x) {
        local_max = fmaxf(local_max, scores[kk]);
    }
    float max_score = block_max(local_max);

    float local_denom = 0.0f;
    for (long long kk = threadIdx.x; kk < rows_k; kk += blockDim.x) {
        local_denom += expf(scores[kk] - max_score);
    }
    float denom = fmaxf(block_sum(local_denom), 1.0e-20f);

    for (long long d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (long long kk = 0; kk < rows_k; ++kk) {
            const __nv_bfloat16* v_vec = v + (kk * num_kv_heads + kv_head) * head_dim;
            float weight = expf(scores[kk] - max_score) / denom;
            acc += weight * __bfloat162float(v_vec[d]);
        }
        out[(q_row * num_q_heads + q_head) * head_dim + d] = __float2bfloat16(acc);
    }
}

extern "C" __global__ void pi05_attention_prefix_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    long long rows_q,
    long long prefix_rows,
    long long suffix_rows,
    long long num_q_heads,
    long long num_kv_heads,
    long long head_dim,
    long long scale_bits,
    __nv_bfloat16* __restrict__ out) {
    extern __shared__ float scores[];
    long long rows_k = prefix_rows + suffix_rows;
    long long q_row = blockIdx.x;
    long long q_head = blockIdx.y;
    if (q_row >= rows_q || q_head >= num_q_heads) return;

    long long group = num_q_heads / num_kv_heads;
    long long kv_head = q_head / group;
    float scale = f32_from_bits_i64(scale_bits);
    const __nv_bfloat16* q_vec = q + (q_row * num_q_heads + q_head) * head_dim;

    for (long long kk = 0; kk < rows_k; ++kk) {
        const __nv_bfloat16* k_vec = k + (kk * num_kv_heads + kv_head) * head_dim;
        float partial = 0.0f;
        for (long long d = threadIdx.x; d < head_dim; d += blockDim.x) {
            partial += __bfloat162float(q_vec[d]) * __bfloat162float(k_vec[d]);
        }
        float score = block_sum(partial) * scale;
        if (threadIdx.x == 0) scores[kk] = score;
        __syncthreads();
    }

    float local_max = -INFINITY;
    for (long long kk = threadIdx.x; kk < rows_k; kk += blockDim.x) {
        local_max = fmaxf(local_max, scores[kk]);
    }
    float max_score = block_max(local_max);

    float local_denom = 0.0f;
    for (long long kk = threadIdx.x; kk < rows_k; kk += blockDim.x) {
        local_denom += expf(scores[kk] - max_score);
    }
    float denom = fmaxf(block_sum(local_denom), 1.0e-20f);

    for (long long d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (long long kk = 0; kk < rows_k; ++kk) {
            const __nv_bfloat16* v_vec = v + (kk * num_kv_heads + kv_head) * head_dim;
            float weight = expf(scores[kk] - max_score) / denom;
            acc += weight * __bfloat162float(v_vec[d]);
        }
        out[(q_row * num_q_heads + q_head) * head_dim + d] = __float2bfloat16(acc);
    }
}

extern "C" __global__ void pi05_euler_update_f32(
    float* __restrict__ x,
    const float* __restrict__ v,
    long long dt_bits,
    long long n) {
    float dt = f32_from_bits_i64(dt_bits);
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += dt * v[i];
    }
}

extern "C" __global__ void pi05_euler_update_bf16(
    float* __restrict__ x,
    const __nv_bfloat16* __restrict__ v,
    long long dt_bits,
    long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float dt = f32_from_bits_i64(dt_bits);
    x[i] += dt * __bfloat162float(v[i]);
}
