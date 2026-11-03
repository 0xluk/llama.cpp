#pragma once
// multi-column i-quant MMVQ. For IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S the expensive part of
// vec_dot_*_q8_1 is the weight decode (grid table lookups, sign expansion, scales), and mul_mat_vec_q used
// to redo it for every dst column. Here a 32-value weight chunk is decoded once into 8 packed int8x4 values
// and reused for all ncols_dst columns (MTP verify batches run with ncols_dst = 2..4).
// Integer sums and the float epilogue are the same as in vecdotq.cuh, so results are bit-exact.

#include "vecdotq.cuh"

static constexpr __host__ __device__ bool mmvq_multi_supported(ggml_type type) {
    return type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S ||
           type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S;
}


// Apply 4 sign bits (bit j -> byte j) to 4 unsigned grid bytes: byte j becomes -g if bit j is set.
// (g ^ 0xff) + 1 == -g without a carry into the next byte because every grid byte is >= 1, so this
// equals __vsub4(g ^ s, s) with s = __vcmpne4(...) but needs ~5 integer ops instead of two emulated SIMD ops.
static __device__ __forceinline__ int mmvq_apply_signs4(const uint32_t g, const uint32_t nib) {
    const uint32_t b = (nib * 0x00204081u) & 0x01010101u;
    const uint32_t m = (b << 8) - b;
    return (int) ((g ^ m) + b);
}

// q8_1 activations of block b of a column: either plain block_q8_1 or the 16-byte aligned x4 layout.
template <bool x4>
static __device__ __forceinline__ void mmvq_load_y(const block_q8_1 * __restrict__ ycol, const int b, int (&u)[8], float & dy) {
    if constexpr (x4) {
        const char * g = (const char *) ycol + (b >> 2) * (4*sizeof(block_q8_1));
        const int4 a = *(const int4 *) (g + 4*sizeof(half2) + (b & 3)*QK8_1);
        const int4 c = *(const int4 *) (g + 4*sizeof(half2) + (b & 3)*QK8_1 + 16);
        u[0] = a.x; u[1] = a.y; u[2] = a.z; u[3] = a.w;
        u[4] = c.x; u[5] = c.y; u[6] = c.z; u[7] = c.w;
        dy = __low2float(((const half2 *) g)[b & 3]);
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            u[k] = get_int_b4(ycol[b].qs, k);
        }
        dy = __low2float(ycol[b].ds);
    }
}

template <ggml_type type> struct mmvq_iq_dec;

template <> struct mmvq_iq_dec<GGML_TYPE_IQ2_XXS> {
    int v[8]; int ls; float d; int iqs;
    __device__ __forceinline__ mmvq_iq_dec(const void * __restrict__ vbq, const int kbx, const int iqs_) : iqs(iqs_) {
        const block_iq2_xxs * bq2 = (const block_iq2_xxs *) vbq + kbx;
        const int q2 = get_int_b2(bq2->qs, iqs);
        const uint8_t * aux8 = (const uint8_t *) &q2;
        const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = ((const uint2*)iq2xxs_grid)[aux8[k0/2]];
            const uint32_t s8 = unpack_ksigns(aux32 >> (7 * k0 / 2)) & 0xFF;
            v[k0 + 0] = mmvq_apply_signs4(grid_pos.x, s8 & 0x0F);
            v[k0 + 1] = mmvq_apply_signs4(grid_pos.y, s8 >> 4);
        }
        ls = aux32 >> 27 | 1;
        d  = __half2float(bq2->d);
    }
    template <bool x4 = false>
    __device__ __forceinline__ float dot(const block_q8_1 * __restrict__ ycol, const int b) const {
        int u[8]; float dy;
        mmvq_load_y<x4>(ycol, b, u, dy);
        int sumi = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            sumi = ggml_cuda_dp4a(v[k], u[k], sumi);
        }
        sumi = sumi * ls / 8;
        const float dd = d * dy;
        return dd * sumi;
    }
};

template <> struct mmvq_iq_dec<GGML_TYPE_IQ2_XS> {
    int v[8]; int ls0; int ls1; float d; int iqs;
    __device__ __forceinline__ mmvq_iq_dec(const void * __restrict__ vbq, const int kbx, const int iqs_) : iqs(iqs_) {
        const block_iq2_xs * bq2 = (const block_iq2_xs *) vbq + kbx;
        const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
        const uint16_t * q2 = (const uint16_t *) &q2_packed;
        ls0 = bq2->scales[iqs/2] & 0x0F;
        ls1 = bq2->scales[iqs/2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint2 grid_pos = ((const uint2*)iq2xs_grid)[q2[l0/2] & 0x1FF];
            const uint32_t s8 = unpack_ksigns(q2[l0/2] >> 9) & 0xFF;
            v[l0 + 0] = mmvq_apply_signs4(grid_pos.x, s8 & 0x0F);
            v[l0 + 1] = mmvq_apply_signs4(grid_pos.y, s8 >> 4);
        }
        d = __half2float(bq2->d);
    }
    template <bool x4 = false>
    __device__ __forceinline__ float dot(const block_q8_1 * __restrict__ ycol, const int b) const {
        int u[8]; float dy;
        mmvq_load_y<x4>(ycol, b, u, dy);
        int sumi0 = 0;
        int sumi1 = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            sumi0 = ggml_cuda_dp4a(v[k], u[k], sumi0);
        }
#pragma unroll
        for (int k = 4; k < 8; ++k) {
            sumi1 = ggml_cuda_dp4a(v[k], u[k], sumi1);
        }
        const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
        const float dd = d * dy;
        return dd * sumi;
    }
};

template <> struct mmvq_iq_dec<GGML_TYPE_IQ2_S> {
    int v[8]; int ls0; int ls1; float d; int iqs;
    __device__ __forceinline__ mmvq_iq_dec(const void * __restrict__ vbq, const int kbx, const int iqs_) : iqs(iqs_) {
        const block_iq2_s * bq2 = (const block_iq2_s *) vbq + kbx;
        const int       qs_packed = get_int_b2(bq2->qs, iqs/2);
        const uint8_t * qs        = (const uint8_t *) &qs_packed;
        const int qh = bq2->qh[iqs/2];
        const int       signs_packed_32 = get_int_b2(bq2->qs, QK_K/32 + iqs/2);
        const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;
        ls0 = bq2->scales[iqs/2] & 0x0F;
        ls1 = bq2->scales[iqs/2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int * grid_pos = (const int *)(iq2s_grid + (qs[l0/2] | ((qh << (8-l0)) & 0x300)));
            v[l0 + 0] = mmvq_apply_signs4(grid_pos[0], signs_packed_8[l0/2] & 0x0F);
            v[l0 + 1] = mmvq_apply_signs4(grid_pos[1], signs_packed_8[l0/2] >> 4);
        }
        d = __half2float(bq2->d);
    }
    template <bool x4 = false>
    __device__ __forceinline__ float dot(const block_q8_1 * __restrict__ ycol, const int b) const {
        int u[8]; float dy;
        mmvq_load_y<x4>(ycol, b, u, dy);
        int sumi0 = 0;
        int sumi1 = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            sumi0 = ggml_cuda_dp4a(v[k], u[k], sumi0);
        }
#pragma unroll
        for (int k = 4; k < 8; ++k) {
            sumi1 = ggml_cuda_dp4a(v[k], u[k], sumi1);
        }
        const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
        const float dd = d * dy;
        return dd * sumi;
    }
};

template <> struct mmvq_iq_dec<GGML_TYPE_IQ3_XXS> {
    int v[8]; int ls; float d; int iqs;
    __device__ __forceinline__ mmvq_iq_dec(const void * __restrict__ vbq, const int kbx, const int iqs_) : iqs(iqs_) {
        const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;
        const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs+1));
        const uint8_t * q3 = (const uint8_t *) &q3_packed;
        const uint32_t aux32 = get_int_b2(bq3->qs, QK_K/16 + iqs/2);
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
            const uint32_t s8 = unpack_ksigns(aux32 >> (7*l0/2)) & 0xFF;
            v[l0 + 0] = mmvq_apply_signs4(grid_pos.x, s8 & 0x0F);
            v[l0 + 1] = mmvq_apply_signs4(grid_pos.y, s8 >> 4);
        }
        ls = aux32 >> 28;
        d  = __half2float(bq3->d);
    }
    template <bool x4 = false>
    __device__ __forceinline__ float dot(const block_q8_1 * __restrict__ ycol, const int b) const {
        int u[8]; float dy;
        mmvq_load_y<x4>(ycol, b, u, dy);
        int sumi = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            sumi = ggml_cuda_dp4a(v[k], u[k], sumi);
        }
        sumi = (ls*sumi + sumi/2)/2;
        const float dd = d * dy;
        return dd * sumi;
    }
};

template <> struct mmvq_iq_dec<GGML_TYPE_IQ3_S> {
    int v[8]; int sc; float d; int iqs;
    __device__ __forceinline__ mmvq_iq_dec(const void * __restrict__ vbq, const int kbx, const int iqs_) : iqs(iqs_) {
        const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;
        const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t * qs        = (const uint8_t *) &qs_packed;
        const int qh = bq3->qh[iqs/2];
        const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
        const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(
                iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
            v[l0 + 0] = mmvq_apply_signs4(grid_pos.x, signs_packed_8[l0/2] & 0x0F);
            v[l0 + 1] = mmvq_apply_signs4(grid_pos.y, signs_packed_8[l0/2] >> 4);
        }
        sc = 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
        d  = __half2float(bq3->d);
    }
    template <bool x4 = false>
    __device__ __forceinline__ float dot(const block_q8_1 * __restrict__ ycol, const int b) const {
        int u[8]; float dy;
        mmvq_load_y<x4>(ycol, b, u, dy);
        int sumi = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            sumi = ggml_cuda_dp4a(v[k], u[k], sumi);
        }
        sumi *= sc;
        const float dd = d * dy;
        return dd * sumi;
    }
};
