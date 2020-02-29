__kernel void gemm1(__global float const* const a,
                        __global float const* const b,
                        __global int* const c,
                        uint const n,
                        uint const m,
                        uint const k)
{
    uint i = get_global_id(0);
    uint l = get_global_id(1);

    for (uint j = 0; j < m; ++j)
        c[i * k + l] += a[i * m + j] * b[j * k + l];
}
