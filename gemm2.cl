__kernel void gemm1(__global float const* const a,
                        __global float const* const b,
                        __global int* const c,
                        uint const n,
                        uint const m,
                        uint const k)
{
    uint i = get_global_id(1);
    uint l = get_global_id(0);

    float local_sum = 0;

    for (uint j = 0; j < m; ++j)
        local_sum += a[i * m + j] * b[j * k + l];

    c[i * k + l] = local_sum;
}
