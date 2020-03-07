__kernel void gemm2(__global float const* const a,
                    __global float const* const b,
                    __global float* const c,
                    uint const n,
                    uint const m,
                    uint const k)
{
    uint i = get_global_id(1);
    uint l = get_global_id(0);

    float local_sum = 0;

    // a: [N x M]
    // b: [M x K]
    for (uint j = 0; j < m; ++j)
        local_sum += a[i * m + j] * b[j * k + l];

    c[i * k + l] = local_sum;
}
