__kernel void par_scan(__global float const* const a,
                       __global float* const c,
                       uint const n)
{
    __local float temp[SCAN_TILE_SIZE];
    int local_i = get_global_id(0);

    temp[local_i] = a[local_i];

    for (int j = 1; local_i - j >= 0; j <<= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        float val = temp[local_i - j];
        barrier(CLK_LOCAL_MEM_FENCE);
        temp[local_i] += val;
    }

    c[local_i] = temp[local_i];
}
