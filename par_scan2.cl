__kernel void local_scan(__global float const* const a,
                         __global float* const c)
{
    __local float temp[SCAN_TILE_SIZE];
    
    int global_i = get_global_id(0);
    int local_i = get_local_id(0);

    temp[local_i] = a[global_i];

    for (int j = 1; local_i - j >= 0; j <<= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        float val = temp[local_i - j];
        barrier(CLK_LOCAL_MEM_FENCE);
        temp[local_i] += val;
    }

    c[global_i] = temp[local_i];
}

__kernel void tiles_sum(__global float* const c,
                        __global float* const temp)
{    
    int global_i = get_global_id(0);
    int tile_i = global_i / SCAN_TILE_SIZE;

    float result = c[global_i];
    for (int i = 1; i <= tile_i; ++i)
        result += c[i * SCAN_TILE_SIZE - 1];

    temp[global_i] = result;
}
