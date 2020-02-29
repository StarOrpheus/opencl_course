__kernel void array_sum(__global const int* const a,
                        __global const int* const b,
                        __global int* const c)
{
    uint gid = get_global_id(0);
    c[gid] = a[gid] + b[gid];
}
