#define TILE_SIZE 32
#define ELEMS_PER_THREAD 4

__kernel void gemm4(__global float const* const a,      /** a: matrix [N x M] */
                    __global float const* const b,      /** b: matrix [M x K] */
                    __global float* const c,            /** c: matrix [N x K] */
                    uint const n,                       /** n = N */
                    uint const m,                       /** m = M */
                    uint const k                        /** k = K */)
{
    uint const global_i     = get_global_id(1) * ELEMS_PER_THREAD;      //!< First row id in result matrix
    uint const global_l     = get_global_id(0);                         //!< Col id in result matrix
    uint const tile_i       = get_local_id(1) * ELEMS_PER_THREAD;       //!< First row id in the current tile
    uint const tile_j       = get_local_id(0);                          //!< Col id in the current tile

    local float A_sub[TILE_SIZE][TILE_SIZE];
    local float B_sub[TILE_SIZE][TILE_SIZE];

    float local_sum[ELEMS_PER_THREAD];
    for (uint i = 0; i < ELEMS_PER_THREAD; ++i)
        local_sum[i] = 0;

    uint const tile_cnt     = m / TILE_SIZE;
    for (uint tile_id = 0; tile_id < tile_cnt; ++tile_id)
    {
        for (uint shift = 0; shift < ELEMS_PER_THREAD; ++shift)
        {
            uint const tiled_row = tile_id * TILE_SIZE + tile_i + shift;    //!< Global row id for second matrix
            uint const tiled_col = tile_id * TILE_SIZE + tile_j;            //!< Global col id for first matrix

            /// Loading them into the current tile buffer
            A_sub[tile_i + shift][tile_j] = a[(global_i + shift) * m + tiled_col];
            B_sub[tile_i + shift][tile_j] = b[tiled_row * k + global_l];
        }

        /// Awaiting local group to fill the buffer
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint shift = 0; shift < ELEMS_PER_THREAD; ++shift)
            for (uint t = 0; t < TILE_SIZE; ++t)
                local_sum[shift] += A_sub[tile_i + shift][t] * B_sub[t][tile_j];

        /// Awaiting local group, not to start loading in the buffer
        /// while it is still in use in prev. loop
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint shift = 0; shift < ELEMS_PER_THREAD; ++shift)
        c[(global_i + shift) * k + global_l] = local_sum[shift];
}
