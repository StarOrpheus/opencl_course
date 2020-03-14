#define TILE_SIZE 32

__kernel void gemm3(__global float const* const a,      /** a: matrix [N x M] */
                    __global float const* const b,      /** b: matrix [M x K] */
                    __global float* const c,            /** c: matrix [N x K] */
                    uint const n,                       /** n = N */
                    uint const m,                       /** m = M */
                    uint const k                        /** k = K */)
{
    uint const global_i     = get_global_id(1);         //!< Row id in result matrix
    uint const global_l     = get_global_id(0);         //!< Col id in result matrix
    uint const tile_i       = get_local_id(1);          //!< Row id in the current tile
    uint const tile_j       = get_local_id(0);          //!< Col id in the current tile

    local float A_sub[TILE_SIZE][TILE_SIZE];    //!< Local buffer for subtiles from the first input matrix
    local float B_sub[TILE_SIZE][TILE_SIZE];    //!< Local buffer for subtiles from the second input matrix

    float local_sum         = 0;
    uint const tile_cnt     = m / TILE_SIZE;
    for (uint tile_id = 0; tile_id < tile_cnt; ++tile_id)
    {
        uint tiled_col = tile_id * TILE_SIZE + tile_j;  //!< Global col id for first matrix
        uint tiled_row = tile_id * TILE_SIZE + tile_i;  //!< Global row id for second matrix

        /// Loading them into the current tile buffer
        A_sub[tile_i][tile_j] = a[global_i * m + tiled_col];
        B_sub[tile_i][tile_j] = b[tiled_row * k + global_l];

        /// Awaiting local group to fill the buffer
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint t = 0; t < TILE_SIZE; ++t)
            local_sum += A_sub[tile_i][t] * B_sub[t][tile_j];

        /// Awaiting local group, not to start loading in the buffer
        /// while it is still in use in prev. loop
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[global_i * k + global_l] = local_sum;
}
