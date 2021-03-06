#ifndef OPENCL_FUN_CONST_H
#define OPENCL_FUN_CONST_H

#ifdef TILE_SIZE
    #error Redefinition of TILE_SIZE
#else
    #define TILE_SIZE 32
#endif

#ifdef ELEMS_PER_THREAD
#error Redifinition of ELEMS_PER_THREAD
#else
#define ELEMS_PER_THREAD 4
#endif

#ifdef SCAN_TILE_SIZE
#error Redifinition of SCAN_TILE_SIZE
#else
#define SCAN_TILE_SIZE 1024
#endif

#endif //OPENCL_FUN_CONST_H
