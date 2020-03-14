#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <omp.h>
#include <CL/opencl.h>

#include "const.h"

#ifndef CHECK_ERR
#define CHECK_ERR(intro, result, exit_label)        \
do {                                                \
    if ((result) < 0)                               \
    {                                               \
        fprintf(stderr, "%s: %d", intro, result);   \
        exit_code = -1;                             \
        goto exit_label;                            \
    }                                               \
} while (false)
#endif

static inline
void fill_array(float* ptr, size_t cnt)
{
    for (size_t i = 0; i < cnt; ++i)
        ptr[i] = (float) i / cnt;
}

static inline
char* load_source_file(char const* file_name)
{
    FILE* source_file = fopen(file_name, "r");
    if (!source_file)
    {
        perror("Error opening kernel file");
        return NULL;
    }

    size_t const file_load_sz = 1024 * 16;
    char* program_code = malloc(file_load_sz);
    if (!program_code)
    {
        fclose(source_file);
        return NULL;
    }

    size_t code_len = fread(program_code, 1, file_load_sz - 1, source_file);
    program_code[code_len] = '\0';

    return program_code;
}

int main()
{
    /// n, m, k are expected to be divisible by tile_size.
    size_t const n = 2048;
    size_t const m = 2048 + 32;
    size_t const k = 2048 + 64;

    size_t const array_mem_sz1 = n * m * sizeof(float);
    size_t const array_mem_sz2 = m * k * sizeof(float);
    size_t const array_mem_sz3 = n * k * sizeof(float);

    char const* const kernel_file_name = "gemm4.cl";
    char const* const kernel_name = "gemm4";

    float* const a = (float*) malloc(array_mem_sz1);
    float* const b = (float*) malloc(array_mem_sz2);
    float* const c = (float*) malloc(array_mem_sz3);

    if (!a || !b || !c)
    {
        perror("Mem alloc failed");
        goto release_matrixes;
    }

    fill_array(a, n * m);
    fill_array(b, m * k);

    int exit_code = 0;
    cl_int error_code;
    cl_uint num_platforms;
    error_code = clGetPlatformIDs(0, 0, &num_platforms);
    CHECK_ERR("Error getting platforms list", error_code, release_matrixes);

    cl_platform_id * const platforms
        = (cl_platform_id *) malloc(num_platforms * sizeof(cl_platform_id));

    error_code = clGetPlatformIDs(num_platforms, platforms, &num_platforms);
    CHECK_ERR("Error getting platforms list", error_code, release_platforms);

    cl_uint         num_devices = 0;
    error_code = clGetDeviceIDs (
            platforms[0], CL_DEVICE_TYPE_GPU,  // using platform[0] as default platform
            0, 0, &num_devices
    );
    CHECK_ERR("Error getting device list", error_code, release_platforms);

    cl_device_id * const gpu_devices
        = (cl_device_id *) malloc(num_devices * sizeof(cl_device_id));

    error_code = clGetDeviceIDs (
            platforms[0], CL_DEVICE_TYPE_GPU,
            num_devices, gpu_devices, &num_devices
    );

    CHECK_ERR("Error getting device list", error_code, release_devices);

    if (!num_devices)
    {
        puts("No GPU devices found :'(");
        free(platforms);
        return -1;
    }

    char device_name[128];
    size_t device_name_len;
    error_code = clGetDeviceInfo (
            gpu_devices[0], CL_DEVICE_NAME,
            128, device_name, &device_name_len
    );
    CHECK_ERR("Error getting device name", error_code, release_devices);

    printf("Target device name: %s\n", device_name);

    cl_context context = clCreateContext(0, 1, gpu_devices, 0, 0, &error_code);
    CHECK_ERR("Error creating context", error_code, release_devices);

    cl_command_queue queue = clCreateCommandQueue (
            context, gpu_devices[0], CL_QUEUE_PROFILING_ENABLE, &error_code
    );
    CHECK_ERR("Error creating command queue", error_code, release_context);

    size_t const num_files = 2;

    char* sources[] = {
            load_source_file("const.h"),
            load_source_file(kernel_file_name)
    };

    if (!sources[0] || !sources[1])
    {
        free(sources[0]);
        free(sources[1]);
        goto release_devices;
    }

    size_t lens[] = {
            strlen(sources[0]),
            strlen(sources[1])
    };

    cl_program program = clCreateProgramWithSource (
            context, 2, sources, lens, &error_code
    );
    CHECK_ERR("Error creating program:", error_code, release_command_queue);

    error_code = clBuildProgram(program, 1, gpu_devices, "", 0, 0);
    if (error_code)
    {
        size_t log_len = 0;
        error_code = clGetProgramBuildInfo (
                program, gpu_devices[0],
                CL_PROGRAM_BUILD_LOG, 0, 0, &log_len
        );
        CHECK_ERR("Error getting build log", error_code, release_context);
        char* const build_log = malloc(log_len);
        error_code = clGetProgramBuildInfo (
                program, gpu_devices[0],
                CL_PROGRAM_BUILD_LOG, log_len, build_log, &log_len
        );
        CHECK_ERR("Error getting build log", error_code, exit4);
        fprintf(stderr, "Kernel compilation error:\n%s\n", build_log);
        exit4:
        free(build_log);
        exit_code = -1;
        goto release_program;
    }

    cl_kernel kernel = clCreateKernel(program, kernel_name, &error_code);
    CHECK_ERR("Error creating kernel", error_code, release_program);

    cl_mem mem1 = clCreateBuffer(context, CL_MEM_READ_ONLY, array_mem_sz1, 0, &error_code);
    CHECK_ERR("Error creating buffer", error_code, release_kernel);
    cl_mem mem2 = clCreateBuffer(context, CL_MEM_READ_ONLY, array_mem_sz2, 0, &error_code);
    CHECK_ERR("Error creating buffer", error_code, release_mem1);
    cl_mem mem3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, array_mem_sz3, 0, &error_code);
    CHECK_ERR("Error creating buffer", error_code, release_mem2);

    error_code = clEnqueueWriteBuffer(queue, mem1, false, 0, array_mem_sz1, a, 0, 0, 0);
    CHECK_ERR("clEnqueueWriteBuffer error", error_code, release_mem3);
    error_code = clEnqueueWriteBuffer(queue, mem2, true, 0, array_mem_sz2, b, 0, 0, 0);
    CHECK_ERR("clEnqueueWriteBuffer error", error_code, release_mem3);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem3);
    clSetKernelArg(kernel, 3, sizeof(cl_uint), &n);
    clSetKernelArg(kernel, 4, sizeof(cl_uint), &m);
    clSetKernelArg(kernel, 5, sizeof(cl_uint), &k);
//    clSetKernelArg(kernel, 6, sizeof(cl_uint), &tile_size);

    size_t work_size[] = {k, n / ELEMS_PER_THREAD};
    size_t local_group_size[] = {TILE_SIZE, TILE_SIZE / ELEMS_PER_THREAD};
    cl_event run_event;
    error_code = clEnqueueNDRangeKernel (
            queue, kernel, 2, NULL,
            work_size, local_group_size, 0, 0, &run_event
    );
    CHECK_ERR("Error enquing kernel", error_code, release_context);
    clEnqueueReadBuffer(queue, mem3, true, 0, array_mem_sz3, c, 0, 0, 0);

    cl_ulong t_start = 0, t_end = 0;
    clGetEventProfilingInfo(run_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, 0);
    clGetEventProfilingInfo(run_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, 0);

    printf("%Lf µs elapsed\n", (t_end - t_start) / (long double) 1000);

#ifndef NDEBUG
    {
        float* const gold = (float*) malloc(array_mem_sz3);
        memset(gold, 0, array_mem_sz3);

        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < m; ++j)
                for (size_t l = 0; l < k; ++l)
                    gold[i * k + l] += a[i * m + j] * b[j * k + l];

        fflush(stdout);

        for (size_t i = 0; i < n; ++i)
        {
            for (size_t l = 0; l < k; ++l)
            {
                float delta = gold[i * k + l] - c[i * k + l];
                float abs_delta = fabsf(delta);
                assert(abs_delta < 0.05);
            }
        }

        free(gold);
    }
#endif
release_mem3:
    clReleaseMemObject(mem3);
release_mem2:
    clReleaseMemObject(mem2);
release_mem1:
    clReleaseMemObject(mem1);
release_kernel:
    clReleaseKernel(kernel);
release_program:
    for (size_t i = 0; i < num_files; ++i)
        free(sources[i]);
    clReleaseProgram(program);
release_command_queue:
    clReleaseCommandQueue(queue);
release_context:
    clReleaseContext(context);
release_devices:
    for (size_t i = 0; i < num_devices; ++i)
        clReleaseDevice(gpu_devices[i]);
    free(gpu_devices);
release_platforms:
    free(platforms);
release_matrixes:
    free(a);
    free(b);
    free(c);
    return exit_code;
}
