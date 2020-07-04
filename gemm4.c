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
        ptr[i] = (float) ((double) rand() / (double) (RAND_MAX));
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

struct gpu_context
{
    cl_device_id        selected_device;

    cl_program          program;

    cl_mem              fst_mattr_buff_in;
    cl_mem              sec_mattr_buff_in;
    cl_mem              thr_mattr_buff_out;
    cl_kernel           kernel;
};

/// Destructor for \ref gpu_context
void release_gpu_context(struct gpu_context* context)
{
// todo
}

struct input_data
{
    size_t in_A_size; //!< Size in floats
    size_t in_B_size; //!< Size in floats
    size_t out_C_size; //!< Size in floats

    float* in_A;
    float* in_B;
    float* out_C;
};

/// Destructor for \ref input_data
void release_input_data(struct input_data* context)
{
    if (!context)
        return;

    if (context->in_A)
        free(context->in_A);
    if (context->in_B)
        free(context->in_B);
    if (context->out_C)
        free(context->out_C);
    free(context);
}

/// Setup device for the specified \ref gpu_context
cl_int select_device(struct gpu_context* context)
{
    cl_int error_code;
    cl_uint num_platforms;

    error_code = clGetPlatformIDs(0, 0, &num_platforms);
    if (error_code)
    {
        fprintf(stderr, "Error getting platforms list!\n");
        return error_code;
    }

    cl_platform_id* platforms = calloc(num_platforms, sizeof(cl_platform_id));

    error_code = clGetPlatformIDs(num_platforms, platforms, &num_platforms);
    if (error_code)
    {
        fprintf(stderr, "Error getting platforms list!\n");
        free(platforms);
        return error_code;
    }

    size_t max_warp_sz = 0;
    cl_uint num_devices = 0;
    size_t max_devices = 42;
    cl_device_id device_list[max_devices];
    char device_name[64];

    for (size_t i = 0; i < num_platforms; ++i)
    {
        error_code = clGetDeviceIDs(
            platforms[i], CL_DEVICE_TYPE_ALL, max_devices, device_list,
            &num_devices
        );

        if (error_code) continue;

        for (size_t j = 0; j < num_devices; ++j)
        {
            if (!context->selected_device)
                context->selected_device = device_list[j];

            size_t ret_sz;
            size_t warp_sz;

            error_code = clGetDeviceInfo(
                device_list[j], CL_DEVICE_NAME, 63, device_name, &ret_sz
            );

            if (error_code)
            {
                clReleaseDevice(device_list[j]);
                continue;
            }

            device_name[ret_sz] = '\0';

            fprintf(stderr, "Found device %s\n", device_name);

            error_code = clGetDeviceInfo(
                device_list[j], CL_DEVICE_WARP_SIZE_NV,
                sizeof(size_t), &warp_sz, &ret_sz
            );

            if (error_code)
            {
                if (context->selected_device != device_list[j])
                    clReleaseDevice(device_list[j]);
                continue;
            }

            if (warp_sz > max_warp_sz)
            {
                if (context->selected_device)
                    clReleaseDevice(device_list[j]);
                context->selected_device = device_list[j];
            }
            else if (context->selected_device != device_list[j])
                clReleaseDevice(device_list[j]);

            fprintf(stderr, "\tDevice warp size: %zu\n", warp_sz);
        }
    }

    if (!context->selected_device)
    {
        free(platforms);
        return error_code;
    }
    else
    {
        cl_uint ret_sz;
        error_code = clGetDeviceInfo(
            context->selected_device, CL_DEVICE_NAME, 63, device_name, &ret_sz
        );

        device_name[ret_sz] = '\0';

        fprintf(stderr, "Selected device: %s\n", device_name);
    }

    return 0;
}

/// Loads and compile the kernel for the specified \ref gpu_context
cl_int load_program(struct gpu_context* context,
                    char const** sources_list, size_t src_list_sz,
                    char const* kernel_name)
{
    // todo
    return 0;
}

/// Setups kernel & kernel structs like mem buffers for the \ref gpu_context
cl_int setup_kernel(struct gpu_context* context, size_t n, size_t m, size_t k)
{
    // todo
    return 0;
}

struct gpu_context* setup_gpu_context(size_t n, size_t m, size_t k,
                                      char const** sources_list,
                                      size_t src_list_sz,
                                      char const* kernel_name,
                                      cl_int* error)
{
    assert(error != 0);
    assert(kernel_name != 0);
    assert(sources_list != 0);

    *error = 0;

    struct gpu_context* const context = calloc(1, sizeof(struct gpu_context));

    *error = select_device(context);
    if (*error)
        goto return_error;

    *error = load_program(context, sources_list, src_list_sz, kernel_name);
    if (*error)
        goto return_error;

    *error = setup_kernel(context, n, m, k);
    if (*error)
        goto return_error;

    return context;

return_error:
    release_gpu_context(context);
    return NULL;
}

struct input_data* generate_input(size_t n, size_t m, size_t k)
{
    struct input_data* data = calloc(1, sizeof(struct input_data));

    data->in_A_size = n * m;
    data->in_B_size = m * k;
    data->out_C_size = n * k;

    data->in_A = calloc(data->in_A_size, sizeof(float));
    data->in_B = calloc(data->in_B_size, sizeof(float));
    data->out_C = calloc(data->out_C_size, sizeof(float));

    if (!data->in_A || !data->in_B || !data->out_C)
        goto error_return;

    fill_array(data->in_A, data->in_A_size);
    fill_array(data->in_B, data->in_B_size);

    return data;

error_return:
    release_input_data(data);
    return NULL;
}

int main()
{
    /// n, m, k are expected to be divisible by tile_size.
    size_t const n = 2048;
    size_t const m = 512;
    size_t const k = 1024;

    char const* const   kernel_file_name = "gemm4.cl";
    char const* const   kernel_name = "gemm4";

    int exit_code = 0;
    cl_int error_code;

    struct gpu_context* context = setup_gpu_context(
        n, m, k, "", 0, kernel_name, &error_code
    );

    release_gpu_context(context);
    return 0;
/*
    error_code = clGetPlatformIDs(0, 0, &num_platforms);
    CHECK_ERR("Error getting platforms list", error_code, release_matrixes);

//    cl_platform_id * const platforms
//        = (cl_platform_id *) malloc(num_platforms * sizeof(cl_platform_id));

    error_code = clGetPlatformIDs(num_platforms, platforms, &num_platforms);
    CHECK_ERR("Error getting platforms list", error_code, release_platforms);

    cl_uint         num_devices = 0;
    error_code = clGetDeviceIDs (
            platforms[0], CL_DEVICE_TYPE_GPU,  // using platform[0] as default platform
            0, 0, &num_devices
    );
    CHECK_ERR("Error getting device list", error_code, release_platforms);

    if (!num_devices)
    {
        error_code = clGetDeviceIDs (
            platforms[0], CL_DEVICE_TYPE_CPU,  // using platform[0] as default platform
            0, 0, &num_devices
        );
    }

    cl_device_id * const gpu_devices
        = (cl_device_id *) malloc(num_devices * sizeof(cl_device_id));

    error_code = clGetDeviceIDs (
            platforms[0], CL_DEVICE_TYPE_GPU,
            num_devices, gpu_devices, &num_devices
    );

    CHECK_ERR("Error getting device list", error_code, release_devices);

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
    */
}
