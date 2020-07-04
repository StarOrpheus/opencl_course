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
    if ((result) != 0)                              \
    {                                               \
        fprintf(stderr, "%s: %d\n", intro, result); \
        goto exit_label;                            \
    }                                               \
} while (false)
#endif

#ifndef CHECK_AND_RET_ERR
#define CHECK_AND_RET_ERR(intro, result)            \
do {                                                \
    if ((result) != 0)                              \
    {                                               \
        fprintf(stderr, "%s: %d\n", intro, result); \
        return result;                              \
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
char* load_source_file(char const* file_name, size_t* len)
{
    FILE* source_file = fopen(file_name, "r");
    if (!source_file)
    {
        perror("Error opening source file");
        return NULL;
    }

    size_t const file_load_sz = 1024 * 1024;
    char* program_code = malloc(file_load_sz);
    if (!program_code)
    {
        fclose(source_file);
        return NULL;
    }

    size_t code_len = fread(program_code, 1, file_load_sz - 1, source_file);
    program_code[code_len] = '\0';
    *len = code_len;

    return program_code;
}

struct gpu_context
{
    size_t n;
    size_t m;
    size_t k;

    cl_device_id        selected_device;

    cl_context          context;
    cl_command_queue    command_queue;
    cl_program          program;

    cl_mem              fst_mattr_buff_in;
    cl_mem              sec_mattr_buff_in;
    cl_mem              thr_mattr_buff_out;
    cl_kernel           kernel;
};

/// Destructor for \ref gpu_context
void release_gpu_context(struct gpu_context* context)
{
    if (context->program)
        clReleaseProgram(context->program);
    if (context->fst_mattr_buff_in)
        clReleaseMemObject(context->fst_mattr_buff_in);
    if (context->sec_mattr_buff_in)
        clReleaseMemObject(context->sec_mattr_buff_in);
    if (context->thr_mattr_buff_out)
        clReleaseMemObject(context->thr_mattr_buff_out);
    if (context->kernel)
        clReleaseKernel(context->kernel);
    if (context->context)
        clReleaseContext(context->context);
    if (context->command_queue)
        clReleaseCommandQueue(context->command_queue);
    if (context->selected_device)
        clReleaseDevice(context->selected_device);
    free(context);
}

struct input_data
{
    size_t n;
    size_t m;
    size_t k;

    size_t in_A_size; //!< Size in floats = N * M
    size_t in_B_size; //!< Size in floats = M * K
    size_t out_C_size; //!< Size in floats = N * K

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

static inline
char const* get_mem_type_str(cl_device_local_mem_type t)
{
    switch (t)
    {
    case CL_LOCAL:
        return "local";
    case CL_GLOBAL:
        return "global";
    default:
        return "other";
    }
}

/// Setup device for the specified \ref gpu_context
cl_int select_device(struct gpu_context* context)
{
    assert(context);

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

    cl_uint num_devices = 0;
    size_t max_devices = 42;
    cl_device_id device_list[max_devices];
    char device_name[64];

    cl_device_local_mem_type mem_type = CL_NONE;
    size_t max_work_group_size = 0;

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

            size_t ret_sz = 0;
            cl_device_local_mem_type cur_mem_type = 0;
            size_t work_group_size = 0;

            error_code = clGetDeviceInfo(
                device_list[j], CL_DEVICE_LOCAL_MEM_TYPE,
                sizeof(cl_device_local_mem_type), &cur_mem_type, &ret_sz
            );

            if (error_code)
            {
                if (context->selected_device != device_list[j])
                    clReleaseDevice(device_list[j]);
                continue;
            }

            error_code = clGetDeviceInfo(
                device_list[j], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                sizeof(size_t), &work_group_size, &ret_sz
            );

            if (error_code)
            {
                if (context->selected_device != device_list[j])
                    clReleaseDevice(device_list[j]);
                continue;
            }

            error_code = clGetDeviceInfo(
                device_list[j], CL_DEVICE_NAME, 63, device_name, &ret_sz
            );

            device_name[ret_sz] = '\0';
            fprintf(
                stderr,
                "Found device \"%s\": mem type %s, max workgroup size %zu\n",
                device_name, get_mem_type_str(cur_mem_type), work_group_size
            );

            if ((mem_type != CL_LOCAL && cur_mem_type == CL_LOCAL)
                || (cur_mem_type == mem_type
                    && max_work_group_size < work_group_size))
            {
                clReleaseDevice(context->selected_device);
                context->selected_device = device_list[j];
                max_work_group_size = work_group_size;
                mem_type = cur_mem_type;
            }
            else if (context->selected_device != device_list[j])
                clReleaseDevice(device_list[j]);
        }
    }

    if (!context->selected_device)
    {
        free(platforms);
        return error_code;
    }
    else
    {
        size_t ret_sz;
        clGetDeviceInfo(
            context->selected_device, CL_DEVICE_NAME, 63, device_name, &ret_sz
        );
        device_name[ret_sz] = '\0';

        fprintf(stderr, "Selected device: %s\n", device_name);
    }

    context->context = clCreateContext(
        0, 1, &context->selected_device, 0, 0, &error_code
    );

    if (error_code)
    {
        release_gpu_context(context);
        return error_code;
    }

    context->command_queue = clCreateCommandQueue(
        context->context, context->selected_device, CL_QUEUE_PROFILING_ENABLE,
        &error_code
    );

    if (error_code)
    {
        release_gpu_context(context);
        return error_code;
    }

    return 0;
}

/// Loads and compile the kernel for the specified \ref gpu_context
cl_int load_program(struct gpu_context* context,
                    char const** sources_list, size_t src_list_sz,
                    char const* kernel_name)
{
    assert(context);
    assert(context->selected_device);
    assert(context->context);

    char const* file_data[src_list_sz];
    size_t lens[src_list_sz];
    cl_int result = 0;

    memset(file_data, 0, sizeof(char const*) * src_list_sz);

    for (size_t i = 0; i < src_list_sz; ++i)
    {
        file_data[i] = load_source_file(sources_list[i], &lens[i]);
        if (!file_data[i])
        {
            result = -1;
            goto return_error;
        }
    }

    context->program = clCreateProgramWithSource(
        context->context, src_list_sz, file_data, lens, &result
    );
    CHECK_ERR("Failed to create clProgram", result, return_error);

    result = clBuildProgram(
        context->program, 1, &context->selected_device, "", 0, 0
    );

    if (result)
    {
        fprintf(stderr, "kernel compilation failed\n");
        size_t log_len = 0;
        cl_int saved_error_code = result;
        char* build_log;

        result = clGetProgramBuildInfo (
            context->program, context->selected_device,
            CL_PROGRAM_BUILD_LOG, 0, 0, &log_len
        );
        CHECK_ERR("Failed to retrieve build's log", result, return_error);

        build_log = malloc(log_len);
        result = clGetProgramBuildInfo(
            context->program, context->selected_device,
            CL_PROGRAM_BUILD_LOG, log_len, build_log, &log_len
        );

        if (result)
        {
            fprintf(stderr, "Failed to retrieve build's log: %d", result);
            free(build_log);
            goto return_error;
        }

        fprintf(stderr, "Kernel compilation log:\n%s\n", build_log);
        result = saved_error_code;
        goto return_error;
    }

return_error:
    for (size_t i = 0; i < src_list_sz; ++i)
        free((void*) file_data[i]);
    return result;
}

/// Setups kernel & kernel structs like mem buffers for the \ref gpu_context
cl_int setup_kernel(struct gpu_context* context,
                    char const* kernel_name)
{
    size_t n = context->n;
    size_t m = context->m;
    size_t k = context->k;

    cl_int result = 0;

    assert(context);
    assert(context->selected_device);
    assert(context->context);
    assert(context->command_queue);

    context->kernel = clCreateKernel(context->program, kernel_name, &result);
    CHECK_AND_RET_ERR("Failed to create kernel", result);

    context->fst_mattr_buff_in = clCreateBuffer(
        context->context, CL_MEM_READ_ONLY, n * m * sizeof(float), 0, &result
    );
    CHECK_ERR("Error creating buffer", result, release_kernel);

    context->sec_mattr_buff_in = clCreateBuffer(
        context->context, CL_MEM_READ_ONLY, m * k * sizeof(float), 0, &result
    );
    CHECK_ERR("Error creating buffer", result, release_mem1);

    context->thr_mattr_buff_out = clCreateBuffer(
        context->context, CL_MEM_READ_WRITE, n * k * sizeof(float), 0, &result
    );
    CHECK_ERR("Error creating buffer", result, release_mem2);

    clSetKernelArg(context->kernel, 0, sizeof(cl_mem), &context->fst_mattr_buff_in);
    clSetKernelArg(context->kernel, 1, sizeof(cl_mem), &context->sec_mattr_buff_in);
    clSetKernelArg(context->kernel, 2, sizeof(cl_mem), &context->thr_mattr_buff_out);
    clSetKernelArg(context->kernel, 3, sizeof(cl_uint), &context->n);
    clSetKernelArg(context->kernel, 4, sizeof(cl_uint), &context->m);
    clSetKernelArg(context->kernel, 5, sizeof(cl_uint), &context->k);

    return 0;

release_mem2:
    clReleaseMemObject(context->sec_mattr_buff_in);
    context->sec_mattr_buff_in = NULL;
release_mem1:
    clReleaseMemObject(context->fst_mattr_buff_in);
    context->fst_mattr_buff_in = NULL;
release_kernel:
    clReleaseKernel(context->kernel);
    context->kernel = NULL;
    return result;
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

    context->n = n;
    context->m = m;
    context->k = k;

    *error = select_device(context);
    if (*error)
        goto return_error;

    *error = load_program(context, sources_list, src_list_sz, kernel_name);
    if (*error)
        goto return_error;

    *error = setup_kernel(context, kernel_name);
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

    data->n = n;
    data->m = m;
    data->k = k;

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

void validate_result(struct input_data* data)
{
    float* const gold = (float*) calloc(data->out_C_size, sizeof(float));
    fprintf(stderr, "Validating results...\n");
    #pragma omp parallel for
    for (size_t i = 0; i < data->n; ++i)
        for (size_t j = 0; j < data->m; ++j)
            for (size_t l = 0; l < data->k; ++l)
                gold[i * data->k + l] += data->in_A[i * data->m + j]
                                         * data->in_B[j * data->k + l];

    for (size_t i = 0; i < data->n; ++i)
        for (size_t l = 0; l < data->k; ++l)
        {
            float delta = gold[i * data->k + l] - data->out_C[i * data->k + l];
            float abs_delta = fabsf(delta);
            assert(abs_delta < 0.05);
        }

    free(gold);
}

int main()
{
    /// n, m, k are expected to be divisible by tile_size.
    size_t const n = 2048;
    size_t const m = 512;
    size_t const k = 1024;

    char const* const   kernel_name = "gemm4";
    char const* const   sources_list[] =
    {
        "const.h",
        "gemm4.cl"
    };

    int exit_code = 0;
    cl_int error_code;

    struct gpu_context* context = setup_gpu_context(
        n, m, k, sources_list, sizeof(sources_list) / sizeof(char const*),
        kernel_name, &error_code
    );
    CHECK_AND_RET_ERR("startup failed", error_code);

    struct input_data* data = generate_input(n, m, k);
    if (!data)
    {
        fprintf(stderr, "Input generation failed!\n");
        return -1;
    }

    error_code = clEnqueueWriteBuffer(
        context->command_queue, context->fst_mattr_buff_in, true, 0,
        n * m * sizeof(float), data->in_A, 0, 0, 0
    );
    CHECK_ERR("clEnqueueWriteBuffer error", error_code, return_error);
    error_code = clEnqueueWriteBuffer(
        context->command_queue, context->sec_mattr_buff_in, true, 0,
        m * k * sizeof(float), data->in_B, 0, 0, 0
    );
    CHECK_ERR("clEnqueueWriteBuffer error", error_code, return_error);

    size_t work_size[] = {k, n / ELEMS_PER_THREAD};
    size_t local_group_size[] = {TILE_SIZE, TILE_SIZE / ELEMS_PER_THREAD};
    cl_event run_event;
    error_code = clEnqueueNDRangeKernel (
            context->command_queue, context->kernel, 2, NULL,
            work_size, local_group_size, 0, 0, &run_event
    );
    CHECK_ERR("Error enquing kernel", error_code, return_error);
    clEnqueueReadBuffer(
        context->command_queue, context->thr_mattr_buff_out, true, 0,
        n * k * sizeof(float), data->out_C, 0, 0, 0
    );

    validate_result(data);

    cl_ulong t_start = 0, t_end = 0;
    clGetEventProfilingInfo(run_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, 0);
    clGetEventProfilingInfo(run_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, 0);

    long double elapsed_time = t_end - t_start;
    long double ops = (long double) n * m * k * 2;

    printf("%.4Lf ms elapsed and ", elapsed_time / 1e6);
    printf("achieved %.4Lf TFlops\n", ops / elapsed_time / 1e3);

return_error:
    release_gpu_context(context);
    release_input_data(data);
    return exit_code;
}
