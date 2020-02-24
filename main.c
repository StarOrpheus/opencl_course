#include <stdbool.h>
#include <assert.h>
#include <stdio.h>

#include <CL/opencl.h>

#ifndef CHECH_ERR
#define CHECK_ERR(intro, result, exit_label)        \
do {                                                \
    if ((result) < 0)                               \
    {                                               \
        exit_code = -1;                             \
        goto exit_label;                            \
    }                                               \
} while (false)
#endif



int main()
{
    char const* const kernel_file_name = "a_plus_b.cl";
    char const* const kernel_name = "a_plus_b";
    size_t const file_load_sz = 1024 * 16;

    int exit_code = 0;
    cl_int error_code;
    cl_uint num_platforms;
    error_code = clGetPlatformIDs(0, 0, &num_platforms);
    CHECK_ERR("Error getting platforms list", error_code, exit0);

    cl_platform_id * const platforms
        = (cl_platform_id *) malloc(num_platforms * sizeof(cl_platform_id));

    error_code = clGetPlatformIDs(num_platforms, platforms, &num_platforms);
    CHECK_ERR("Error getting platforms list", error_code, exit1);

    cl_uint         num_devices = 0;
    error_code = clGetDeviceIDs (
            platforms[0], CL_DEVICE_TYPE_GPU,  // using platform[0] as default platform
            0, 0, &num_devices
    );
    CHECK_ERR("Error getting device list", error_code, exit1);

    cl_device_id * const gpu_devices
        = (cl_device_id *) malloc(num_devices * sizeof(cl_device_id));

    error_code = clGetDeviceIDs (
            platforms[0], CL_DEVICE_TYPE_GPU,
            num_devices, gpu_devices, &num_devices
    );

    CHECK_ERR("Error getting device list", error_code, exit2);

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
    CHECK_ERR("Error getting device name", error_code, exit2);

    printf("Target device name: %s\n", device_name);

    cl_context context = clCreateContext(0, 1, gpu_devices, 0, 0, &error_code);
    CHECK_ERR("Error creating context", error_code, exit2);

    cl_command_queue queue = clCreateCommandQueue (
            context, gpu_devices[0], 0, &error_code
    );
    CHECK_ERR("Error creating command queue", error_code, exit2);

    FILE* kernel_file = fopen(kernel_file_name, "r");
    if (!kernel_file)
    {
        perror("Error opening kernel file");
        goto exit2;
    }

    char* program_code = malloc(file_load_sz);
    size_t code_len = fread(program_code, 1, file_load_sz, kernel_file);
    program_code[file_load_sz - 1] = '\0';

    cl_program program = clCreateProgramWithSource (
            context, 1,
            &program_code, &code_len, &error_code
    );
    CHECK_ERR("Error creating program:", error_code, exit3);

    error_code = clBuildProgram(program, 1, gpu_devices, "", 0, 0);
    if (error_code)
    {
        size_t log_len = 0;
        error_code = clGetProgramBuildInfo (
                program, gpu_devices[0],
                CL_PROGRAM_BUILD_LOG, 0, 0, &log_len
        );
        CHECK_ERR("Error getting build log", error_code, exit3);
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
        goto exit3;
    }

    cl_kernel kernel = clCreateKernel(program, kernel_name, &error_code);
    CHECK_ERR("Error creating kernel", error_code, exit3);

    cl_int a, b, c;
    puts("Enter two numbers: ");
    scanf("%d %d", &a, &b);

    cl_mem mem1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(a), 0, &error_code);
    CHECK_ERR("Error creating buffer", error_code, exit3);
    cl_mem mem2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(b), 0, &error_code);
    CHECK_ERR("Error creating buffer", error_code, exit3);
    cl_mem mem3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(c), 0, &error_code);
    CHECK_ERR("Error creating buffer", error_code, exit3);

    clEnqueueWriteBuffer(queue, mem1, false, 0, sizeof(a), &a, 0, 0, 0);
    clEnqueueWriteBuffer(queue, mem2, false, 0, sizeof(b), &b, 0, 0, 0);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem3);

    size_t work_offset = 0;
    size_t work_size = 1;
    cl_event run_event;
    clEnqueueNDRangeKernel(queue, kernel, 1, &work_offset, &work_size, 0, 0, 0, &run_event);
    clEnqueueReadBuffer(queue, mem3, true, 0, sizeof(cl_int), &c, 0, 0, 0);
    cl_ulong t_start = 0, t_end = 0;
    clGetEventProfilingInfo(run_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, 0);
    clGetEventProfilingInfo(run_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, 0);

    printf("%d + %d = %d\n", a, b, c);
    printf("%lu ns elapsed\n", t_end - t_start);

exit3:
    free(program_code);
exit2:
    free(gpu_devices);
exit1:
    free(platforms);
exit0:
    return exit_code;
}
