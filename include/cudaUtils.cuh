#ifndef CUDAUTILS_CUH
#define CUDAUTILS_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <defines.h>

namespace gpu
{
#define THREADS_PER_BLOCK 128
#define BLOCK_COLS 32
#define BLOCK_ROWS (THREADS_PER_BLOCK / BLOCK_COLS)
#define STEP_1 65536
#define STEP_2 409600
#define ROLL_1 1
#define ROLL_2 4
#define ROLL_3 16

#define CREATE_CUB_DATA(x) cub_data x = CUB_INIT
#define CREATE_CUBLAS_DATA(x) cublas_data x = CUBLAS_INIT
#define CU_ERR_CHECK(err) cu_err_check(err, __FILE__, __LINE__)
#define CU_LAST_ERR_CHECK() cu_last_err_check(__FILE__, __LINE__)
#define CU_MALLOC(ptr_dir, size) cu_malloc(ptr_dir, size, __FILE__, __LINE__)
#define CU_MALLOC_HOST(ptr_dir, size) cu_malloc_host(ptr_dir, size, __FILE__, __LINE__)
#define CU_COPY(dst, src, size, kind) cu_copy(dst, src, size, kind, __FILE__, __LINE__)
#define CU_COPY_ASYNC(dst, src, size, kind, stream) cu_copy_async(dst, src, size, kind, stream, __FILE__, __LINE__)
#define CU_FREE(ptr) cu_free(ptr, __FILE__, __LINE__)
#define CU_FREE_HOST(ptr) cu_free_host(ptr, __FILE__, __LINE__)
#define CUBLAS_ERR_CHECK(err) cublas_err_check(err, __FILE__, __LINE__)

    typedef std::vector<cudaStream_t> stream_pack;

    typedef struct
    {
        float *d_result;
        void *d_temp_storage;
        size_t temp_storage_bytes;
        size_t size;
    } cub_data;

#define CUB_INIT                   \
    {                              \
        .d_result = nullptr,       \
        .d_temp_storage = nullptr, \
        .temp_storage_bytes = 0,   \
        .size = 0                  \
    }

    typedef struct
    {
        float *one; // cuda constants
        float *cero;
        cublasHandle_t handle;
    } cublas_data;

#define CUBLAS_INIT       \
    {                     \
        .one = nullptr,   \
        .cero = nullptr,  \
        .handle = nullptr \
    }

    // single ops
    void inv_var(float *var, cudaStream_t stream);
    void neg_var(float *var, cudaStream_t stream);
    void set_var(float *var, float val, cudaStream_t stream);
    void set_inv(float *var, float val, cudaStream_t stream);
    void mul_var(float *var, float val, cudaStream_t stream);
    void add_var(float *var, float val, cudaStream_t stream);

    // errors
    void cu_err_check(cudaError_t &err, const char *file, int line);
    void cu_last_err_check(const char *file, int line);

    // memory
    template <class T>
    void cu_malloc(T **data, size_t size, const char *file, int line)
    {
        cudaError_t err = cudaMalloc(data, size * sizeof(T));

        if (err != cudaSuccess)
        {
            std::cout << RED << "failed to allocate memory, error " << cudaGetErrorString(err) << ", file: " << file << ", line: " << line << RESET << "\n";
            throw std::runtime_error("cu_malloc");
        }
    }

    template <>
    void cu_malloc<void>(void **data, size_t size, const char *file, int line); // void specialization

    template <class T>
    void cu_malloc_host(T **data, size_t size, const char *file, int line)
    {
        cudaError_t err = cudaMallocHost(data, size * sizeof(T));

        if (err != cudaSuccess)
        {
            std::cout << RED << "failed to allocate pinned host memory, error " << cudaGetErrorString(err) << ", file: " << file << ", line: " << line << RESET << "\n";
            throw std::runtime_error("cu_malloc_host");
        }
    }
    template <>
    void cu_malloc_host(void **data, size_t size, const char *file, int line); // void specialization

    template <class T>
    void cu_copy(T *dst, const T *src, size_t size, cudaMemcpyKind kind, const char *file, int line)
    {
        cudaError_t err = cudaMemcpy(dst, src, size * sizeof(T), kind);

        if (err != cudaSuccess)
        {
            std::cout << RED << "failed to copy memory, error " << cudaGetErrorString(err) << ", file: " << file << ", line: " << line << RESET << "\n";
            throw std::runtime_error("cu_copy");
        }
    }

    template <class T>
    void cu_copy_async(T *dst, const T *src, size_t size, cudaMemcpyKind kind, cudaStream_t stream, const char *file, int line)
    {
        cudaError_t err = cudaMemcpyAsync(dst, src, size * sizeof(T), kind, stream);

        if (err != cudaSuccess)
        {
            std::cout << RED << "failed to async copy memory, error " << cudaGetErrorString(err) << ", file: " << file << ", line: " << line << RESET << "\n";
            throw std::runtime_error("cu_copy_async");
        }
    }

    void cu_copy_kernel(float *dst, const float *src, int size, cudaStream_t stream);

    template <class T>
    void cu_free(T *data, const char *file, int line)
    {
        cudaError_t err = cudaSuccess;
        err = cudaFree(data);

        if (err != cudaSuccess)
        {
            std::cout << RED << "failed to release memory, error " << cudaGetErrorString(err) << ", file: " << file << ", line: " << line << RESET << "\n";
            throw std::runtime_error("cu_free");
        }
    }

    template <class T>
    void cu_free_host(T *data, const char *file, int line)
    {
        cudaError_t err = cudaSuccess;
        err = cudaFreeHost(data);

        if (err != cudaSuccess)
        {
            std::cout << RED << "failed to release pinned host memory, error " << cudaGetErrorString(err) << ", file: " << file << ", line: " << line << RESET << "\n";
            throw std::runtime_error("cu_free_host");
        }
    }

    typedef struct
    {
        float free_mem;
        float used_mem;
    } mem_stats;

    mem_stats mem_stats_mb();
    cudaStream_t create_stream();
    void destroy_stream(cudaStream_t stream);
    void sync_device();
    void sync_stream(cudaStream_t stream);

    // cublas
    void cublas_init(cublas_data &cublas, cudaStream_t stream);
    void cublas_free(cublas_data &cublas);
    void cublas_err_check(cublasStatus_t &err, const char *file, int line);

    // cub
    void cub_init(cub_data &data);
    void cub_free(cub_data &data);

    template <class T, class U>
    void cub_prepare(cub_data &cub, const float *__restrict__ data, int size, cudaStream_t stream)
    {
        if (cub.size != size)
        {
            cub.temp_storage_bytes = 0;
            CU_FREE(cub.d_temp_storage);
            cub.d_temp_storage = nullptr;

            cudaError_t err = cudaSuccess;
            err = U::Reduce(cub.d_temp_storage, cub.temp_storage_bytes, data, cub.d_result, size, T(), 0, stream);
            CU_ERR_CHECK(err);
            CU_MALLOC(&cub.d_temp_storage, cub.temp_storage_bytes);
            cub.size = size;
        }
    }

    // print
    template <class T>
    void print(const T *data, int size, cudaStream_t stream)
    {
        sync_stream(stream);
        std::vector<T> copy((size_t)size);
        CU_COPY(copy.data(), data, size, cudaMemcpyDeviceToHost);

        for (auto &i : copy)
            std::cout << i << " ";

        std::cout << "\n";
    }

    template <class T>
    void print_matrix(const T *data, int rows, int cols, cudaStream_t stream)
    {
        sync_stream(stream);
        std::vector<T> copy((size_t)rows * cols);
        CU_COPY(copy.data(), data, rows * cols, cudaMemcpyDeviceToHost);
        T *d = copy.data();

        for (int i = 0; i < rows; i++)
        {
            std::cout << "row " << i << "\n";
            for (int j = 0; j < cols; j++)
                std::cout << d[i * cols + j] << " ";
            std::cout << "\n";
        }

        std::cout << "\n";
    }

}

#endif