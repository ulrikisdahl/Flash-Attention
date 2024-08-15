#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h> 
#include <vector>
#include <iostream>
#include <assert.h>
#include <ATen/Dispatch.h>

// #define CUDA_CHECK_ERROR()  
// do { 
//     cudaError_t err = cudaGetLastError();  
//     if (err != cudaSuccess) { 
//         std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;  
//         exit(EXIT_FAILURE); 
//     } 
// } while (0)

/**
 * @brief computes the l_ij, m_i_new, l_i_new statistics
 */
template <typename scalar_t>
__device__ 
void kernel_compute_statistics(
    scalar_t* scores,
    scalar_t* local_rowmax,
    scalar_t global_rowmax_old,
    scalar_t global_rowsum_old,
    scalar_t* global_rowmax_new,
    scalar_t* global_rowsum_new,
    int num_rows_per_tile,
    int thread_idx,
    int local_row_idx,
    int dimension)
{
    if(thread_idx % dimension == 0){ 
        //compute rowsums for S_ij
        float l_ij = 0.0f;
        float m_i_new = 0.0f;
        float l_i_new = 0.0f;
        for (int i = 0; i < num_rows_per_tile; i++){
            l_ij += scores[local_row_idx * num_rows_per_tile + i]; //l_ij doenst need to be written to SRAM
        }
        //compute new global rowmax statistics 
        m_i_new = fmax(global_rowmax_old, local_rowmax[local_row_idx]);
        global_rowmax_new[local_row_idx] = m_i_new; //Reuse the shared memory allocated to Q_i, since we dont use it anymore after computing S_ij

        //compute new global rowsum statistics
        l_i_new = expf(global_rowmax_old - m_i_new) * global_rowsum_old + expf(local_rowmax[local_row_idx] - m_i_new) * l_ij; 
        global_rowsum_new[local_row_idx] = l_i_new;  
    }
}

/**
 * @brief Each thread (within threshold) computes a rowmax
 */
template <typename scalar_t>
__device__
void kernel_reduction_max( //TODO: not actually reduction yet
    scalar_t* scores, 
    scalar_t* local_rowmax, 
    int num_rows_per_tile, 
    int dimension,  
    int thread_idx,
    int block_idx)
{
    if(thread_idx < num_rows_per_tile){ //S_ij is square, so num_rows_per_tile accounts for both row and column dimension
        float max_val = -INFINITY;
        for (int i = 0; i < num_rows_per_tile; i++){
            auto s_ij = scores[thread_idx * num_rows_per_tile + i];
            max_val = fmax(max_val, s_ij);
        }
        local_rowmax[thread_idx] = max_val; //Resue Q_i allocated memory
    }   
}

/**
 * @brief Computes the inner product between Q_i and K^T_j
 */
template <typename scalar_t>
__device__
void inner_product_matmul(
    scalar_t* Q_i, 
    scalar_t* K_j, 
    scalar_t* scores, 
    int num_rows_per_block,
    int dimension, 
    int thread_idx, 
    int thread_idx_limit,
    float scaling_factor)
{
    if (thread_idx < thread_idx_limit){
        //each threads computes one output value
        float temp = 0.0f;
        int local_matrix_row_index = thread_idx / num_rows_per_block;
        for(int k = 0; k < dimension; k++){
            temp += Q_i[local_matrix_row_index * dimension + k] * K_j[(thread_idx % num_rows_per_block) * dimension + k]; //Q_i * K^T_j
        }
        scores[thread_idx] = scaling_factor * temp;
    }
}

/**
 * @brief Each thread creates a element of the output product
 */
template <typename scalar_t>
__device__
float outer_product_matmul(
    scalar_t* scores,
    scalar_t* V_j, 
    int num_rows_per_block,
    int dimension,
    int thread_idx,
    int thread_idx_limit
    )
{
    if(thread_idx < thread_idx_limit){ //TODO: fix edge case for when last tile does not have same amount of rows
        float temp = 0.0f;
        for (int k = 0; k < num_rows_per_block; k++){
            temp += scores[(thread_idx / dimension) * num_rows_per_block + k] * V_j[k * dimension + (thread_idx % dimension)];
        }
        return temp;
    };
    return 0.0f;
}

/**
 * From: https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
 */
template <typename T>
__device__
T* shared_memory_proxy()
{
    extern __shared__ unsigned char memory[];
    return reinterpret_cast<T*>(memory);
}


/**
 * @brief Forward attention kernel
 */
template <typename scalar_t>
__global__ 
void forward_attention_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> query,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> key,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> value,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> outputs,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rowmax_statistics, 
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rowsum_statistics,
    int batch_size, int sequence_length, int dimension,
    int block_size,
    int num_rows_per_block,
    int num_blocks_per_sample)
{
    //SRAM
    auto sharedMemory = shared_memory_proxy<scalar_t>();
    scalar_t* Q_i = &sharedMemory[0];
    scalar_t* K_j = Q_i + block_size;
    scalar_t* V_j = Q_i + 2*block_size; 
    scalar_t* scores = &sharedMemory[0]; //Reuse Q_i allocated SRAM space
    scalar_t* local_rowmax = scores + num_rows_per_block * num_rows_per_block; 
    scalar_t* global_rowmax_new = local_rowmax + num_rows_per_block; 
    scalar_t* global_rowsum_new = global_rowmax_new + num_rows_per_block; 
    
    //compute indexes
    int batch_idx = blockIdx.x; 
    int local_row_idx = threadIdx.x / dimension; 
    int col_idx = threadIdx.x % dimension; // global_col_idx == local_col_idx in this sense

    //scaling factor
    float scaling_factor = 1.0f / (sqrtf(static_cast<float>(dimension)));

    if(batch_idx < batch_size && local_row_idx < num_rows_per_block){ 
        for(int j = 0; j < num_blocks_per_sample; j++){
            //Load K_j, V_j to SRAM
            K_j[threadIdx.x] = key[batch_idx][j * num_rows_per_block + local_row_idx][col_idx]; // K_j
            V_j[threadIdx.x] = value[batch_idx][j * num_rows_per_block + local_row_idx][col_idx]; // V_j - Not very coalessed for when we do our matmuls later.... 

            for(int i = 0; i < num_blocks_per_sample; i++){ //i gives us which tile we are on for Q along the row-axis

                int global_row_idx_i = i * num_rows_per_block + local_row_idx;

                //Load Q_i, m_i, l_i to SRAM - O_i is unecessary 
                Q_i[threadIdx.x] = query[batch_idx][global_row_idx_i][col_idx]; 
                
                //Compute attention scores Q_i*K^T_j
                __syncthreads(); //necessary because utilized threads all come from the first row in the tile, but some of them operate on values from other rows in the tile
                inner_product_matmul(Q_i, K_j, scores, num_rows_per_block, dimension, threadIdx.x, num_rows_per_block * num_rows_per_block, scaling_factor); 
                // __syncthreads(); 
 
                //compute statistics - brute force it for now...
                kernel_reduction_max(scores, local_rowmax, num_rows_per_block, dimension, threadIdx.x, blockIdx.x);
                __syncthreads();

                if(threadIdx.x < num_rows_per_block*num_rows_per_block){ 
                    scores[threadIdx.x] = expf(scores[threadIdx.x] - local_rowmax[threadIdx.x / num_rows_per_block]); //P_ij 
                }
                __syncthreads();

                auto global_rowmax_old = rowmax_statistics[batch_idx][global_row_idx_i];
                auto global_rowsum_old = rowsum_statistics[batch_idx][global_row_idx_i]; 
                kernel_compute_statistics(scores, local_rowmax, global_rowmax_old, global_rowsum_old, global_rowmax_new, global_rowsum_new, num_rows_per_block, threadIdx.x, local_row_idx, dimension); //pretty sure its good
                __syncthreads();
            
                //compute attention outputs (from here on out its all element-wise so we dont need to sync threads)
                auto m_i_new = global_rowmax_new[local_row_idx];
                scalar_t old_output_adjusted = (global_rowsum_old * expf(global_rowmax_old - m_i_new)) * outputs[batch_idx][global_row_idx_i][col_idx]; 
                scalar_t local_attention_adjusted = outer_product_matmul(scores, V_j, num_rows_per_block, dimension, threadIdx.x, num_rows_per_block * dimension); //TODO: num_rows_per_block*dimension doesnt account for edge_case of non-divisible total-rows
                local_attention_adjusted = expf(local_rowmax[local_row_idx] - m_i_new) * local_attention_adjusted; 

                //Write to global memory (HBM)
                outputs[batch_idx][global_row_idx_i][col_idx] = (1 / (global_rowsum_new[local_row_idx])) * (old_output_adjusted + local_attention_adjusted); 
                if(threadIdx.x < num_rows_per_block){
                    rowmax_statistics[batch_idx][i * num_rows_per_block + threadIdx.x % num_rows_per_block] = global_rowmax_new[threadIdx.x];
                    rowsum_statistics[batch_idx][i * num_rows_per_block + threadIdx.x % num_rows_per_block] = global_rowsum_new[threadIdx.x]; 
                }
            }
        }
    }
}



std::vector<torch::Tensor> forward_attention(torch::Tensor query, torch::Tensor key, torch::Tensor value)
{
    constexpr int num_threads_per_block = 1024;
    int B = query.sizes()[0];
    int N = query.sizes()[1];
    int D = query.sizes()[2];
    int num_rows_per_block = num_threads_per_block / D; 
    int num_blocks_per_sample = ceil(N / num_rows_per_block);
    dim3 GRID(B); 
    int BLOCK_SIZE = num_rows_per_block * D;

    assert(num_rows_per_block * D > num_rows_per_block * num_rows_per_block + 3 * num_rows_per_block);
    assert(BLOCK_SIZE <= num_threads_per_block);
    assert(num_rows_per_block < D);
    assert(D <= num_threads_per_block);
    if (N % num_rows_per_block != 0){
        printf("Warning: The number of rows in a sample is not divisble by the number of rows per tile. This will lead to idle threads!");
    }

    torch::Tensor outputs = torch::zeros({B, N, D}, query.options());
    torch::Tensor rowmax_statistics = torch::full({B, N}, -INFINITY, query.options());
    torch::Tensor rowsum_statistics = torch::zeros({B, N}, query.options()); //l_i
    torch::Tensor debug_out = torch::zeros({B, N, D}, query.options()).to(torch::kInt);

    AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "forward_attention", ([&]{
        forward_attention_kernel<<<GRID, num_threads_per_block, 3 * BLOCK_SIZE * sizeof(float)>>>(
            query.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            key.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            value.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            outputs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), 
            rowmax_statistics.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), 
            rowsum_statistics.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), 
            B, N, D,
            BLOCK_SIZE,
            num_rows_per_block,
            num_blocks_per_sample
        );
    }));

    return {outputs, rowmax_statistics, rowsum_statistics};
}


// int main(){

//     torch::Tensor query = torch::rand({32, 32, 256}, torch::device(torch::kCUDA));
//     torch::Tensor key = torch::rand({32, 32, 256}, torch::device(torch::kCUDA));
//     torch::Tensor value = torch::rand({32, 32, 256}, torch::device(torch::kCUDA));

//     cudaDeviceSynchronize();

//     auto result = forward_attention(query, key, value);
//     float x = expf(-INFINITY - 0.2);
//     float y = expf(-0.2);
//     float z = expf(0.2);

//     cudaDeviceSynchronize();
//     // CUDA_CHECK_ERROR();

//     return 0;
// }

