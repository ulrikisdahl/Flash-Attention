#include <torch/extensions.h>
#include <vector>
#include <iostream>
#include <assert.h>

constexpr int SRAM_SIZE = 49152; //Shared memory size per thread block for the Ada Lovelace architecture


/**
 * @brief computes the l_ij, m_i_new, l_i_new statistics
 */
__device__ 
void kernel_compute_statistics(
    int* scores,
    int* local_rowmax,
    float global_rowmax_old,
    float global_rowsum_old,
    int* global_rowmax_new,
    int* global_rowsum_new,
    int num_rows_per_tile,
    int thread_idx)
{
    if(thread_idx < num_rows_per_tile){
        //compute rowsums for S_ij
        float l_ij = 0;
        for (int i = 0; i < num_rows_per_tile; i++){
            l_ij += scores[thread_idx * num_rows_per_tile + i]; //l_ij doenst need to be written to SRAM
        }

        //compute new global rowmax statistics 
        m_i_new = max(global_rowmax_old, local_rowmax[thread_idx]);
        global_rowmax_new[thread_idx] = m_i_new; //Reuse the shared memory allocated to Q_i, since we dont use it anymore after computing S_ij

        //compute new global rowsum statistics
        l_i_new = exp(global_rowmax_old - m_i_new) * global_rowsum_old + exp(local_rowmax[thread_idx] - m_i_new) * l_ij; 
        global_rowsum_new[thread_idx] = l_i_new;  

        //save old global statistics
        // global_rowmax_old[thread_idx] = m_i_old;
        // global_rowsum_old[thread_idx] = l_i_old; 
    }
}

/**
 * @brief Each thread (within threshold) computes a rowmax
 * @param sram_offset offset to the S_ij matrix in shared memory
 */
__device__
void kernel_reduction_max( //TODO: optimize
    int* scores, 
    int* local_rowmax, 
    int num_rows_per_tile, 
    int dimension,  
    int thread_idx)
{
    if(thread_idx < num_rows_per_tile){ //S_ij is square, so num_rows_per_tile accounts for both row and column dimension
        float max_val = -INFINITY;
        for (int i = 0; i < num_rows_per_tile; i++){
            auto s_ij = scores[thread_idx * num_rows_per_tile + i];
            max_val = max_val >= s_ij ? max_val : s_ij;
        }
        local_rowmax[thread_idx] = max_val; //Resue Q_i allocated memory
    }   
}

/**
 * @brief Takes in a shared memory and computes a matrix multiplication on two matrices A and B inside the shared memory given offsets A_SRAM_offset and B_SRAM_offset
 * @param sharedMemory
 * @param A_SRAM_offset
 * @param B_SRAM_offset
 * @param C_SRAM_offset
 * @param dimension number elemetns along the shared dimension of A and B
 * @param thread_idx
 * @param thread_idx_limit address for final threadIdx.x that is a part of the computation
 */
__device__
void inner_product_matmul(
    int* Q_i, 
    int* K_j, 
    int* scores, 
    int dimension, 
    int thread_idx, 
    int thread_idx_limit)
{
    if (thread_idx < thread_idx_limit){
        //each threads computes one output value
        float temp = 0;
        int local_matrix_row_index = thread_idx / (sqrtf32(thread_idx_limit)); //TODO
        for(int k = 0; k < dimension; k++){
            temp += Q_i[local_matrix_row_index * dimension + k] * K_j[(thread_idx % dimension) * dimension + k]; 
        }
        scores[thread_idx] = temp;
    }
}

/**
 * @brief Each thread creates a element of the output product
 */
__device__
void outer_product_matmul(
    int* scores,
    int* V_j, 
    int num_rows_per_block,
    int dimension,
    int thread_idx,
    int thread_idx_limit,
    )
{
    if(thread_idx < thread_idx_limit){ //TODO: fix edge case for when last tile does not have same amount of rows
        float temp = 0;
        for (int k = 0; k < thread_limit){
            temp += scores[(thread_idx / dimension) * num_rows_per_block + k] * V_j[k * dimension + thread_idx];
        }
        return temp;
    }
}

/**
 * From: https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
 */
template <typename T>
__device__
T* shared_memory_proxy()
{
    __shared__ unsigned char memory[SRAM_SIZE];
    return reinterpret_cast<T*>(memory);
}


/**
 * threadIdx.x gives the thread index
 * tile == block
 */
template <typename scalar_t>
__global__ 
void forward_attention_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> query,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> key,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> value,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> outputs,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rowsum_statistics,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rowmax_statistics, 
    int batch_size, int sequence_length, int dimension,
    int block_size,
    int num_rows_per_block,
    int num_blocks_per_sample)
{
    //SRAM
    auto sharedMemory = shared_memory_proxy<scalar_t>(); //Need to be dynamic     <-----
    float* Q_i = &sharedMemory[0];
    float* K_j = &sharedMemory[block_size];
    float* V_j = &sharedMemory[2*block_size];
    float* scores = &sharedMemory[3*block_size]; //[3*block_size, 3*block_size + num_rows**2)
    float* local_rowmax = &sharedMemory[0];
    float* global_rowmax_new = &sharedMemory[num_rows_per_block];
    float* global_rowsum_new = &sharedmemory[2*num_rows_per_block];
    
    //compute indexes
    int batch_idx = blockIdx.x / num_blocks_per_sample; 
    int tile_idx = blockidx.x % num_blocks_per_sample; //the tile number within a sample
    int global_row_idx = tile_idx * num_rows_per_block + (threadIdx.x / dimension); 
    int local_row_idx = threadIdx.x / dimension; 
    int col_idx = threadIdx.x % dimension; // global_col_idx == local_col_idx in this sense

    if(batch_idx < batch_size && tile_idx < num_blocks_per_sample && local_row_idx < num_rows_per_block && global_row_idx < N){ //Think (global_row_idx < N) should fix the edge case of total rows not being divisible by rows_per_block 
        for(int j = 0; j < num_blocks_per_sample; j++){
            //Load K_j, V_j to SRAM
            K_j[threadIdx.x] = key[batch_idx][j * num_rows_per_block * dimension + local_row_idx][col_idx]; // K_j
            V_j[threadIdx.x] = value[batch_idx][j * num_rows_per_block * dimension + local_row_idx][col_idx]; // V_j 
            __syncthreads(); //Necessary?

            for(int i = 0; i < num_blocks_per_sample; i++){ //i gives us which tile we are on for Q along the row-axis
                //Load Q_i, m_i, l_i to SRAM - O_i is unecessary 
                Q_i[threadIdx.x] = query[batch_idx][i * num_rows_per_block * dimension + local_row_idx][col_idx]; 
                
                //Compute attention scores Q_i*K^T_j
                kernel_matmul(Q_i, K_j, scores, dimension, threadIdx.x, num_rows_per_block * num_rows_per_block);
                __syncthreads(); 
 
                //compute statistics - brute force it for now...
                kernel_reduction_max(scores, local_rowmax, num_rows_per_block, dimension, threadIdx.x); 
                __syncthreads();

                if(threadIdx.x < num_rows_per_block*num_rows_per_block){ //might be faster to move this to kernel_reduction_max?
                    scores[threadIdx.x] = exp(scores[threadIdx.x] - local_rowmax[threadIdx.x / num_rows_per_block]); //P_ij 
                }
                __syncthreads(); 

                auto global_rowmax_old = rowmax_statistics[batch_idx][local_row_idx];
                auto global_rowsum_old = rowmax_statistics[batch_idx][local_row_idx]; 
                kernel_compute_statistics(scores, local_rowmax, global_rowmax_old, global_rowsum_old, global_rowmax_new, global_rowsum_new, num_rows_per_tile, threadIdx.x);
                __syncthreads();
            
                //compute attention outputs (from here on out its all element-wise so we dont need to sync threads)
                auto old_output_adjusted = (global_rowsum_old * exp(global_rowmax_old - global_rowmax_new[local_row_idx])) * outputs[batch_idx][global_row_idx][col_idx]; //  <------ Change so that we dont load O_i into SRAM (unesessary)
                auto local_attention_adjusted = outer_product_matmul(scores, V_j, num_rows_per_block, dimension, thread_idx, num_rows_per_block * dimension);
                local_attention_adjusted = exp(local_rowmax[local_row_idx] - global_rowmax_new[local_row_idx]) * local_attention_adjusted;   
                
                //Write to global memory (HBM)
                outputs[batch_idx][row_idx][col_idx] = (1 / global_rowsum_new[row_idx_local]) * (old_output_adjusted + local_attention_adjusted); //global row idx = (blockIdx.x % num_blocks_per_sampe) * num_rows_per_block + (threadIdx.x / dimension), where (blockIdx.x % num_blocks_per_sample) is the block/tile offset in the sample 
                if(threadIdx.x < dimension){
                    rowmax_statistics[batch_idx][threadIdx.x] = global_rowmax_new[threadIdx.x];
                    rowsum_statistics[batch_idx][threadIdx.x] = global_rowsum_new[threadIdx.x]; 
                }
            }
        }
    }
}

std::vector<torch::tensor> forward_attention_kernel(torch::Tensor query, torch::Tensor key, torch::Tensor value)
{
    int B, N, D = query.siezs()[0], query.sizes()[1], query.sizes()[2]; //Batch size, N sequences, Dimensions
    int BLOCK_ROWS = ceil(SRAM_SIZE / (3*D)); //how to avoid the threads that are not used accessing global out of bounds?
    BLOCK_ROWS = ceil(SRAM / (3*D + BLOCK_ROWS)); //???????
    int num_rows_per_block = N / BLOCK_COLS;
    int num_blocks_per_sample = ceil((N*D) / block_size); //have to round up incase there are "extra rows" in the input
    dim3 GRID(B * N * D); 

    assert(num_rows_per_block * dimension > 3 * num_rows_per_block);
    
    torch::Tensor outputs = torch::zeros({B, N, D}, query.options());
    // auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor rowsum_statistics = torch::zeros({B, N}, query.options()); //l_i
    torch::Tensor rowmax_statistics = torch::zeros({B, N}, query.options()); //m_i <---- SET TO -INF

    AT_DISTPATCH_FLOATING_TYPES(output_grad.scalar_type(), "forward_attention", [&]{
        forward_attention_kernel<<<GRID, BLOCK_SIZE, SRAM_SIZE>>>(
            query.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            key.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            value.pack_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            outputs.pack_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), 
            rowsum_statistics.pack_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), 
            rowmax_statistics.pack_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), 
            B, N, D,
            BLOCK_SIZE,
            num_rows_per_block,
            num_blocks_per_sample
        );
    })
}





//SRAM ranges
    //[0, block_size) = Q-tile
    //[block_size, 2*block_size) = K-tile
    //[2*block_size, 3*block_size) = V-tile
    //[3*block_size, 4*block_size) = Output-tile
    //[4*block_size, 4*block_size + num_rows_per_block**2) = S_ij
    //[4*block_size + num_rows_per_block**2, 4*block_size + num_rows_per_block**3) = rowmax statistics (local to S_ij)      <---- Can we not use this and rather reuse the Q_i allocated space
    //[4*block_size + num_rows_per_block**3, 4*block_size + num_rows_per_block**4) rowsum statistics (local)
    //[0, num_rows_per_block) rowmax statistics m_ij (local to S_ij) 
    //[num_rows_per_block, 2*num_rows_per_block] old rowmax statistics (global)
    //[2*num_rows_per_block, 3*num_rows_per_block] old rowsum statistics (global)
    //[3*num_rows_per_block, 4*num_rows_per_block] new rowmax statistics (global)      <---- Dont really need to store l_ij after we have computed this! 
    //[4*num_rows_per_block, 5*num_rows_per_block] new rowsum statistics (global)