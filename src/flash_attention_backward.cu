#include <vector>
#include <torch/extension.h>

template <typename scalar_t>
__device__
void inner_product_matmul(
     scalar_t* A, 
    scalar_t* B, 
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
            temp += A[local_matrix_row_index * dimension + k] * B[(thread_idx % num_rows_per_block) * dimension + k]; //Q_i * K^T_j
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
    scalar_t* A_ij,
    scalar_t* B_i, 
    int num_rows_per_block,
    int dimension,
    int thread_idx,
    int thread_idx_limit,
    float scaling_factor)
{
    if(thread_idx < thread_idx_limit){ //TODO: fix edge case for when last tile does not have same amount of rows
        float temp = 0.0f;
        for (int k = 0; k < num_rows_per_block; k++){
            temp += A_ij[(thread_idx / dimension) * num_rows_per_block + k] * B_i[k * dimension + (thread_idx % dimension)];
        }
        return scaling_factor * temp;
    };
    return 0.0f;
}

/**
 * @brief Each thread creates a element of the output product
 */
template <typename scalar_t>
__device__
float outer_product_transposed(
    scalar_t* A_ij,
    scalar_t* B_i, 
    int num_rows_per_block,
    int dimension,
    int thread_idx,
    int thread_idx_limit,
    float scaling_factor)
{
    if(thread_idx < thread_idx_limit){ //TODO: fix edge case for when last tile does not have same amount of rows
        float temp = 0.0f;
        for (int k = 0; k < num_rows_per_block; k++){
            temp += A_ij[k * num_rows_per_block + (thread_idx / dimension)] * B_i[k * dimension + (thread_idx % dimension)];
        }
        return scaling_factor * temp;
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

template <typename scalar_t>
__global__
void backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> query,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> key,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> value,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> outputs,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_outputs,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rowmax_statistics, 
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rowsum_statistics,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_query, 
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_key,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_value,
    int batch_size, int sequence_length, int dimension,
    int block_size,
    int num_rows_per_block,
    int num_blocks_per_sample)
{

    auto sharedMemory = shared_memory_proxy<scalar_t>();
    scalar_t* Q_i = &sharedMemory[0];
    scalar_t* K_j = Q_i + block_size;
    scalar_t* V_j = Q_i + 2*block_size; 
    scalar_t* d_Q_i = Q_i + 3*block_size;
    scalar_t* d_O_i = Q_i + 4*block_size;
    scalar_t* o_hadamard = d_O_i; //Reuse d_O_i memory allocation
    // scalar_t* scores = &sharedMemory[0]; //S_ij
    scalar_t* scores = Q_i + 5*block_size;
    scalar_t* P_ij = scores + num_rows_per_block * num_rows_per_block;
    scalar_t* d_P_ij = P_ij + num_rows_per_block * num_rows_per_block;
    scalar_t* D_i = P_ij + num_rows_per_block * num_rows_per_block; //could be an idea to save this to a variable....
    scalar_t* d_S_ij = scores; //Reuse S_ij allocation (d_S_ij will be of same size)

    //compute indexes
    int batch_idx = blockIdx.x; 
    int local_row_idx = threadIdx.x / dimension; 
    int col_idx = threadIdx.x % dimension;

    float scaling_factor = 1.0f / (sqrtf(static_cast<float>(dimension)));    
    for (int j = 0; j < num_rows_per_block; j++){
        K_j[threadIdx.x] = key[batch_idx][j * num_rows_per_block + local_row_idx][col_idx];
        V_j[threadIdx.x] = value[batch_idx][j * num_rows_per_block + local_row_idx][col_idx];
        float d_K_j = 0.0f;
        float d_V_j = 0.0f;

        for (int i = 0; i < num_rows_per_block; i++){
            int global_row_idx_i = i * num_rows_per_block + local_row_idx; 
            Q_i[threadIdx.x] = query[batch_idx][global_row_idx_i][col_idx];
            d_O_i[threadIdx.x] = d_outputs[global_row_idx_i][col_idx];

            //compute S_ij
            inner_product_matmul(Q_i, K_j, scores, num_rows_per_block, dimension, threadIdx.x, num_rows_per_block*num_rows_per_block, scaling_factor);

            //P_ij
            if(threadIdx.x < num_rows_per_block * num_rows_per_block){
                auto global_rowmax = rowmax_statistics[batch_idx][i * num_rows_per_block + threadIdx.x / num_rows_per_block];
                auto global_rowsum = rowsum_statistics[batch_idx][i * num_rows_per_block + threadIdx.x / num_rows_per_block];
                scores[threadIdx.x] = (1 / global_rowsum) * expf(scores[threadIdx.x] - global_rowsum);
            }
            __syncthreads();

            //update d_Vj
            d_V_j += outer_product_transposed(scores, d_O_i, num_rows_per_block, dimension, threadIdx.x, num_rows_per_block * dimension, 1.0f); 

            //compute d_P_ij
            inner_product_matmul(d_O_i, V_j, d_P_ij, num_rows_per_block, dimension, threadIdx.x, num_rows_per_block*num_rows_per_block, 1.0f); //Dont want to use scaling factor here
            __syncthreads();

            //compute Di
            auto d_i = d_O_i[threadIdx.x] + outputs[batch_idx][global_row_idx_i][col_idx];
            o_hadamard[threadIdx.x] = d_i;
            __syncthreads();

            //recompute local rowsum
            if(threadIdx.x % dimension == 0){
                float temp = 0.0f;
                for (int k = 0; k < dimension; k++){ //TODO: Implement sum reduction algorithm to utilize more threads
                    temp += o_hadamard[local_row_idx * dimension + k];
                }
                D_i[threadIdx.x / dimension] = temp; 
            }
            __syncthreads();

            //compute d_S_ij
            if(threadIdx.x < num_rows_per_block * num_rows_per_block){
                d_S_ij[threadIdx.x] = P_ij[threadIdx.x] * (d_P_ij[threadIdx.x] - D_i[threadIdx.x / num_rows_per_block]);
            }
            __syncthreads();

            //update gradients
            auto q_gradient = outer_product_matmul(d_S_ij, K_j, num_rows_per_block, dimension, threadIdx.x, num_rows_per_block * dimension, scaling_factor);
            auto k_gradient = outer_product_transposed(d_S_ij, Q_i, num_rows_per_block, dimension, threadIdx.x, num_rows_per_block * dimension, scaling_factor);
            d_query[global_row_idx_i][col_idx] += q_gradient;
            d_K_j += k_gradient;  

            __syncthreads();
        }

        //update d_K_j and d_V_j
        d_key[j * num_rows_per_block + local_row_idx][col_idx] = d_K_j;
        d_value[j * num_rows_per_block + local_row_idx][col_idx] = d_V_j;
    }
}


std::vector<torch::Tensor> backward(
    torch::Tensor query, 
    torch::Tensor key, 
    torch::Tensor value, 
    torch::Tensor outputs, 
    torch::Tensor d_outputs,
    torch::Tensor rowmax_statistics,
    torch::Tensor rowsum_statistics)
{
    constexpr int num_threads_per_block = 1024;
    int B = query.sizes()[0];
    int N = query.sizes()[1];
    int D = query.sizes()[2];
    int num_rows_per_block = floor(num_threads_per_block / D);
    int num_rows_per_block_squared = num_rows_per_block * num_rows_per_block;
    int BLOCK_SIZE = num_rows_per_block * D;
    int num_blocks_per_sample = N / num_rows_per_block;
    dim3 GRID(B);

    torch::Tensor d_query = torch::zeros({N, D}, query.options());
    torch::Tensor d_key = torch::zeros({N, D}, query.options()); 
    torch::Tensor d_value = torch::zeros({N, D}, query.options());

    AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "backward", ([&]{
        backward_kernel<<<GRID, num_threads_per_block, sizeof(scalar_t) * (5*BLOCK_SIZE + 3*num_rows_per_block_squared)>>>(
            query.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            key.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            value.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            outputs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), 
            d_outputs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), //Think this should not have the batch channel (therefore only 2)??? 
            rowmax_statistics.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), 
            rowsum_statistics.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), 
            d_query.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), 
            d_key.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), 
            d_value.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            B, N, D,
            BLOCK_SIZE,
            num_rows_per_block,
            num_blocks_per_sample 
        );
    }));

    return {d_query, d_key, d_value};
}
