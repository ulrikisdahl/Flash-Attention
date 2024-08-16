#pragma once

#include <torch/extension.h>
#include <vector>


std::vector<torch::Tensor> backward(
    torch::Tensor query, 
    torch::Tensor key, 
    torch::Tensor value, 
    torch::Tensor outputs, 
    torch::Tensor d_outputs,
    torch::Tensor rowmax_statistics,
    torch::Tensor rowsum_statistics);