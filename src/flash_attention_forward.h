#pragma once

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> forward_attention(
    torch::Tensor query,
    torch::Tensor key, 
    torch::Tensor value);