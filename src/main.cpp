#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include "flash_attention_forward.h"
// #include "fused_kernel_backward.h"

PYBIND11_MODULE(flash_attention_extension, handle){
    handle.def("flash_attention_forward", &forward_attention);
}
