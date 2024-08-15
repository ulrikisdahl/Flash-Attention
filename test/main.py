"""
Compare CUDA Flash Attention implementation with official pytorch attention modules

Allclose-test:
- Tests if outputs are the same. If the outputs are close (within margin of error), the test will success.
"""

from modules import FlashAttention, TorchAttention
import torch
import torch.nn.functional as F

torch.manual_seed(1)

#Testing script
if __name__ == "__main__":
        device = "cuda"
        d_model = 256

        query, key, value = torch.randn((32, 32, d_model), dtype=torch.float32, device=device), torch.randn((32, 32, d_model), dtype=torch.float32, device=device), torch.randn((32, 32, d_model), dtype=torch.float32, device=device)

        torch_attention = TorchAttention()
        vanilla_attention_outputs = torch_attention(query, key, value)

        flash_attention_outputs = FlashAttention.apply(query, key, value)

        pytorch_flash_attention = F.scaled_dot_product_attention(query, key, value)

        test = torch.allclose(vanilla_attention_outputs, flash_attention_outputs, atol=1e-3)
        print(f"All close test: {test}")

        #Further verification
        print(f"Average value in torch-attention: {torch.mean(vanilla_attention_outputs)}")
        print(f"Average value in scaled_dot_prod-attention: {torch.mean(pytorch_flash_attention)}") 
        print(f"Average value in flash-attention: {torch.mean(flash_attention_outputs)}") 
