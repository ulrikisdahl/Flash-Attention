from modules import FlashAttention, TorchAttention
import torch

if __name__ == "__main__":
    d_model = 256
    device = "cuda"
    query, key, value = torch.randn((32, 32, d_model), dtype=torch.float32, device=device), torch.randn((32, 32, d_model), dtype=torch.float32, device=device), torch.randn((32, 32, d_model), dtype=torch.float32, device=device)

    torch_attention = TorchAttention()


    #BENCHMARK 1
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        torch_attention(query, key, value)
        
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        FlashAttention.apply(query, key, value)


    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    

