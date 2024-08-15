import torch
import torch.nn as nn
from torch.autograd import Function
from flash_attention_extension import flash_attention_forward #, flash_attention_backward

device = "cuda"
dtype = torch.float32

class FlashAttention(Function):
    @staticmethod
    def forward(ctx, query, key, value):
        """
        """
        output, rowmax_statistics, rowsum_statistics = flash_attention_forward(query, key, value)
        ctx.save_for_backward(rowmax_statistics, rowsum_statistics)
        return output
    
    @staticmethod
    def backward(ctx, output_grad, rowmax_, rowsum_):
        """
        """
        # rowmax_statistics, rowsum_statistics = ctx.saved_tensors
        # query_grad, key_grad, val_grad = flash_attention_backward(output_grad, rowmax_statistics, rowsum_statistics)
        # return query_grad, key_grad, val_grad
        return output_grad


class FlashAttentionModule(nn.Module):
    def __init__(self, seq, d_model):
        super(FlashAttentionModule, self).__init__()

        #He initialization
        self.query = nn.Linear

    def forward(self, x):
        Q = torch.matmul(x, self.W_q)
        K = torch.matmul(x, self.W_k)
        V = torch.matmul(x, self.W_v)

        return FlashAttention.apply(Q, K, V)
    

class TorchAttention(nn.Module):
    def __init__(self):
        super(TorchAttention, self).__init__()
        # self.query = nn.Linear(in_features=d_model, out_features=d_model); 
        # self.key = nn.Linear(in_features=d_model, out_features=d_model);
        # self.value = nn.Linear(in_features=d_model, out_features=d_model)
        
    def forward(self, query, key, value):
        """
        Args:
        query: query tensor of shape (batch, seq, d_model)
        key: key tensor of shape (batch, seq, d_model)
        value: value tensor of shape (batch, seq, d_model)
        """
        sqrt_term = torch.sqrt(torch.tensor(query.shape[-1], dtype=torch.float32, device=query.device))
        similarity_scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt_term 
        attention_scores = nn.functional.softmax(similarity_scores, dim=-1)
        attention_output = torch.einsum("bqk, bkv -> bqv", attention_scores, value)

        return attention_output



