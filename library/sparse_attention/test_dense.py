import torch
from sparse_attention_cpu import SparseAttentionServer
import pytest
import random
import math


@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("max_len_delta", [129])
@pytest.mark.parametrize("group_size", [1,4,8])
@pytest.mark.parametrize("batch_size", [1,4])
@pytest.mark.parametrize("num_layers", [1])
@pytest.mark.parametrize("num_attention_heads", [32])
@pytest.mark.parametrize("head_dim", [128])
def test_dense_attention(
seq_len: int,
max_len_delta: int,
group_size: int,
batch_size: int,
num_layers: int,
num_attention_heads: int,
head_dim: int
):  
    
    layer = random.randint(0, num_layers-1)
    max_length = seq_len + max_len_delta
    num_key_value_heads = num_attention_heads // group_size
    key = torch.randn(size=(batch_size, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16)
    value = torch.randn(size=(batch_size, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16)
    key_norm = key.norm(p=2, dim=-1).float()
    
    max_value_expsum = torch.zeros(size=(2, batch_size * num_attention_heads), dtype=torch.float32)
    
    output = torch.zeros(size=(batch_size * num_attention_heads, head_dim), dtype=torch.bfloat16)
    
    attn_server = SparseAttentionServer()
    attn_server.alloc(num_layers, num_attention_heads, num_key_value_heads, head_dim, batch_size, max_length)
    
    for i in range(batch_size):
        attn_server.fill(layer, i, key[i], value[i], key_norm[i])
    
    query = torch.randn(size=(batch_size, num_attention_heads, 1, head_dim), dtype=torch.float32)
    key = key[:,:,None,:,:].repeat(1, 1, group_size, 1, 1).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    value = value[:,:,None,:,:].repeat(1, 1, group_size, 1, 1).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    score_ref = torch.matmul(query, key.transpose(2,3).float()) / math.sqrt(head_dim)
    
    nnz = torch.full(fill_value=seq_len, size=(batch_size * num_attention_heads,)).int()
    
    attn_server.full_attention(layer, output, max_value_expsum, query, nnz)
    score = attn_server.get_score()
    
    score = score.view(batch_size * num_attention_heads, max_length)
    score_ref = score_ref.view(batch_size * num_attention_heads, seq_len)
    value = value.view(batch_size * num_attention_heads, seq_len, head_dim)
    
    for i in range(batch_size * num_attention_heads):
        ref = score_ref[i]
        ref = torch.softmax(ref, dim=-1)
        
        
        assert torch.allclose(score[i][:nnz[i]], ref, rtol=1e-2, atol=1e-2)
        assert torch.abs(score[i][:nnz[i]].sum() - 1) <= 1e-2
        v = value[i]
        o = torch.matmul(ref.unsqueeze(0), v.float())
        
        assert torch.allclose(output[i].float(), o, rtol=1e-2, atol=1e-2)

test_dense_attention(
seq_len=64,
max_len_delta=32,
group_size=4,
batch_size=1,
num_layers=6,
num_attention_heads=32,
head_dim=128
)
