import torch
from lsh import LSH
import pytest
import random
@pytest.mark.parametrize("K", [4,8])
@pytest.mark.parametrize("L", [50, 100])
@pytest.mark.parametrize("seq_len", [1024, 4096, 8192])
@pytest.mark.parametrize("max_len_delta", [128, 1024])
@pytest.mark.parametrize("group_size", [4,8])
@pytest.mark.parametrize("bsz", [1,4])
@pytest.mark.parametrize("num_layers", [1,2])
@pytest.mark.parametrize("num_attention_heads", [32])
def test_batch_retrieve(
K: int,
L: int,
max_len_delta: int,
seq_len: int,
group_size: int,
bsz: int,
num_layers: int,
num_attention_heads: int
):  
    layer = random.randint(0, num_layers-1)
    lsh_retriever = LSH()
    num_key_values_heads = int(num_attention_heads // group_size)
    lsh_retriever.alloc(K, L, num_layers, num_attention_heads, num_key_values_heads, bsz, seq_len + max_len_delta)
    num_buckets = int(2**K)
    hash_code :torch.Tensor = torch.randint(low=0, high=num_buckets, size=(bsz, num_key_values_heads, L, seq_len), dtype=torch.int16)
    sorted_code, sorted_indices = hash_code.sort()
    for i in range(bsz):
        lsh_retriever.fill(layer, i, sorted_code[i], sorted_indices[i].int())
    
    # for i in range(bsz):
    #      lsh_retriever.fastfill(layer, i, hash_code[i])
    query = torch.randint(low=0, high=num_buckets, size=(bsz * num_attention_heads, L), dtype=torch.int32)
    results = torch.zeros((bsz * num_attention_heads, seq_len + max_len_delta), dtype=torch.int32)
    nnz = torch.zeros((bsz * num_attention_heads), dtype=torch.int32)
    lsh_retriever.batch_retrieve(layer, query, results, nnz)
    
    
    hash_code = hash_code[:,:,None,:,:].repeat(1, 1, group_size, 1, 1)
    hash_code = hash_code.reshape(bsz * num_attention_heads, L, seq_len)
    ref_mask = (hash_code == query[:,:,None]).int().sum(dim=1) > 1
    
    ref_nnz = ref_mask.int().sum(dim=-1).int()
    print(nnz, ref_nnz)
    assert torch.allclose(nnz, ref_nnz)
    
    
    
    ref_mask = ref_mask.reshape(bsz * num_attention_heads, seq_len)
    for i in range(bsz * num_attention_heads):
        nnz_i = nnz[i]
        result_i = results[i][:nnz_i]
        ref_mask_i = ref_mask[i]
        assert ref_mask_i[result_i].int().sum() == nnz_i
    
    
    query = torch.randint(low=0, high=num_buckets, size=(bsz * num_attention_heads, L), dtype=torch.int32)
    results = torch.zeros((bsz * num_attention_heads, seq_len + max_len_delta), dtype=torch.int32)
    nnz = torch.zeros((bsz * num_attention_heads), dtype=torch.int32)
    lsh_retriever.batch_retrieve(layer, query, results, nnz)
    
    
    ref_mask = (hash_code == query[:,:,None]).int().sum(dim=1) > 1
    
    ref_nnz = ref_mask.int().sum(dim=-1).int()
    assert torch.allclose(nnz, ref_nnz)
    
    
    ref_mask = ref_mask.reshape(bsz * num_attention_heads, seq_len)
    for i in range(bsz * num_attention_heads):
        nnz_i = nnz[i]
        result_i = results[i][:nnz_i]
        ref_mask_i = ref_mask[i]
        assert ref_mask_i[result_i].int().sum() == nnz_i
    
test_batch_retrieve(K=2, L=4, seq_len=128, group_size=1, bsz=1, num_layers=1, num_attention_heads=1, max_len_delta=16)
