import torch
import argparse
import torch.utils.benchmark as benchmark
from sparse_attention_cpu import SparseAttentionServer
import time

def attention(attn_server, output, max_value_expsum, query, query_norm, ind, nnz):
    attn_server.full_attention(0, output, max_value_expsum, query, nnz)

def bench(
head_dim: int,
seq_len: int,
group_size: int,
bsz: int,
num_attention_heads: int,
max_nnz: int
):
    
    
    num_key_value_heads = num_attention_heads // group_size
    key = torch.randn(size=(bsz, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16)
    value = torch.randn(size=(bsz, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16)
    key_norm = key.norm(p=2, dim=-1).float()
    
    max_value_expsum = torch.zeros(size=(2, bsz * num_attention_heads), dtype=torch.float32)
    
    output = torch.zeros(size=(bsz * num_attention_heads, head_dim), dtype=torch.bfloat16)
    
    attn_server = SparseAttentionServer()
    attn_server.alloc(1, num_attention_heads, num_key_value_heads, head_dim, bsz, seq_len)
    
    for i in range(bsz):
        attn_server.fill(0, i, key[i], value[i], key_norm[i])
    
    query = torch.randn(size=(bsz, num_attention_heads, 1, head_dim), dtype=torch.float32)
    query_norm = query.norm(p=2, dim=-1)
    
    #nnz = torch.randint(low=1, high=max_nnz, size=(bsz * num_attention_heads,)).int()
    nnz = torch.ones(size=(bsz * num_attention_heads,)).int() * max_nnz
    ind = torch.zeros((bsz * num_attention_heads, seq_len)).int()

    for i in range(bsz * num_attention_heads):
            ind[i][:nnz[i]] = torch.randperm(seq_len)[:nnz[i]].int()
    
    T = 128
    for _ in range(16):
        attention(attn_server, output, max_value_expsum, query, query_norm, ind, nnz)
    
    t1 = time.time()
    for _ in range(T):
        attention(attn_server, output, max_value_expsum, query, query_norm, ind, nnz)
    t2 = time.time()
    
    memory = num_key_value_heads * seq_len * bsz * head_dim * 4 / (1024**3)
    
    print("Bandwidth = {:.2f} GB/s".format(T * memory / (t2 - t1)))
    # t_op = benchmark.Timer(
    #     stmt="attention(attn_server, output, max_value_expsum, query, query_norm, ind, nnz)",
    #     globals={"attention": attention, "attn_server": attn_server, 
    #     "output": output,  "max_value_expsum": max_value_expsum, "query": query,
    #     "query_norm": query_norm, "ind": ind, "nnz":nnz},
    #     label="batch_search",
    #     num_threads=64
    # )

    # print("MAX_NNZ = {}, Avg NNZ = {}, SEQ_LEN = {}, BSZ = {}".format(nnz.max(), nnz.float().mean(), seq_len, bsz * num_attention_heads))
    # print(t_op.timeit(128))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--head_dim", type=int, default=128)
    argparser.add_argument("--seq", type=int, default=131072)
    argparser.add_argument("--bsz", type=int, default=2)
    argparser.add_argument("--num_attention_heads", type=int, default=32)
    argparser.add_argument("--num_key_value_heads", type=int, default=8)
    argparser.add_argument("--max_nnz", type=int, default=131072)
    args = argparser.parse_args()
    bench(args.head_dim, args.seq, args.num_attention_heads // args.num_key_value_heads, args.bsz, args.num_attention_heads, args.max_nnz)
    
