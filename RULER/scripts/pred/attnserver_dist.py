from transformers import LlamaConfig
import torch
from lsh import LSH 
from sparse_attention_cpu import SparseAttentionServer
import flashinfer
import torch.distributed as dist
import math
import torch.nn.functional as F
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
class QuestAttnServer:
    """Use Max and Min landmarks for retrieval"""
    def __init__(self, 
        config :LlamaConfig,
        batch_size :int = 1,
        max_length :int = 32*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        sparse_budget: int = 2048,
        chunk_size=16,
        ) -> None:
        
        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads // self.world_size
        self.num_key_value_heads = config.num_key_value_heads // self.world_size
        self.workload = 0
        self.decode_tokens = 1
        self.sparse_budget = int(sparse_budget)
        self.chunk_size = chunk_size
        
        self.hidden_size = config.hidden_size // self.world_size
        self.k_cache_cpu = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            self.num_key_value_heads,
            self.max_length,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache_cpu = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            self.num_key_value_heads,
            self.max_length,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0

        self.k_landmark_max = []
        self.k_landmark_min = []
        self.dense_layers = [0,1] # better than [0,16] in Quest
        
    def print_stats(self):
        print(f"QuestCache | sparse budget {self.sparse_budget} | chunk size {self.chunk_size} | cached {self.kv_offset}")

    def register_k_landmark(self, k_landmark_max, k_landmark_min):
        self.k_landmark_max.append(k_landmark_max.clone())
        self.k_landmark_min.append(k_landmark_min.clone())

    def fill(self, 
        layer_idx:int,
        request_id: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        seq_len:int
            ):
        
        key_cache = key_cache[:seq_len].transpose(0,1).unsqueeze(0)
        value_cache = value_cache[:seq_len].transpose(0,1).unsqueeze(0)
        
        incoming = key_cache.shape[-2] # [bsz, num_kv_heads, incoming, head_dim]
        self.prefill = incoming
        self.sparse_budget = int (0.0375 * self.prefill)
        self.v_cache_cpu[layer_idx][:, :, :incoming] = value_cache.clone()
        self.k_cache_cpu[layer_idx][:, :, :incoming] = key_cache.clone()

        self.chunks = incoming // self.chunk_size - 32 // self.chunk_size
        self.select_sets = self.sparse_budget // self.chunk_size
        
        self.chunk_end = self.chunks * self.chunk_size
        
        #assert self.select_sets * self.chunk_size == self.sparse_budget, f"({self.select_sets}) * {self.chunk_size} != {self.sparse_budget}"

        key_states_roped_ctx = key_cache[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim)
        
        k_landmark_max = key_states_roped_ctx.min(dim=-2).values
        k_landmark_min = key_states_roped_ctx.max(dim=-2).values

        # register rest_idxed landmarks to k_landmark
        self.register_k_landmark(k_landmark_max, k_landmark_min)

        if layer_idx == self.num_layers - 1:
            assert self.sparse_budget < incoming
            self.kv_offset += incoming

    def collect_kv(self, layer_idx, query_states):
        
        if layer_idx in self.dense_layers:
            self.incoming_q_len = query_states.shape[-2]
            gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len
            k = self.k_cache_cpu[layer_idx][:,:,:self.prefill+gen_offset]
            v = self.v_cache_cpu[layer_idx][:,:,:self.prefill+gen_offset]
            return k, v
        else:
            self.incoming_q_len = query_states.shape[-2] # 1
            min_cache = repeat_kv(self.k_landmark_min[layer_idx], self.num_key_value_groups)
            max_cache = repeat_kv(self.k_landmark_max[layer_idx], self.num_key_value_groups)
            min_value = min_cache * query_states
            max_value = max_cache * query_states

            heuristic = torch.max(min_value, max_value)
            heuristic = heuristic.sum(dim=-1)
            
            heuristic = heuristic.reshape(1, self.num_key_value_heads, self.num_key_value_groups, -1)
            heuristic = heuristic.sum(dim=-2, keepdim=True)
            
            topk_chunk = heuristic.topk(k=self.select_sets, dim=-1).indices

            position_ids = (topk_chunk.unsqueeze(-1) * self.chunk_size + torch.arange(self.chunk_size, device=topk_chunk.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)).view(1, self.num_key_value_heads, -1) # [bsz, 8, select_sets * chunk_size]

            key_ = self.k_cache_cpu[layer_idx].gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
            value_ = self.v_cache_cpu[layer_idx].gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))

            gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

            ret_k = torch.cat([key_, self.k_cache_cpu[layer_idx][:,:,self.chunk_end:self.prefill+gen_offset]], dim = 2)
            ret_v = torch.cat([value_, self.v_cache_cpu[layer_idx][:,:,self.chunk_end:self.prefill+gen_offset]], dim = 2)

            return ret_k, ret_v
        
    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            ):

        incoming = new_k_cache.shape[-2]
        self.k_cache_cpu[layer_idx][:, :, self.kv_offset:self.kv_offset + incoming].copy_(new_k_cache)
        self.v_cache_cpu[layer_idx][:, :, self.kv_offset:self.kv_offset + incoming].copy_(new_v_cache)

        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming
            self.gen_offset += incoming

    def clear(self):
        self.k_cache_cpu.zero_()
        self.v_cache_cpu.zero_()
        self.k_landmark_max.clear()
        self.k_landmark_min.clear()

        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0

    def get_kv_len(self):
        return self.kv_offset
    
    def build_table(self, 
        layer_idx:int,
        request_id: int,
        seq_len:int):
        
        pass
    
    def plan(self):
        pass
    
    def alloc_buffer(self, seq_len:int):
        pass
    
    def decode(
        self, 
        query_states:torch.Tensor, 
        key_states:torch.Tensor, 
        value_states:torch.Tensor,
        layer_idx:int):
        
        self.update_kv_cache(key_states, value_states, layer_idx)
        k, v = self.collect_kv(layer_idx, query_states)
        query_states = query_states[0].transpose(0,1)[0].contiguous()
        
        o = flashinfer.decode.single_decode_with_kv_cache(
            q=query_states,
            k=k[0].contiguous(),
            v=v[0].contiguous(),
            kv_layout="HND"
        )
        
        
        hidden_states = o.reshape(1, 1, self.hidden_size)
            
        return hidden_states
class LSHSparseAttnServer:

    def __init__(self, 
        config :LlamaConfig,
        K: int = 10,
        L: int = 150,
        batch_size :int = 1,
        num_sink_tokens :int = 4,
        num_local_tokens :int = 64,
        generation_buffer :int = 256,
        max_length: int = 8192,
        dense_layers: list[int] = [0, 16, 32, 48, 64],
        device :str = 'cuda:0',
        dtype = torch.bfloat16) -> None:
        
        self.K = K
        self.L = L
        self.config = config
        self.length = num_sink_tokens + num_local_tokens + generation_buffer
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.num_layers = config.num_hidden_layers
        self.batch_size = batch_size
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        self.num_key_value_heads = config.num_key_value_heads // self.world_size
        self.num_attention_heads = config.num_attention_heads // self.world_size
        self.hidden_size = config.hidden_size // self.world_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dense_layers = dense_layers
        self.num_sink_tokens = num_sink_tokens
        self.num_local_tokens = num_local_tokens
        
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
         
        self.avg_k = [torch.zeros(
            self.batch_size,
            self.num_key_value_heads,
            1,
            self.head_dim,
            device=self.device,
            dtype=self.dtype
        ) for _ in range(self.num_layers)] 
        
        self.workload = 0
        self.decode_tokens = 1
        self.prefill_len = 0
        self.attn_server = SparseAttentionServer()
        self.attn_server.alloc(self.num_layers, self.num_attention_heads, self.num_key_value_heads, self.head_dim, batch_size, max_length)
        self.lsh_retriever = LSH()
        self.lsh_retriever.alloc(self.K, self.L, self.num_layers, self.num_attention_heads, self.num_key_value_heads, batch_size, max_length)
        self.hash_func = torch.randn((self.head_dim, self.K * self.L), device=self.device, dtype=self.dtype)
        dist.broadcast(self.hash_func, 0)
        self.binary_pack = [int(2**i) for i in range(self.K)]
        self.binary_pack = torch.Tensor(self.binary_pack).to(device=self.device, dtype=torch.float16)
        
        self.nnz = torch.zeros((self.batch_size * self.num_attention_heads,)).to(torch.int32)
        self.results_lsh_cpu = torch.zeros((self.batch_size * self.num_attention_heads, self.max_length)).to(torch.int32)
        self.max_value_expsum = torch.ones((2, self.batch_size * self.num_attention_heads)).to(torch.float32).pin_memory()
        self.output_cuda = torch.zeros((self.batch_size * self.num_attention_heads, self.head_dim), dtype=torch.bfloat16).to(self.device)
        self.max_value_expsum_cuda = torch.ones((self.batch_size * self.num_attention_heads)).to(torch.float32).to(self.device)
        self.output = torch.zeros((self.batch_size * self.num_attention_heads, self.head_dim), dtype=torch.bfloat16).pin_memory()
        self.pinned_hashcode = torch.zeros((self.batch_size * self.num_attention_heads, self.L), dtype=torch.int32).pin_memory()
        self.pinned_query = torch.zeros((self.batch_size * self.num_attention_heads, self.head_dim), dtype=torch.bfloat16).pin_memory()
        self.chunk_size = 8192

        self.hash_code_buffer =  torch.zeros((self.num_key_value_heads, self.L, max_length), dtype=torch.int16, device=self.device)
        self.hash_code_buffer_cpu :torch.Tensor = None
        self.sorted_hash_values_buffer :torch.Tensor = None
        self.sorted_hash_indices_buffer :torch.Tensor = None
        
        self.max_num_pages = self.batch_size
        self.page_size = self.length
        self.kv_page_indices = torch.arange(self.max_num_pages).int().to(self.device)
        self.kv_page_indptr = torch.arange(self.batch_size + 1).int().to(self.device)
        self.kv_last_page_len = torch.zeros(self.batch_size).int().to(self.device)
        self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        self.workspace_buffer, "HND"
        )
        
        self.dense_max_num_pages = self.batch_size
        self.dense_page_size = self.max_length
        self.dense_kv_page_indices = torch.arange(self.dense_max_num_pages).int().to(self.device)
        self.dense_kv_page_indptr = torch.arange(self.batch_size + 1).int().to(self.device)
        self.dense_kv_last_page_len = torch.zeros(self.batch_size).int().to(self.device)
        self.dense_workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.dense_decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        self.dense_workspace_buffer, "HND"
        )
        
        self.flashinfer_kv_cache = [
        torch.zeros(
            self.max_num_pages, 
            2, 
            self.num_key_value_heads, 
            self.dense_page_size if i in self.dense_layers else self.page_size, 
            self.head_dim, 
            dtype=torch.bfloat16, 
            device=self.device
        ) for i in range(self.num_layers)
        ]
        
        
        
    def alloc_buffer(self, seq_len:int):
        self.sorted_hash_values_buffer =  torch.zeros((self.num_key_value_heads, self.L, seq_len - self.num_sink_tokens - self.num_local_tokens), dtype=torch.int16, device="cpu").pin_memory()
        self.sorted_hash_indices_buffer =  torch.zeros((self.num_key_value_heads, self.L, seq_len - self.num_sink_tokens - self.num_local_tokens), dtype=torch.int32, device="cpu").pin_memory()
        
    def fill(self, 
        layer_idx:int,
        request_id: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        seq_len:int):
        
        self.prefill_len = seq_len
        if layer_idx in self.dense_layers:
            
            self.flashinfer_kv_cache[layer_idx][request_id][0].copy_(key_cache.transpose(0,1))
            self.flashinfer_kv_cache[layer_idx][request_id][1].copy_(value_cache.transpose(0,1))
            self.dense_kv_last_page_len[request_id] = seq_len
            
        else:
            sink_tokens_key = key_cache[:self.num_sink_tokens]
            sink_tokens_value = value_cache[:self.num_sink_tokens]
            
            local_tokens_key = key_cache[seq_len-self.num_local_tokens:seq_len]
            local_tokens_value = value_cache[seq_len-self.num_local_tokens:seq_len]
            
            key = torch.cat([sink_tokens_key, local_tokens_key], dim=0).transpose(0,1)
            value = torch.cat([sink_tokens_value, local_tokens_value], dim=0).transpose(0,1)
            
            
            offload_key = key_cache[self.num_sink_tokens: seq_len-self.num_local_tokens]
            offload_value = value_cache[self.num_sink_tokens: seq_len-self.num_local_tokens]
            
            offload_key = offload_key.transpose(0,1).contiguous()
            offload_value = offload_value.transpose(0,1).contiguous()
            
            avg_k = offload_key.mean(dim=1, keepdim=True)
            
            key = key - avg_k
            offload_key = offload_key - avg_k
            kn = offload_key.norm(p=2, dim=-1).float()
            
            self.avg_k[layer_idx][request_id] = avg_k
            
            
            self.flashinfer_kv_cache[layer_idx][request_id][0][...,:self.num_sink_tokens + self.num_local_tokens,:].copy_(key)
            self.flashinfer_kv_cache[layer_idx][request_id][1][...,:self.num_sink_tokens + self.num_local_tokens,:].copy_(value)
            self.kv_last_page_len[request_id] = (self.num_sink_tokens + self.num_local_tokens)
            
            
            offload_len = offload_key.shape[1]
            num_iter = (offload_len // self.chunk_size) if (not offload_len % self.chunk_size) else (offload_len // self.chunk_size + 1)
            
            for i in range(num_iter):
                start = i * self.chunk_size
                end = min((i+1) * self.chunk_size, offload_len)
                hash_code = torch.matmul(offload_key[:,start:end,:], self.hash_func)
                hash_code = hash_code > 0
                hash_code = hash_code.reshape(-1, self.K).to(torch.float16)
                hash_code = torch.mv(hash_code, self.binary_pack)
                hash_code = hash_code.reshape(self.num_key_value_heads, -1, self.L)
                hash_code = hash_code.transpose(1,2).contiguous().to(torch.int16)
                self.hash_code_buffer[:,:,start:end].copy_(hash_code)
            
            
            
            offload_key = offload_key.cpu()
            offload_value = offload_value.cpu()
            kn = kn.cpu()
            self.attn_server.fill(layer_idx, request_id, offload_key, offload_value, kn)
            
    
    def build_table(self, 
        layer_idx:int,
        request_id: int,
        seq_len:int):
        
        
        if layer_idx not in self.dense_layers:
            offload_len = seq_len - self.num_sink_tokens - self.num_local_tokens
            for i in range(self.num_key_value_heads):
                    sorted_hash_values, sorted_hash_indices = self.hash_code_buffer[i,:,:offload_len].sort()
                    self.sorted_hash_values_buffer[i].copy_(sorted_hash_values)
                    self.sorted_hash_indices_buffer[i].copy_(sorted_hash_indices)

            self.lsh_retriever.fill(layer_idx, request_id, 
                    self.sorted_hash_values_buffer, 
                    self.sorted_hash_indices_buffer)
        
        
    def plan(self):
        
        self.kv_last_page_len += 1
        self.decode_wrapper.plan(
        self.kv_page_indptr,
        self.kv_page_indices,
        self.kv_last_page_len,
        self.num_attention_heads,
        self.num_key_value_heads,
        self.head_dim,
        self.page_size,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        data_type=torch.bfloat16
    )     
        
        self.dense_kv_last_page_len += 1
        self.dense_decode_wrapper.plan(
        self.dense_kv_page_indptr,
        self.dense_kv_page_indices,
        self.dense_kv_last_page_len,
        self.num_attention_heads,
        self.num_key_value_heads,
        self.head_dim,
        self.dense_page_size,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        data_type=torch.bfloat16
    )     
    
    
    
    def decode(
        self, 
        query_states:torch.Tensor, 
        key_states:torch.Tensor, 
        value_states:torch.Tensor,
        layer_idx:int):
        
        if layer_idx in self.dense_layers:
            
            key_states = key_states.reshape(self.batch_size, self.num_key_value_heads, self.head_dim)
            value_states = value_states.reshape(self.batch_size, self.num_key_value_heads, self.head_dim)
            
            flashinfer.append_paged_kv_cache(
                key_states,
                value_states,
                self.dense_kv_page_indptr,
                self.flashinfer_kv_cache[layer_idx],
                self.dense_kv_page_indices,
                self.dense_kv_page_indptr,
                self.dense_kv_last_page_len,
                kv_layout="HND"
            )
            
            q = query_states.reshape(self.batch_size, self.num_attention_heads, self.head_dim)
            hidden_states = self.dense_decode_wrapper.run(
            q, 
            self.flashinfer_kv_cache[layer_idx]
            )
            
            hidden_states = hidden_states.reshape(self.batch_size, 1, self.hidden_size)
           
            return hidden_states
        
        else:
            
            
            bsz, _, q_len, _ = query_states.shape
            norm_q = query_states.reshape(-1, self.head_dim)
            norm_q = norm_q / norm_q.norm(p=2, dim=-1, keepdim=True) 
            q_hashcode = torch.matmul(norm_q, self.hash_func).gt(0)
            q_hashcode = q_hashcode.reshape(-1, self.K).to(torch.float16)
            q_hashcode = torch.mv(q_hashcode, self.binary_pack).int()
            q_hashcode = q_hashcode.reshape(self.batch_size * self.num_attention_heads, self.L)
            
            self.pinned_hashcode.copy_(q_hashcode)
            self.pinned_query.copy_(query_states.reshape(self.batch_size * self.num_attention_heads, self.head_dim))
            
            key_states = key_states - self.avg_k[layer_idx]
            
            key_states = key_states.reshape(self.batch_size, self.num_key_value_heads, self.head_dim)
            value_states = value_states.reshape(self.batch_size, self.num_key_value_heads, self.head_dim)
            
            
            flashinfer.append_paged_kv_cache(
                key_states,
                value_states,
                self.kv_page_indptr,
                self.flashinfer_kv_cache[layer_idx],
                self.kv_page_indices,
                self.kv_page_indptr,
                self.kv_last_page_len,
                kv_layout="HND"
            )
            
            q = query_states.reshape(self.batch_size, self.num_attention_heads, self.head_dim)
            gpu_hidden_states, gpu_lse = self.decode_wrapper.run_return_lse(
            q, 
            self.flashinfer_kv_cache[layer_idx]
            )
            
            
            self.lsh_retriever.batch_retrieve(layer_idx,self.pinned_hashcode, self.results_lsh_cpu, self.nnz)
            self.decode_tokens = self.decode_tokens + 1
            self.workload += self.nnz.float().mean() / self.prefill_len
            self.attn_server.attention(layer_idx, self.K, self.L, self.output, self.max_value_expsum, self.pinned_query.float(), self.pinned_query.float().norm(p=2, dim=-1), self.results_lsh_cpu, self.nnz)
            
            self.max_value_expsum_cuda.copy_(self.max_value_expsum[1], non_blocking=True)
            self.output_cuda.copy_(self.output, non_blocking=True)

            cpu_lse = self.max_value_expsum_cuda.reshape(self.batch_size, self.num_attention_heads)
            
            cpu_hidden_states = self.output_cuda.reshape(self.batch_size, self.num_attention_heads, self.head_dim)
            hidden_states, _ = flashinfer.merge_state(gpu_hidden_states, gpu_lse, cpu_hidden_states, cpu_lse)
        
            hidden_states = hidden_states.reshape(bsz, q_len, self.hidden_size)
            
            return hidden_states 
    def clear(self):
        
        self.nnz.zero_()
        self.results_lsh_cpu.zero_()
        self.max_value_expsum.zero_()
        self.output_cuda.zero_()
        self.max_value_expsum_cuda.zero_()
        self.output.zero_()
        self.pinned_hashcode.zero_()
        self.pinned_query.zero_()
        for i in range(self.num_layers):
            self.avg_k[i].zero_()
            self.flashinfer_kv_cache[i].zero_()
            
        self.kv_last_page_len.zero_()
        self.dense_kv_last_page_len.zero_()
        self.lsh_retriever.clear()
        self.attn_server.clear()

class LSHSparseAttnServerMasked:

    def __init__(self, 
        config :LlamaConfig,
        K: int = 10,
        L: int = 150,
        batch_size :int = 1,
        num_sink_tokens :int = 4,
        num_local_tokens :int = 64,
        generation_buffer :int = 256,
        max_length: int = 8192,
        dense_layers: list[int] = [0, 16, 32, 48, 64],
        device :str = 'cuda:0',
        dtype = torch.bfloat16) -> None:
        
        self.K = K
        self.L = L
        self.config = config
        self.length = num_sink_tokens + num_local_tokens + generation_buffer
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.num_layers = config.num_hidden_layers
        self.batch_size = batch_size
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.workload = 0
        self.decode_tokens = 1        
        self.num_key_value_heads = config.num_key_value_heads // self.world_size
        self.num_attention_heads = config.num_attention_heads // self.world_size
        self.hidden_size = config.hidden_size // self.world_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dense_layers = dense_layers
        self.num_sink_tokens = num_sink_tokens
        self.num_local_tokens = num_local_tokens
        
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
         
        self.avg_k = [torch.zeros(
            self.batch_size,
            self.num_key_value_heads,
            1,
            self.head_dim,
            device=self.device,
            dtype=self.dtype
        ) for _ in range(self.num_layers)] 
        
        
        self.hash_func = torch.randn((self.head_dim, self.K * self.L), device=self.device, dtype=self.dtype)
        dist.broadcast(self.hash_func, 0)
        self.binary_pack = [int(2**i) for i in range(self.K)]
        self.binary_pack = torch.Tensor(self.binary_pack).to(device=self.device, dtype=torch.float16)
        
        self.max_num_pages = self.batch_size
        self.page_size = self.length
        self.kv_page_indices = torch.arange(self.max_num_pages).int().to(self.device)
        self.kv_page_indptr = torch.arange(self.batch_size + 1).int().to(self.device)
        self.kv_last_page_len = torch.zeros(self.batch_size).int().to(self.device)
        self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        self.workspace_buffer, "HND"
        )
        
        self.dense_max_num_pages = self.batch_size
        self.dense_page_size = self.max_length
        self.dense_kv_page_indices = torch.arange(self.dense_max_num_pages).int().to(self.device)
        self.dense_kv_page_indptr = torch.arange(self.batch_size + 1).int().to(self.device)
        self.dense_kv_last_page_len = torch.zeros(self.batch_size).int().to(self.device)
        self.dense_workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.dense_decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        self.dense_workspace_buffer, "HND"
        )
        
        self.flashinfer_kv_cache = [
        torch.zeros(
            self.max_num_pages, 
            2, 
            self.num_key_value_heads, 
            self.dense_page_size if i in self.dense_layers else self.page_size, 
            self.head_dim, 
            dtype=torch.bfloat16, 
            device=self.device
        ) for i in range(self.num_layers)
        ]
        
        self.sparse_kv_cache = [
        torch.zeros(
            self.batch_size, 
            2, 
            self.num_key_value_heads, 
            self.dense_page_size,
            self.head_dim, 
            dtype=torch.bfloat16, 
            device=self.device
        ) for _ in range(self.num_layers)
        ]
        
        self.hash_code = [
        torch.zeros(
            self.batch_size, 
            self.num_key_value_heads, 
            self.max_length,
            self.L, 
            dtype=torch.int32, 
            device=self.device
        ) for _ in range(self.num_layers)
        ]
        
        self.knorm = [
        torch.zeros(
            self.batch_size, 
            self.num_key_value_heads, 
            self.max_length, 
            dtype=torch.float32, 
            device=self.device
        ) for _ in range(self.num_layers)
        ]
        
        self.prefill = 0
    def alloc_buffer(self, seq_len:int):
        pass
        
    def fill(self, 
        layer_idx:int,
        request_id: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        seq_len:int):
        
        if layer_idx in self.dense_layers:
            
            self.flashinfer_kv_cache[layer_idx][request_id][0].copy_(key_cache.transpose(0,1))
            self.flashinfer_kv_cache[layer_idx][request_id][1].copy_(value_cache.transpose(0,1))
            self.dense_kv_last_page_len[request_id] = seq_len
            
        else:
            sink_tokens_key = key_cache[:self.num_sink_tokens]
            sink_tokens_value = value_cache[:self.num_sink_tokens]
            
            local_tokens_key = key_cache[seq_len-self.num_local_tokens:seq_len]
            local_tokens_value = value_cache[seq_len-self.num_local_tokens:seq_len]
            
            key = torch.cat([sink_tokens_key, local_tokens_key], dim=0).transpose(0,1)
            value = torch.cat([sink_tokens_value, local_tokens_value], dim=0).transpose(0,1)
            
            
            offload_key = key_cache[self.num_sink_tokens: seq_len-self.num_local_tokens]
            offload_value = value_cache[self.num_sink_tokens: seq_len-self.num_local_tokens]
            
            offload_key = offload_key.transpose(0,1).contiguous()
            offload_value = offload_value.transpose(0,1).contiguous()
            
            avg_k = offload_key.mean(dim=1, keepdim=True)
            
            key = key - avg_k
            offload_key = offload_key - avg_k
            kn = offload_key.norm(p=2, dim=-1).float()
            
            self.avg_k[layer_idx][request_id] = avg_k
            
            
            self.flashinfer_kv_cache[layer_idx][request_id][0][...,:self.num_sink_tokens + self.num_local_tokens,:].copy_(key)
            self.flashinfer_kv_cache[layer_idx][request_id][1][...,:self.num_sink_tokens + self.num_local_tokens,:].copy_(value)
            self.kv_last_page_len[request_id] = (self.num_sink_tokens + self.num_local_tokens)
            
            
            offload_len = offload_key.shape[1]
            self.sparse_kv_cache[layer_idx][request_id][0][...,:offload_len,:].copy_(offload_key)
            self.sparse_kv_cache[layer_idx][request_id][1][...,:offload_len,:].copy_(offload_value)
            self.knorm[layer_idx][request_id][...,:offload_len].copy_(kn)
            
            hash_code = torch.matmul(offload_key, self.hash_func) > 0
            hash_code = hash_code.reshape(-1, self.K).to(torch.float16)
            
            hash_code = torch.mv(hash_code, self.binary_pack)
            hash_code = hash_code.reshape(self.num_key_value_heads, offload_len, self.L)
            self.hash_code[layer_idx][request_id][:,:offload_len,:].copy_(hash_code)
            self.prefill = offload_len
    def build_table(self, 
        layer_idx:int,
        request_id: int,
        seq_len:int):
        
        pass
        
        
    def plan(self):
        
        self.kv_last_page_len += 1
        self.decode_wrapper.plan(
        self.kv_page_indptr,
        self.kv_page_indices,
        self.kv_last_page_len,
        self.num_attention_heads,
        self.num_key_value_heads,
        self.head_dim,
        self.page_size,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        data_type=torch.bfloat16
    )     
        
        self.dense_kv_last_page_len += 1
        self.dense_decode_wrapper.plan(
        self.dense_kv_page_indptr,
        self.dense_kv_page_indices,
        self.dense_kv_last_page_len,
        self.num_attention_heads,
        self.num_key_value_heads,
        self.head_dim,
        self.dense_page_size,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        data_type=torch.bfloat16
    )     
    
    
    
    def decode(
        self, 
        query_states:torch.Tensor, 
        key_states:torch.Tensor, 
        value_states:torch.Tensor,
        layer_idx:int):
        
        if layer_idx in self.dense_layers:
            
            key_states = key_states.reshape(self.batch_size, self.num_key_value_heads, self.head_dim)
            value_states = value_states.reshape(self.batch_size, self.num_key_value_heads, self.head_dim)
            
            flashinfer.append_paged_kv_cache(
                key_states,
                value_states,
                self.dense_kv_page_indptr,
                self.flashinfer_kv_cache[layer_idx],
                self.dense_kv_page_indices,
                self.dense_kv_page_indptr,
                self.dense_kv_last_page_len,
                kv_layout="HND"
            )
            
            q = query_states.reshape(self.batch_size, self.num_attention_heads, self.head_dim)
            hidden_states = self.dense_decode_wrapper.run(
            q, 
            self.flashinfer_kv_cache[layer_idx]
            )
            
            hidden_states = hidden_states.reshape(self.batch_size, 1, self.hidden_size)
           
            return hidden_states
        
        else:
            
            
            bsz, _, q_len, _ = query_states.shape
            norm_q = query_states.reshape(-1, self.head_dim)
            norm_q = norm_q / norm_q.norm(p=2, dim=-1, keepdim=True) 
            q_hashcode = torch.matmul(norm_q, self.hash_func).gt(0)
            q_hashcode = q_hashcode.reshape(-1, self.K).to(torch.float16)
            q_hashcode = torch.mv(q_hashcode, self.binary_pack).int()
            q_hashcode = q_hashcode.reshape(self.batch_size * self.num_key_value_heads, self.num_key_value_groups, 1, self.L)
            
            k_hashcode = self.hash_code[layer_idx][0,:,:self.prefill,:].unsqueeze(1)
            mask = (k_hashcode == q_hashcode).int().sum(dim=-1) > 1
            mask = mask.reshape(self.num_attention_heads, self.prefill)
            self.workload += mask.float().mean()
            self.decode_tokens += 1
            cpu_key_cache = self.sparse_kv_cache[layer_idx][:,0, :,:self.prefill]
            cpu_value_cache = self.sparse_kv_cache[layer_idx][:,1, :,:self.prefill]
            
            cpu_key_cache = cpu_key_cache[:,:,None,:,:].repeat(1, 1, self.num_key_value_groups, 1, 1).reshape(1, self.num_attention_heads, self.prefill, self.head_dim)
            cpu_value_cache = cpu_value_cache[:,:,None,:,:].repeat(1, 1, self.num_key_value_groups, 1, 1).reshape(self.num_attention_heads, self.prefill, self.head_dim)
            
            cpu_attn_score = torch.matmul(query_states, cpu_key_cache.transpose(2,3)).to(torch.float32)
            key_norm = self.knorm[layer_idx][:,:,:self.prefill][:,:,None,:].repeat(1,1,self.num_key_value_groups,1).reshape(1, self.num_attention_heads, 1, self.prefill)
            
            cos_similarity = (cpu_attn_score / (key_norm * query_states.norm(p=2, dim=-1, keepdim=True).float())).to(torch.float32)
                    
            theta = torch.arccos(cos_similarity)
                    
            weight = 1 - theta / torch.pi
            weight = 1 - (1 - weight**self.K)**self.L - self.L * ((1 - weight**self.K)**(self.L - 1)) * (weight**self.K)
                    
                    
                    
            cpu_attn_score = cpu_attn_score / math.sqrt(self.head_dim)
            cpu_attn_score = cpu_attn_score - torch.log(weight + 1e-4)
            cpu_attn_score = cpu_attn_score[0,:,0,:]
            cpu_attn_score = cpu_attn_score.masked_fill(~mask, -torch.inf)
            cpu_lse = torch.logsumexp(cpu_attn_score, dim=-1).unsqueeze(0)
            cpu_lse = cpu_lse / math.log(2)
            cpu_attn_score = cpu_attn_score.softmax(dim=-1).unsqueeze(-2)
            cpu_hidden_states = torch.matmul(cpu_attn_score.to(self.dtype), cpu_value_cache).transpose(0,1)
            
            
            
            key_states = key_states - self.avg_k[layer_idx]
            
            key_states = key_states.reshape(self.batch_size, self.num_key_value_heads, self.head_dim)
            value_states = value_states.reshape(self.batch_size, self.num_key_value_heads, self.head_dim)
            
            
            flashinfer.append_paged_kv_cache(
                key_states,
                value_states,
                self.kv_page_indptr,
                self.flashinfer_kv_cache[layer_idx],
                self.kv_page_indices,
                self.kv_page_indptr,
                self.kv_last_page_len,
                kv_layout="HND"
            )
            
            q = query_states.reshape(self.batch_size, self.num_attention_heads, self.head_dim)
            gpu_hidden_states, gpu_lse = self.decode_wrapper.run_return_lse(
            q, 
            self.flashinfer_kv_cache[layer_idx]
            )
            
            
            
            
            
            hidden_states, _ = flashinfer.merge_state(gpu_hidden_states, gpu_lse, cpu_hidden_states, cpu_lse)
        
            hidden_states = hidden_states.reshape(bsz, q_len, self.hidden_size)
            
            return hidden_states 
    def clear(self):
        
        
        for i in range(self.num_layers):
            self.avg_k[i].zero_()
            self.flashinfer_kv_cache[i].zero_()
            self.sparse_kv_cache[i].zero_()
            self.hash_code[i].zero_()
            
        self.kv_last_page_len.zero_()
        self.dense_kv_last_page_len.zero_()
        self.prefill = 0
class AttnServer:

    def __init__(self, 
        config :LlamaConfig,
        K: int = 10,
        L: int = 150,
        batch_size :int = 1,
        num_sink_tokens :int = 4,
        num_local_tokens :int = 64,
        generation_buffer :int = 256,
        max_length: int = 8192,
        dense_layers: list[int] = list(range(32)),
        device :str = 'cuda:0',
        dtype = torch.bfloat16) -> None:
        
        self.workload = 0
        self.decode_tokens = 1
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.K = K
        self.L = L
        self.config = config
        self.length = num_sink_tokens + num_local_tokens + generation_buffer
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.num_layers = config.num_hidden_layers
        self.batch_size = batch_size
        self.num_key_value_heads = config.num_key_value_heads // self.world_size
        self.num_attention_heads = config.num_attention_heads // self.world_size
        self.hidden_size = config.hidden_size // self.world_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dense_layers = dense_layers
        self.num_sink_tokens = num_sink_tokens
        self.num_local_tokens = num_local_tokens
        
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
         
        self.avg_k = [torch.zeros(
            self.batch_size,
            self.num_key_value_heads,
            1,
            self.head_dim,
            device=self.device,
            dtype=self.dtype
        ) for _ in range(self.num_layers)] 
        
        self.attn_server = SparseAttentionServer()
        self.attn_server.alloc(self.num_layers, self.num_attention_heads, self.num_key_value_heads, self.head_dim, batch_size, max_length)
        
        
        self.nnz = torch.zeros((self.batch_size * self.num_attention_heads,)).to(torch.int32)
        self.max_value_expsum = torch.ones((2, self.batch_size * self.num_attention_heads)).to(torch.float32).pin_memory()
        self.output_cuda = torch.zeros((self.batch_size * self.num_attention_heads, self.head_dim), dtype=torch.bfloat16).to(self.device)
        self.max_value_expsum_cuda = torch.ones((self.batch_size * self.num_attention_heads)).to(torch.float32).to(self.device)
        self.output = torch.zeros((self.batch_size * self.num_attention_heads, self.head_dim), dtype=torch.bfloat16).pin_memory()
        self.pinned_query = torch.zeros((self.batch_size * self.num_attention_heads, self.head_dim), dtype=torch.bfloat16).pin_memory()


        self.max_num_pages = self.batch_size
        self.page_size = self.length
        self.kv_page_indices = torch.arange(self.max_num_pages).int().to(self.device)
        self.kv_page_indptr = torch.arange(self.batch_size + 1).int().to(self.device)
        self.kv_last_page_len = torch.zeros(self.batch_size).int().to(self.device)
        self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        self.workspace_buffer, "HND", use_tensor_cores=True
        )
        
        self.dense_max_num_pages = self.batch_size
        self.dense_page_size = self.max_length
        self.dense_kv_page_indices = torch.arange(self.dense_max_num_pages).int().to(self.device)
        self.dense_kv_page_indptr = torch.arange(self.batch_size + 1).int().to(self.device)
        self.dense_kv_last_page_len = torch.zeros(self.batch_size).int().to(self.device)
        self.dense_workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.dense_decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        self.dense_workspace_buffer, "HND", use_tensor_cores=True
        )
        
        self.flashinfer_kv_cache = [
        torch.zeros(
            self.max_num_pages, 
            2, 
            self.num_key_value_heads, 
            self.dense_page_size if i in self.dense_layers else self.page_size, 
            self.head_dim, 
            dtype=torch.bfloat16, 
            device=self.device
        ) for i in range(self.num_layers)
        ]
        
        
        
    def alloc_buffer(self, seq_len:int):
        pass
        
    def fill(self, 
        layer_idx:int,
        request_id: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        seq_len:int):
        
        if layer_idx in self.dense_layers:
            
            self.flashinfer_kv_cache[layer_idx][request_id][0].copy_(key_cache.transpose(0,1))
            self.flashinfer_kv_cache[layer_idx][request_id][1].copy_(value_cache.transpose(0,1))
            self.dense_kv_last_page_len[request_id] = seq_len
            
        else:
            sink_tokens_key = key_cache[:self.num_sink_tokens]
            sink_tokens_value = value_cache[:self.num_sink_tokens]
            
            local_tokens_key = key_cache[seq_len-self.num_local_tokens:seq_len]
            local_tokens_value = value_cache[seq_len-self.num_local_tokens:seq_len]
            
            key = torch.cat([sink_tokens_key, local_tokens_key], dim=0).transpose(0,1)
            value = torch.cat([sink_tokens_value, local_tokens_value], dim=0).transpose(0,1)
            
            
            offload_key = key_cache[self.num_sink_tokens: seq_len-self.num_local_tokens]
            offload_value = value_cache[self.num_sink_tokens: seq_len-self.num_local_tokens]
            
            offload_key = offload_key.transpose(0,1).contiguous()
            offload_value = offload_value.transpose(0,1).contiguous()
            
            avg_k = offload_key.mean(dim=1, keepdim=True)
            
            key = key - avg_k
            offload_key = offload_key - avg_k
            kn = offload_key.norm(p=2, dim=-1).float()
            
            self.avg_k[layer_idx][request_id] = avg_k
            
            
            self.flashinfer_kv_cache[layer_idx][request_id][0][...,:self.num_sink_tokens + self.num_local_tokens,:].copy_(key)
            self.flashinfer_kv_cache[layer_idx][request_id][1][...,:self.num_sink_tokens + self.num_local_tokens,:].copy_(value)
            self.kv_last_page_len[request_id] = (self.num_sink_tokens + self.num_local_tokens)
            
            
            self.nnz[request_id * self.num_attention_heads: (request_id + 1) * self.num_attention_heads]  = offload_key.shape[1]
            offload_key = offload_key.cpu()
            offload_value = offload_value.cpu()
            kn = kn.cpu()
            self.attn_server.fill(layer_idx, request_id, offload_key, offload_value, kn)
            
    
    def build_table(self, 
        layer_idx:int,
        request_id: int,
        seq_len:int):
        
        
        pass
        
    def plan(self):
        
        self.kv_last_page_len += 1
        self.decode_wrapper.plan(
        self.kv_page_indptr,
        self.kv_page_indices,
        self.kv_last_page_len,
        self.num_attention_heads,
        self.num_key_value_heads,
        self.head_dim,
        self.page_size,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        data_type=torch.bfloat16
    )     
        
        self.dense_kv_last_page_len += 1
        self.dense_decode_wrapper.plan(
        self.dense_kv_page_indptr,
        self.dense_kv_page_indices,
        self.dense_kv_last_page_len,
        self.num_attention_heads,
        self.num_key_value_heads,
        self.head_dim,
        self.dense_page_size,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        data_type=torch.bfloat16
    )     
    
    
    
    def decode(
        self, 
        query_states:torch.Tensor, 
        key_states:torch.Tensor, 
        value_states:torch.Tensor,
        layer_idx:int):
        
        if layer_idx in self.dense_layers:
            
            key_states = key_states.reshape(self.batch_size, self.num_key_value_heads, self.head_dim)
            value_states = value_states.reshape(self.batch_size, self.num_key_value_heads, self.head_dim)
            
            flashinfer.append_paged_kv_cache(
                key_states,
                value_states,
                self.dense_kv_page_indptr,
                self.flashinfer_kv_cache[layer_idx],
                self.dense_kv_page_indices,
                self.dense_kv_page_indptr,
                self.dense_kv_last_page_len,
                kv_layout="HND"
            )
            
            q = query_states.reshape(self.batch_size, self.num_attention_heads, self.head_dim)
            hidden_states = self.dense_decode_wrapper.run(
            q, 
            self.flashinfer_kv_cache[layer_idx]
            )
            hidden_states = hidden_states.reshape(self.batch_size, 1, self.hidden_size)
           
            return hidden_states
        
        else:
            
            
            bsz, _, q_len, _ = query_states.shape
            
            self.pinned_query.copy_(query_states.reshape(self.batch_size * self.num_attention_heads, self.head_dim))
            
            key_states = key_states - self.avg_k[layer_idx]
            
            key_states = key_states.reshape(self.batch_size, self.num_key_value_heads, self.head_dim)
            value_states = value_states.reshape(self.batch_size, self.num_key_value_heads, self.head_dim)
            
            
            flashinfer.append_paged_kv_cache(
                key_states,
                value_states,
                self.kv_page_indptr,
                self.flashinfer_kv_cache[layer_idx],
                self.kv_page_indices,
                self.kv_page_indptr,
                self.kv_last_page_len,
                kv_layout="HND"
            )
            
            q = query_states.reshape(self.batch_size, self.num_attention_heads, self.head_dim)
            gpu_hidden_states, gpu_lse = self.decode_wrapper.run_return_lse(
            q, 
            self.flashinfer_kv_cache[layer_idx]
            )
            
            self.attn_server.full_attention(layer_idx, self.output, self.max_value_expsum, self.pinned_query.float(), self.nnz)
            self.max_value_expsum_cuda.copy_(self.max_value_expsum[1], non_blocking=True)
            self.output_cuda.copy_(self.output, non_blocking=True)

            cpu_lse = self.max_value_expsum_cuda.reshape(self.batch_size, self.num_attention_heads)
            
            cpu_hidden_states = self.output_cuda.reshape(self.batch_size, self.num_attention_heads, self.head_dim)
            hidden_states, _ = flashinfer.merge_state(gpu_hidden_states, gpu_lse, cpu_hidden_states, cpu_lse)
        
            hidden_states = hidden_states.reshape(bsz, q_len, self.hidden_size)
            
            return hidden_states 
    def clear(self):
        self.nnz.zero_()
        self.max_value_expsum.zero_()
        self.output_cuda.zero_()
        self.max_value_expsum_cuda.zero_()
        self.output.zero_()
        self.pinned_query.zero_()
        self.workspace_buffer.zero_()
        self.dense_workspace_buffer.zero_()
        for i in range(self.num_layers):
            self.avg_k[i].zero_()
            self.flashinfer_kv_cache[i].zero_()
            
        self.kv_last_page_len.zero_()
        self.dense_kv_last_page_len.zero_()
        self.attn_server.clear()


