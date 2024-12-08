#pragma once

#include<torch/extension.h>
#include<math.h>
#include "fbgemm/FbgemmConvert.h"
#include<immintrin.h>
#include<vector>
#include<omp.h>
using namespace fbgemm;
#define ATTENTION_THREADS 64



class SparseAttentionServer{

    public:
        SparseAttentionServer();
        ~SparseAttentionServer();
        void alloc(int num_layers, int num_attention_heads, int num_key_value_heads, int head_dim, int batch_size, int max_length);
        void fill(int layer_id, int request_id, torch::Tensor k, torch::Tensor v, torch::Tensor kn);
        void clear();
        void attention_wrapper(int layer_id, int K, int L, torch::Tensor output_pt, torch::Tensor max_value_expsum_pt, torch::Tensor query_pt, torch::Tensor query_norm_pt, torch::Tensor ind_pt, torch::Tensor nnz_pt);
        void attention_wrapper_bf16(int layer_id, int K, int L, torch::Tensor output_pt, torch::Tensor max_value_expsum_pt, torch::Tensor query_pt, torch::Tensor query_norm_pt, torch::Tensor ind_pt, torch::Tensor nnz_pt);
        void attention(int layer_id, int K, int L, torch::Tensor output_pt, torch::Tensor max_value_expsum_pt, torch::Tensor query_pt, torch::Tensor query_norm_pt, torch::Tensor ind_pt, torch::Tensor nnz_pt);
        void attention_bf16(int layer_id, int K, int L, torch::Tensor output_pt, torch::Tensor max_value_expsum_pt, torch::Tensor query_pt, torch::Tensor query_norm_pt, torch::Tensor ind_pt, torch::Tensor nnz_pt);
        void dynamic_attention(int layer_id, int K, int L, torch::Tensor output_pt, torch::Tensor max_value_expsum_pt, torch::Tensor query_pt, torch::Tensor query_norm_pt, torch::Tensor ind_pt, torch::Tensor nnz_pt);
        void dynamic_attention_bf16(int layer_id, int K, int L, torch::Tensor output_pt, torch::Tensor max_value_expsum_pt, torch::Tensor query_pt, torch::Tensor query_norm_pt, torch::Tensor ind_pt, torch::Tensor nnz_pt);
        void full_attention(int layer_id, torch::Tensor output_pt, torch::Tensor max_value_expsum_pt, torch::Tensor query_pt, torch::Tensor nnz_pt);
        void scheduled_attention(int layer_id, int K, int L, torch::Tensor output_pt, torch::Tensor max_value_expsum_pt, torch::Tensor query_pt, torch::Tensor query_norm_pt, torch::Tensor ind_pt, torch::Tensor nnz_pt);
        void scheduled_attention_bf16(int layer_id, int K, int L, torch::Tensor output_pt, torch::Tensor max_value_expsum_pt, torch::Tensor query_pt, torch::Tensor query_norm_pt, torch::Tensor ind_pt, torch::Tensor nnz_pt);
        torch::Tensor get_key_cache(int layer_id);
        torch::Tensor get_value_cache(int layer_id);
        torch::Tensor get_key_norm(int layer_id);
        torch::Tensor get_score();
    private:
        int num_layers;
        int num_attention_heads;
        int num_key_value_heads;
        int head_dim;
        int num_attention_groups;
        int batch_size;
        int max_length;
        bool require_transform;
        bool allocated;
        std::vector<bfloat16 *>key_cache;
        std::vector<bfloat16 *>value_cache;
        
        std::vector<float *>key_norm;
        float* attention_score;
        float* query_buffer;
        bfloat16* output_buffer;
};
