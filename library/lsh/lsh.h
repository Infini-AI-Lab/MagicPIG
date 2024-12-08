#pragma once
#include<cstdint>
#include<cstdio>
#include<cstring>
#include<iostream>
#include<torch/extension.h>
#include<cassert>
#include<algorithm>
#include<vector>
#include<math.h>
#include<immintrin.h>
#define LSH_THREADS 64

class LSH{
    public: 
        LSH();
        ~LSH();
        void alloc(int K, int L, int num_layers, int num_attention_heads, int num_key_value_heads, int batch_size, int max_length);
        void fill(int layer_id, int request_id, torch::Tensor sorted_hash_code_pt, torch::Tensor sorted_indices_pt);
        void fastfill(int layer_id, int request_id, torch::Tensor hash_code_pt);
        void copy(torch::Tensor query_pt);
        void clear();
        void batch_retrieve(int layer_id, torch::Tensor query_pt, torch::Tensor results_pt, torch::Tensor nnz_pt);
        torch::Tensor get_table_start(int layer_id);
        torch::Tensor get_table_end(int layer_id);
        torch::Tensor get_table(int layer_id);
        torch::Tensor get_mask();
    private:
        int K;
        int L;
        int num_buckets;
        int num_layers;
        int num_attention_heads;
        int num_key_value_heads;
        int num_attention_groups;
        int batch_size;
        int max_length;
        bool allocated;
        uint8_t* mask;
        std::vector<int*> table_start;
        std::vector<int*> table_end;
        std::vector<int*> table;
        int retrieve(int layer_id, int head_id, const int* __restrict query, int* __restrict results);
};