#include "lsh.h"

static inline void fast_memcpy_avx512(int* dest, const int* src, size_t count) {
    const size_t avx512_width = 16; // Number of 32-bit integers in 512 bits
    size_t i = 0;

    // Calculate the number of full AVX-512 vectors to process
    size_t vec_count = count / avx512_width;

    // Process data in AVX-512 chunks
    for (; i < vec_count * avx512_width; i += avx512_width) {
        // Load 16 integers from src (unaligned)
        __m512i data = _mm512_loadu_si512(reinterpret_cast<const void*>(src + i));

        // Store 16 integers to dest (unaligned)
        _mm512_storeu_si512(reinterpret_cast<void*>(dest + i), data);
    }

    // Handle remaining elements that don't fit into a full AVX-512 vector
    for (; i < count; ++i) {
        dest[i] = src[i];
    }
}

LSH::LSH(){
    this->allocated = false;
}

LSH::~LSH(){

    if (this->allocated) {
        for (int i = 0; i < this->num_layers; ++i){
            delete [] this->table_start[i];
            delete [] this->table_end[i];
            delete [] this->table[i];
        }
        delete [] this->mask;
        //cudaFreeHost(this->query_buffer);
        //cudaStreamDestroy(this->stream);
    }
    this->allocated = false;
}

void LSH::alloc(
int K, 
int L, 
int num_layers, 
int num_attention_heads, 
int num_key_value_heads, 
int batch_size,
int max_length)
{

    this->K = K;
    this->L = L;
    this->num_buckets = static_cast<int>(pow(2, this->K));
    this->num_layers = num_layers;
    this->num_attention_heads = num_attention_heads;
    this->num_key_value_heads = num_key_value_heads;
    this->batch_size = batch_size;
    this->max_length = max_length;
    this->num_attention_groups = static_cast<int>(this->num_attention_heads / this->num_key_value_heads);
    for (int i = 0; i < this->num_layers; ++i){

        int * start = new int [this->batch_size * this->num_key_value_heads * this->L * this->num_buckets]; 
        memset(start, 0, this->batch_size * this->num_key_value_heads * this->L * this->num_buckets * sizeof(int));
        this->table_start.push_back(start);

        int * end = new int [this->batch_size * this->num_key_value_heads * this->L * this->num_buckets]; 
        memset(end, 0, this->batch_size * this->num_key_value_heads * this->L * this->num_buckets * sizeof(int));
        this->table_end.push_back(end);

        int * content = new int [this->batch_size * this->num_key_value_heads * this->L * this->max_length]; 
        memset(content, 0, this->batch_size * this->num_key_value_heads * this->L * this->max_length * sizeof(int));
        this->table.push_back(content);

        // std::vector<std::vector<std::vector<std::vector<int>>>> ht_li(this->batch_size * this->num_key_value_heads, 
        // std::vector<std::vector<std::vector<int>>>(this->L, 
        // std::vector<std::vector<int>>(this->num_buckets, std::vector<int>(0))));
        // this->ht.push_back(ht_li);

    }

    
    this->mask = new uint8_t [this->batch_size * this->num_attention_heads * this->max_length];
    memset(mask, 0, this->batch_size * this->num_attention_heads * this->max_length * sizeof(uint8_t));
    //cudaHostAlloc((void**)&query_buffer, this->batch_size * this->num_attention_heads * this->L * sizeof(int), cudaHostAllocDefault);
    //cudaStreamCreate(&this->stream);
    this->allocated = true;

}

void LSH::fastfill(
int layer_id, 
int request_id, 
torch::Tensor hash_code_pt
)
{

int seq_len = hash_code_pt.size(2);
int offset = request_id * this->num_key_value_heads;

// #pragma omp parallel for schedule(static,1) num_threads(LSH_THREADS)
// for (int l = 0; l < this->L; ++l){
//     for(int h = 0; h < this->num_key_value_heads; ++h){
//         for (int s = 0; s < seq_len; ++s){
//             const int v = hash_code_pt[h][l][s].item<int>();
//             this->ht[layer_id][offset + h][l][v].push_back(s);
//         }
//     }
// }

int * hash_code = static_cast<int *>(hash_code_pt.data_ptr());
int stride = this->num_key_value_heads * this->L * this->num_buckets;
int * start = this->table_start[layer_id] + request_id * stride;
int * end = this->table_end[layer_id] + request_id * stride;

for(int i = 0;  i < this->num_key_value_heads; ++i){
        const int * v_i = hash_code + i * (this->L * seq_len);
        int *ms_i = start + i * (this->L * num_buckets);
        int *me_i = end+ i * (this->L * num_buckets);
        for(int j=0; j < this->L; ++j)
            {
            const int * v_ij = v_i + j * seq_len;
            int *ms_ij = ms_i + j * num_buckets;
            int *me_ij = me_i + j * num_buckets;
            int *mo_ij = new int [num_buckets];
            memset(mo_ij, 0, num_buckets * sizeof(int));
            for(int k=0; k<seq_len; ++k){
                const int v = v_ij[k];
                me_ij[v] += 1;
            }

            for(int b=1; b<this->num_buckets; ++b){
                ms_ij[b] =  me_ij[b-1];
                me_ij[b] += ms_ij[b]; 
            }
            }
}


}
void LSH::fill(
int layer_id, 
int request_id, 
torch::Tensor sorted_hash_code_pt,
torch::Tensor sorted_indices_pt
)
{

assert(layer_id >= 0);
assert(layer_id < this->num_layers);

assert(request_id >= 0);
assert(request_id < this->batch_size);

assert(sorted_hash_code_pt.size(0) == this->num_key_value_heads);
assert(sorted_hash_code_pt.size(1) == this->L);
assert(sorted_hash_code_pt.size(2) <= this->max_length);

int seq_len = sorted_hash_code_pt.size(2);
int stride = this->num_key_value_heads * this->L * this->num_buckets;
int16_t * sorted_hash_code = static_cast<int16_t *>(sorted_hash_code_pt.data_ptr());

int * start = this->table_start[layer_id] + request_id * stride;
int * end = this->table_end[layer_id] + request_id * stride;

#pragma omp parallel for collapse(2) schedule(static)
for(int j=0; j < this->L; ++j){
    for(int i = 0;  i < this->num_key_value_heads; ++i){
        const int16_t * v_i = sorted_hash_code + i * (this->L * seq_len);
        int *ms_i = start + i * (this->L * num_buckets);
        int *me_i = end+ i * (this->L * num_buckets);
        const int16_t * v_ij = v_i + j * seq_len;
        int *ms_ij = ms_i + j * num_buckets;
        int *me_ij = me_i + j * num_buckets;
        for(int k=0; k<seq_len; ++k){
            const int v = static_cast<int>(v_ij[k]);
            if(me_ij[v]==0) {
                ms_ij[v] = k;
                me_ij[v] = k+1;
            } else {
                me_ij[v] = me_ij[v] + 1;
            }
        }
            
    }
 }

assert(sorted_indices_pt.size(0) == this->num_key_value_heads);
assert(sorted_indices_pt.size(1) == this->L);
assert(sorted_indices_pt.size(2) == seq_len);

int * sorted_indices = static_cast<int *>(sorted_indices_pt.data_ptr());
int * content = this->table[layer_id] + request_id * this->num_key_value_heads * this->L * this->max_length;

#pragma omp parallel for schedule(static,1) num_threads(LSH_THREADS)
for (int i = 0; i < this->num_key_value_heads * this->L; ++i){
    fast_memcpy_avx512(content + i * max_length, sorted_indices + i * seq_len, seq_len);
}
}

void LSH::copy( 
torch::Tensor query_pt)
{
    //cudaMemcpyAsync(this->query_buffer, query_pt.data_ptr(), this->batch_size * this->num_attention_heads * this->L * sizeof(int), cudaMemcpyDeviceToHost, this->stream);
}


void LSH::batch_retrieve(
int layer_id, 
torch::Tensor query_pt, 
torch::Tensor results_pt, 
torch::Tensor nnz_pt)
{
    
int * query = static_cast<int *>(query_pt.data_ptr());

//int * query = this->query_buffer;
int * results = static_cast<int *>(results_pt.data_ptr());
int * nnz = static_cast<int *>(nnz_pt.data_ptr());

assert(query_pt.size(0) == this->batch_size * this->num_attention_heads);
assert(query_pt.size(1) == this->L);
assert(results_pt.size(0) == this->batch_size * this->num_attention_heads);
assert(results_pt.size(1) == this->max_length);
assert(nnz_pt.size(0) == this->batch_size * this->num_attention_heads);
//cudaStreamSynchronize(this->stream);
#pragma omp parallel for schedule(static,1) num_threads(LSH_THREADS)
for (int head_id = 0; head_id < this->batch_size * this->num_attention_heads; ++head_id)
{
    nnz[head_id] = this->retrieve(
        layer_id,
        head_id,
        query,
        results
    );
}


}

int LSH::retrieve(
int layer_id, 
int head_id, 
const int* __restrict query, 
int* __restrict results)
{

        // 预计算 group_id 以减少冗余计算
        int group_id = head_id / this->num_attention_groups;
        int* __restrict start = this->table_start[layer_id] + group_id * this->L * this->num_buckets;
        int* __restrict end = this->table_end[layer_id] + group_id * this->L * this->num_buckets;
        int* __restrict content = this->table[layer_id] + group_id * this->L * this->max_length;
        const int* __restrict q = query + head_id * this->L;
        int* __restrict result = results + head_id * this->max_length;
        uint8_t* __restrict tmask = reinterpret_cast<uint8_t*>(mask + head_id * this->max_length);
        
        // 初始化 tmask
        memset(tmask, 0, this->max_length);
        
        int offset = 0;
        int* result_ptr = result;
        
        for (int i = 0; i < this->L; ++i){
            int q_i = q[i];
            int* __restrict m_start = start + i * this->num_buckets;
            int* __restrict m_end = end + i * this->num_buckets;
            int* __restrict m_content = content + i * this->max_length;
            int start_pos = m_start[q_i];
            int end_pos = m_end[q_i];
            for (int j = start_pos; j < end_pos; ++j){
                int idx = m_content[j];
                uint8_t mask_val = tmask[idx];
                // 提示编译器 mask_val == 0 是更可能的情况
                if (__builtin_expect(mask_val == 0, 1)){
                    tmask[idx] = 1;
                }
                else if (__builtin_expect(mask_val == 1, 0)){
                    tmask[idx] = 2;
                    *result_ptr++ = idx;
                }
            }
        }
        
        offset = result_ptr - result;
        return offset;
}




void LSH::clear()
{
#pragma omp parallel for schedule(static,1) num_threads(LSH_THREADS)
for (int i = 0; i < this->num_layers; ++i){

        memset(this->table_start[i], 0, this->batch_size * this->num_key_value_heads * this->L * this->num_buckets * sizeof(int));
        
        memset(this->table_end[i], 0, this->batch_size * this->num_key_value_heads * this->L * this->num_buckets * sizeof(int));
        memset(this->table[i], 0, this->batch_size * this->num_key_value_heads * this->L * this->max_length * sizeof(int));
    }

    memset(this->mask, 0, this->batch_size * this->num_attention_heads * this->max_length * sizeof(uint8_t));

}

torch::Tensor LSH::get_mask()
{

auto options = torch::TensorOptions().dtype(torch::kInt8);
torch::Tensor tensor = torch::from_blob(this->mask, {this->batch_size, this->num_attention_heads, this->max_length}, options);
return tensor;
}

PYBIND11_MODULE(lsh, m) {
    py::class_<LSH>(m, "LSH")
        .def(py::init<>())
        .def("alloc", &LSH::alloc)
        .def("fill", &LSH::fill)
        .def("clear", &LSH::clear)
        .def("copy", &LSH::copy)
        .def("fastfill", &LSH::fastfill)
        .def("batch_retrieve", &LSH::batch_retrieve)
        .def("get_mask", &LSH::get_mask);
}