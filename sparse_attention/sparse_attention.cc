#include "sparse_attention.h"

const __m512 LOG2E_VEC = _mm512_set1_ps(1.44269504089f); // log2(e)
const __m512 MAGIC_FLOAT_BIAS = _mm512_set1_ps(12582912.0f); // Bias for integer part
const __m512 ONE_VEC = _mm512_set1_ps(1.0f);
const __m512 LN2_PART_VEC = _mm512_set1_ps(0.69314718056f); // ln(2) for scaling

// Polynomial coefficients for 2^f approximation (Taylor expansion for small values)
const __m512 EXP_POLY_COEFFS[4] = {
    _mm512_set1_ps(1.0000000000f),
    _mm512_set1_ps(0.6931471806f),
    _mm512_set1_ps(0.2402265069f),
    _mm512_set1_ps(0.05550410866f),
};

// Approximation of exp using AVX512
__m512 avx512_exp_ps(__m512 x) {
    // Step 1: Scale x to base-2 exponent
    __m512 scaled = _mm512_mul_ps(x, LOG2E_VEC);

    // Step 2: Split into integer and fractional parts
    __m512i int_part = _mm512_cvttps_epi32(scaled); // Integer part
    __m512 frac_part = _mm512_sub_ps(scaled, _mm512_cvtepi32_ps(int_part)); // Fractional part

    // Step 3: Compute 2^integer_part using bit manipulation
    __m512 int_exp = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(int_part, _mm512_set1_epi32(127)), 23));

    // Step 4: Compute 2^frac_part using polynomial approximation
    __m512 poly = EXP_POLY_COEFFS[3];
    poly = _mm512_fmadd_ps(poly, frac_part, EXP_POLY_COEFFS[2]);
    poly = _mm512_fmadd_ps(poly, frac_part, EXP_POLY_COEFFS[1]);
    poly = _mm512_fmadd_ps(poly, frac_part, EXP_POLY_COEFFS[0]);

    // Step 5: Combine integer and fractional parts
    return _mm512_mul_ps(int_exp, poly);
}

void qk_kernel(
const bfloat16 *key, 
const int *ind,
const float *query, 
float *attn_weight,
const int HEAD_DIM,
const int nnz) 
{  
  memset(attn_weight, 0, sizeof(float) * nnz);
  float w_f32[16];
  int nnz_16 = (nnz % 16 == 0) ? (nnz): (nnz + 16 - nnz % 16);
  for (int io = 0; io < (nnz_16 / 16); ++io) {
    
    for (int jo = 0; jo < (HEAD_DIM / 16); ++jo) {
      __m512 x_tile = _mm512_loadu_ps(query + jo * 16);
      for (int ii = 0; ii < 16; ++ii) {
        const int i = io * 16 + ii;
        const int row = ind[i];

        Bfloat16ToFloat_avx512(key + row * HEAD_DIM + jo * 16, w_f32, 16);
        __m512 w_tile = _mm512_loadu_ps(w_f32);

        __m512 wx = _mm512_mul_ps(w_tile, x_tile);
        float sum = _mm512_reduce_add_ps(wx);
        attn_weight[io * 16 + ii] += sum;
      }
    }

  }
}
#ifdef __AVX512BF16__
void qk_kernel_bf16_impl(
const bfloat16 *key, 
const int *ind,
const bfloat16 *query, 
float *attn_weight,
const int HEAD_DIM,
const int nnz) 
{  
    memset(attn_weight, 0, sizeof(float) * nnz);
    int nnz_16 = (nnz % 16 == 0) ? nnz : (nnz + 16 - nnz % 16);

    for (int io = 0; io < (nnz_16 / 16); ++io) {
        for (int jo = 0; jo < (HEAD_DIM / 32); ++jo) {
            // Load 32 bfloat16 values from query into x_tile
            __m512bh x_tile = __m512bh(_mm512_loadu_epi16(reinterpret_cast<const void*>(query + jo * 32)));

            for (int ii = 0; ii < 16; ++ii) {
                const int i = io * 16 + ii;
                const int row = ind[i];

                // Load 32 bfloat16 values from key into w_tile
                __m512bh w_tile = __m512bh(_mm512_loadu_epi16(reinterpret_cast<const void*>(key + row * HEAD_DIM + jo * 32)));

                // Compute dot product of x_tile and w_tile
                __m512 result = _mm512_dpbf16_ps(_mm512_setzero_ps(), w_tile, x_tile);

                // Reduce and sum the result into a scalar
                float sum = _mm512_reduce_add_ps(result);

                // Update attn_weight
                attn_weight[io * 16 + ii] += sum;
            }
        }
    }
}
#endif

void qk_kernel_full(
    const bfloat16 *key,      // k 的输入，形状为 max_length x 128
    const float *query,       // q 的输入，形状为  num_attention_groups x 128
    float *attn_weight,       // 输出，形状为 num_attention_groups x max_length
    const int num_attention_groups,
    const int HEAD_DIM,       // 每个向量的维度，这里是 128
    const int nnz,           // 只计算k的前nnz <= max_length
    const int max_length
) {
    const int HEAD_DIM_16 = HEAD_DIM / 16; // 每次处理 16 个维度
    const int TILE_SIZE = 64; // 块大小，处理 TILE_SIZE 行 key 一次
    
    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int ko_start = 0; ko_start < nnz; ko_start += TILE_SIZE) {
        float key_tile[TILE_SIZE][HEAD_DIM]; // 局部缓存 key 块，存为 float 格式
        int tile_rows = (ko_start + TILE_SIZE <= nnz) ? TILE_SIZE : (nnz - ko_start);
        
        // 加载当前块的 key 并转换为 float
        for (int i = 0; i < tile_rows; ++i) {
            for (int j = 0; j < HEAD_DIM_16; ++j) {
                Bfloat16ToFloat_avx512(
                    key + (ko_start + i) * HEAD_DIM + j * 16,
                    key_tile[i] + j * 16,
                    16
                );
            }
        }
        
        // 遍历 query 的每一行
        for (int qi = 0; qi < num_attention_groups; ++qi) {
            const float* query_vec = query + qi * HEAD_DIM; // 当前查询向量
            
            for (int j = 0; j < HEAD_DIM_16; ++j) {
    // 预加载 query 的一部分到局部变量
    __m512 q_tile = _mm512_loadu_ps(query_vec + j * 16);

    // 遍历当前块的所有 key 行
    for (int ki = 0; ki < tile_rows; ++ki) {
        if (j == 0) {
            // 初始化累加器，仅在每行的第一轮计算时清零
            attn_weight[qi * max_length + ko_start + ki] = 0.0f;
        }

        // 加载 key 块
        __m512 k_tile = _mm512_loadu_ps(key_tile[ki] + j * 16);

        // 逐元素乘法并累加到结果中
        __m512 prod = _mm512_mul_ps(q_tile, k_tile);
        attn_weight[qi * max_length + ko_start + ki] += _mm512_reduce_add_ps(prod);
    }
}

        }
    }
}



void transform_kernel(
float *score, 
const int nnz,
float q_norm,
float *k_norm,
const int k,
const int l,
const float sqrt_dim,
const int *indices) { 
  
  for(int j = 0; j < nnz; ++j){
        const int ind = indices[j];
        float norm = (q_norm * k_norm[ind]);
        float theta = acosf(score[j] / norm);
        float proba = (1 - theta / M_PI);
        float p = powf(proba, k);
        float q = 1 - p;
        float w = 1  - powf(q, l-1) * (l * p + q);
        score[j] = score[j] / sqrt_dim - logf(w + 1e-4);
    }
}

void softmax_kernel(
float *score, 
const int nnz,
float *max_value,
float *expsum) { 
  
  const int nnz_16 = (nnz % 16 == 0) ? nnz : nnz + 16 - nnz % 16;

    // Step 1: Find the maximum value (reduction max)
    float max_val = *std::max_element(score, score + nnz);
    __m512 max_vec = _mm512_set1_ps(max_val);

    // Step 2: Compute exp(x - max) and sum of exponents
    __m512 sum_vec = _mm512_setzero_ps();
    for (int i = 0; i < nnz / 16; ++i) {
        __m512 scores = _mm512_loadu_ps(score + i * 16);
        __m512 shifted_scores = _mm512_sub_ps(scores, max_vec); // x - max
        __m512 exp_scores = avx512_exp_ps(shifted_scores); // exp(x - max)
        _mm512_storeu_ps(score + i * 16, exp_scores); // Store back
        sum_vec = _mm512_add_ps(sum_vec, exp_scores); // Accumulate sum
    }

    // Step 3: Horizontal sum of `sum_vec`
    float exp_sum = 0.0f;
    float temp[16];
    _mm512_storeu_ps(temp, sum_vec);
    for (int i = 0; i < 16; ++i) {
        exp_sum += temp[i];
    }

    // Process remaining elements (tail loop)
    for (int i = (nnz / 16) * 16; i < nnz; ++i) {
        float shifted_score = score[i] - max_val;
        float exp_score = expf(shifted_score);
        score[i] = exp_score;
        exp_sum += exp_score;
    }

    // Step 4: Normalize scores and store
    __m512 sum = _mm512_set1_ps(exp_sum);
    for (int i = 0; i < nnz / 16; ++i) {
        __m512 scores = _mm512_loadu_ps(score + i * 16);
        __m512 normalized_scores = _mm512_div_ps(scores, sum);
        _mm512_storeu_ps(score + i * 16, normalized_scores);
    }

    // Normalize tail elements
    for (int i = (nnz / 16) * 16; i < nnz; ++i) {
        score[i] /= exp_sum;
    }

    // Step 5: Compute final max_value and expsum
    max_value[0] = max_val * M_LOG2E;
    expsum[0] = log2f(exp_sum) + max_value[0];
}

void softmax_kernel_optimized(
    float *score,
    const int nnz,
    const float head_dim_scale, // sqrt(head_dim)
    float *max_value,
    float *expsum) {

    int nnz_16 = (nnz + 15) & ~15; // Round up to the nearest multiple of 16

    // Scale the scores by 1/sqrt(head_dim)
    __m512 scale_vec = _mm512_set1_ps(1.0f / head_dim_scale);

    // Step 1: Compute max value using AVX512
    __m512 max_vec = _mm512_set1_ps(-INFINITY);
    for (int i = 0; i < nnz_16; i += 16) {
        __m512 s_vec = _mm512_maskz_loadu_ps(i < nnz ? 0xFFFF : ((1 << (nnz % 16)) - 1), score + i);
        s_vec = _mm512_mul_ps(s_vec, scale_vec); // Scale scores
        max_vec = _mm512_max_ps(max_vec, s_vec);
    }
    float max_scalar = _mm512_reduce_max_ps(max_vec);

    // Step 2: Compute exponential and sum using AVX512
    __m512 exp_sum_vec = _mm512_set1_ps(0.0f);
    __m512 max_scalar_vec = _mm512_set1_ps(max_scalar);
    for (int i = 0; i < nnz_16; i += 16) {
        __m512 s_vec = _mm512_maskz_loadu_ps(i < nnz ? 0xFFFF : ((1 << (nnz % 16)) - 1), score + i);
        s_vec = _mm512_mul_ps(s_vec, scale_vec); // Scale scores
        __m512 exp_vec = avx512_exp_ps(_mm512_sub_ps(s_vec, max_scalar_vec)); // Subtract max before exp
        _mm512_mask_storeu_ps(score + i, i < nnz ? 0xFFFF : ((1 << (nnz % 16)) - 1), exp_vec);
        exp_sum_vec = _mm512_add_ps(exp_sum_vec, exp_vec);
    }
    float exp_sum_scalar = _mm512_reduce_add_ps(exp_sum_vec);

    // Step 3: Normalize the scores
    __m512 exp_sum_scalar_vec = _mm512_set1_ps(exp_sum_scalar);
    for (int i = 0; i < nnz_16; i += 16) {
        __m512 s_vec = _mm512_maskz_loadu_ps(i < nnz ? 0xFFFF : ((1 << (nnz % 16)) - 1), score + i);
        __m512 norm_vec = _mm512_div_ps(s_vec, exp_sum_scalar_vec);
        _mm512_mask_storeu_ps(score + i, i < nnz ? 0xFFFF : ((1 << (nnz % 16)) - 1), norm_vec);
    }

    // Save max_value and expsum
    max_value[0] = max_scalar * M_LOG2E;
    expsum[0] = log2f(exp_sum_scalar) + max_value[0];
}


void softmax_kernel_full(
float *score, 
const int nnz,
float sqrt_dim,
float *max_value,
float *expsum
) { 
  int nnz_16 = (nnz % 16 == 0) ? (nnz): (nnz + 16 - nnz % 16);
  __m512 hdiv = _mm512_set1_ps(sqrt_dim);
  for (int i = 0; i < nnz_16 / 16; ++i){
        __m512 s_i = _mm512_loadu_ps(score + 16 * i);
        s_i = _mm512_div_ps(s_i, hdiv);
        _mm512_storeu_ps(score + 16 * i, s_i);
    }
  float m = *std::max_element(score,score+nnz);
  float exp_sum = 0.f;
  for (int i = 0; i < nnz; ++i){
      score[i] = expf32(score[i] - m);
      exp_sum += score[i];
    }

  __m512 sum = _mm512_set1_ps(exp_sum);
  for (int i = 0; i < nnz_16 / 16; ++i){
        __m512 s_i = _mm512_loadu_ps(score + 16 * i);
        s_i = _mm512_div_ps(s_i, sum);
        _mm512_storeu_ps(score + 16 * i, s_i);
    }
  
  max_value[0] = m * M_LOG2E;
  expsum[0] = log2f32(exp_sum) + max_value[0];
}

void wv_kernel(
const bfloat16 *value, 
const int *ind,
const float *attn_weight, 
bfloat16 *output, 
const int HEAD_DIM,
const int nnz) {
    float y_f32[16];
    float w_f32[16];
    for (int io = 0; io < (HEAD_DIM / 16); ++io) {
        memset(y_f32, 0, sizeof(float) * 16);
        __m512 y_acc = _mm512_loadu_ps(y_f32);
        for(int jo = 0; jo < nnz; ++jo)
          {
            const int row = ind[jo];
            __m512 x_j = _mm512_set1_ps(attn_weight[jo]);

            Bfloat16ToFloat_avx512(value + row * HEAD_DIM + io * 16, w_f32, 16);
            __m512 w_tile = _mm512_loadu_ps(w_f32);
            //__m512 w_tile = _mm512_loadu_ps(value + row * HEAD_DIM + io * 16);
            y_acc = _mm512_fmadd_ps(w_tile, x_j, y_acc);
          }
        
        _mm512_storeu_ps(y_f32, y_acc);
        FloatToBfloat16_avx512(y_f32, output + io * 16, 16);
    }
}


void wv_kernel_dim128(
const bfloat16 *value, 
const int *ind,
const float *attn_weight, 
bfloat16 *output, 
const int HEAD_DIM,
const int nnz) {

    __m512 y[8];
    float v_f32[16];
    float y_f32[16];
    memset(v_f32, 0, 16 * sizeof(float));
    memset(y_f32, 0, 16 * sizeof(float));
    for (int i = 0; i < 8; ++i){
        y[i] = _mm512_setzero_ps();
    }
    for (int i = 0; i < nnz; ++i){
        __m512 w_i = _mm512_set1_ps(attn_weight[i]);
        const int row = ind[i];

        for (int j = 0; j < 8; ++j){
            Bfloat16ToFloat_avx512(value + row * HEAD_DIM + j * 16, v_f32, 16);
            __m512 v_tile = _mm512_loadu_ps(v_f32);
             y[j] = _mm512_fmadd_ps(v_tile, w_i, y[j]);

        }

    }
    for (int i = 0; i < 8; ++i){
         _mm512_storeu_ps(y_f32, y[i]);
          FloatToBfloat16_avx512(y_f32, output + i * 16, 16);

    }
}


void wv_kernel_dim128_full(
    const bfloat16 *value,        // value 的输入，形状为 max_length x 128
    const float *attn_weight,     // attn_weight 的输入，形状为 num_q x max_length
    bfloat16 *output,             // 输出，形状为 num_q x 128
    const int HEAD_DIM,           // 每个向量的维度，这里为 128
    const int max_length,         // value 的行数
    const int nnz,                // 每行前 nnz 的非零部分
    const int num_q               // 查询向量的数量，必须是 1, 4 或 8
) {
    if (num_q != 4 && num_q != 8 && num_q != 1) {
        return; // 不支持其他数量
    }

    // 全局结果数组
    __m512 y[8][8]; // 每个查询向量最多分成 8 个块（128维）
    for (int q = 0; q < num_q; ++q) {
        for (int i = 0; i < 8; ++i) {
            y[q][i] = _mm512_setzero_ps();
        }
    }

    // 临时数组分配放在线程中，避免 false sharing
    #pragma omp parallel num_threads(ATTENTION_THREADS)
    {
        __m512 y_local[8][8]; // 每个线程的局部累积结果
        float v_f32[16];      // bfloat16 转换到 float
        for (int q = 0; q < num_q; ++q) {
            for (int i = 0; i < 8; ++i) {
                y_local[q][i] = _mm512_setzero_ps(); // 初始化局部向量
            }
        }

        // 使用 OpenMP 并行化非零值主循环
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < nnz; ++i) {
            for (int j = 0; j < 8; ++j) {
                Bfloat16ToFloat_avx512(value + i * HEAD_DIM + j * 16, v_f32, 16);
                __m512 v_tile = _mm512_loadu_ps(v_f32);

                for (int q = 0; q < num_q; ++q) {
                    __m512 w_i = _mm512_set1_ps(attn_weight[q * max_length + i]);
                    y_local[q][j] = _mm512_fmadd_ps(v_tile, w_i, y_local[q][j]);
                }
            }
        }

        // 合并结果到全局变量
        #pragma omp critical
        {
            for (int q = 0; q < num_q; ++q) {
                for (int j = 0; j < 8; ++j) {
                    y[q][j] = _mm512_add_ps(y[q][j], y_local[q][j]);
                }
            }
        }
    }

    // 将结果转换回 bfloat16 并存储
    float y_f32[16]; // 临时存储 float 类型结果
    for (int q = 0; q < num_q; ++q) {
        for (int i = 0; i < 8; ++i) {
            _mm512_storeu_ps(y_f32, y[q][i]);
            FloatToBfloat16_avx512(y_f32, output + q * HEAD_DIM + i * 16, 16);
        }
    }
}




void wv_kernel_dim128_dim_parallel(
const bfloat16 *value, 
const int *ind,
const float *attn_weight, 
bfloat16 *output, 
const int offset,
const int COMPUTE_HEAD_DIM,
const int HEAD_DIM,
const int nnz) {

    __m512 y[4];
    float v_f32[16];
    float y_f32[16];
    memset(v_f32, 0, 16 * sizeof(float));
    memset(y_f32, 0, 16 * sizeof(float));
    for (int i = 0; i < 4; ++i){
        y[i] = _mm512_setzero_ps();
    }
    for (int i = 0; i < nnz; ++i){
        __m512 w_i = _mm512_set1_ps(attn_weight[i]);
        const int row = ind[i];
        for (int j = 0; j < 4; ++j){
            Bfloat16ToFloat_avx512(value + row * HEAD_DIM + j * 16 + offset, v_f32, 16);
            __m512 v_tile = _mm512_loadu_ps(v_f32);
             y[j] = _mm512_fmadd_ps(v_tile, w_i, y[j]);

        }

    }
    for (int i = 0; i < 4; ++i){
         _mm512_storeu_ps(y_f32, y[i]);
          FloatToBfloat16_avx512(y_f32, output + i * 16 + offset, 16);

    }
}

void wv_kernel_dim_parallel(
const bfloat16 *value, 
const int *ind,
const float *attn_weight, 
bfloat16 *output, 
const int offset,
const int COMPUTE_HEAD_DIM,
const int HEAD_DIM,
const int nnz) {
    float y_f32[16];
    float w_f32[16];
    for (int io = 0; io < (COMPUTE_HEAD_DIM / 16); ++io) {
        memset(y_f32, 0, sizeof(float) * 16);
        __m512 y_acc = _mm512_loadu_ps(y_f32);
        for(int jo = 0; jo < nnz; ++jo)
          {
            const int row = ind[jo];
            __m512 x_j = _mm512_set1_ps(attn_weight[jo]);
            Bfloat16ToFloat_avx512(value + row * HEAD_DIM + io * 16 + offset, w_f32, 16);
            __m512 w_tile = _mm512_loadu_ps(w_f32);
            y_acc = _mm512_fmadd_ps(w_tile, x_j, y_acc);
          }
        
        _mm512_storeu_ps(y_f32, y_acc);
        FloatToBfloat16_avx512(y_f32, output + io * 16 + offset, 16);
    }
}

SparseAttentionServer::SparseAttentionServer(){
    this->allocated = false;
}

SparseAttentionServer::~SparseAttentionServer(){
    if(this->allocated){
        for (int i = 0; i < this->num_layers; ++i)
            {
                delete [] this->key_cache[i];
                delete [] this->value_cache[i];
                delete [] this->key_norm[i];
            }
        delete [] this->attention_score;
        delete [] this->query_buffer;
        delete [] this->output_buffer;
        this->allocated = false;
      
    }

}

void SparseAttentionServer::alloc(
int num_layers, 
int num_attention_heads, 
int num_key_value_heads, 
int head_dim, 
int batch_size, 
int max_length)
{   
    this->num_layers = num_layers;
    this->num_attention_heads = num_attention_heads;
    this->num_key_value_heads = num_key_value_heads;
    this->num_attention_groups = static_cast<int>(this->num_attention_heads/this->num_key_value_heads);
    this->head_dim = head_dim;
    this->batch_size = batch_size;
    this->max_length = max_length;
    this->require_transform = true;

    
    for(int i = 0; i < this->num_layers; ++i){
        bfloat16 * k = new bfloat16[this->batch_size * this->num_key_value_heads * this->max_length * this->head_dim];
        bfloat16 * v = new bfloat16[this->batch_size * this->num_key_value_heads * this->max_length * this->head_dim];

        memset(k, 0, this->batch_size * this->num_key_value_heads * this->max_length * this->head_dim * sizeof(bfloat16));
        memset(v, 0, this->batch_size * this->num_key_value_heads * this->max_length * this->head_dim * sizeof(bfloat16));

        float * kn = new float[this->batch_size * this->num_key_value_heads * this->max_length];
        memset(kn, 0, this->batch_size * this->num_key_value_heads * this->max_length * sizeof(float));
        this->key_cache.push_back(k);
        this->value_cache.push_back(v);
        this->key_norm.push_back(kn);
    }
    this->attention_score = new float[this->batch_size * this->num_attention_heads * this->max_length];
    memset(this->attention_score, 0, this->batch_size * this->num_attention_heads * this->max_length * sizeof(float));
    this->allocated = true;

    this->query_buffer = new float[ATTENTION_THREADS * this->head_dim];
    this->output_buffer = new bfloat16[ATTENTION_THREADS * this->head_dim];
}


void SparseAttentionServer::clear()
{
  
    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for(int i = 0; i < this->num_layers; ++i){
        
        memset(this->key_cache[i], 0, this->batch_size * this->num_key_value_heads * this->max_length * this->head_dim * sizeof(bfloat16));
        memset(this->value_cache[i], 0, this->batch_size * this->num_key_value_heads * this->max_length * this->head_dim * sizeof(bfloat16));
        memset(this->key_norm[i], 0, this->batch_size * this->num_key_value_heads * this->max_length * sizeof(float));
    
    }
    memset(this->attention_score, 0, this->batch_size * this->num_attention_heads * this->max_length * sizeof(float));
}


void SparseAttentionServer::fill(
int layer_id, 
int request_id, 
torch::Tensor k, 
torch::Tensor v, 
torch::Tensor kn)
{
    int stride = this->num_key_value_heads * this->max_length * this->head_dim;
    bfloat16 * key = this->key_cache[layer_id] + request_id * stride;
    bfloat16 * value = this->value_cache[layer_id] + request_id * stride;

    //float * key = this->key_cache[layer_id] + request_id * stride;
    //float * value = this->value_cache[layer_id] + request_id * stride;
    float * k_norm = this->key_norm[layer_id] + request_id * this->num_key_value_heads * this->max_length;

    int seq_len = k.size(1);

    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < this->num_key_value_heads; ++i){
        memcpy(key + i * this->max_length * this->head_dim, static_cast<bfloat16 *>(k.data_ptr()) + i * seq_len * this->head_dim, seq_len * this->head_dim * sizeof(bfloat16));
        memcpy(value + i * this->max_length * this->head_dim, static_cast<bfloat16 *>(v.data_ptr()) + i * seq_len * this->head_dim, seq_len * this->head_dim * sizeof(bfloat16));

        //memcpy(key + i * this->max_length * this->head_dim, static_cast<float *>(k.data_ptr()) + i * seq_len * this->head_dim, seq_len * this->head_dim * sizeof(float));
        //memcpy(value + i * this->max_length * this->head_dim, static_cast<float *>(v.data_ptr()) + i * seq_len * this->head_dim, seq_len * this->head_dim * sizeof(float));
        memcpy(k_norm + i * this->max_length, static_cast<float *>(kn.data_ptr()) + i * seq_len, seq_len * sizeof(float));
    }
}

void SparseAttentionServer::attention_wrapper(
int layer_id, 
int K,
int L,
torch::Tensor output_pt,
torch::Tensor max_value_expsum_pt,
torch::Tensor query_pt, 
torch::Tensor query_norm_pt,
torch::Tensor ind_pt, 
torch::Tensor nnz_pt)
{

#ifdef __AVX512BF16__
this->attention_wrapper_bf16(
    layer_id, 
    K,
    L,
    output_pt,
    max_value_expsum_pt,
    query_pt, 
    query_norm_pt,
    ind_pt, 
    nnz_pt);
#else
if (this->batch_size * this->num_attention_heads == 32 && ATTENTION_THREADS == 64)
{
    this->scheduled_attention(
    layer_id, 
    K,
    L,
    output_pt,
    max_value_expsum_pt,
    query_pt, 
    query_norm_pt,
    ind_pt, 
    nnz_pt);
}
else if (this->batch_size * this->num_attention_heads > ATTENTION_THREADS){
    this->dynamic_attention(
    layer_id, 
    K,
    L,
    output_pt,
    max_value_expsum_pt,
    query_pt, 
    query_norm_pt,
    ind_pt, 
    nnz_pt);
} else
{
    this->attention(
    layer_id, 
    K,
    L,
    output_pt,
    max_value_expsum_pt,
    query_pt, 
    query_norm_pt,
    ind_pt, 
    nnz_pt);
}
#endif

};
#ifdef __AVX512BF16__
void SparseAttentionServer::attention_wrapper_bf16(
int layer_id, 
int K,
int L,
torch::Tensor output_pt,
torch::Tensor max_value_expsum_pt,
torch::Tensor query_pt, 
torch::Tensor query_norm_pt,
torch::Tensor ind_pt, 
torch::Tensor nnz_pt)
{

if (this->batch_size * this->num_attention_heads == 32 && ATTENTION_THREADS == 64)
{
    this->scheduled_attention_bf16(
    layer_id, 
    K,
    L,
    output_pt,
    max_value_expsum_pt,
    query_pt, 
    query_norm_pt,
    ind_pt, 
    nnz_pt);
}
else if (this->batch_size * this->num_attention_heads > ATTENTION_THREADS){
    this->dynamic_attention_bf16(
    layer_id, 
    K,
    L,
    output_pt,
    max_value_expsum_pt,
    query_pt, 
    query_norm_pt,
    ind_pt, 
    nnz_pt);
} else
{
    this->attention_bf16(
    layer_id, 
    K,
    L,
    output_pt,
    max_value_expsum_pt,
    query_pt, 
    query_norm_pt,
    ind_pt, 
    nnz_pt);
}


};
#endif

void SparseAttentionServer::dynamic_attention(
int layer_id, 
int K,
int L,
torch::Tensor output_pt,
torch::Tensor max_value_expsum_pt,
torch::Tensor query_pt, 
torch::Tensor query_norm_pt,
torch::Tensor ind_pt, 
torch::Tensor nnz_pt)
{
    bfloat16 * key = this->key_cache[layer_id];
    bfloat16 * value = this->value_cache[layer_id];
    auto q_f32 = query_pt.to(torch::kFloat32);
    float * query = static_cast<float *>(q_f32.data_ptr());
    bfloat16 * output = static_cast<bfloat16 *>(output_pt.data_ptr());
    int * ind = static_cast<int *>(ind_pt.data_ptr());
    int * nnz = static_cast<int *>(nnz_pt.data_ptr());
    float * query_norm = static_cast<float *>(query_norm_pt.data_ptr());
    float * max_value =  static_cast<float *>(max_value_expsum_pt.data_ptr());
    float * expsum = max_value + this->batch_size * this->num_attention_heads;

    #pragma omp parallel for schedule(dynamic,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < this->batch_size * this->num_attention_heads; ++i){
        qk_kernel(
          key + (i/this->num_attention_groups) * this->max_length * this->head_dim,
          ind + i * this->max_length,
          query + i * this->head_dim,
          this->attention_score + i * this->max_length,
          this->head_dim,
          nnz[i]
        );
        transform_kernel(
          this->attention_score + i * this->max_length,
          nnz[i],
          query_norm[i],
          this->key_norm[layer_id] + (i/this->num_attention_groups) * this->max_length, 
          K,
          L,
          sqrtf(this->head_dim),
          ind + i * this->max_length
        );
        softmax_kernel(
          this->attention_score + i * this->max_length,
          nnz[i],
          max_value + i,
          expsum + i
        );
        wv_kernel(
            value + (i/this->num_attention_groups) * this->max_length * this->head_dim,
            ind + i * this->max_length,
            this->attention_score + i * this->max_length,
            output + i * this->head_dim,
            this->head_dim,
            nnz[i]
        );
      }

}
#ifdef __AVX512BF16__
void SparseAttentionServer::dynamic_attention_bf16(
int layer_id, 
int K,
int L,
torch::Tensor output_pt,
torch::Tensor max_value_expsum_pt,
torch::Tensor query_pt, 
torch::Tensor query_norm_pt,
torch::Tensor ind_pt, 
torch::Tensor nnz_pt)
{
    bfloat16 * key = this->key_cache[layer_id];
    bfloat16 * value = this->value_cache[layer_id];
    bfloat16 * query = static_cast<bfloat16 *>(query_pt.data_ptr());
    bfloat16 * output = static_cast<bfloat16 *>(output_pt.data_ptr());
    int * ind = static_cast<int *>(ind_pt.data_ptr());
    int * nnz = static_cast<int *>(nnz_pt.data_ptr());
    float * query_norm = static_cast<float *>(query_norm_pt.data_ptr());
    float * max_value =  static_cast<float *>(max_value_expsum_pt.data_ptr());
    float * expsum = max_value + this->batch_size * this->num_attention_heads;

    #pragma omp parallel for schedule(dynamic,1) num_threads(64)
    for (int i = 0; i < this->batch_size * this->num_attention_heads; ++i){
        qk_kernel_bf16_impl(
          key + (i/this->num_attention_groups) * this->max_length * this->head_dim,
          ind + i * this->max_length,
          query + i * this->head_dim,
          this->attention_score + i * this->max_length,
          this->head_dim,
          nnz[i]
        );
        transform_kernel(
          this->attention_score + i * this->max_length,
          nnz[i],
          query_norm[i],
          this->key_norm[layer_id] + (i/this->num_attention_groups) * this->max_length, 
          K,
          L,
          sqrtf(this->head_dim),
          ind + i * this->max_length
        );
        softmax_kernel(
          this->attention_score + i * this->max_length,
          nnz[i],
          max_value + i,
          expsum + i
        );
        wv_kernel(
            value + (i/this->num_attention_groups) * this->max_length * this->head_dim,
            ind + i * this->max_length,
            this->attention_score + i * this->max_length,
            output + i * this->head_dim,
            this->head_dim,
            nnz[i]
        );
      }

}
#endif
void SparseAttentionServer::attention(
int layer_id, 
int K,
int L,
torch::Tensor output_pt,
torch::Tensor max_value_expsum_pt,
torch::Tensor query_pt, 
torch::Tensor query_norm_pt,
torch::Tensor ind_pt, 
torch::Tensor nnz_pt)
{
    bfloat16 * key = this->key_cache[layer_id];
    bfloat16 * value = this->value_cache[layer_id];
    auto q_f32 = query_pt.to(torch::kFloat32);
    float * query = static_cast<float *>(q_f32.data_ptr());
    bfloat16 * output = static_cast<bfloat16 *>(output_pt.data_ptr());
    int * ind = static_cast<int *>(ind_pt.data_ptr());
    int * nnz = static_cast<int *>(nnz_pt.data_ptr());
    float * query_norm = static_cast<float *>(query_norm_pt.data_ptr());
    float * max_value =  static_cast<float *>(max_value_expsum_pt.data_ptr());
    float * expsum = max_value + this->batch_size * this->num_attention_heads;

    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < this->batch_size * this->num_attention_heads; ++i){
        qk_kernel(
          key + (i/this->num_attention_groups) * this->max_length * this->head_dim,
          ind + i * this->max_length,
          query + i * this->head_dim,
          this->attention_score + i * this->max_length,
          this->head_dim,
          nnz[i]
        );
        transform_kernel(
          this->attention_score + i * this->max_length,
          nnz[i],
          query_norm[i],
          this->key_norm[layer_id] + (i/this->num_attention_groups) * this->max_length, 
          K,
          L,
          sqrtf(this->head_dim),
          ind + i * this->max_length
        );
        softmax_kernel(
          this->attention_score + i * this->max_length,
          nnz[i],
          max_value + i,
          expsum + i
        );
        wv_kernel(
            value + (i/this->num_attention_groups) * this->max_length * this->head_dim,
            ind + i * this->max_length,
            this->attention_score + i * this->max_length,
            output + i * this->head_dim,
            this->head_dim,
            nnz[i]
        );
      }

}

#ifdef __AVX512BF16__
void SparseAttentionServer::attention_bf16(
int layer_id, 
int K,
int L,
torch::Tensor output_pt,
torch::Tensor max_value_expsum_pt,
torch::Tensor query_pt, 
torch::Tensor query_norm_pt,
torch::Tensor ind_pt, 
torch::Tensor nnz_pt)
{
    bfloat16 * key = this->key_cache[layer_id];
    bfloat16 * value = this->value_cache[layer_id];
    
    bfloat16 * query = static_cast<bfloat16 *>(query_pt.data_ptr());
    bfloat16 * output = static_cast<bfloat16 *>(output_pt.data_ptr());
    int * ind = static_cast<int *>(ind_pt.data_ptr());
    int * nnz = static_cast<int *>(nnz_pt.data_ptr());
    float * query_norm = static_cast<float *>(query_norm_pt.data_ptr());
    float * max_value =  static_cast<float *>(max_value_expsum_pt.data_ptr());
    float * expsum = max_value + this->batch_size * this->num_attention_heads;

    #pragma omp parallel for schedule(static,1) num_threads(64)
    for (int i = 0; i < this->batch_size * this->num_attention_heads; ++i){
        qk_kernel_bf16_impl(
          key + (i/this->num_attention_groups) * this->max_length * this->head_dim,
          ind + i * this->max_length,
          query + i * this->head_dim,
          this->attention_score + i * this->max_length,
          this->head_dim,
          nnz[i]
        );
        transform_kernel(
          this->attention_score + i * this->max_length,
          nnz[i],
          query_norm[i],
          this->key_norm[layer_id] + (i/this->num_attention_groups) * this->max_length, 
          K,
          L,
          sqrtf(this->head_dim),
          ind + i * this->max_length
        );
        softmax_kernel(
          this->attention_score + i * this->max_length,
          nnz[i],
          max_value + i,
          expsum + i
        );
        wv_kernel(
            value + (i/this->num_attention_groups) * this->max_length * this->head_dim,
            ind + i * this->max_length,
            this->attention_score + i * this->max_length,
            output + i * this->head_dim,
            this->head_dim,
            nnz[i]
        );
      }

}
#endif
void SparseAttentionServer::full_attention(
int layer_id, 
torch::Tensor output_pt,
torch::Tensor max_value_expsum_pt,
torch::Tensor query_pt,
torch::Tensor nnz_pt)
{
    bfloat16 * key = this->key_cache[layer_id];
    bfloat16 * value = this->value_cache[layer_id];
    auto q_f32 = query_pt.to(torch::kFloat32);
    float * query = static_cast<float *>(q_f32.data_ptr());
    bfloat16 * output = static_cast<bfloat16 *>(output_pt.data_ptr());
    int * nnz = static_cast<int *>(nnz_pt.data_ptr());
    float * max_value =  static_cast<float *>(max_value_expsum_pt.data_ptr());
    float * expsum = max_value + this->batch_size * this->num_attention_heads;
    for (int i = 0; i < this->batch_size * this->num_key_value_heads; ++i){
        qk_kernel_full(
          key + i * this->max_length * this->head_dim,
          query + i * this->head_dim * this->num_attention_groups,
          this->attention_score + i * this->max_length * this->num_attention_groups,
          this->num_attention_groups,
          this->head_dim,
          nnz[i],
          this->max_length
        );
    }
    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < this->batch_size * this->num_attention_heads; ++i){
        softmax_kernel_optimized(
          this->attention_score + i * this->max_length,
          nnz[i],
          sqrtf32(this->head_dim),
          max_value + i,
          expsum + i
        );
      }

    for (int i = 0; i < this->batch_size * this->num_key_value_heads; ++i){
        wv_kernel_dim128_full(
            value + i * this->max_length * this->head_dim,
            this->attention_score + i * this->max_length * this->num_attention_groups,
            output + i * this->head_dim * this->num_attention_groups,
            this->head_dim,
            this->max_length,
            nnz[i],
            this->num_attention_groups
        );
      }

}

void SparseAttentionServer::scheduled_attention(
int layer_id, 
int K,
int L,
torch::Tensor output_pt,
torch::Tensor max_value_expsum_pt,
torch::Tensor query_pt, 
torch::Tensor query_norm_pt,
torch::Tensor ind_pt, 
torch::Tensor nnz_pt)
{
    bfloat16 * key = this->key_cache[layer_id];
    bfloat16 * value = this->value_cache[layer_id];
    auto q_f32 = query_pt.to(torch::kFloat32);
    float * query = static_cast<float *>(q_f32.data_ptr());
    bfloat16 * output = static_cast<bfloat16 *>(output_pt.data_ptr());
    int * ind = static_cast<int *>(ind_pt.data_ptr());
    int * nnz = static_cast<int *>(nnz_pt.data_ptr());
    float * query_norm = static_cast<float *>(query_norm_pt.data_ptr());
    float * max_value =  static_cast<float *>(max_value_expsum_pt.data_ptr());
    float * expsum = max_value + this->batch_size * this->num_attention_heads;

    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < ATTENTION_THREADS; ++i){
        int p = int (i / 2);
        int offset = int(nnz[p] / 2);
        int offset_16 = (offset % 16 == 0) ? (offset): (offset + 16 - offset % 16);
        int ith_offset = (i % 2 == 0) ? (0) : (offset_16);
        int workload = (i % 2 == 0) ? (offset_16) : (nnz[p] - offset_16);
        if (workload > 0){
          qk_kernel(
            key + (p/this->num_attention_groups) * this->max_length * this->head_dim,
            ind + p * this->max_length + ith_offset,
            query + p * this->head_dim,
            this->attention_score + p * this->max_length + ith_offset,
            this->head_dim,
            workload
          );

          transform_kernel(
          this->attention_score + p * this->max_length + ith_offset,
          workload,
          query_norm[p],
          this->key_norm[layer_id] + (p/this->num_attention_groups) * this->max_length, 
          K,
          L,
          sqrtf(this->head_dim),
          ind + p * this->max_length + ith_offset
        );
        }
      }
    

    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < this->batch_size * this->num_attention_heads; ++i){
        softmax_kernel(
          this->attention_score + i * this->max_length,
          nnz[i],
          max_value + i,
          expsum + i
        );
    }

    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < ATTENTION_THREADS; ++i){

      int p = int (i / 2);
      int ith_offset = (i % 2 == 0) ? (0) : int(this->head_dim / 2);



      wv_kernel_dim128_dim_parallel(
            value + (p/this->num_attention_groups) * this->max_length * this->head_dim,
            ind + p * this->max_length,
            this->attention_score + p * this->max_length,
            output + p * this->head_dim,
            ith_offset,
            int(this->head_dim / 2),
            this->head_dim,
            nnz[p]
        );
      
    }



}
#ifdef __AVX512BF16__
void SparseAttentionServer::scheduled_attention_bf16(
int layer_id, 
int K,
int L,
torch::Tensor output_pt,
torch::Tensor max_value_expsum_pt,
torch::Tensor query_pt, 
torch::Tensor query_norm_pt,
torch::Tensor ind_pt, 
torch::Tensor nnz_pt)
{
    bfloat16 * key = this->key_cache[layer_id];
    bfloat16 * value = this->value_cache[layer_id];
    bfloat16 * query = static_cast<bfloat16 *>(query_pt.data_ptr());
    bfloat16 * output = static_cast<bfloat16 *>(output_pt.data_ptr());
    int * ind = static_cast<int *>(ind_pt.data_ptr());
    int * nnz = static_cast<int *>(nnz_pt.data_ptr());
    float * query_norm = static_cast<float *>(query_norm_pt.data_ptr());
    float * max_value =  static_cast<float *>(max_value_expsum_pt.data_ptr());
    float * expsum = max_value + this->batch_size * this->num_attention_heads;

    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < ATTENTION_THREADS; ++i){
        int p = int (i / 2);
        int offset = int(nnz[p] / 2);
        int offset_16 = (offset % 16 == 0) ? (offset): (offset + 16 - offset % 16);
        int ith_offset = (i % 2 == 0) ? (0) : (offset_16);
        int workload = (i % 2 == 0) ? (offset_16) : (nnz[p] - offset_16);

        if (workload > 0){
          qk_kernel_bf16_impl(
            key + (p/this->num_attention_groups) * this->max_length * this->head_dim,
            ind + p * this->max_length + ith_offset,
            query + p * this->head_dim,
            this->attention_score + p * this->max_length + ith_offset,
            this->head_dim,
            workload
          );

          transform_kernel(
          this->attention_score + p * this->max_length + ith_offset,
          workload,
          query_norm[p],
          this->key_norm[layer_id] + (p/this->num_attention_groups) * this->max_length, 
          K,
          L,
          sqrtf(this->head_dim),
          ind + p * this->max_length + ith_offset
        );
        }
      }
    

    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < this->batch_size * this->num_attention_heads; ++i){
        softmax_kernel(
          this->attention_score + i * this->max_length,
          nnz[i],
          max_value + i,
          expsum + i
        );
    }

    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < ATTENTION_THREADS; ++i){

      int p = int (i / 2);
      int ith_offset = (i % 2 == 0) ? (0) : int(this->head_dim / 2);
     
      wv_kernel_dim128_dim_parallel(
            value + (p/this->num_attention_groups) * this->max_length * this->head_dim,
            ind + p * this->max_length,
            this->attention_score + p * this->max_length,
            output + p * this->head_dim,
            ith_offset,
            int(this->head_dim/2),
            this->head_dim,
            nnz[p]
        );
      
    }



}
#endif
torch::Tensor SparseAttentionServer::get_key_cache(int layer_id)
{
auto options = torch::TensorOptions().dtype(torch::kBFloat16);
//auto options = torch::TensorOptions().dtype(torch::kFloat32);
torch::Tensor tensor = torch::from_blob(this->key_cache[layer_id], {this->batch_size, this->num_key_value_heads, this->max_length, this->head_dim}, options);
return tensor;
}

torch::Tensor SparseAttentionServer::get_value_cache(int layer_id)
{
auto options = torch::TensorOptions().dtype(torch::kBFloat16);
//auto options = torch::TensorOptions().dtype(torch::kFloat32);
torch::Tensor tensor = torch::from_blob(this->value_cache[layer_id], {this->batch_size, this->num_key_value_heads, this->max_length, this->head_dim}, options);
return tensor;
}

torch::Tensor SparseAttentionServer::get_key_norm(int layer_id)
{
auto options = torch::TensorOptions().dtype(torch::kFloat32);
torch::Tensor tensor = torch::from_blob(this->key_norm[layer_id], {this->batch_size, this->num_key_value_heads, this->max_length}, options);
return tensor;
}

torch::Tensor SparseAttentionServer::get_score()
{
auto options = torch::TensorOptions().dtype(torch::kFloat32);
torch::Tensor tensor = torch::from_blob(this->attention_score, {this->batch_size, this->num_attention_heads, this->max_length}, options);
return tensor;
}

PYBIND11_MODULE(sparse_attention_cpu, m) {
    py::class_<SparseAttentionServer>(m, "SparseAttentionServer")
        .def(py::init<>())
        .def("alloc", &SparseAttentionServer::alloc)
        .def("fill", &SparseAttentionServer::fill)
        .def("attention", &SparseAttentionServer::attention)
        #ifdef __AVX512BF16__
        .def("attention_bf16", &SparseAttentionServer::attention_bf16)
        #endif
        .def("full_attention", &SparseAttentionServer::full_attention)
        .def("scheduled_attention", &SparseAttentionServer::scheduled_attention)
        .def("attention_wrapper", &SparseAttentionServer::attention_wrapper)
        #ifdef __AVX512BF16__
        .def("attention_wrapper_bf16", &SparseAttentionServer::attention_wrapper_bf16)
        #endif
        .def("get_key_cache", &SparseAttentionServer::get_key_cache)
        .def("get_value_cache", &SparseAttentionServer::get_value_cache)
        .def("get_key_norm", &SparseAttentionServer::get_key_norm)
        .def("get_score", &SparseAttentionServer::get_score)
        .def("clear", &SparseAttentionServer::clear);
}
