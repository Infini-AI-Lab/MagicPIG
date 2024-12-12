<div align="center">
<h1><img src="assets/MGPF.jpg" height="40px" align="top"/> MagicPIG-v0.2: LSH sampling<br>for Efficient LLM Generation
</h1>

Zhuoming Chen<sup>1</sup>, Ranajoy Sadhukhan<sup>1</sup>, Zihao Ye<sup>2</sup>, Yang Zhou<sup>1</sup>, Jianyu Zhang<sup>3,4</sup>, Niklas Nolte<sup>4</sup>, <br> Yuandong Tian<sup>4</sup>, Matthijs Douze<sup>4</sup>, Leon Bottou<sup>3,4</sup>, Zhihao Jia<sup>1</sup>, Beidi Chen<sup>1</sup>

<sup>1</sup> Carnegie Mellon University,   <sup>2</sup>University of Washington, <sup>3</sup>New York University, <sup>4</sup>FAIR 

----------
*For exploring the possibility of **GPU-CPU** system powered by Locality-Sensitive-Hashing.*
</div>

<div align="center">
[<a href="https://arxiv.org/abs/2410.16179">Paper</a>] | [<a href="www.lsh-ai.com">Blog</a>]
</div>
<br>

----------

## *Latest News 📣*

- [2024.12] Use FlashInfer to compute the GPU attention parts.
- [2024.12] More efficient and easy-to-use CPU sparse attention.
- [2024.12] Overlap hash table construction and prefilling to hide CPU overhead.

<table>
<tr>
<td> <img src="assets/Throughput_Improved.jpg" align="top"/>
<td> <img src="assets/batch_size_vs_throughput.png" align="top" width="1200"/>
</tr>
<caption>
MagicPIG improves decoding throughput in various regimes by 1.76-4.99x compared with GPU-only attention. Here, we use L20 GPUs with Intel 8563C to simulate different sizes of VRAM. 
</caption>
</table>

<table>
<tr>
<td> <img src="assets/RULER.png" align="top"/>
<td> <img src="assets/Reasoning.png" align="top"/>
</tr>
<caption>
MagicPIG achieves higher downstream accuracy in retrieval and reasoning tasks than Quest, the state-of-the-art baseline with lower computation.  K10L150 and K11L300 correspond to 2% computation cost while K10L170 corresponds to 2.5%.
</caption>
</table>

## Installation
**Commands:**

    conda create -n magicpig
  
    conda activate magicpig
  
    bash install.sh


**Hardware requirements:** 

Basic: Intel CPUs supporting AVX512. 

BFloat16: Intel CPUs supporting AVX512_BF16, GCC Version $\geq$ 11.

Recommended Python version: 3.9/3.10. 


## Generation

**Commands:**

    cd models
    numactl -C 0-31,52-83 -m 0,1 \
    python generation.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --M 8192 \ 
    --G 256 \
    --K 10 \ 
    --L 170 \ 
    --template meta-llama3 \
    --data ../data/story.txt

**Explanations:**
 `--model` : Name or path for a huggingface model (Only Llamas are supported currently).

 `--M`: Maximum sequence length for pre-allocated the VRAM. It should be larger than context length + generation length.

 `--G`: Generation length.

 `--K, --L`: LSH hyper-parameter (when K=0, we use full attention).

 `--template`: Chat template (only support meta-llama3 and meta-llama2 currently).

 `--data`: Source data for generation (.txt file).


## Benchmark

**Commands:**

    cd models
    numactl -C 0-31,52-83 -m 0,1 \
    python bench.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --B 1 \
    --P 98000 \ 
    --M 98304 \ 
    --K 10 \ 
    --L 150


**Explanations:**
 `--model` : Name or path for a huggingface model (Only Llamas are supported currently).

 `--M`: Maximum sequence length for pre-allocated the VRAM. It should be larger than `--M` (by at least 192).

 `--P`: Actual context length for benchmarking. 

 `--B`: Batch Size.

 `--K, --L`: LSH hyper-parameter (when K=0, we use full attention).
 

## Evaluations

### Install RULER environments

**Commands:**

    cd evaluations/RULER
    pip install -r requirements.txt

### Run RULER Benchmark

**Commands:**

    cd evaluations/RULER
    python download_nltk.py
    bash run.sh llama3-8b-chat-128k synthetic $K $L

replace `K` and `L` with the hyper-parameter you want to evaluate. 

Currently, we support the following models.

`llama3-8b-chat-128k`: [[meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)],  `llama3-8b-chat-512k`: [[princeton-nlp/Llama-3-8B-ProLong-512k-Instruct](https://huggingface.co/princeton-nlp/Llama-3-8B-ProLong-512k-Instruct)],  `mistral-7b-chat-512k`: [[aws-prototyping/MegaBeam-Mistral-7B-512k](https://huggingface.co/aws-prototyping/MegaBeam-Mistral-7B-512k)], `llama3-70b-chat-128k`: [[meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)],


**Notice:**
This will call the compiled lsh and sparse_attention_cpu to execute the proposed systems in the paper. **Require lsh and sparse_attention_cpu are successfully installed.**


Not all users/developpers have AVX512 machines. **You can still test the accuracy of MagicPIG even if you cannot finish the installation.**.

We implement another two versions of MagicPIG for different purposes.

### Tensor Parallelism (GPU + Mask)

We implement a mathematically equivalent version with tensor parallelism.  

**Commands:**

    cd evaluations/RULER
    python download_nltk.py
    bash run_tensor_parallel.sh llama3-8b-chat-128k synthetic $K $L

replace K and L with the hyper-parameter you want to evaluate. 


### Single GPU (Huggingface + Mask)

We implement a mathematically equivalent version with huggingface for easy-exporting to other evaluation frameworks (e.g., lm-eval-harness, LongBench).

**Commands:**

    cd evaluations/RULER
    python download_nltk.py
    bash run_single_gpu.sh llama3-8b-chat-128k synthetic $K $L 4 64 $method 0

replace K and L with the hyper-parameter you want to evaluate.

`$method`: 0: MagicPIG; 1: Quest; 2: TopK 3: Oracle Sampling

`$K`: LSH hyper-parameter for MagicPIG and Page Size for Quest

`$L`: LSH hyper-parameter for MagicPIG and number of selected pages for Quest

Pipeline parallelism can be enabled with Accelerate by adding more GPU ids in `Line 26` of `run_single_gpu.sh`.


----------
This project was made possible thanks to a collaboration with

<a href="https://www.cmu.edu"><img src="https://upload.wikimedia.org/wikipedia/commons/9/9b/Carnegie_Mellon_wordmark.svg" height="20"></a> &nbsp;&nbsp;&nbsp;
<a href="https://www.washington.edu/"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/University_of_Washington_signature.svg/2560px-University_of_Washington_signature.svg.png" height="25"></a> &nbsp;&nbsp;&nbsp;
<a href="https://www.nyu.edu/"><img src="https://upload.wikimedia.org/wikipedia/en/thumb/5/58/NYU_logo.svg/2560px-NYU_logo.svg.png" height="21"></a> &nbsp;&nbsp;&nbsp;
<a href="https://ai.meta.com/research/"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Meta_Platforms_Inc._logo.svg/2560px-Meta_Platforms_Inc._logo.svg.png" height="21"></a> &nbsp;&nbsp;&nbsp;

----------

## Reference
```bibtex
@article{chen2024magicpig,
  title={Magicpig: Lsh sampling for efficient llm generation},
  author={Chen, Zhuoming and Sadhukhan, Ranajoy and Ye, Zihao and Zhou, Yang and Zhang, Jianyu and Nolte, Niklas and Tian, Yuandong and Douze, Matthijs and Bottou, Leon and Jia, Zhihao and others},
  journal={arXiv preprint arXiv:2410.16179},
  year={2024}
}
```


