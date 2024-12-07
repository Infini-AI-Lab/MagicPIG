# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import requests
import torch
from typing import Dict, List, Optional
CHUNK_SIZE = 1024

class HuggingFaceModel:
    def __init__(self, name_or_path: str, K, L, S, W, Q, QR, approx=False, **generation_kwargs) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        if Q == 0:
            from llama_ref import LlamaForCausalLM
        elif Q == 1:
            from llama_quest import LlamaForCausalLM
        elif Q == 2 or Q == 3:
            from llama_topk import LlamaForCausalLM
        elif Q == 5:
            from transformers import LlamaForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
        
        if Q!=5:
            model_kwargs = {"attn_implementation": "eager"}
        else: 
            model_kwargs = {"attn_implementation": "flash_attention_2"}
        
        
        self.pipeline = None
        self.model :LlamaForCausalLM = LlamaForCausalLM.from_pretrained(name_or_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16, **model_kwargs)
        self.model.config.K = K
        self.model.config.L = L if (Q in [0, 1]) else QR
        self.model.config.window = W
        self.model.config.QR = QR
        self.approx = (Q!=5)
        self.model.config.cache_mode = "topk" if Q == 2 else "topp"
        self.model.eval()
        self.Q = Q
        print(K,L,S,W,Q,QR)
        if self.approx:
            self.model.set_sparse_attn(sparse=S, window_size=W, kernel_size=5, random_sparse=0.1, vsparse=1.0)

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')
        self.mat = []
        self.decode_tokens = []
        self.sparse = []
    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        if self.pipeline is None:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            seq_len = inputs["input_ids"].shape[1]
            num_chunk = (seq_len // CHUNK_SIZE - 2) if (seq_len % CHUNK_SIZE == 0) else (seq_len // CHUNK_SIZE - 1)
            past_key_values = None
            if self.approx:
                self.model.select_kv(False)
            with torch.inference_mode():
                for chunk_id in range(num_chunk):
                    outputs = self.model(input_ids=inputs["input_ids"][:,chunk_id * CHUNK_SIZE : (chunk_id + 1) * CHUNK_SIZE],
                                         attention_mask = inputs["attention_mask"][:,: (chunk_id + 1) * CHUNK_SIZE],
                                        past_key_values=past_key_values,
                                        use_cache=True)
                    past_key_values = outputs.past_key_values
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            if self.approx:
                self.model.select_kv(True)
            if self.Q != 0 and self.Q != 3:
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    **self.generation_kwargs
                )
            else:
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    **self.generation_kwargs,
                    return_dict_in_generate=True
                )
                cache = output.past_key_values
                sparse = cache.sparse
                self.sparse.append(sparse)
                self.decode_tokens.append(cache.decode_tokens)
                output = output.sequences
            if self.approx:
                self.model.select_kv(False)
            generated_text = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            torch.cuda.empty_cache()
        else:
            output = self.pipeline(text_inputs=prompt, **self.generation_kwargs,)
            assert len(output) == 1
            generated_text = output[0]["generated_text"]
            
        # remove the input form the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :]
                
        if self.stop is not None:
            for s in self.stop:
                generated_text = generated_text.split(s)[0]
        return {'text': [generated_text]}


class MambaModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.device = "cuda"
        self.model = MambaLMHeadModel.from_pretrained(name_or_path, device=self.device, dtype=torch.bfloat16)
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')
        self.max_genlen = self.generation_kwargs.pop('max_new_tokens')
        self.minp = 0.0

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        # tokenize
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        max_length = input_ids.shape[1] + self.max_genlen

        # generate
        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            **self.generation_kwargs,
        )
        assert len(out.sequences) == 1
        # detok
        return {'text': [self.tokenizer.decode(out.sequences[0][input_ids.shape[1] :])]}
