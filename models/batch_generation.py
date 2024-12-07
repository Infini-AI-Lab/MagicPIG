from llama import LLM
import argparse
import torch
from transformers import AutoTokenizer
import jsonlines
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",help='model')
parser.add_argument('--T', type=int, default=2000, help='repeat times')
parser.add_argument('--B', type=int, default=2, help='batch size')
parser.add_argument('--M', type=int, default=4096, help='max length')
parser.add_argument('--D', type=int, default=1, help='dec length')
parser.add_argument('--G', type=int, default=32, help='generation length')
parser.add_argument('--K', type=int, default=10, help='K')
parser.add_argument('--L', type=int, default=150, help='K')
args = parser.parse_args()
print(args)
MAX_LEN = args.M
DEC_LEN = args.D
GEN_LEN = args.G
BATCH_SIZE = args.B
MODEL_NAME = args.model
DTYPE = torch.bfloat16
DEVICE = "cuda:0"
T = args.T
WARM_UP = 10

with open("../data/data4k.jsonl") as f:
    d = jsonlines.Reader(f)
    for idx, item in enumerate(d):
        data = item
        break

llm = LLM(K=args.K, L=args.L, max_length=MAX_LEN, model_name=args.model, batch_size=BATCH_SIZE, device=DEVICE, dtype=DTYPE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
text = data["input"]
input_ids = tokenizer.encode(text=text, return_tensors="pt").to(device=DEVICE)
PREFIX_LEN = input_ids.shape[1]

position_ids = torch.arange(MAX_LEN, device=DEVICE).unsqueeze(0).repeat(BATCH_SIZE, 1)

batch_logits = []
for i  in range(BATCH_SIZE):
    logits = llm.prefill(input_ids, i)
    batch_logits.append(logits)

logits = torch.cat(batch_logits, dim=0)
generated_tokens = []
prefix_len = input_ids.shape[1]
for k in range(GEN_LEN):
    input_ids = logits.argmax(dim=-1)
    logits = llm.inference(input_ids=input_ids, position_ids=position_ids[:,prefix_len + k:prefix_len + k + 1])
    generated_tokens.append(input_ids)
    if input_ids[0].item() in [128000, 128001, 128008, 128009]:
                break
generated_tokens = torch.cat(generated_tokens, dim=1).to(device="cpu")
decoded_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(decoded_texts)
    
    
    


