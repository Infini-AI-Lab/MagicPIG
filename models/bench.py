from llama import LLM
import argparse
import torch
from transformers import AutoTokenizer
import jsonlines
import time
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",help='model')
parser.add_argument('--B', type=int, default=1, help='batch size')
parser.add_argument('--M', type=int, default=98304, help='max length')
parser.add_argument('--D', type=int, default=1, help='dec length')
parser.add_argument('--P', type=int, default=98000, help='prefill length')
parser.add_argument('--G', type=int, default=128, help='generation length')
parser.add_argument('--K', type=int, default=10, help='K')
parser.add_argument('--L', type=int, default=150, help='L')
args = parser.parse_args()
print(args)
MAX_LEN = args.M
DEC_LEN = args.D
GEN_LEN = args.G
B = args.B
MODEL_NAME = args.model
DTYPE = torch.bfloat16
PREFIX_LEN = args.P
DEVICE = "cuda:0"
WARM_UP = 32

with open("../data/data.jsonl") as f:
    d = jsonlines.Reader(f)
    for idx, item in enumerate(d):
        data = item
        break

llm = LLM(K=args.K, L=args.L, max_length=MAX_LEN, model_name=args.model, batch_size=B, device=DEVICE, dtype=DTYPE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
text = data["input"]
input_ids = tokenizer.encode(text=text, return_tensors="pt").to(device=DEVICE)
input_ids = input_ids[:,:PREFIX_LEN].repeat(B, 1)
position_ids = torch.arange(MAX_LEN, device=DEVICE).unsqueeze(0).repeat(B, 1)

for i in range(B):
    logits = llm.prefill(input_ids=input_ids[i:i+1], request_id=i)

generated = input_ids[0].tolist()
for k in range(WARM_UP):
    logits = llm.inference(input_ids=input_ids[:, 128+k:128+k+1], position_ids=position_ids[:,PREFIX_LEN + k:PREFIX_LEN + k + 1])

torch.cuda.synchronize()
t1 = time.time()
for k in range(GEN_LEN):
    logits = llm.inference(input_ids=input_ids[:, WARM_UP+k:WARM_UP+k+1], position_ids=position_ids[:,WARM_UP + PREFIX_LEN + k: WARM_UP + PREFIX_LEN + k + 1])

torch.cuda.synchronize()
t2 = time.time()

print("Decoding Latency {:.2f} ms/token".format(1000 * (t2 - t1)/GEN_LEN))
print("Decoding Throughput {:.2f} token/s".format(B * GEN_LEN / (t2 - t1)))
