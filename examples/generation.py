import sys
sys.path.append("..")
from models.llama import LLM
import argparse
import torch
from transformers import AutoTokenizer
import jsonlines
from models.template import Templates
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",help='model')
parser.add_argument('--M', type=int, default=8192, help='max length')
parser.add_argument('--D', type=int, default=1, help='dec length')
parser.add_argument('--G', type=int, default=256, help='generation length')
parser.add_argument('--K', type=int, default=10, help='K')
parser.add_argument('--L', type=int, default=150, help='K')
parser.add_argument('--data', type=str, default="../data/story.txt", help='source data file')
parser.add_argument('--template', type=str, default="meta-llama3", help='chat template')
args = parser.parse_args()
print(args)
MAX_LEN = args.M
DEC_LEN = args.D
GEN_LEN = args.G
MODEL_NAME = args.model
DTYPE = torch.bfloat16
DEVICE = "cuda:0"
chat_template = Templates[args.template]
llm = LLM(K=args.K, L=args.L, max_length=MAX_LEN, model_name=args.model, batch_size=1, device=DEVICE, dtype=DTYPE, generation_buffer=args.G + 32)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
with open(args.data, "r", encoding="utf-8") as file:
    content = file.read()
    content = chat_template.format(content)
    input_ids = tokenizer.encode(text=content, return_tensors="pt")
    context = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(context)
    input_ids = input_ids.to(DEVICE)
    PREFIX_LEN = input_ids.shape[1]
    position_ids = torch.arange(MAX_LEN, device=DEVICE).unsqueeze(0)
    generated = llm.generate(input_ids, max_tokens=args.G)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print("\033[32m" + text + "\033[0m")
    



