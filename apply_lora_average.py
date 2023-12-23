import argparse
from transformers import AutoTokenizer,AutoModelForCausalLM
import os 
import torch
import random
import gc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--m1", type=str, default="")
    parser.add_argument("--m2", type=str, default="")
    parser.add_argument("--llama-path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--p", type=float, default=0.5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.llama_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.llama_path)

    adapter1 = torch.load(args.m1 + "/adapter_model.bin")
    adapter2 = torch.load(args.m2 + "/adapter_model.bin")

    sd = model.named_parameters()

    print("merging", flush=True)


    for name,val in sd:
        if  ('self_attn.q_proj' not in name) and ('self_attn.v_proj' not in name):
            continue

        str_a = 'base_model.model.' + name[:-6] + 'lora_A.weight'
        str_b = 'base_model.model.' + name[:-6] + 'lora_B.weight'
        A1 = adapter1[str_a]
        B1 = adapter1[str_b]
        A2 = adapter2[str_a]
        B2 = adapter2[str_b]

        W1 = torch.matmul(A1,B1)
        W2 = torch.matmul(A2,B2)

        print(A1.size())
        print(B1.size())
        print(W1.size())

        val.add_(W1, alpha=args.p)
        val.add_(W2, alpha=1-args.p)


    print("merged")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    
    tokenizer.save_pretrained(args.save_path)
    model.save_pretrained(args.save_path)




if __name__ == "__main__":
    with torch.no_grad():
        main()