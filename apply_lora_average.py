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

    print(adapter1)


    for name,val in sd:
        print(name)
        continue
        val1.mul_(args.p)
        val2.mul_(1-args.p)
        val1.add_(val2)

    print(adapter1.keys())
    return 

    print("merged")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for (name1, val1) in model1.named_parameters():
        if name1 == "model.embed_tokens.weight":
            print(val1)
        else:
            continue
    
    tokenizer2.save_pretrained(args.save_path)
    model1.save_pretrained(args.save_path)




if __name__ == "__main__":
    with torch.no_grad():
        main()