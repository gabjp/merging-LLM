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
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--merge-embeddings", type=int, default=0)
    args = parser.parse_args()

    tokenizer1 = AutoTokenizer.from_pretrained(args.m1, use_fast=False)
    model1 = AutoModelForCausalLM.from_pretrained(args.m1)

    tokenizer2 = AutoTokenizer.from_pretrained(args.m2, use_fast=False)
    model2 = AutoModelForCausalLM.from_pretrained(args.m2)

    sd1 = model1.named_parameters()
    sd2 = model2.named_parameters()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("merging", flush=True)

    sd = {}

    keys = list(sd1.keys())
    print(keys)

    # Use vicuna

    if args.merge_embeddings == 0:

        for key in keys:
            print(key, flush=True)
            if key == "model.embed_tokens.weight":
                continue
            elif key == "lm_head.weight":
                continue
            else:
                sd1[key].mul_(args.p)
                sd2[key].mul_(1-args.p)
                sd1[key].add_(sd2[key])


    # drop and merge

    else:

        for key in keys:
            print(key, flush=True)
            if key == "model.embed_tokens.weight":
                sd1[key].mul_(args.p)
                sd2[key].mul_(1-args.p)
                sd1[key].add_(sd2[key])
            elif key == "lm_head.weight":
                sd1[key].mul_(args.p)
                sd2[key].mul_(1-args.p)
                sd1[key].add_(sd2[key])
            else:
                sd1[key].mul_(args.p)
                sd2[key].mul_(1-args.p)
                sd1[key].add_(sd2[key])



    print("merged")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    
    tokenizer1.save_pretrained(args.save_path)
    model1.save_pretrained(args.save_path)




if __name__ == "__main__":
    with torch.no_grad():
        main()