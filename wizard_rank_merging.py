import argparse
from transformers import AutoTokenizer,AutoModelForCausalLM
import os 
import torch
import random
import gc
from peft import PeftModel 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1", type=str, default="")
    parser.add_argument("--m2", type=str, default="")
    parser.add_argument("--llama-path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--save-path", type=str, default="")
    args = parser.parse_args()

    tokenizer1 = AutoTokenizer.from_pretrained(args.m1, use_fast=False)
    tokenizer2 = AutoTokenizer.from_pretrained(args.m2, use_fast=False)

    model1 = AutoModelForCausalLM.from_pretrained(args.m1)
    model2 = AutoModelForCausalLM.from_pretrained(args.m2)
    base = AutoModelForCausalLM.from_pretrained(args.llama_path)


    sd_base = list(base.named_parameters())
    sd_model1 = list(model1.named_parameters())
    sd_model2 = list(model2.named_parameters())


    m1_sums = []
    m2_sums = []

    for ((n,v1),(_,v2), (_,v3)) in zip(sd_base, sd_model1, sd_model2):
        print(f"computing rank for layer {n}")
        print(v1.dype)
        print(v2.dype)
        print(v3.dype)

        if n == "model.embed_tokens.weight" or n == "lm_head.weight":
            delta_1 = torch.sum(torch.abs(v2[:-1]-v1))
            delta_2 = torch.sum(torch.abs(v3[:-1]-v1))
        else:
            delta_1 = torch.sum(torch.abs(v2-v1))
            delta_2 = torch.sum(torch.abs(v3-v1))
        
        m1_sums.append((delta_1,n))
        m2_sums.append((delta_2,n))
    
    print(len(m1_sums))

    m1_sums = sorted(m1_sums, key=lambda tup: tup[0])
    m2_sums = sorted(m2_sums, key=lambda tup: tup[0])

    layers_rank_m1 = {}
    layers_rank_m2 = {}

    for i in range(len(m1_sums)):
        layers_rank_m1[m1_sums[i][1]] = i+1
        layers_rank_m2[m2_sums[i][1]] = i+1

    print(m1_sums)
    print(m2_sums)

    print(layers_rank_m1)
    print(layers_rank_m2)

    it = zip(sd_base, sd_model1, sd_model2)

    for ((n,v1),(_,v2), (_,v3)) in it:

        p = layers_rank_m1[n] / (layers_rank_m1[n] + layers_rank_m2[n])

        print("rank m1", layers_rank_m1[n])
        print("rank m2", layers_rank_m2[n])
        print(f"merging layer {n} p={p}")
        print()

        v2.mul_(p)
        v2.add_(v3 * (1 - p))

        print(v2.dtype)

    print("merged")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    tokenizer1.save_pretrained(args.save_path)
    model1.save_pretrained(args.save_path)



if __name__ == "__main__":
    with torch.no_grad():
        main()