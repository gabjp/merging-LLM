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

    tokenizer = AutoTokenizer.from_pretrained(args.llama_path, use_fast=False)

    llama1 = AutoModelForCausalLM.from_pretrained(args.llama_path)
    llama2 = AutoModelForCausalLM.from_pretrained(args.llama_path)
    llama_base = AutoModelForCausalLM.from_pretrained(args.llama_path)

    model1_unmerged = PeftModel.from_pretrained(llama_base, args.m1)
    model1 = PeftModel.from_pretrained(llama1, args.m1)
    model2 = PeftModel.from_pretrained(llama2, args.m2)

    model1.merge_adapter()
    model2.merge_adapter()

    sd_model1_unmerged = model1_unmerged.named_parameters()
    sd_model1 = model1.named_parameters()
    sd_model2 = model2.named_parameters()


    m1_sums = []
    m2_sums = []

    for ((n,v1),(_,v2), (_,v3)) in zip(sd_model1_unmerged, sd_model1, sd_model2):
        delta_1 = torch.sum(torch.abs(v2-v1))
        delta_2 = torch.sum(torch.abs(v3-v1))
        if delta_1 != 0 and delta_2 != 0:
            m1_sums.append((delta_1,n))
            m2_sums.append((delta_2,n))
    
    print(len(m1_sums))

    m1_sums = sorted(m1_sums, key=lambda tup: tup[0])
    m2_sums = sorted(m2_sums, key=lambda tup: tup[0])

    layers_rank_m1 = {}
    layers_rank_m2 = {}

    for i in range(len(m1_sums)):
        layers_rank_m1[m1_sums[i]] = i+1
        layers_rank_m2[m1_sums[i]] = i+1

    print(m1_sums)
    print(m2_sums)

    print(layers_rank_m1)
    print(layers_rank_m2)

    it = zip(sd_model1_unmerged, sd_model1, sd_model2)

    for ((n,v1),(_,v2), (_,v3)) in it:
        if 'query_key_value.weight' in n or 'dense_h_to_4h.weight' in n or 'dense_4h_to_h.weight' in n:

            p = layers_rank_m1[n] / (layers_rank_m1[n] + layers_rank_m2[n])

            print("rank m1", layers_rank_m1[n])
            print("rank m2", layers_rank_m2[n])

            ((n,v1),(_,v2), (_,v3)) = next(it)
            print(f"merging {n} p={p}")
            v1.mul_(p)
            v1.add_(v3 * (1 - p))

            ((n,v1),(_,v2), (_,v3)) = next(it)
            print(f"merging {n} p={p}")
            v1.mul_(p)
            v1.add_(v3 * (1 - p))
            print()
    print("merged")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    tokenizer.save_pretrained(args.save_path)
    model1_unmerged.save_pretrained(args.save_path)



if __name__ == "__main__":
    with torch.no_grad():
        main()