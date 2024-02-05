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
    base = AutoModelForCausalLM.from_pretrained(args.llama_path)

    model1 = PeftModel.from_pretrained(llama1, args.m1).merge_and_unload()
    model2 = PeftModel.from_pretrained(llama2, args.m2).merge_and_unload()


    sd_base = list(base.named_parameters())
    sd_model1 = list(model1.named_parameters())
    sd_model2 = list(model2.named_parameters())


    m1_sums = [ [] for i in range(32) ]
    m2_sums = [ [] for i in range(32) ]

    for ((n,v1),(_,v2), (_,v3)) in zip(sd_base, sd_model1, sd_model2):

        layer_name = n.split(".")
        if v2==v3==v1:
            print(f"skipping layer {n}")
            continue
        print(f"computing rank for layer {n}")

        layer_num = int(layer_name[2])

        delta_1 = torch.mean(torch.abs(v2-v1))
        delta_2 = torch.mean(torch.abs(v3-v1))

        m1_sums[layer_num].append((delta_1,n))
        m2_sums[layer_num].append((delta_2,n))
    
    for i in range(32):
        m1_sums[i] = sorted(m1_sums[i], key=lambda tup: tup[0])
        m2_sums[i] = sorted(m2_sums[i], key=lambda tup: tup[0])

    layers_rank_m1 = {}
    layers_rank_m2 = {}

    for j in range(32):
        for i in range(3):
            layers_rank_m1[m1_sums[j][i][1]] = i+1
            layers_rank_m2[m2_sums[j][i][1]] = i+1

    print(m1_sums)
    print()
    print(m2_sums)
    print()
    print(layers_rank_m1)
    print()
    print(layers_rank_m2)

    it = zip(sd_base, sd_model1, sd_model2)

    for ((n,v1),(_,v2), (_,v3)) in it:
        if 'query_key_value.weight' in n or 'dense_h_to_4h.weight' in n or 'dense_4h_to_h.weight' in n:

            p = layers_rank_m1[n] / (layers_rank_m1[n] + layers_rank_m2[n])

            print("rank m1", layers_rank_m1[n])
            print("rank m2", layers_rank_m2[n])
            print(f"merging layer {n} p={p}")
            print()

            v2.mul_(p)
            v2.add_(v3 * (1 - p))

    print("merged")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    tokenizer.save_pretrained(args.save_path)
    model1.save_pretrained(args.save_path)



if __name__ == "__main__":
    with torch.no_grad():
        main()