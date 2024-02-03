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


    m1_sums = [ 0 for i in range(32) ]
    m2_sums = [ 0 for i in range(32) ]

    it = zip(sd_base, sd_model1, sd_model2)

    print("computing rank")

    for ((n,v1),(_,v2), (_,v3)) in it:

        layer_name = n.split(".")
        if layer_name[1] != "h":
            print(f"skipping layer {n}")
            continue

        layer_num = int(layer_name[2])

        delta_1 = torch.sum(torch.abs(v2-v1))
        delta_2 = torch.sum(torch.abs(v3-v1))
        
        m1_sums[layer_num] += delta_1
        m2_sums[layer_num] += delta_2
    
    m1_sums = [(m1_sums[i], i) for i in range(32)]
    m2_sums = [(m2_sums[i], i) for i in range(32)]

    m1_sums = sorted(m1_sums, key=lambda tup: tup[0])
    m2_sums = sorted(m2_sums, key=lambda tup: tup[0])

    layers_rank_m1 = {}
    layers_rank_m2 = {}

    for i in range(len(m1_sums)):
        layers_rank_m1[m1_sums[i][1]] = i+1
        layers_rank_m2[m2_sums[i][1]] = i+1

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
            layer_name = n.split(".")
            layer_num = int(layer_name[2])

            p = layers_rank_m1[layer_num] / (layers_rank_m1[layer_num] + layers_rank_m2[layer_num])

            print("rank m1", layers_rank_m1[layer_num])
            print("rank m2", layers_rank_m2[layer_num])
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