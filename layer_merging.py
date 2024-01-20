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

    it = zip(sd_model1_unmerged,sd_model1, sd_model2)


    for ((n,v1),(_,v2), (_,v3)) in it:
        if 'query_key_value.weight' in n or 'dense_h_to_4h.weight' in n or 'dense_4h_to_h.weight' in n:
            # COMPARE DELTA AND MERGE NEXT 2 MATRICES (LORA ADAPTERS)

            delta_1 = torch.sum(torch.abs(v2-v1))
            delta_2 = torch.sum(torch.abs(v3-v1))

            print(n)
            print(f"delta 1 {delta_1}")
            print(f"delta 2 {delta_2}")

            if delta_1 >= delta_2:
                p = 0.9
            else:
                p = 0.1

            ((n,v1),(_,v2), (_,v3)) = next(it)
            print(f"merging {n} p={p}")
            v1.mul_(p)
            v1.add_(v3 * p)

            ((n,v1),(_,v2), (_,v3)) = next(it)
            print(f"merging {n} p={p}")
            v1.mul_(p)
            v1.add_(v3 * p)

        print("merged")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    tokenizer.save_pretrained(args.save_path)
    model1_unmerged.save_pretrained(args.save_path)



if __name__ == "__main__":
    with torch.no_grad():
        main()