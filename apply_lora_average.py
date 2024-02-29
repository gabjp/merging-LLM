import argparse
from transformers import AutoTokenizer,AutoModelForCausalLM
import os 
import torch
import random
import gc
from peft import PeftModel 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--m1", type=str, default="")
    parser.add_argument("--m2", type=str, default="")
    parser.add_argument("--m3", type=str, default="")
    parser.add_argument("--llama-path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--p", type=float, default=0.5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.llama_path, use_fast=False)
    llama1 = AutoModelForCausalLM.from_pretrained(args.llama_path)
    llama2 = AutoModelForCausalLM.from_pretrained(args.llama_path)

    model1 = PeftModel.from_pretrained(llama1, args.m1).merge_and_unload()
    model2 = PeftModel.from_pretrained(llama2, args.m2).merge_and_unload()

    sd1 = model1.named_parameters()
    sd2 = model2.named_parameters()

    if args.m3 != "":
        llama3 = AutoModelForCausalLM.from_pretrained(args.llama_path)
        model3 = PeftModel.from_pretrained(llama3, args.m3).merge_and_unload()
        sd3 = model3.named_parameters()

        for ((name1,val1),(name2,val2), (name3,val3)) in zip(sd1, sd2, sd3):
            print(f"merging {name1}")
            val1.copy_(torch.stack([val1, val2, val3], dim=0).mean(dim=0))

    else:
        for ((name1,val1),(name2,val2)) in zip(sd1,sd2):
            print(f"merging {name1}")
            print(val1)
            print(val2)
            val1.mul_(args.p)
            val2.mul_(1-args.p)
            val1.add_(val2)
    


    print("merged")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    
    tokenizer.save_pretrained(args.save_path)
    model1.save_pretrained(args.save_path)




if __name__ == "__main__":
    with torch.no_grad():
        main()