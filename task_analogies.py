import argparse
from transformers import AutoTokenizer,AutoModelForCausalLM
import os 
import torch
import random
import gc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--ma", type=str, default="")
    parser.add_argument("--mb", type=str, default="")
    parser.add_argument("--mc", type=str, default="")
    args = parser.parse_args()

    tokenizer_a = AutoTokenizer.from_pretrained(args.ma, use_fast=False)
    model_a = AutoModelForCausalLM.from_pretrained(args.ma)

    tokenizer_b = AutoTokenizer.from_pretrained(args.mb, use_fast=False)
    model_b = AutoModelForCausalLM.from_pretrained(args.mb)

    tokenizer_c = AutoTokenizer.from_pretrained(args.mc, use_fast=False)
    model_c = AutoModelForCausalLM.from_pretrained(args.mc)


    sda = model_a.named_parameters()
    sdb = model_b.named_parameters()
    sdc = model_c.named_parameters()

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("merging", flush=True)


    for ((name1,val1),(name2,val2), (name3,val3)) in zip(sda,sdb, sdc):
        val2.add_(val3)
        val2.sub_(val1)


    print("merged")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    tokenizer_b.save_pretrained(args.save_path)
    model_b.save_pretrained(args.save_path)




if __name__ == "__main__":
    with torch.no_grad():
        main()