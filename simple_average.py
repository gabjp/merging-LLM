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
    args = parser.parse_args()

    tokenizer1 = AutoTokenizer.from_pretrained(args.m1, use_fast=False)
    model1 = AutoModelForCausalLM.from_pretrained(args.m1, do_sample=True)

    tokenizer2 = AutoTokenizer.from_pretrained(args.m2, use_fast=False)
    model2 = AutoModelForCausalLM.from_pretrained(args.m2, do_sample=True)


    sd1 = model1.named_parameters()
    sd2 = model2.named_parameters()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("merging", flush=True)


    for ((name1,val1),(name2,val2)) in zip(sd1,sd2):
        val1.mul_(args.p)
        val2.mul_(1-args.p)
        val1.add_(val2)


    print("merged")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    
    tokenizer1.save_pretrained(args.save_path)
    model1.save_pretrained(args.save_path)


if __name__ == "__main__":
    with torch.no_grad():
        main()