import argparse
from transformers import AutoTokenizer,AutoModelForCausalLM
import os 
import torch
import random
import gc
import numpy as np

def layer(string):
    return 0 if string == 'weight' else int(string) + 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--m1", type=str, default="")
    parser.add_argument("--m2", type=str, default="")
    parser.add_argument("--start-p", type=float, default=0.5)
    parser.add_argument("--end-p", type=float, default=0.9)
    parser.add_argument("--start-head", type=int, default=0)
    args = parser.parse_args()

    tokenizer1 = AutoTokenizer.from_pretrained(args.m1, use_fast=False)
    model1 = AutoModelForCausalLM.from_pretrained(args.m1)

    tokenizer2 = AutoTokenizer.from_pretrained(args.m2, use_fast=False)
    model2 = AutoModelForCausalLM.from_pretrained(args.m2)


    sd1 = model1.named_parameters()
    sd2 = model2.named_parameters()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("merging", flush=True)

    p_list = np.linspace(args.start_p, args.end_p, num = 33) # 32 layers + embeddings
    
    for ((name1,val1),(name2,val2)) in zip(sd1,sd2):
        if len(name1.split(".")) >=2: string = name1.split(".")[2]
        else: string = name1.split(".")[-1]
        p = p_list[layer(string)] # Retrieve layer number
        print(f"{name1} -- [{p}/{1-p}]")
        val1.mul_(p)
        val2.mul_(1-p)
        val1.add_(val2)
        


    print("merged")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    
    tokenizer2.save_pretrained(args.save_path)
    model1.save_pretrained(args.save_path)




if __name__ == "__main__":
    with torch.no_grad():
        main()