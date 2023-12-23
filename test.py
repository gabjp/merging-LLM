import argparse
from transformers import AutoTokenizer,AutoModelForCausalLM
import os 
import torch
import random
import gc
import numpy as np
from peft import PeftConfig
import pickle
from safetensors.torch import load_file

def layer(string_list):
    if len(string_list) >=3 and string_list[2] == 'weight' and string_list[1] != 'norm':
        idx = 0
    elif len(string_list) >=3 and string_list[2] != 'weight':
        idx = int(string_list[2]) + 1
    else:
        idx = 33
    return idx

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

    #tokenizer3 = AutoTokenizer.from_pretrained(args.m3, use_fast=False)
    #model3 = AutoModelForCausalLM.from_pretrained(args.m3)

    #config = PeftConfig.from_pretrained(args.m1)

    sd1 = model1.named_parameters()
    sd2 = model1.named_parameters()
    #sd3 = model1.named_parameters()

    for ((name1,val1),(name2,val2)) in zip(sd1,sd2):
            print(name1)
            print(val1)
            val1.mul_(0.5)
            val2.mul_(1-0.5)
            val1.add_(val2)
            print(val1)
            

    #sd1 = model1.named_parameters()
    #sd2 = model2.named_parameters()





if __name__ == "__main__":
    with torch.no_grad():
        main()