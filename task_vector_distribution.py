import argparse
from transformers import AutoTokenizer,AutoModelForCausalLM
import os 
import torch
import random
import gc
from peft import PeftModel 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--llama-path", type=str, default="meta-llama/Llama-2-7b-hf")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.llama_path, use_fast=False)
    llama = AutoModelForCausalLM.from_pretrained(args.llama_path)

    model = PeftModel.from_pretrained(llama, args.model_path).merge_and_unload()


    sd_llama = llama.named_parameters()
    sd_model = model.named_parameters()

    for ((name1,val_llama),(name2,val_model)) in zip(sd_llama,sd_model):
        val_model.sub_(val_llama)
        print(name1)
        print(f"Sum: {torch.sum(torch.abs(val_model))}")
        print(f"Mean {torch.mean(torch.abs(val_model))}")
        print()


if __name__ == "__main__":
    with torch.no_grad():
        main()