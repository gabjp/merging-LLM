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
    parser.add_argument("--v", type=int, default=0)
    parser.add_argument("--llama-path", type=str, default="meta-llama/Llama-2-7b-hf")
    args = parser.parse_args()


    llama = AutoModelForCausalLM.from_pretrained(args.llama_path)
    llama_base = AutoModelForCausalLM.from_pretrained(args.llama_path)

    model = PeftModel.from_pretrained(llama_base, args.model_path).merge_adapter()


    sd_llama = llama.named_parameters()
    sd_model = model.named_parameters()

    print("MERGED")
    for (name, val) in sd_model:
        print(name)

    print()
    print("BASE")
    for (name, val) in sd_llama:
        print(name)

    #for ((name1,val_llama),(name2,val_model)) in zip(sd_llama,sd_model):


        # val_model.sub_(val_llama)
        # if torch.sum(torch.abs(val_model)) != 0 or args.v == 1:
        #     print(name1)
        #     print(f"Sum: {torch.sum(torch.abs(val_model))}")
        #     print(f"Mean: {torch.mean(torch.abs(val_model))}")
        #     print()


if __name__ == "__main__":
    with torch.no_grad():
        main()