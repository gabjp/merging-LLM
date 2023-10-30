import argparse
from transformers import AutoTokenizer,AutoModelForCausalLM
import os 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--m1", type=str, default="")
    parser.add_argument("--m2", type=str, default="")
    parser.add_argument("--p", type=float, default=0.5)
    args = parser.parse_args()

    tokenizer1 = AutoTokenizer.from_pretrained(args.m1, use_fast=False)
    model1 = AutoModelForCausalLM.from_pretrained(args.m1)

    tokenizer2 = AutoTokenizer.from_pretrained(args.m2, use_fast=False)
    model2 = AutoModelForCausalLM.from_pretrained(args.m2)

    sd1 = model1.state_dict()
    sd2 = model2.state_dict()

    print("loaded")

    for key in sd1.keys():
        if sd1[key].size() != sd2[key].size():
            print(key, sd1[key].size(), sd2[key].size())

    sd = {key : (sd1[key]/2 + sd2[key]/2) for key in sd1.keys()}

    print("merged")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    model1.load_state_dict(sd)
    
    tokenizer1.save_pretrained(args.save_path)
    model1.save_pretrained(args.save_path)




if __name__ == "__main__":
    main()