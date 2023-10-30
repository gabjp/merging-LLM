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

    print(model1.state_dict().keys())
    print(model2.state_dict().keys())

    #if not os.path.exists(args.save_path):
    #    os.makedirs(args.save_path)
    
    #tokenizer.save_pretrained(args.save_path)
    #model.save_pretrained(args.save_path)




if __name__ == "__main__":
    main()