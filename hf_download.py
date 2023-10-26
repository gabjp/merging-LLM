import argparse
from transformers import AutoTokenizer,AutoModelForCausalLM
import os 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--model-id", type=str, default="")
    args = parser.parse_args()

    os.environ['HF_HOME'] = '/tmp' 

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_id)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    tokenizer.save_pretrained(args.save_path)
    model.save_pretrained(args.save_path)




if __name__ == "__main__":
    main()