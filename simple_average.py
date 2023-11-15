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
    parser.add_argument("--merge-embeddings", type=int, default=0)
    args = parser.parse_args()

    tokenizer1 = AutoTokenizer.from_pretrained(args.m1, use_fast=False)
    model1 = AutoModelForCausalLM.from_pretrained(args.m1)

    tokenizer2 = AutoTokenizer.from_pretrained(args.m2, use_fast=False)
    model2 = AutoModelForCausalLM.from_pretrained(args.m2)


    model1.resize_token_embeddings(32001)

    input_embeddings = model1.get_input_embeddings().weight.data
    output_embeddings = model1.get_output_embeddings().weight.data

    input_embeddings_avg = input_embeddings[:-1].mean(dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings[:-1].mean(dim=0, keepdim=True)

    input_embeddings[-1:] = input_embeddings_avg
    output_embeddings[-1:] = output_embeddings_avg



    sd1 = model1.named_parameters()
    sd2 = model2.named_parameters()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("merging", flush=True)


    # Use vicuna

    if args.merge_embeddings == 0:
        pass

        #for key in keys:
        #    print(key, flush=True)
        #    if key == "model.embed_tokens.weight":
        #        continue
        #    elif key == "lm_head.weight":
        #        continue
        #    else:
        #        sd1[key].mul_(args.p)
        #        sd2[key].mul_(1-args.p)
        #        sd1[key].add_(sd2[key])


    # drop and merge

    else:

        for ((name1,val1),(name2,val2)) in zip(sd1,sd2):
            if name1 == "model.embed_tokens.weight":
                print(val1)
                val1.mul_(args.p)
                val2.mul_(1-args.p)
                val1.add_(val2)
            elif name1 == "lm_head.weight":
                val1.mul_(args.p)
                val2.mul_(1-args.p)
                val1.add_(val2)
            else:
                val1.mul_(args.p)
                val2.mul_(1-args.p)
                val1.add_(val2)



    print("merged")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for (name1, val1) in model1.named_parameters():
        if name1 == "model.embed_tokens.weight":
            print(val1)
        else:
            continue
    
    tokenizer2.save_pretrained(args.save_path)
    model1.save_pretrained(args.save_path)




if __name__ == "__main__":
    with torch.no_grad():
        main()