import argparse
from transformers import AutoTokenizer,AutoModelForCausalLM
import os 
import torch
import random

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


    t1 = sd1["model.embed_tokens.weight"]
    t2 = sd2["model.embed_tokens.weight"][0:32000,:]

    norms = torch.norm(t1-t2, dim=1)
    print("average distance between model embeddings")
    print(torch.mean(norms))

    combinations = [(i,j) for i in range(32000) for j in range(i+1, 32000)]
    random.shuffle(combinations)
    combinations = combinations[0:32000]
    id1 = [elem[0] for elem in combinations]
    id2 = [elem[1] for elem in combinations]

    norms = torch.norm(t1[id1]-t1[id2], dim=1)
    print("average distance between embeddings of model 1")
    print(torch.mean(norms))

    norms = torch.norm(t2[id1]-t2[id2], dim=1)
    print("average distance between embeddings of model 2")
    print(torch.mean(norms))

    return 


    sd = {key : (sd1[key]/2 + sd2[key]/2) for key in sd1.keys()}

    print("merged")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    model1.load_state_dict(sd)
    
    tokenizer1.save_pretrained(args.save_path)
    model1.save_pretrained(args.save_path)




if __name__ == "__main__":
    main()