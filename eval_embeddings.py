import argparse
from transformers import AutoTokenizer,AutoModelForCausalLM
import os 
import torch
import random
import gc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1", type=str, default="")
    parser.add_argument("--m2", type=str, default="")
    args = parser.parse_args()

    tokenizer1 = AutoTokenizer.from_pretrained(args.m1, use_fast=False)
    model1 = AutoModelForCausalLM.from_pretrained(args.m1)

    tokenizer2 = AutoTokenizer.from_pretrained(args.m2, use_fast=False)
    model2 = AutoModelForCausalLM.from_pretrained(args.m2)

    sd1 = model1.state_dict()
    sd2 = model2.state_dict()

    embeddings_1 = sd1["model.embed_tokens.weight"]
    embeddings_2 = sd2["model.embed_tokens.weight"]

    print(f"Embedding 1: {embeddings_1.size}")
    print(f"Embedding 2: {embeddings_2.size}")

    v1 = tokenizer1.get_vocab()
    v2 = tokenizer2.get_vocab()

    # Find the differences between the two dictionaries
    keys = set(v1.keys()).union(set(v2.keys()))
    differences = {}
    for key in keys:
        if v1.get(key) != v2.get(key):
            differences[key] = [v1.get(key), v2.get(key)]

    # Print the differences
    print("Differences between the two vocabs:")
    print(differences)





if __name__ == "__main__":
    with torch.no_grad():
        main()