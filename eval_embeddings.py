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

    print(tokenizer1.eos_token)
    print(tokenizer1.eos_token_id)

    print(tokenizer2.eos_token)
    print(tokenizer2.eos_token_id)

    sd1 = model1.state_dict()
    sd2 = model2.state_dict()

    embeddings_1 = sd1["model.embed_tokens.weight"]
    embeddings_2 = sd2["model.embed_tokens.weight"]

    print(f"Embedding 1: {embeddings_1.size()}")
    print(f"Embedding 2: {embeddings_2.size()}")

    v1 = tokenizer1.get_vocab()
    v2 = tokenizer2.get_vocab()

    print(v2[13,])
    print(v2[829])
    print(v2[29879])
    print(v2[29958])

    return

    # Find the differences between the two dictionaries
    keys = set(v1.keys()).union(set(v2.keys()))
    differences = {}
    for key in keys:
        if v1.get(key) != v2.get(key):
            differences[key] = [v1.get(key), v2.get(key)]

    # Print the differences
    print("Differences between the two vocabs:")
    print(differences)

    #compute distances between models

    t1 = sd1["model.embed_tokens.weight"][0:32000,:]
    t2 = sd2["model.embed_tokens.weight"][0:32000,:]

    norms = torch.norm(t1-t2, dim=1)
    print("average distance between model embeddings")
    print(torch.mean(norms))

    #compute distances inside models

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





if __name__ == "__main__":
    with torch.no_grad():
        main()