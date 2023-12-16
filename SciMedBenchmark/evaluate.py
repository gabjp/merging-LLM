import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, QuestionAnsweringPipeline, AutoModelForCausalLM
import random
random.seed(0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="")
parser.add_argument("--model-name", type=str, default="")
args = parser.parse_args()

def load_data():
    f = open('data/pubmed_qa_post/test.json')
    pubmedqa = json.load(f)
    f = open('data/sciq_post/test.json')
    sciq = json.load(f)
    f = open('data/boolq_post/test.json')
    boolq = json.load(f)

    return pubmedqa, sciq, boolq

def question_template(question, options):
    prompt = f""" A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n Human: {question}\n{options}\nAssistant: """
    return prompt

def main():
    #Load Data
    pubmedqa, sciq, boolq = load_data()

    pubmedqa_questions = [elem["instruction"] for elem in pubmedqa]
    pubmedqa_answer = [elem["output"] for elem in pubmedqa]
    pubmedqa_options = [elem["input"] for elem in pubmedqa]
   
    sciq_questions = [elem["instruction"] for elem in sciq]
    sciq_answer = [elem["output"] for elem in sciq]
    sciq_options = [elem["input"] for elem in sciq]

    boolq_questions = [elem["instruction"] for elem in boolq]
    boolq_answer = [elem["output"] for elem in boolq]
    boolq_options = [elem["input"] for elem in boolq]
    
    #Load Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    model.eval()

    #Generate Answer
    print("evaluating sciq")
    response = []
    sciq_prompts = [question_template(question, options) for question, options in zip(sciq_questions, sciq_options)]
    sciq_lens = [len(question) for question in sciq_prompts]
    for idx in range(0, len(sciq_prompts)):
        inputs = tokenizer(sciq_prompts[idx], padding=False, return_tensors="pt", truncation=True, max_length=2048).to(device)
        output = model.generate(**inputs, do_sample=False, max_new_tokens=64, min_new_tokens=2)
        for out in output.tolist():
            decoded = tokenizer.decode(out, skip_special_tokens=True)
            #print(decoded)
            response.append(decoded)
    preds = [out[i:] for out, i in zip(response, sciq_lens)]
    print(preds)

    sciq_count = [int((true in pred.partition('\n')[0]) and (len(pred.partition('\n')[0]) <= 2.5*len(true))) for true,pred in zip(sciq_answer, preds)]
    print(f"Sciq acc (contained first line and length): {sum(sciq_count)/len(sciq_count)}")
    
    print("evaluating pubmedqa")
    response = []
    pubmedqa_prompts = [question_template(question, options) for question, options in zip(pubmedqa_questions, pubmedqa_options)]
    pubmedqa_lens = [len(question) for question in pubmedqa_prompts]
    for idx in range(0, len(pubmedqa_prompts)):
        inputs = tokenizer(pubmedqa_prompts[idx], padding=False, return_tensors="pt", truncation=True, max_length=2048).to(device)
        output = model.generate(**inputs, do_sample=False, max_new_tokens=64, min_new_tokens=2)
        for out in output.tolist():
            decoded = tokenizer.decode(out, skip_special_tokens=True)
            #print(decoded)
            response.append(decoded)
    preds = [out[i:] for out, i in zip(response, pubmedqa_lens)]
    print(preds)

    pubmedqa_count = [int((true in pred.split()[0]) and (len(pred.partition('\n')[0]) <= 2.5*len(true))) for true,pred in zip(pubmedqa_answer, preds)]
    print([pred.split()[0] for pred in preds])
    print(f"pubmedqa acc (contained first word and first line length): {sum(pubmedqa_count)/len(pubmedqa_count)}")

    print("evaluating boolq")
    response = []
    boolq_prompts = [question_template(question, options) for question, options in zip(boolq_questions, boolq_options)]
    boolq_lens = [len(question) for question in boolq_prompts]
    for idx in range(0, len(boolq_prompts)):
        inputs = tokenizer(boolq_prompts[idx], padding=False, return_tensors="pt", truncation=True, max_length=2048).to(device)
        output = model.generate(**inputs, do_sample=False, max_new_tokens=64, min_new_tokens=2)
        for out in output.tolist():
            decoded = tokenizer.decode(out, skip_special_tokens=True)
            #print(decoded)
            response.append(decoded)
    preds = [out[i:] for out, i in zip(response, boolq_lens)]
    print(preds)

    boolq_count = [int((true in pred.split()[0]) and (len(pred.partition('\n')[0]) <= 2.5*len(true))) for true,pred in zip(boolq_answer, preds)]
    print([pred.split()[0] for pred in preds])
    print(f"boolq acc (contained first word and first line length): {sum(boolq_count)/len(boolq_count)}")
    

if __name__ == "__main__":
    main()