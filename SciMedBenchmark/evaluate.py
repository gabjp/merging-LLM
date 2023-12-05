import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, QuestionAnsweringPipeline, AutoModelForCausalLM


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="")
parser.add_argument("--model-name", type=str, default="")
args = parser.parse_args()

def load_data():
    f = open('data/test_PubMedQA.json')
    pubmedqa = json.load(f)
    f = open('data/test_SciQ.json')
    sciq = json.load(f)

    return pubmedqa, sciq

def main():
    #Load Data
    pubmedqa, sciq = load_data()

    #Load Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)#.to(device)

    pipeline = QuestionAnsweringPipeline(model=model, framework="pt", tokenizer=tokenizer, device=device)

    #Generate Answer
    pubmedqa_questions = [pubmedqa[key]["QUESTION"] for key in pubmedqa.keys()]
    pubmedqa_answer = [pubmedqa[key]["final_decision"] for key in pubmedqa.keys()]
    pubmedqa_context = [pubmedqa[key]["CONTEXTS"] for key in pubmedqa.keys()]

    sciq_questions = [elem["question"] for elem in sciq]
    sciq_answer = [elem["correct_answer"] for elem in sciq]
    sciq_context = [elem["support"] for elem in sciq]

    print("Answering sciq questions...")
    sciq_predictions = pipeline(context=sciq_context[0], question=sciq_questions[0])

    print("Answering sciq questions...")
    pubmedqa_predictions = pipeline(context=pubmedqa_context[0], question=pubmedqa_questions[0])

    print(sciq_predictions)

    print(pubmedqa_predictions)



    #Save
    #Compute Metrics
    #Save

    pass

if __name__ == "__main__":
    main()