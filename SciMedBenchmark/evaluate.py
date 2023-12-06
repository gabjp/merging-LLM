import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, QuestionAnsweringPipeline, AutoModelForCausalLM
import random
random.seed(0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
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

def question_template(question, contexts, options):
    prompt = f"""Answer the following [QUESTION] based on the given [CONTEXTS], if available. \n[QUESTION]\n{question}\n[Contexts]"""
    if contexts[0] == "":
        prompt+= "No context available \n"
    else:
        for i in range(0, len(contexts)):
            prompt += f"Context {i+1}: {contexts[i]}\n"
    prompt+= "[OPTIONS]\n"
    for i in range(len(options)):
        prompt+= f"{i+1}. {options[i]}\n"
    prompt+= "The answer must be given in only one line, containing only the text of the right answer"

def main():
    #Load Data
    pubmedqa, sciq = load_data()

    pubmedqa_questions = [pubmedqa[key]["QUESTION"] for key in pubmedqa.keys()]
    pubmedqa_answer = [pubmedqa[key]["final_decision"] for key in pubmedqa.keys()]
    pubmedqa_context = [pubmedqa[key]["CONTEXTS"] for key in pubmedqa.keys()]
    pubmed_options = ['yes', 'no', 'maybe']

    sciq_questions = [elem["question"] for elem in sciq]
    sciq_answer = [elem["correct_answer"] for elem in sciq]
    sciq_context = [elem["support"] for elem in sciq]
    sciq_options = [random.sample([elem['distractor1'], elem['distractor2'], elem['distractor3'], elem['correct_answer']], 4) for elem in sciq]

    print(question_template(pubmedqa_questions[0], pubmedqa_context[0], pubmed_options))
    print(question_template(sciq_questions[0], sciq_context[0], sciq_options[0]))
    return
    #Load Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)

    #Generate Answer
    pipeline = QuestionAnsweringPipeline(model=model, framework="pt", tokenizer=tokenizer, device=device)

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