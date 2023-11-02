import json

ques = []

with open("flask_evaluation.jsonl", "r") as jsonl:
    for line in jsonl:
        ques_json = json.loads(line)
        if "Coding" in ques_json["domain_labeled"] or "Culture" in ques_json["domain_labeled"]:
            ques.append(ques_json)

print(len(ques))

with open("coding_culture.jsonl", "w") as file:
    for x in ques:
        json.dump(x, file)
        file.write("\n")