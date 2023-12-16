import json
import random
random.seed(0)

def load_pubmedqa():
    f = open('data/pubmed_qa/train_set.json')
    train = json.load(f)

    f = open('data/pubmed_qa/dev_set.json')
    val = json.load(f)

    f = open('data/pubmed_qa/test_set.json')
    test = json.load(f)
    

    return train, val, test

def pre_pubmedqa():
    train, val, test = load_pubmedqa()
    post_train = []
    post_val = []
    post_test = []

    for (pre, post) in zip([train,val,test],[post_train,post_val,post_test]):
        for key in pre.keys():
            post_task = {}
            context_str = "Context:\n"
            for context in pre[key]["CONTEXTS"]:
                context_str += context
                context_str += "\n"
            post_task["input"] = context_str + "Answer 'yes', 'no' or 'maybe'."
            post_task["instruction"] = pre[key]["QUESTION"]
            post_task["output"] = pre[key]["final_decision"]
            post.append(post_task)

    with open('data/pubmed_qa_post/train.json', 'w', encoding='utf-8') as f:
        json.dump(post_train + post_val, f, ensure_ascii=False, indent=4)
    with open('data/pubmed_qa_post/test.json', 'w', encoding='utf-8') as f:
        json.dump(post_test, f, ensure_ascii=False, indent=4)

def load_sciq():
    f = open('data/sciq/train.json')
    train = json.load(f)

    f = open('data/sciq/valid.json')
    val = json.load(f)

    f = open('data/sciq/test.json')
    test = json.load(f)
    

    return train, val, test

def pre_sciq():
    train, val, test = load_sciq()
    post_train = []
    post_val = []
    post_test = []

    for (pre, post) in zip([train,val,test],[post_train,post_val,post_test]):
        for task in pre:
            post_task = {}
            sciq_options = [task['distractor1'], task['distractor2'], task['distractor3'], task['correct_answer']]
            random.shuffle(sciq_options)
            post_task["input"] = f"Context:\n{task['support']}\nAnswer '{sciq_options[0]}', '{sciq_options[1]}', '{sciq_options[2]}' or '{sciq_options[3]}'."
            post_task["instruction"] = task["question"]
            post_task["output"] = task["correct_answer"]
            post.append(post_task)

    with open('data/sciq_post/train.json', 'w', encoding='utf-8') as f:
        json.dump(post_train + post_val, f, ensure_ascii=False, indent=4)
    with open('data/sciq_post/test.json', 'w', encoding='utf-8') as f:
        json.dump(post_test, f, ensure_ascii=False, indent=4)

def load_boolq():
    train = []

    with open('data/boolq/train.jsonl', 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        train.append(json.loads(json_str))

    val = []

    with open('data/boolq/dev.jsonl', 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        val.append(json.loads(json_str))

   
    return train, val

def pre_boolq():
    train, val = load_boolq()
    post_train = []
    post_val = []

    for (pre, post) in zip([train,val],[post_train,post_val]):
        for task in pre:
            post_task = {}
            post_task["input"] = "Context:\n" + task["passage"] + "\nAnswer 'true' or 'false'."
            post_task["instruction"] = task["question"]
            post_task["output"] = task["answer"]
            post.append(post_task)

    with open('data/boolq_post/train.json', 'w', encoding='utf-8') as f:
        json.dump(post_train, f, ensure_ascii=False, indent=4)
    with open('data/boolq_post/test.json', 'w', encoding='utf-8') as f:
        json.dump(post_val, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    pre_boolq()