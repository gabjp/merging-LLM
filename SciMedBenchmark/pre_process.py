import json

def load_pubmedqa():
    f = open('data/pubmed_qa/train_set.json')
    train = json.load(f)

    f = open('data/pubmed_qa/dev_set.json')
    val = json.load(f)

    f = open('data/pubmed_qa/test_set.json')
    test = json.load(f)
    

    return train, val, test

def main():
    train, val, test = load_pubmedqa()
    post_train = []
    post_val = []
    post_test = []

    for (pre, post) in zip([train,val,test],[post_train,post_val,post_test]):
        for key in pre.keys():
            post_task = {"input": "Answer 'yes', 'no' or 'maybe'."}
            post_task["instruction"] = pre[key]["QUESTION"]
            post_task["output"] = pre[key]["final_decision"]
            post.append(post_task)

    with open('data/pubmed_qa_post/train.json', 'w', encoding='utf-8') as f:
        json.dump(post_train + post_val, f, ensure_ascii=False, indent=4)
    with open('data/pubmed_qa_post/test.json', 'w', encoding='utf-8') as f:
        json.dump(post_test, f, ensure_ascii=False, indent=4)
    


if __name__ == "__main__":
    main()