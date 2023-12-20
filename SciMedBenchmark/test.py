import json

f = open('SciMedBenchmark/data/newsqa/combined-newsqa-data-v1.json')
dict = json.load(f)

count= 0

for story in dict['data']:
    for q in story['questions']:
        count+=1

print(count)

print(dict['data'][100]['text'])