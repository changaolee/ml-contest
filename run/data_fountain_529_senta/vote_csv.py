import pandas as pd
import csv

results = {'bert_base.csv': 0.69339421919,
           'skep_hidden_fusion.csv': 0.69258945423,
           'bert_hidden_fusion.csv': 0.69966576165}

dfs = [(pd.read_csv(filename), weight) for filename, weight in results.items()]

results = []

num = len(dfs[0][0])
for i in range(num):
    vote = {}
    _id = dfs[0][0].iloc[i]['id']
    for j in range(len(dfs)):
        df, weight = dfs[j]
        label = df.iloc[i]['class']
        vote[label] = vote.get(label, 0.) + weight
    label = sorted(vote, key=lambda x: vote[x])[-1]
    results.append([_id, label])

with open("result.csv", "w", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "class"])
    for line in results:
        qid, label = line
        writer.writerow([qid, label])
