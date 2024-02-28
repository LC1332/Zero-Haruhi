import json
import sys
score_dict = {}
path = sys.argv[1]
with open(path, "r") as f:
    records = json.load(f)
for record in records:
    if record['metric_en'] not in score_dict:
        score_dict[record['metric_en']] = []
    score_dict[record['metric_en']].append(record[record['metric_en']])

for metric in score_dict:
    print(metric)
    print(sum(score_dict[metric]) / len(score_dict[metric]))
