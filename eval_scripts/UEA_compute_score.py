import json
import os

root = 'results_newly/user_emotion_analysis'
ground_truth = json.load(open('datasets/user_emotion_analysis/ground_truth.json'))

results = {}

for file in os.listdir(root):
    if 'score' in file:
        continue
    data_file = os.path.join(root, file)
    data = json.load(open(data_file))
    model = file.strip('.json')
    result = []
    for user_id in data:
        for i, item in enumerate(data[user_id]):
            if list(item.values())[0] == ground_truth[user_id][i]["emotion"]:
                result.append(1)
            else:
                result.append(0)
        
    results[model] = sum(result) / len(result) * 100

print(json.dumps(results, indent=4))
