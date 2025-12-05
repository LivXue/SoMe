import json
import os

root = 'scores/misinformation_detection'
ground_truth = json.load(open('datasets/misinformation_detection/ground_truth.json'))

results = {}

for file in os.listdir(root):
    data_file = os.path.join(root, file)
    data = json.load(open(data_file))
    model = file.strip('.json')
    result = []
    for json_name, value in data.items():
        if value == ground_truth[json_name]["label"]:
            result.append(1)
        else:
            result.append(0)
        
    results[model] = sum(result) / len(result) * 100

print(json.dumps(results, indent=4))
