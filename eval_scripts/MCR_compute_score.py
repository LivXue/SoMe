import json
import os

root = 'scores/media_content_recommend'
ground_truth = json.load(open('datasets/media_content_recommend/ground_truth.json'))

results = {}

for file in os.listdir(root):
    data_file = os.path.join(root, file)
    data = json.load(open(data_file))
    model = file.strip('.json')
    result = []
    for user_id in data:
        for i, item in enumerate(data[user_id]):
            if list(item.values())[0] == ground_truth[user_id][i]["label"]:
                result.append(1)
            else:
                result.append(0)
        
    results[model] = sum(result)/len(result) * 100
output_file = f"scores/media_content_recommend/model_mean_score.json"  

print(json.dumps(results, ensure_ascii=False, indent=4))
