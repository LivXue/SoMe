import json
import os

root = 'scores/user_comment_silmulation'
ground_truth = json.load(open('datasets/user_comment_silmulation/ground_truth.json'))

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
output_file = f"scores/user_comment_silmulation/model_mean_score.json"  
    
with open(output_file, "w", encoding="utf8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(json.dumps(results, ensure_ascii=False, indent=4))
