import json
import os

root = 'scores/user_behavior_prediction'
ground_truth = json.load(open('datasets/user_behavior_prediction/ground_truth.json'))

key_mapping = {
    "like": "点赞",
    "comment": "评论",
    "repost": "转发"
}

new_ground_truth= {key_mapping.get(k, k): v for k, v in ground_truth.items()}

results = {}

for file in os.listdir(root):
    data_file = os.path.join(root, file)
    data = json.load(open(data_file))
    model = file.strip('.json')
    result = []
    for interaction, users in data.items():
        for user_id, perd_result in users.items():
            if user_id not in new_ground_truth[interaction]:
                continue
            for key, value in perd_result.items():
                if value != "错误":
                    result.append(1)
                else:
                    result.append(0)
        
    results[model] = sum(result) / len(result) * 100
    
print(json.dumps(results, indent=4))
