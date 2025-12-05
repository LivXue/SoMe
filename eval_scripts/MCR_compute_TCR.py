import json
import os
root = 'scores/media_content_recommend'

results = {}
for file in os.listdir(root):
    if file  == 'model_mean_score.json':
        continue
    file_path = os.path.join(root, file)
    data = json.load(open(file_path))
    model = file.strip('.json')
    result = []
    
    for user_id, samples in data.items():
        for item in samples:
            for idx, text in item.items():
                if text == "错误":
                    result.append(1)
                else:
                    result.append(0)
                        
    results[model] = 100 - sum(result) / len(result) * 100
    
print(json.dumps(results, indent=4))
