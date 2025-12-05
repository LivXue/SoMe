import json
import os
root = 'scores/misinformation_detection'

results = {}
for file in os.listdir(root):
    file_path = os.path.join(root, file)
    data = json.load(open(file_path))
    model = file.strip('.json')
    result = []
    
    for idx, text in data.items():
        if text != "error":
            result.append(1)
        else:
            result.append(0)
                        
    results[model] = sum(result) / len(result) * 100
    
print(json.dumps(results, indent=4))
