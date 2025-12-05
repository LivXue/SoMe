import json
import os
root = 'scores/user_emotion_analysis'

results = {}
for file in os.listdir(root):
    file_path = os.path.join(root, file)
    data = json.load(open(file_path))
    model = file.strip('.json')
    result = []
    
    for user, posts in data.items():
        for post in posts:
            for pid, emo in post.items():
                if emo != "错误":
                    result.append(1)
                else:
                    result.append(0)
                        
    results[model] = sum(result) / len(result) * 100
    
print(json.dumps(results, indent=4))
