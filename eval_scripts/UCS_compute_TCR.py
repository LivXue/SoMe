import json
import os
root = 'scores/user_comment_silmulation'

results = {}
for file in os.listdir(root):
    file_path = os.path.join(root, file)
    data = json.load(open(file_path))
    model = file.strip('.json')
    result = []
    
    for user, posts in data.items():
        for post in posts:
            for pid, res in post.items():
                if res != "错误":
                    result.append(1)
                else:
                    result.append(0)
                        
    results[model] = sum(result) / len(result) * 100
    
print(json.dumps(results, indent=4))
