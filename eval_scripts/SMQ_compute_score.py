import json 
import re
import os

from tqdm import tqdm


root_path = 'scores/social_media_question_answering'

for file_path in os.listdir(root_path):
    json_path = os.path.join(root_path, file_path)
    data = json.load(open(json_path))
    
    output_file = f"scores/social_media_question_answering/{file_path}"  
    results = {}
    for topic, value in data.items():
        
        if topic in results:
            continue
        
        result = re.findall(r"\d+", value)
        assert len(result) == 1, f"Bad results {result} of {file_path}"
        age = sum([int(x) for x in result])/len(result) * 20
        results[topic] = age
        
    with open(output_file, "w", encoding="utf8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"{file_path}: {sum(results.values())/len(results)}")
        
        