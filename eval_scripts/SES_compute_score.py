import json 
import re
import os

from tqdm import tqdm


root_path = 'scores/streaming_event_summary'

for file_path in os.listdir(root_path):
    json_path = os.path.join(root_path, file_path)
    data = json.load(open(json_path))
    
    results = {}
    for topic, value in data.items():
        
        if topic in results:
            continue
        
        result = re.findall(r"\d+", value)
        age = sum([int(x) for x in result]) * 20 / len(result)
        results[topic] = age
        
    print(f"{file_path}:{sum(results.values())/len(results)}")
    # with open(output_file, "w", encoding="utf8") as f:
    #     json.dump(results, f, ensure_ascii=False, indent=4)
        
        