import json 
import re
import os

from tqdm import tqdm


root_path = 'scores/realtime_event_detection'
 
for file_path in os.listdir(root_path):
    json_path = os.path.join(root_path, file_path)
    data = json.load(open(json_path))
    
    output_file = f"scores/realtime_event_detection/{file_path}"  
    results = {}
    for topic, value in data.items():
        
        if topic in results:
            continue
        
        result = re.findall(r"\d+", value)
        assert len(result) == 3, f"Bad results {result} of {file_path}"
        age = sum([int(x) for x in result]) / len(result) * 20
        results[topic] = age
        
    print(f"{file_path}: {sum(results.values())/len(results)}")
        
        