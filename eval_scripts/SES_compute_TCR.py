import json
import os
import re

from tqdm import tqdm
from openai import OpenAI

def remove_think_tags(text):
    # 使用正则表达式匹配<think>到</think>之间的内容，并将其替换为空字符串
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()


prompt = '''
    请仔细阅读话题事件摘要参考答案“<place_holder>”，并根据参考答案对给出的话题事件摘要内容进行评分，结果只能输出 “正确” 或 “错误”：
    输出格式要求：
    - 如果给出的话题事件摘要内容与参考答案在格式上基本相符，比如有提到事件，时间等关键信息，但是对应的内容上与参考答案完全不符，则输出：正确
    - 如果给出的话题事件摘要内容与参考答案无任何关联，比如格式和内容上都不符，或者判定不出是否有关联，则输出：错误
    注意：请严格按照上述要求输出结果，不要添加任何解释、说明或多余内容。
'''
client = OpenAI(
            api_key="mysecrettoken123",
            base_url="http://0.0.0.0:8000/v1",
        )
ground_truth = json.load(open("datasets/realtime_event_detection/ground_truth.json"))
root = 'scores/realtime_event_detection'
raw_data_path = 'results/realtime_event_detection'

model_results = {}
for file in tqdm(os.listdir(root)):
    results = {}
    file_path = os.path.join(root, file)
    data = json.load(open(file_path))
    model = file.strip('.json')
    
    raw_file_path = os.path.join(raw_data_path, file)
    raw_data = json.load(open(raw_file_path))
    
    zero_keys = [key for key, value in data.items() if value == 0]
    for key in tqdm(zero_keys):
        if key in results:
            continue
        current_prompt = prompt.replace("<place_holder>", ground_truth[key])
        if len(raw_data[key]) > 5000:
            results[key]= "错误"
            continue
        query = f"给出的事件摘要内容为“{raw_data[key]}”"
        chat_response = client.chat.completions.create(
                    model="Qwen3-32B",
                    messages=[
                        {"role": "system", "content": current_prompt},
                        {"role": "user", "content": query},
                    ],
                )
        response = chat_response.choices[0].message.content
        results[key]= remove_think_tags(response)
        
    result = []
    for json_name, text in results.items():
        if text == "正确":
            result.append(1)
        else:
            result.append(0)
                         
    model_results[model] = sum(result) / len(data) * 100

print(json.dumps(model_results, ensure_ascii=False, indent=4))
