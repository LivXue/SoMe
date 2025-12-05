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
    请仔细阅读问题“<place_holder>”，并对给出的回答进行判断，结果只能输出 “正确” 或 “错误”：
    输出格式要求：
    - 如果给出的回答试图对问题提供了答案，则输出：正确
    - 如果给出的回答没有对问题提供回答，或者提到在回答的过程中发生了错误，则输出：错误
    注意：请严格按照上述要求输出结果，不要添加任何解释、说明或多余内容**。
'''
client = OpenAI(
            api_key="mysecrettoken123",
            base_url="http://0.0.0.0:8002/v1",
        )
ground_truth = json.load(open("datasets/social_media_question_answering/ground_truth.json"))
root = 'scores/social_media_question_answering'
raw_data_path = 'results/social_media_question_answering'

model_results = {}
for file in tqdm(os.listdir(root)):
    results = {}
    file_path = os.path.join(root, file)
    data = json.load(open(file_path))
    
    raw_file_path = os.path.join(raw_data_path, file)
    raw_data = json.load(open(raw_file_path))
    
    result = []
    
    zero_keys = [key for key, value in data.items() if value.split("：")[-1] == "0"]
    for key in tqdm(zero_keys):
        if len(raw_data[key]) > 5000:
            results[key]= "错误"
            continue
        current_prompt = prompt.replace("<place_holder>", key)
        query = f"给出的回答为“{raw_data[key]}”"
        chat_response = client.chat.completions.create(
                    model="Qwen3-32B",
                    messages=[
                        {"role": "system", "content": current_prompt},
                        {"role": "user", "content": query},
                    ],
                )
        response = chat_response.choices[0].message.content
        results[key]= remove_think_tags(response)

    for json_name, text in tqdm(results.items()):
        if text == "错误":
            result.append(1)
        else:
            result.append(0)
                         
    model_results[file.split('.json')[0]] = 100 - sum(result) / len(data) * 100

print(json.dumps(model_results, ensure_ascii=False, indent=4))
