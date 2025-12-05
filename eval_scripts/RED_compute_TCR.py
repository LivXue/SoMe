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
    请对给出的文本内容进行评分，结果只能输出 “正确” 或 “错误”：
    输出格式要求：
    - 如果给出的文本内容提供了话题事件摘要，比如有提到事件，时间等关键信息，或认为没有发生的事件需要摘要，则输出：“正确”
    - 如果给出的文本内容没有提供话题事件摘要，只包含无关信息，则输出：错误
    注意：请严格按照上述要求输出结果，不要添加任何解释、说明或多余内容。
'''
client = OpenAI(
            api_key="mysecrettoken123",
            base_url="http://0.0.0.0:8002/v1",
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
    
    result = []
    
    zero_keys = [key for key, value in data.items() if value == 0]
    for key in tqdm(zero_keys):
        if key in results:
            continue
        # llama may trap in infinite loop
        if len(raw_data[key]) > 5000:
            results[key]= "错误"
            continue
        query = f"给出的文本内容为“{raw_data[key]}”"
        chat_response = client.chat.completions.create(
                    model="Qwen3-32B",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": query},
                    ],
                )
        response = chat_response.choices[0].message.content
        results[key]= remove_think_tags(response)

    for json_name, text in tqdm(results.items()):
        if text == "正确":
            result.append(1)
        else:
            result.append(0)
                         
    model_results[model] = 100 - sum(result) / len(data) * 100
    print(f"{model}: {model_results[model]}")

#output =f"scores/realtime_event_detection/TCRs.json"
# with open(output, "w", encoding="utf8") as f:
#     json.dump(model_results, f, ensure_ascii=False, indent=4)

print(json.dumps(model_results, ensure_ascii=False, indent=4))
