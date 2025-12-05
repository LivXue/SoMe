import json
import os
import re
import argparse

from tqdm import tqdm
from openai import OpenAI


args = argparse.ArgumentParser()
args.add_argument('--result_path', type=str, default='results/social_media_question_answering', description='Path to the result files')
args.add_argument('--output_path', type=str, default='scores/social_media_question_answering', description='Path to the output files')
args.add_argument('--setting_path', type=str, default='eval_scripts/settings.json', description='Path to the setting file')
args = args.parse_args()

settings = json.load(open(args.setting_path, 'r'))

def remove_think_tags(text):
    # 使用正则表达式匹配<think>到</think>之间的内容，并将其替换为空字符串
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()



prompt = '''
    请仔细阅读所给出问题的参考答案<place_holder>，并根据参考答案对该问题的回答内容进行评分，评分范围为 0 到 5 分，评分维度为：
    - 准确性（Accuracy）：是否准确反映参考答案中的关键信息；
    请根据该维度为问题的回答内容打分，并输出综合评判后的评分，
    准确性评分标准定义如下：
    - 5 分：完全符合参考答案内容，没有错误或遗漏；
    - 4 分：基本符合参考答案内容，存在少量轻微错误或遗漏；
    - 3 分：部分符合参考答案内容，存在一些错误或遗漏，但不影响整体理解；
    - 2 分：少部分符合参考答案内容，存在较多错误或遗漏，影响了部分理解；
    - 1 分：勉强符合参考答案内容，存在严重错误或大量遗漏；
    - 0 分：完全不符合参考答案内容，信息错误或空白；
    输出格式要求：
    - 如果评分为：准确性：2，则输出：准确性：2
    注意：请严格按照上述要求输出结果，不要添加任何解释、说明或多余内容。
'''
client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
        )
ground_truth = json.load(open("datasets/social_media_question_answering/ground_truth.json"))
pattern = r"准确性：\d+"


for file_path in tqdm(os.listdir(args.result_path)):
    json_path = os.path.join(args.result_path, file_path)
    raw_data = json.load(open(json_path))

    output_file = os.path.join(args.output_path, file_path)
    if os.path.exists(output_file):
        results = json.load(open(output_file))
    else:
        results = {}
    for topic, summary in tqdm(raw_data.items()):
        if topic in results:
            continue
        current_prompt = prompt.replace("<place_holder>", ground_truth[topic])
        if len(summary) > 5000:
            results[topic]= "准确性：0"
            continue
        query = f"该问题的回答内容为“{summary}”"
        while True:
            chat_response = client.chat.completions.create(
                        model=settings.model,
                        messages=[
                            {"role": "system", "content": current_prompt},
                            {"role": "user", "content": query},
                        ],
                    )
            response = chat_response.choices[0].message.content
            print(f"{raw_data[topic]}\n{response}\n")
            score = remove_think_tags(response)
            if re.match(pattern, score):
                results[topic]= score
                break
                
    #print(json.dumps(results, ensure_ascii=False, indent=4))  
        with open(output_file, "w", encoding="utf8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)