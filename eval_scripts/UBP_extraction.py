import json
import os
import re
import argparse

from tqdm import tqdm
from openai import OpenAI


args = argparse.ArgumentParser()
args.add_argument('--result_path', type=str, default='results/user_behavior_prediction', description='Path to the result files')
args.add_argument('--output_path', type=str, default='scores/user_behavior_prediction', description='Path to the output files')
args.add_argument('--setting_path', type=str, default='eval_scripts/settings.json', description='Path to the setting file')
args = args.parse_args()

settings = json.load(open(args.setting_path, 'r'))

def remove_think_tags(text):
    # 使用正则表达式匹配<think>到</think>之间的内容，并将其替换为空字符串
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

prompt = '''
    请仔细阅读给出的文本内容，并从中提取出最终的判断结果，判断该用户是否可能对该贴文发出<place_holder>行为。结果只能输出 “是” 或 “否”，不需要任何解释或额外内容。
    输出格式要求：
    - 如果文本内容的判断结果为肯定，则输出：是
    - 如果文本内容的判断结果为否定，则输出：否
    - 如果文本内容的判断结果为不确定，或者没有判断结果，则输出：错误
    注意：请严格按照上述要求输出结果。
    示例输入：
    - "根据用户“奥特曼与小怪兽看星星”（ID：5288580817）的贴文分析，其主要活跃领域集中在明星粉丝互动（尤其是孙千）、转发微博和少量原创内容。用户对明星相关内容的互动较多，但对其他类型的内容（如祈祷平安类）的互动较少。\n\n原帖内容为“🙏祈祷平安”，发布者为“李小璐Super璐”，属于情感类内容。从用户的历史行为来看，其点赞和互动主要集中在明星粉丝活动，对情感类内容的互动较少。因此，综合判断：\n\n否"
    示例输出：
    - "否"
'''
client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
        )


for file_path in tqdm(os.listdir(args.result_path)):
    json_path = os.path.join(args.result_path, file_path)
    raw_data = json.load(open(json_path))
    
    output_file = os.path.join(args.output_path, file_path)
    if os.path.exists(output_file):
        results = json.load(open(output_file))
    else:
        results = {}
    for interaction, samples in tqdm(raw_data.items()):
        if interaction not in results:
            results[interaction] = {}
        for user_id, user_samples in tqdm(samples.items()):
            if user_id not in results[interaction]:
                results[interaction][user_id] = {}
            for idx, text in user_samples.items():
                if idx in results[interaction][user_id]:
                    continue
                if text in ["是", "否"]:
                    results[interaction][user_id][idx] = text
                    continue
                elif len(text) > 5000:
                    results[interaction][user_id][idx] =  "错误"
                    continue
                
                query = text
                current_prompt = prompt.replace("<place_holder>", interaction)
                while True:
                    chat_response = client.chat.completions.create(
                                model=settings.model,
                                messages=[
                                    {"role": "system", "content": current_prompt},
                                    {"role": "user", "content": query},
                                ],
                            )
                    response = chat_response.choices[0].message.content
                    print(f"{raw_data[interaction][user_id][idx]}\n{response}\n")
                    answer = remove_think_tags(response)
                    if answer in ["是", "否", "错误"]:
                        results[interaction][user_id][idx] = answer
                        break

            with open(output_file, "w", encoding="utf8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
                