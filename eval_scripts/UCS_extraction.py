import json
import os
import re
import argparse

from tqdm import tqdm
from openai import OpenAI


args = argparse.ArgumentParser()
args.add_argument('--result_path', type=str, default='results/user_comment_silmulation', description='Path to the result files')
args.add_argument('--output_path', type=str, default='scores/user_comment_silmulation', description='Path to the output files')
args.add_argument('--setting_path', type=str, default='eval_scripts/settings.json', description='Path to the setting file')
args = args.parse_args()

settings = json.load(open(args.setting_path, 'r'))

def remove_think_tags(text):
    # 使用正则表达式匹配<think>到</think>之间的内容，并将其替换为空字符串
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()


prompt = '''
    请仔细阅读给出的文本内容，并从中提取出最终的判断结果，判断该评论是否可能由该用户发表。结果只能输出 “是” 或 “否”，不需要任何解释或额外内容。
    输出格式要求：
    - 如果文本内容的判断结果为肯定，则输出：是
    - 如果文本内容的判断结果为否定，则输出：否
    - 如果文本内容的判断结果为不确定，或者没有判断结果，则输出：错误
    注意：请严格按照上述要求输出结果。
    示例输入：
    - "根据用户7391946667（亲玺宝宝）的贴文内容和评论风格分析，其发言主要集中在易烊千玺相关话题、日常问候和少量活动推广。评论“今天生日 开年svip求你让我抽个”涉及生日和SVIP抽奖内容，但该用户的历史贴文中并未表现出类似的抽奖或生日相关互动风格，且其语言风格（如频繁使用“晚安”“早上好”等简单问候）与评论的随意请求语气不一致。\n\n此外，评论中提到的“开年svip”与该用户近期贴文内容无直接关联，且其IP定位为重庆，而评论发布的帖子发布于北京，地点也不一致。\n\n综合判断：  \n否"
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
        
    for user_id, samples in tqdm(list(raw_data.items())[:1000]):
        if user_id not in results:
            results[user_id] = []
        for idx, text in tqdm(list(samples.items())[:4]):
            if user_id in results and str(idx) in [list(dd.keys())[0] for dd in results[user_id]]:
                continue
            elif text in ["是", "否"]:
                results[user_id].append({idx: text})
                continue
            elif len(text) > 5000:
                results[user_id].append({idx: "错误"})
                continue
            query = f"给出的文本内容为“{text}”"
            while True:
                chat_response = client.chat.completions.create(
                            model=settings.model,
                            messages=[
                                {"role": "system", "content": prompt},
                                {"role": "user", "content": query},
                            ],
                        )
                response = chat_response.choices[0].message.content
                print(f"{raw_data[user_id][idx]}\n{response}\n")
                answer = remove_think_tags(response)
                if answer in ["是", "否", "错误"]:
                    results[user_id].append({idx: answer})
                    break
            
        #print(json.dumps(results, ensure_ascii=False, indent=4))  
        with open(output_file, "w", encoding="utf8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)