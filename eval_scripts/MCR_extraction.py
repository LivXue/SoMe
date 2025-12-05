import json
import os
import re
import argparse

from tqdm import tqdm
from openai import OpenAI


args = argparse.ArgumentParser()
args.add_argument('--result_path', type=str, default='results/media_content_recommend', description='Path to the result files')
args.add_argument('--output_path', type=str, default='scores/media_content_recommend', description='Path to the output files')
args.add_argument('--setting_path', type=str, default='eval_scripts/settings.json', description='Path to the setting file')
args = args.parse_args()

settings = json.load(open(args.setting_path, 'r'))

def remove_think_tags(text):
    # 使用正则表达式匹配<think>到</think>之间的内容，并将其替换为空字符串
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()


prompt = '''
    请仔细阅读给出的文本内容，并从中提取出最终的判断结果，判断该用户是否可能对该贴文感兴趣。结果只能输出 “是” 或 “否”，不需要任何解释或额外内容。
    输出格式要求：
    - 如果文本内容的判断结果为肯定，则输出：是
    - 如果文本内容的判断结果为否定，则输出：否
    - 如果文本内容的判断结果为不确定，或者没有判断结果，则输出：错误
    注意：请严格按照上述要求输出结果。
    示例输入：
    - "根据用户搜索结果中的信息，7391946667用户发布的内容主要是关于易烊千玺的演唱会、电影以及日常的个人感想或生活动态。而从具体的贴文中可以提取出该用户可能感兴趣的领域或话题包括：演唱会、电影、偶像崇拜、日常个人生活等。\n\n再看用户发的5条内容中，只有第五条提到了#易烊千玺2025演唱会礐峃#，期待演唱会#易烊千玺演唱会#，这表示该用户对易烊千玺的演唱会表现出明显的兴趣。\n\n再 compares the user's interests with the given post content: \"怀着敬畏与好奇，再次走进“中国天眼”，星空会告诉我们坚持探索的答案。致敬科技工作者，致敬科学精神，#科学闪耀中国#\"，该贴文主要内容是关于“中国天眼”和科技工作者，主题和用户兴趣领域不匹配。\n\n最终判断，用户可能对这条贴文不感兴趣。"
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
    for user_id, samples in tqdm(raw_data.items()):
        if user_id not in results:
            results[user_id] = []
        for idx, text in tqdm(samples.items()):
            if user_id in results and str(idx) in [list(dd.keys())[0] for dd in results[user_id]]:
                continue
            if text in ["是", "否"]:
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
            