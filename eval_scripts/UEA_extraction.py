import json
import os
import re
import argparse

from tqdm import tqdm
from openai import OpenAI


args = argparse.ArgumentParser()
args.add_argument('--result_path', type=str, default='results/user_emotion_analysis', description='Path to the result files')
args.add_argument('--output_path', type=str, default='scores/user_emotion_analysis', description='Path to the output files')
args.add_argument('--setting_path', type=str, default='eval_scripts/settings.json', description='Path to the setting file')
args = args.parse_args()

settings = json.load(open(args.setting_path, 'r'))

def remove_think_tags(text):
    # 使用正则表达式匹配<think>到</think>之间的内容，并将其替换为空字符串
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()


prompt = '''
    请仔细阅读给出的文本内容，并从中提取出最终的判断结果，判断用户会对该帖子产生哪种情绪。结果只能输出一种情绪，不需要任何解释或额外内容。
    输出格式要求：
    - 如果对文本内容产生的情绪被判断为积极，则输出：积极
    - 如果对文本内容产生的情绪被判断为愤怒，则输出：愤怒
    - 如果对文本内容产生的情绪被判断为悲伤，则输出：悲伤
    - 如果对文本内容产生的情绪被判断为恐惧，则输出：恐惧
    - 如果对文本内容产生的情绪被判断为惊奇，则输出：惊奇
    - 如果对文本内容产生的情绪被判断为无情绪，则输出：无情绪
    - 如果对文本内容产生的情绪判断结果为不确定，或者没有判断结果，则输出：错误
    注意：请严格按照上述要求输出结果。
    示例输入：
    -  "根据提供的用户信息和帖子内容，可以看出该用户是鹿晗的粉丝，经常转发和评论关于鹿晗的帖子。用户的个人简介和IP定位显示其位于香港，可能对中国内地的娱乐新闻和明星动态感兴趣。\n\n结合帖子内容，用户似乎对鹿晗的演唱会和音乐作品非常感兴趣，多次转发和评论相关的帖子。此外，用户也对鹿晗的个人生活和动态感兴趣，例如其自拍和日常生活的帖子。\n\n因此，对于给定的帖子{\"内容\": \"周末见🔥🔥🔥 ​​​\", \"发布时间\": \"2024-11-19 23:24\", \"发布者\": \"VueChen_\", \"发布地点\": \"发布于 北京\", \"转发量\": 405411, \"评论量\": 44368, \"点赞量\": 44368}，用户可能会对其产生**积极**的情绪。因为该帖子的内容是关于周末的期待和祝福，可能会让用户感到开心和期待。同时，帖子的发布者VueChen_可能是用户感兴趣的博主或明星，进一步增加了用户对帖子的积极情绪。"
    示例输出：
    - "积极"
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
    for user_id, samples in tqdm(list(raw_data.items())):
        if user_id not in results:
            results[user_id] = []
        for i, (idx, text) in enumerate(tqdm(list(samples.items()))):
            if str(idx) in [list(dd.keys())[0] for dd in results[user_id]]:
                continue
            if text in ["积极", "愤怒", "悲伤", "恐惧", "惊奇", "无情绪"]:
                results[user_id].append({idx: text})
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
                if answer in ["积极", "愤怒", "悲伤", "恐惧", "惊奇", "无情绪", "错误"]:
                    results[user_id].append({idx: answer})
                    break
            
        with open(output_file, "w", encoding="utf8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)