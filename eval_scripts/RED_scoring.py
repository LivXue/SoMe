import json
import os
import re
import argparse

from tqdm import tqdm
from openai import OpenAI


args = argparse.ArgumentParser()
args.add_argument('--result_path', type=str, default='results/realtime_event_detection', description='Path to the result files')
args.add_argument('--output_path', type=str, default='scores/realtime_event_detection', description='Path to the output files')
args.add_argument('--setting_path', type=str, default='eval_scripts/settings.json', description='Path to the setting file')
args = args.parse_args()

settings = json.load(open(args.setting_path, 'r'))

def remove_think_tags(text):
    # 使用正则表达式匹配<think>到</think>之间的内容，并将其替换为空字符串
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()


prompt = '''
    请仔细阅读给出实时事件的参考答案<place_holder>，并根据参考答案对给出的实时事件内容进行总体评分，评分范围为 0 到 5 分，评分维度包括：
    - 准确性（Accuracy）：内容中的信息是否准确无误？时间、地点、人物和事件等关键细节是否与提供的参考答案一致；
    - 完整性（Completeness）：内容是否涵盖了参考答案中提到的所有重要方面？包括但不限于标题、摘要、时间、地点等；
    - 相关性（Relevance）：是否与参考答案中的主题和内容相关，未引入无关信息；
    请从三个评分维度为给出的实时事件内容进行总体打分，并输出综合评判后的评分，
    准确性评分标准定义如下：
    - 5 分：完全符合参考答案内容，没有错误或遗漏；
    - 4 分：基本符合参考答案内容，存在少量轻微错误或遗漏；
    - 3 分：部分符合参考答案内容，存在一些错误或遗漏，但不影响整体理解；
    - 2 分：少部分符合参考答案内容，存在较多错误或遗漏，影响了部分理解；
    - 1 分：勉强符合参考答案内容，存在严重错误或大量遗漏；
    - 0 分：完全不符合参考答案内容，信息错误或空白；
    完整性评分标准定义如下：
    - 5 分：完全符合参考答案内容，涵盖了所有关键时间点、地点和重要事件；
    - 4 分：基本符合参考答案内容，遗漏了少量次要信息；
    - 3 分：部分符合参考答案内容，遗漏了一些重要事件或时间点或地点等关键信息；
    - 2 分：少部分符合参考答案内容，遗漏较多关键信息；
    - 1 分：提及的信息非常有限，几乎未涵盖参考答案中的关键信息；
    - 0 分：完全未涵盖参考答案内容，信息缺失严重或空白；
    相关性评分标准定义如下：
    - 5 分：内容高度相关，聚焦于参考答案中提到的重要细节；
    - 4 分：基本相关，主体与参考答案中提到内容相关，但包含少量无关信息；
    - 3 分：部分相关，主题有一定关联，但引入较多无关内容；
    - 2 分：相关性较弱，大部分内容偏离参考答案中的内容；
    - 1 分：几乎不相关，仅有个别词汇或概念与参考答案沾边；
    - 0 分：完全不相关，回答内容与参考答案给出的内容无任何关联；
    请为每个维度分别打分。
    输出格式要求：
    - 如果三个维度的评分为：准确性：2，完整性：2，相关性：2，则输出：准确性：2，完整性：2，相关性：2
    注意：请严格按照上述要求输出结果，不要添加任何解释、说明或多余内容。
'''
client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
        )
ground_truth = json.load(open("datasets/realtime_event_detection/ground_truth.json"))
pattern = r"准确性：\d+，完整性：\d+，相关性：\d+"


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
        current_prompt = prompt.replace("<place_holder>", json.dumps(ground_truth[topic], ensure_ascii=False))
        # llama may trap in infinite loop
        if len(summary) > 5000:
            results[topic]= "准确性：0，完整性：0，相关性：0"
            continue
        query = f"给出的事件摘要内容为“{summary}”"
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

