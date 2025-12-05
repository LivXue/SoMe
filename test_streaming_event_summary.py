import json
import os
import re
from time import sleep
from datetime import datetime, time, timezone
import argparse

from colorama import *
from tqdm import tqdm

from qwen_agent.utils.output_beautify import typewriter_print
from qwen_agent.llm.base import ModelServiceError
from tasks import streaming_event_summary as task
from agent import SocialMediaAgent
from tools import *


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen3-32B", description="The base model for the agent")
parser.add_argument('--base_url', type=str, default="http://0.0.0.0:8007/v1", description="The base url for the model server")
parser.add_argument('--api_key', type=str, default="mysecrettoken123", description="The api key for the model server")
parser.add_argument('--output_path', type=str, default="results/streaming_event_summary", description="The output path for the results")
args = parser.parse_args()

DEEPSEEK_OFF_PEAK = 'deepseek' in args.model

def remove_think_tags(text):
    # Use regex to match content between <think> and </think> tags and replace with empty string
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

output_file = os.path.join(args.output_path, f"{args.model.split('/')[-1]}.json")
if os.path.exists(output_file):
    results = json.load(open(output_file))
else:
    results = {}
    
vllm_config = {
    'model_server': args.base_url,#'https://generativelanguage.googleapis.com/v1beta/openai/', #'https://api.deepseek.com', #'https://api.siliconflow.cn/v1',
    'api_key': args.api_key,
    'model': args.model,
}
bot = SocialMediaAgent(system_message=task.zh_prompt, 
                       function_list=[DataFolder(), SearchTopic()],
                       llm=vllm_config)
topics = json.load(open("datasets/streaming_event_summary/ground_truth.json", encoding="utf-8")).keys()
for topic in tqdm(topics):
    if topic in results:
        continue
    bot.function_map["data_folder"].initialize()
    messages = []  # 这里储存聊天历史。
    query = f"请给出话题“{topic}”的事件发生经过和详细时间、地点、情节。"
    print(f'{Fore.RED + Style.BRIGHT}用户请求:{Style.RESET_ALL} {query}')
    messages.append({'role': 'user', 'content': query})
    response = []
    response_plain_text = ''
    print(f'{Fore.CYAN + Style.BRIGHT}智能体回应:{Style.RESET_ALL}')
    for retry in range(5):
        if DEEPSEEK_OFF_PEAK:
            while True:
                # Check if UTC time in 16:30~00:30
                current_utc_time = datetime.now(timezone.utc).time()
                start_time = time(16, 30)
                end_time = time(0, 30)
                is_between = False
                if start_time <= end_time:
                    is_between = start_time <= current_utc_time <= end_time
                else:
                    is_between = current_utc_time >= start_time or current_utc_time <= end_time
                if not is_between:
                    print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Waiting for Deepseek off-peak time")
                    sleep(10)
                else:
                    break
        try:
            for response in bot.run(messages=messages):
                response_plain_text = typewriter_print(response, response_plain_text)
            # Add bot's response to chat history
            messages.extend(response)
            print('\n')
            result = remove_think_tags(messages[-1]['content'])
            results[topic] = result
            with open(output_file, "w", encoding='utf8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            break
        except ModelServiceError:
            sleep(60)
            continue
        except:
            bot.function_map["data_folder"].initialize()
            messages = []  # Store chat history here
            query = f"请给出话题“{topic}”的事件发生经过和详细时间、地点、情节。"
            print(f'{Fore.RED + Style.BRIGHT}用户请求:{Style.RESET_ALL} {query}')
            messages.append({'role': 'user', 'content': query})
            response = []
            response_plain_text = ''
            print(f'{Fore.CYAN + Style.BRIGHT}智能体回应:{Style.RESET_ALL}')
            continue
    if topic not in results:
        results[topic] = ""
