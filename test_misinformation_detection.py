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
import tasks.misinformation_detection as task
from agent import SocialMediaAgent
from tools.en import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Meta-Llama-3.1-8B-Instruct", description="The base model for the agent")
args.add_argument('--base_url', type=str, default="http://0.0.0.0:8007/v1", description="The base url for the model server")
args.add_argument('--api_key', type=str, default="mysecrettoken123", description="The api key for the model server")
args.add_argument('--output_path', type=str, default="results/misinformation_detection", description="The output path for the results")
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
    'model_server': args.base_url, #'https://api.deepseek.com', #'https://api.moonshot.cn/v1',
    'api_key': args.api_key,
    'model': args.model,
}
bot = SocialMediaAgent(system_message=task.en_prompt, 
                       function_list=['knowledge_retrieve'],
                       llm=vllm_config)
twitters = json.load(open("datasets/misinformation_detection/ground_truth.json", encoding="utf-8"))
for tid in tqdm(twitters):
    if tid in results:
        continue
    messages = []  # Store chat history here
    query = f"Please determine if the twitter '{twitters[tid]['claim']}' is true."
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
            for response in bot.run(messages=messages, lang='en'):
                response_plain_text = typewriter_print(response, response_plain_text)
            # Add bot's response to chat history
            messages.extend(response)
            print('\n')
            result = remove_think_tags(messages[-1]['content'])
            results[tid] = result
            with open(output_file, "w", encoding='utf8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            break
        except ModelServiceError:
            sleep(60)
            continue
        except:
            messages = []  # Store chat history here
            query = f"Please determine if the twitter '{twitters[tid]['claim']}' is true."
            print(f'{Fore.RED + Style.BRIGHT}用户请求:{Style.RESET_ALL} {query}')
            messages.append({'role': 'user', 'content': query})
            response = []
            response_plain_text = ''
            print(f'{Fore.CYAN + Style.BRIGHT}智能体回应:{Style.RESET_ALL}')
            continue
    if tid not in results:
        results[tid] = ""
