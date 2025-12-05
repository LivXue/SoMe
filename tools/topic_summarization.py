import os
import re
#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json5
import numpy as np
from tqdm import tqdm

from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.agent import BasicAgent


def remove_think_tags(text):
    # Use regular expression to match content between <think> and </think> and replace it with empty string
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()


def jaccard_similarity(str1, str2):
    set1, set2 = set(str1), set(str2)
    return len(set1 & set2) / len(set1 | set2)

def find_most_similar(target, candidates):
    best_index, best_match = max(enumerate(candidates), key=lambda x: jaccard_similarity(target, x[1]))
    return best_index, best_match  # 返回索引和字符串

def get_summary_prompt(question):
    prompt = f'''你是事件分析专家，请根据提供的若干个新闻事件，总结出一个核心事件，输出这个核心事件的摘要，把该事件的发生时间提取出来，并且生成核心事件的新闻来源编号。你需要生成一个JSON。这个JSON的key是summary、time

其中，summary需要你根据新闻列表的内容判断是否有事件，如何有的话，输出核心事件，如果没有事件的话输出"无"。
   - summary禁止出现["引起"、"引发"、"关注"、"讨论"、"热议"]这些词汇
   - summary需要突出事件发生的人物、地点、以及事件经过
   - summary里面禁止出现年、月、日时间信息
   - summary需要包含事件发生的具体原因。
   - summary需要严格控制在100到200字，summary格式为：某人因为某原因在某地做了某事情。不得出现["引发"、"引发"、"反映"]这些词汇, 保证summary的长度在100到200字之间

time需要你从提供的若干个新闻事件中把核心事件的最早发生时间提取出来，时间尽量精确到"日"，如果得不到事件发生时间，输出"xxxx.xx.xx"
  - 格式为xxxx.xx.xx  分别代表"年", "月", "日"
  - 如果年月日任一部分缺乏，直接用x来补充, "年"用"xxxx", "月"用"xx"，"日"用"xx"，例如：3月7日，输出xxxx.03.07，2024年10月，输出2024.10.xx
  - 如果日期为农历日期，如正月十五，改为：xxxx.01.15, 腊月初九改为xxxx.12.09


生成结果示例：
示例1：{{"summary": "共和党候选人唐纳德·特朗普击败民主党候选人希拉里·克林顿，当选美国第45任总统。此次选举特朗普获得304张选举人票，超过胜选所需的270张，而希拉里获得227张选举人票。尽管希拉里在普选票数上领先，但因美国选举人团制度，特朗普最终胜出。选举期间，特朗普以“让美国再次伟大”为竞选口号，强调移民政策调整、经济改革及“美国优先”等议题, "time": "xxxx.xx.xx"}}
示例2：{{"summary": "北京市发布新的人才引进政策，旨在优化人才流动机制，吸引高层次专业人才来京发展。新政策涵盖户籍办理、居住证积分、住房保障等方面，为符合条件的国内外人才提供更加便捷的落户及工作环境。同时，政策对科技创新、重点产业、高精尖领域的人才给予特别支持，并加强对海外高层次人才的引进力度。北京市政府表示，此次调整旨在推动首都高质量发展，增强城市竞争力，为人才提供更优的职业发展环境", "time": "2025.05.15"}}
示例3：{{"summary": "美国总统唐纳德·特朗普签署总统备忘录，依据“301调查”结果，宣布对从中国进口的商品加征关税，涉及约600亿美元的中国商品。此举旨在应对美国对华贸易逆差，并指控中国存在不公平贸易行为。随后，美国贸易代表办公室（USTR）公布加税商品清单，涵盖高科技、机械、航空等多个领域。此后，中美双方展开多轮贸易磋商，并相继出台反制措施。", "time": "xxxx.xx.xx"}}
示例4：{{"summary": "京沪高速因道路施工导致部分路网故障，造成车辆通行受阻，不少司机反映行车缓慢，部分路段出现拥堵。相关部门已紧急派遣抢修人员进行修复，同时加强交通引导，确保道路尽快恢复畅通。针对部分受影响的路段，交管部门发布绕行建议，并提醒司机及时关注实时路况信息，合理规划出行路线。部分车主对此表示理解，但也有人认为施工安排应更加科学，以减少对交通的影响。目前，抢修工作仍在进行，预计将在短时间内完成修复。", "time": "xxxx.06.16"}}                                                                                                                                                                                                                   
示例5：{{"summary": "18岁青年李某近日被诱拐至缅甸，家属发现后立即报警求助。警方接到报警后，迅速展开了跨国营救行动。经过多方努力，相关部门成功定位到李某的具体位置，并在当地警方协助下，成功将其解救。在营救过程中，李某未受重伤，已被安全护送回国。家属对警方的迅速反应和全力营救表示感谢。专家呼吁加强对青少年的安全教育和跨境犯罪的防范工作。", "time": "xxxx.08.13"}}

以下是事件：
{question}

要求:
    请严格输出JSON格式
    summary禁止出现["引起"、"引发"、"关注"、"讨论"、"热议"]这些词汇
'''
    return prompt

def get_title_prompt(question):

    prompt = f'''我将给你提供一段摘要。你需要根据这个摘要生成一个JSON。这个JSON的key是title，location。
title需要你根据摘要内容总结出一个标题出来，30个字左右, 禁止出现["引起"、"引发"、"反映"、"公众关注"、"讨论"、"热议"]这些词汇。如果摘要里面有事件发生的地点的话，title也需要包含地点。 
   - title客观陈述事件内容，需要反映摘要的完整核心内容，不能改变原始意思，突出
        1. 地点：指出事件发生的地点。
        2. 人物或主体：涉及事件的主要人物、机构或相关主体。
        3. 内容：概括事件的核心内容，禁止出现"引起xx"、"引发xx"等反映事件影响的内容
        4. 原因或目的: 事件发生的原因或目的
   - title里面不需要时间。title格式为: 某再某地人因为某原因做了某事情。
        title示例1：热心市民在社区公园救助受伤流浪猫，联系救助机构为流浪猫提供干净住所
        title示例2: 浙江嘉兴一大学被曝禁止学生拿外卖进校园，校方回应称措施为保障食品安全

location需要你提取摘要里面的地点，包括国家、城市、地标、地区、街道、社区等。如果没有地点的话，输出"未提到任何地点"，如果摘要里面的地点不在中国的话，输出"国际", location示例："北京"、"上海"、"嘉兴"

输出示例:
输入：2025年1月17日，王毅外长访问非洲多国，强调中非合作及绿色发展
正确输出: {{"title": "王毅外长访问非洲多国强调中非合作及绿色发展", "location": "国际"}}
错误输出: {{"title": "王毅外长访问非洲多国强调中非合作及绿色发展，反映大国外交", "location": "国际"}}

输入：北京等地出现违法强拆行为，损害百姓权益，政府警告将会严惩此类危害人民群众财产的行为
正确输出: {{"title": "北京等地出现违法强拆行为损害百姓权益, 政府警告将严惩此类行为", "location": "北京"}}
错误输出: {{"title": "北京等地出现违法强拆行为损害百姓权益, 政府警告将严惩此类行为，引发公众热议", "location": "北京"}}

输入：一辆拉粪三轮车横穿逆行导致交通事故，交警提醒遵守交通规则
正确输出: {{"title": "拉粪三轮车横穿逆行导致交通事故, 交警提醒遵守交通规则", "location": "未提到任何地点"}}
错误输出: {{"title": "拉粪三轮车横穿逆行导致交通事故, 社会公众对其态度表示否定，交警提醒遵守交通规则", "location": "未提到任何地点"}}

输入: 3月4日,公安部网安局通报，多名网民因捏造并传播关于80后死亡率的虚假信息被查处。
正确输出: {{"title": "9人因捏造“80后死亡率”谣言被处罚", "location": "未提到任何地点"}}
错误输出: {{"title": "9人因捏造“80后死亡率”谣言被处罚，引发公众对网络造谣危害的思考", "location": "未提到任何地点"}}

要求：你需要严格输出JSON格式
      title禁止出现["引起"、"引发"、"反映"、"公众关注"、"讨论"、"热议]"这些词汇

输入摘要：{question}

 '''
    return prompt

class SummarizeAgent(object):
    def __init__(self,
                 llm
                 ):
        self.zh_prompt = """ 对于获取到的用户信息和贴文，你根据提供的若干个社交媒体贴文，总结出一个核心舆情事件，输出这个核心舆情事件的摘要，把该事件的发生时间、地点和经过提取出来。注意：
- 如果文本内容属于无关的个人言论、无意义的信息、广告和推广信息、短剧或者综艺节目、科普内容等不属于舆情事件的内容，则跳过该簇的信息，输出：无(一个字)
- 将有意义的簇的信息进行输出，只要对簇形成一个总体的内容分析，不用每个帖子详细分析,一句话输出，不要把推理过程写出来
你总是用中文回复用户"""
        self.agent = BasicAgent(system_message=self.zh_prompt, 
                       llm=llm)
        self.location = None

    def summarize(self, cluster):
        results = []
        for key, value in tqdm(cluster.items()):
            try:
                result = self.process_cluster((key, value), self.location)
                if result:
                    results.append(result)
            except:
                continue

        return results
    
    def process_cluster(self, cluster_data, location):
        key, value = cluster_data
        content_list = []
        content_ocr_list = []
        for item in value:
            if not item['like_count']:
                item['like_count'] = 0

            if item['like_count'] >= 0 and len(item['content']) > 10:
                content = item['content']
                content_ocr = item['ocr']
                content_list.append(content)
                content_ocr_list.append(content_ocr)


        top_k_content_list = content_list[:10]
        if len(top_k_content_list) == 0:
            return None 
        top_k_content_num_list = ['新闻' + str(num) + ': ' + value for num, value in enumerate(top_k_content_list)]

        # 生成 summary

        summary_prompt = get_summary_prompt('\n'.join(top_k_content_num_list))
        messages= [{'role': 'user', 'content': summary_prompt}]
        summary_res = remove_think_tags(self.agent._run(messages)[0]['content']).replace('\n','')
        # summary_res = self.get_model_res(summary_prompt, self.summary_description_json_schema)
        if summary_res.startswith('```json'):
            summary_res = summary_res[7:-3]
        summary_res = json5.loads(summary_res)
        summary = summary_res['summary']
        #无法总结出簇的摘要，在kafka里面暂时放弃这个簇
        if summary == '无':
            return None

        time = summary_res['time']

        # 根据 summary，生成标签 keyword、title、label
        summary_temp = summary.replace('引起', '').replace('引发', '').replace('反映', '').replace('关注', '').replace('热议', '')
        title_info_prompt = get_title_prompt(summary_temp)
        messages= [{'role': 'user', 'content': title_info_prompt}]
        title_info = remove_think_tags(self.agent._run(messages)[0]['content']).replace('\n','')
        title_info = json5.loads(title_info)
        title = title_info['title']
        location = title_info['location']

        return {
            '标题': title,
            '摘要': summary,
            '地点': location,
            '时间': time,
        }


@register_tool('topic_summarization')
class TopicSummarization(BaseTool):
    def __init__(self, llm):
        super().__init__()
        # `description` is used to tell the agent the function of the tool.
        self.description = "社交媒体聚类摘要服务，输入存放聚类结果的文件夹，总结每个聚类的主要话题内容"
        # `parameters` tells the agent what input parameters the tool has.
        self.parameters = [{
            'name': 'folder_name',
            'type': 'string',
            'description': '文件夹名称',
            'required': True
        }]
        
        self.agent = SummarizeAgent(llm=llm)
        self.data_cluster = {}
        self.data_vector = {}
    
    def call(self, params: str, data_folder, **kwargs):
        params_json = json5.loads(params)
        folder_name = params_json['folder_name']
        location, start_time, end_time = folder_name.split("_")[:3]
        self.agent.location = location
        cluster = data_folder.data_folders[folder_name]
        cluster_key = data_folder.data_folders[f"{folder_name}_key"]
        clusters = {}
        for k, v in zip(cluster_key, cluster):
            clusters[k] = v
            
        results = self.agent.summarize(clusters)
        
        #trans = {'title': '标题', 'summary': '摘要', 'location': '地点', 'time': '时间'}
        data_folder.data_folders[f"{location}_{start_time}_{end_time}_summarization"] = results#[{k: res[k] for k in trans} for res in results]
        data_folder.show_funcs[f"{location}_{start_time}_{end_time}_summarization"] = self.show
        
        return f"位于{location}的从{start_time}到{end_time}的{len(results)}个事件摘要已存放在文件夹'{location}_{start_time}_{end_time}_summarization'中。可以通过工具'data_folder'调用。"
    
    @staticmethod
    def show(posts: list, start_idx, end_idx) -> str:
        """
        Display posts within the specified range
        :param posts: Posts list
        :param start_idx: Start index
        :param end_idx: End index
        :return: Post content within the specified range
        """
        if not posts:
            return "没有找到符合条件的话题。"
        posts = posts[start_idx:end_idx]
        
        if start_idx < 0:
            raise ValueError("起始索引不能小于0。")
        elif end_idx > len(posts):
            raise ValueError(f"结束索引不能超过帖子列表的长度{len(posts)}。")
        elif start_idx >= end_idx:
            raise ValueError("起始索引必须小于结束索引。")

        result = ""
        for i, post in enumerate(posts):
            result += f"事件 {i + 1}:\n" 
            for key, value in post.items():
                result += f"***{key}***: {value}\n"
            result += "\n"
            
        return result
        