import json
import json5
from datetime import datetime

from qwen_agent.tools.base import BaseTool, register_tool
from config import post_key2entry


@register_tool('post_search')
class SearchPosts(BaseTool):
    # `description` is used to tell the agent the function of the tool.
    description = "社交媒体贴文搜索服务，输入搜索地点、开始时间、结束时间，将对应地点发表于开始时间和结束时间之间的贴文存放入文件夹'{location}_{start_time}_{end_time}'。"
    # `parameters` tells the agent what input parameters the tool has.
    parameters = [{
        'name': 'location',
        'type': 'string',
        'description': '贴文的关联地点',
        'required': True
    },
                  {
        'name': 'start_time',
        'type': 'string',
        'description': '贴文发表时段的开始时刻，格式为%Y-%m-%d %H:%M:%S',
        'required': True
    },
                  {
        'name': 'end_time',
        'type': 'string',
        'description': '贴文发表时段的结束时刻，格式为%Y-%m-%d %H:%M:%S',
        'required': True
    }]

    def call(self, params: str, data_folder, **kwargs):
        """
        Search posts based on location, start time and end time
        :param location: Location
        :param start_time: Start time %Y-%m-%d %H:%M:%S
        :param end_time: End time %Y-%m-%d %H:%M:%S
        :return: Search results list
        """
        params = json5.loads(params)
        location, start_time, end_time = params['location'], params['start_time'], params['end_time']
        date_form = "%Y-%m-%d %H:%M:%S"
        file_name = f"./database/raw_data/{start_time.split(' ')[0]}_{location}_sensitive.jsonl"

        # Filter posts that meet the conditions
        filtered_posts = []
        filtered_posts_idx = []
        for line in open(file_name, 'r', encoding='utf-8'):
            post = json.loads(line.strip())
            if datetime.strptime(start_time, date_form) <= datetime.strptime(post["post_publish_time"], date_form) < datetime.strptime(end_time, date_form):
                succinct_post = {post_key2entry[k]: post[k] for k in post_key2entry}
                filtered_posts.append(succinct_post)
                filtered_posts_idx.append(succinct_post['unique_id'])

        data_folder.data_folders[f"{location}_{start_time}_{end_time}"] = filtered_posts
        data_folder.data_folders[f"{location}_{start_time}_{end_time}_idx"] = filtered_posts_idx
        data_folder.show_funcs[f"{location}_{start_time}_{end_time}"] = SearchPosts.show
        
        return f"{len(filtered_posts)} 条符合条件的帖子已存储在数据文件夹 '{location}_{start_time}_{end_time}' 中。可以通过工具'data_folder'调用。\n"
    
    @staticmethod
    def show(posts: list, start_idx: int, end_idx: int) -> str:
        """
        Display posts within the specified range
        :param posts: Posts list
        :param start_idx: Start index
        :param end_idx: End index
        :return: Post content within the specified range
        """
        if not posts:
            return "没有找到符合条件的帖子。"
        
        if start_idx < 0:
            raise ValueError("起始索引不能小于0。")
        elif end_idx > len(posts):
            raise ValueError(f"结束索引不能超过帖子列表的长度{len(posts)}。")
        elif start_idx >= end_idx:
            raise ValueError("起始索引必须小于结束索引。")
         
        result = ""
        for i in range(start_idx, end_idx):
            post = posts[i]
            result += f"帖子 {i + 1}:\n"
            for key, value in post.items():
                if key == "unique_id":
                    continue
                result += f"***{key}***: {value[:512]}\n"
            result += "\n"

        return result.strip()