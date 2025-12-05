import json
import json5

from qwen_agent.tools.base import BaseTool, register_tool
from config import topic_key2entry


@register_tool('topic_search')
class SearchTopic(BaseTool):
    # `description` is used to tell the agent the function of the tool.
    description = "社交媒体话题搜索服务，输入搜索话题名称，将对应话题的贴文存放入文件夹'{topic_name}'。"
    # `parameters` tells the agent what input parameters the tool has.
    parameters = [{
        'name': 'topic_name',
        'type': 'string',
        'description': '贴文的关联话题名称',
        'required': True
    }]
    
    def call(self, params: str, data_folder, **kwargs):
        """
        Search posts based on topic
        :param topic_name: Topic name
        :return: Search results list
        """
        params = json5.loads(params)
        topic_name = params['topic_name']
        file_name = f"./database/topic_data/{topic_name}.json"

        # Filter posts that meet the conditions
        filtered_posts = []
        try:
            topic = json.load(open(file_name, 'r', encoding='utf-8'))
        except:
            return f"话题“{topic_name}”不存在，请检查话题名！"
        for post in topic['media_info'].values():
            succinct_post = {topic_key2entry[k]: post[k] for k in topic_key2entry}
            filtered_posts.append(succinct_post)

        data_folder.data_folders[f"{topic_name}"] = filtered_posts
        data_folder.show_funcs[f"{topic_name}"] = self.show
        
        return f"{len(filtered_posts)} 条符合条件的帖子已存储在数据文件夹 '{topic_name}' 中。可以通过工具'data_folder'调用。\n"
    
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
                result += f"***{key}***: {value}\n"
            result += "\n"

        return result.strip()