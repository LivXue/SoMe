import json
import json5

from qwen_agent.tools.base import BaseTool, register_tool
from config import weibo_key2entry


@register_tool('user_search')
class SearchUser(BaseTool):
    # `description` is used to tell the agent the function of the tool.
    description = "社交媒体用户信息搜索服务，输入用户ID，可以查询到该ID用户的个人信息和发表的贴文。用户发表的贴文存放入文件夹'{uid}_weibo'。"
    # `parameters` tells the agent what input parameters the tool has.
    parameters = [{
        'name': 'uid',
        'type': 'string',
        'description': '用户的ID',
        'required': True
    }]
    
    def call(self, params: str, data_folder, **kwargs):
        """
        Search user information and Weibo posts published by the user based on user ID
        :param uid: User id
        :return: Search results list
        """
        params = json5.loads(params)
        uid = params['uid']
        file_name = f"./database/user_data/{uid}/{uid}.json"

        # Find and filter user information and Weibo posts based on file_name
        user_weibo = []
        try:
            use_data = json.load(open(file_name, encoding='utf-8'))
        except FileNotFoundError:
            return f"未找到用户ID为 {uid} 的数据，请检查用户ID是否正确。\n"
        
        description = f"该用户的信息包括：\n用户ID：{use_data["user"]["id"]}\n用户名：{use_data["user"]["name"]}\n关注数：{use_data["user"]["followers_count"]}\n粉丝数：{use_data["user"]["friends_count"]}\n转赞评数：{use_data["user"]["total_rcl_count"]}\n认证信息：{use_data["user"]["verified_reason"]}\n个人简介：{use_data["user"]["description"]}\nIP定位：{use_data["user"]["location"]}\n"

        for weibo in use_data["weibo"]:
            weibo_post = {weibo_key2entry[k]: weibo[k] for k in weibo_key2entry}
            user_weibo.append(weibo_post)
        
        data_folder.data_folders[f"{uid}_weibo"] = user_weibo
        data_folder.show_funcs[f"{uid}_weibo"] = SearchUser.show
        
        return f"用户发表的 {len(user_weibo)} 条微博已存储在数据文件夹 '{uid}_weibo' 中。可以通过工具'data_folder'调用。\n{description}"
    
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
                if key in ["id", "pic_num", "pictures", "video", "rca_list"]:
                    continue
                result += f"***{key}***: {value}\n"
            result += "\n"

        return result.strip()
    
                    
        
    


