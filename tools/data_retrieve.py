import os
#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json5
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from qwen_agent.tools.base import BaseTool, register_tool
from config import embedding_model_path, embedding_model_device


@register_tool('post_retrieve')
class RetrievePosts(BaseTool):
    # `description` is used to tell the agent the function of the tool.
    description = "社交媒体贴文检索服务，输入检索查询、数据文件夹名称和topk，从对应文件夹中检索匹配查询语义的贴文，并返回topk个最相关的贴文。"
    # `parameters` tells the agent what input parameters the tool has.
    parameters = [{
        'name': 'query',
        'type': 'string',
        'description': '检索查询',
        'required': True
    },
                  {
        'name': 'folder_name',
        'type': 'string',
        'description': '需要查询的数据文件夹名称',
        'required': True
    },
                  {
        'name': 'topk',
        'type': 'int',
        'description': '需要返回的相关贴文个数',
        'required': True
    }]
    # initialize the embedding model
    model = SentenceTransformer(embedding_model_path, 
                                device=embedding_model_device, 
                                model_kwargs={"torch_dtype": torch.bfloat16}, 
                                tokenizer_kwargs={"padding_side": "left"})
    emb_base = np.load("./database/emb_data/topic_data.npy", allow_pickle=True).item()
    
    def call(self, params: str, data_folder, **kwargs):
        params = json5.loads(params)
        query, folder_name, topk = params['query'], params['folder_name'], params['topk']
        if folder_name not in data_folder.data_folders:
            raise ValueError(f"文件夹'{folder_name}'不存在。请先创建文件夹或检查名称是否正确。")
        
        query_embedding = self.model.encode([query], prompt_name="query")
        docoments = data_folder.data_folders[folder_name]
        
        # load the document embeddings
        embeddings = self.emb_base
        if folder_name not in embeddings:
            raise ValueError(f"话题'{folder_name}'的数据不存在。")
        document_embeddings = np.stack([emb for emb in embeddings[folder_name].values()], axis=0)
        
        # compute similarity and rank the documents
        similarity = self.model.similarity(query_embedding, document_embeddings)
        similarity = similarity.squeeze(0)  # remove the batch dimension
        ranked_indices = torch.argsort(similarity, descending=True)[:topk]
        
        # retrieve the top-k documents
        retrieved_posts = [docoments[i] for i in ranked_indices]
        return self.show(retrieved_posts, 0, topk)

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