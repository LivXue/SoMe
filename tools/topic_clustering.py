from numba import jit
import numpy as np
import time
import os
#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json5
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from qwen_agent.tools.base import BaseTool, register_tool
from config import embedding_model_path, embedding_model_device



@jit(nopython=True)
def vector_similarity_1(avg_vector, vector):
    oneSimilarity = avg_vector @ vector.T

    return oneSimilarity


class Single_Pass_Cluster(object):
    def __init__(self,
                 theta=0.5
                 ):

        self.bge_model = SentenceTransformer(embedding_model_path, 
                                device=embedding_model_device, 
                                model_kwargs={"torch_dtype": torch.bfloat16}, 
                                tokenizer_kwargs={"padding_side": "left"})
        self.theta = theta


    def get_bge_vector_representation(self, word_segmentation, model):
        embeddings = model.encode(word_segmentation, normalize_embeddings=True)
        return embeddings


    def getBGEMaxSimilarity(self, dictTopic, vector):
        # Calculate the text similarity between the newly entered document and existing documents. Here, cosine similarity is used.

        maxValue = 0
        maxIndex = -1
        # print(dictTopic)
        for k, cluster in dictTopic.items():
            avgVector = cluster['avgVector']
            # avgVector = np.mean(cluster, axis=0)
            # vector = vector.astype(avgVector.dtype)
            oneSimilarity = vector_similarity_1(avgVector, vector)

            if oneSimilarity > maxValue:
                maxValue = oneSimilarity
                maxIndex = k
        return maxIndex, maxValue

    def single_pass(self, corpus, data_title, data_content, data_ocr, theta, unique_id, like_count, dictTopic=None, clusterTopic=None, start_cluster_id=1):
        if dictTopic == None:
            dictTopic = {}
        if clusterTopic == None:
            clusterTopic = {}
            numTopic = start_cluster_id
        else:
            # int_numbers = [int(num) for num in str_numbers]
            numTopic = max(int(num) for num in list(dictTopic.keys())) + 1
        cnt = 1
        for vector, source_title, source_content, source_ocr, unique_id, like_count in zip(corpus, data_title, data_content, data_ocr, unique_id, like_count):
            if numTopic == start_cluster_id:
                dictTopic[numTopic] = {}
                dictTopic[numTopic]['vectors'] = []
                dictTopic[numTopic]['vectors'].append(vector)
                dictTopic[numTopic]['avgVector'] = vector

                clusterTopic[numTopic] = []
                clusterTopic[numTopic].extend([{'unique_id': unique_id, 'like_count': like_count, 'title': source_title, 'content': source_content, 'ocr': source_ocr}])
                
                lastTopic = {'unique_id': unique_id,'cluster_id': numTopic}
                numTopic += 1
            else:
                maxIndex, maxValue = self.getBGEMaxSimilarity(dictTopic, vector)
                # 以第一篇文档为种子，建立一个主题，将给定语句分配到现有的、最相似的主题中
                if maxValue > theta:
                    dictTopic[maxIndex]['vectors'].append(vector)
                    dictTopic[maxIndex]['avgVector'] = np.mean(dictTopic[maxIndex]['vectors'], axis=0)
                    clusterTopic[maxIndex].extend([{'unique_id': unique_id, 'like_count': like_count,  'title': source_title, 'content': source_content, 'ocr': source_ocr}])
                    lastTopic = {'unique_id': unique_id, 'cluster_id': maxIndex}

                # 或者创建一个新的主题
                else:
                    dictTopic[numTopic] = {}
                    dictTopic[numTopic]['vectors'] = []
                    dictTopic[numTopic]['vectors'].append(vector)
                    dictTopic[numTopic]['avgVector'] = vector

                    clusterTopic[numTopic] = []
                    clusterTopic[numTopic].extend([{'unique_id': unique_id, 'like_count': like_count, 'title': source_title, 'content': source_content, 'ocr': source_ocr}])
                    lastTopic = {'unique_id': unique_id, 'cluster_id': maxIndex}
                    numTopic += 1
            cnt += 1

        return dictTopic, clusterTopic, lastTopic

    def fit_transform(self, theta=0.5, dictT=None, clusterT=None, query=None, query_emb=None, start_cluster_id=1):

        # Synthesize the above functions to get the final clustering results: including cluster labels, number of each cluster, key topic words and key sentences
        start_time1 = time.time()
        data_title = [query['标题']]
        data_content = [query['内容']]
        data_ocr = [query['OCR']]
        # 内容 OCR unique_id
        # datMat = [query['title']]
        # datMat_ori = [query['content']]
        # datMat_rewrite = [query['ocr']]
        if 'like_count' in query.keys():
            like_count = [query['like_count']]
        else:
            like_count = [0]
        # 
        unique_id = [query['unique_id']]


        # 得到文本数据的空间向量表示
        corpus_bge = query_emb#self.get_bge_vector_representation(data_content, self.bge_model)
        dictTopic, clusterTopic,lastTopic = self.single_pass(corpus_bge, data_title, data_content, data_ocr, theta, unique_id, like_count, dictT, clusterT, start_cluster_id)
        # 按聚类语句数量对聚类结果进行降序排列，找到重要的聚类群
        return dictTopic, clusterTopic,lastTopic
    
    
@register_tool('topic_clustering')
class TopicClustering(BaseTool):
    # `description` is used to tell the agent the function of the tool.
    description = "社交媒体话题聚类服务，输入文件夹名称，将对应文件夹中的贴文进行聚类。"
    # `parameters` tells the agent what input parameters the tool has.
    parameters = [{
        'name': 'folder_name',
        'type': 'string',
        'description': '文件夹名称',
        'required': True
    }]
    
    single_pass_cluster = Single_Pass_Cluster()
    data_cluster = {}
    data_vector = {}
    
    def call(self, params: str, data_folder, **kwargs):
        params_json = json5.loads(params)
        folder_name = params_json['folder_name']
        location, start_time, end_time = folder_name.split("_")
        posts = data_folder.data_folders[folder_name] 
        posts_idx = data_folder.data_folders[f"{folder_name}_idx"]
        date = start_time.split(" ")[0]
        posts_emb = np.load(f"./database/emb_data/{date}_{location}.npy", allow_pickle=True).item()
        dictT = None
        clusterT = None
        for item_idx, item in zip(posts_idx, posts):
            #start_2 = time.time()
            dictT, clusterT, lastTopic = self.single_pass_cluster.fit_transform(theta=0.5,dictT=dictT, clusterT=clusterT, query=item, query_emb=np.array(posts_emb[item_idx])[None, :])
        
        data_folder.data_folders[f"{location}_{start_time}_{end_time}_cluster"] = [clusterT[k] for k in clusterT.keys()]
        data_folder.data_folders[f"{location}_{start_time}_{end_time}_cluster_key"] = [k for k in clusterT.keys()]
        data_folder.show_funcs[f"{location}_{start_time}_{end_time}_cluster"] = self.show
        
        return f"文件夹'{folder_name}'中的帖子已经完成聚类，聚类结果存放在文件夹'{location}_{start_time}_{end_time}_cluster'中。可以通过工具'data_folder'调用。"
    
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
            return "没有找到符合条件的帖子。"
        posts = posts[start_idx]
        start_idx, end_idx = 0, len(posts)
        end_idx = min(end_idx, 5)
        
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
                if key in ["id", "source", "pic_num", "pictures", "video", "rca_list"]:
                    continue
                result += f"***{key}***: {value}\n"
            result += "\n"
            
        return result
        