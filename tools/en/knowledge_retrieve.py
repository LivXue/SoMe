import os
#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import json5
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from qwen_agent.tools.base import BaseTool, register_tool
from config import embedding_model_path, embedding_model_device, knowledge_path, knowledge_emb_path


@register_tool('knowledge_retrieve')
class RetrieveKnowledge(BaseTool):
    # `description` is used to tell the agent the function of the tool.
    description = "Service for social media knowledge retrieval. Enter the retrieval query and topk, retrieve the knowledge documents matching the query semantics from the knowledge base, and return the topk most relevant knowledge documents."
    # `parameters` tells the agent what input parameters the tool has.
    parameters = [{
        'name': 'query',
        'type': 'string',
        'description': 'retrieval query',
        'required': True
    },
                  {
        'name': 'topk',
        'type': 'int',
        'description': 'number of returned knowledge documents',
        'required': True
    }]
    # initialize the embedding model
    model = SentenceTransformer(embedding_model_path, 
                                device=embedding_model_device, 
                                model_kwargs={"torch_dtype": torch.bfloat16}, 
                                tokenizer_kwargs={"padding_side": "left"})
    documents = json.load(open(knowledge_path, encoding='utf-8'))
    document_embeddings = np.load(knowledge_emb_path)
    
    def call(self, params: str, **kwargs):
        params = json5.loads(params)
        query, topk = params['query'], params['topk']
        
        if topk <= 0:
            raise ValueError("The value of topk should be positive integer!")
        
        query_embedding = self.model.encode([query], prompt_name="query")
        
        # compute similarity and rank the documents
        similarity = self.model.similarity(query_embedding, self.document_embeddings)
        similarity = similarity.squeeze(0)  # remove the batch dimension
        ranked_indices = torch.argsort(similarity, descending=True)[:topk]
        
        # retrieve the top-k documents
        retrieved_posts = [self.documents[i] for i in ranked_indices]
        return self.show(retrieved_posts)
    
    @staticmethod
    def show(docs):
        result = "The following are results from the knowledge base:\n\n"
        
        for i, doc in enumerate(docs):
            result += f"Document {i+1}:\n***Content***: {doc["content"]}\n***Source URL***: {doc["link"]}\n\n"
            
        return result
