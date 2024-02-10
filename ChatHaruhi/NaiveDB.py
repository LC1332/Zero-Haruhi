import random
import string
import os
from math import sqrt

class NaiveDB:
    def __init__(self):
        self.verbose = False
        self.init_db()

    def init_db(self):
        if self.verbose:
            print("call init_db")
        self.stories = []
        self.norms = []
        self.vecs = []
        self.flags = [] # 用于标记每个story是否可以被搜索
        self.metas = [] # 用于存储每个story的meta信息
        self.last_search_ids = [] # 用于存储上一次搜索的结果

    def build_db(self, stories, vecs, flags = None, metas = None):
        self.stories = stories
        self.vecs = vecs
        self.flags = flags if flags else [True for _ in self.stories]
        self.metas = metas if metas else [{} for _ in self.stories]
        self.recompute_norm()

    def save(self, file_path):
        print( "warning! directly save folder from dbtype NaiveDB has not been implemented yet, try use role_from_hf to load role instead" )

    def load(self, file_path):
        print( "warning! directly load folder from dbtype NaiveDB has not been implemented yet, try use role_from_hf to load role instead" )

    def recompute_norm( self ):
        # 补全这部分代码，self.norms 分别存储每个vector的l2 norm
        # 计算每个向量的L2范数
        self.norms = [sqrt(sum([x**2 for x in vec])) for vec in self.vecs]

    def get_stories_with_id(self, ids ):
        return [self.stories[i] for i in ids]
    
    def clean_flag(self):
        self.flags = [True for _ in self.stories]

    def close_last_search(self):
        for id in self.last_search_ids:
            self.flags[id] = False

    def search(self, query_vector , n_results):

        if self.verbose:
            print("call search")

        if len(self.norms) != len(self.vecs):
            self.recompute_norm()

        # 计算查询向量的范数
        query_norm = sqrt(sum([x**2 for x in query_vector]))

        idxs = list(range(len(self.vecs)))

        # 计算余弦相似度
        similarities = []
        for vec, norm, idx in zip(self.vecs, self.norms, idxs ):
            if len(self.flags) == len(self.vecs) and not self.flags[idx]:
                continue

            dot_product = sum(q * v for q, v in zip(query_vector, vec))
            if query_norm < 1e-20:
                similarities.append( (random.random(), idx) )
                continue
            cosine_similarity = dot_product / (query_norm * norm)
            similarities.append( ( cosine_similarity, idx) )

        # 获取最相似的n_results个结果， 使用第0个字段进行排序
        similarities.sort(key=lambda x: x[0], reverse=True)
        self.last_search_ids = [x[1] for x in similarities[:n_results]]

        top_indices = [x[1] for x in similarities[:n_results]]
        return top_indices

        
    
    