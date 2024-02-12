import zhipuai

aclient = None

client = None

import os
from zhipuai import ZhipuAI

def init_client():
    
    # 将client设置为全局变量
    global client
    
    # 将ZHIPUAI_API_KEY作为参数值传递给OS
    api_key = os.environ("ZHIPUAI_API_KEY")
    if api_key is None:
        raise ValueError("环境变量'ZHIPUAI_API_KEY'未设置，请确保已经定义了API密钥")

def init_aclient():
    
    # 将aclient设置为全局变量
    global aclient

    # 将ZHIPUAI_API_KEY作为参数值传递给OS
    api_key = os.environ("ZHIPUAI_API_KEY")
    if api_key is None:
        raise ValueError("环境变量'ZHIPUAI_API_KEY'未设置，请确保已经定义了API密钥")
    
    # 异步调用








    


    
