import openai

aclient = None

client = None

import os
from openai import OpenAI

def init_client():
    # 将client设置为全局变量，以便在其他函数中使用
    global client

    # 检查是否存在API_KEY环境变量
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("环境变量'OPENAI_API_KEY'未设置。请确保已经定义了API密钥。")
    
    # 检查是否存在API_BASE环境变量，并据此设置base_url参数
    api_base = os.getenv("OPENAI_API_BASE")
    if api_base:
        client = OpenAI(base_url=api_base, api_key=api_key)
    else:
        client = OpenAI(api_key=api_key)

    

def get_response( message ):
    if client is None:
        init_client()
    response = client.chat.completions.create(\
        model="gpt-3.5-turbo-0125",\
        messages = message, \
        max_tokens = 300, \
        temperature = 0.1 )
    return response.choices[0].message.content

from openai import AsyncOpenAI

def init_aclient():
    # 将aclient设置为全局变量，以便在其他函数中使用
    global aclient

    # 检查是否存在API_KEY环境变量
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("环境变量'OPENAI_API_KEY'未设置。请确保已经定义了API密钥。")
    
    # 检查是否存在API_BASE环境变量，并据此设置base_url参数
    api_base = os.getenv("OPENAI_API_BASE")
    if api_base:
        aclient = AsyncOpenAI(base_url=api_base, api_key=api_key)
    else:
        aclient = AsyncOpenAI(api_key=api_key)

async def async_get_response( message ):
    if aclient is None:
        init_aclient()
    response = await aclient.chat.completions.create(\
        model="gpt-3.5-turbo-0125",\
        messages = message, \
        max_tokens = 300, \
        temperature = 0.1 )
    return response.choices[0].message.content
    