import os
from openai import OpenAI, AsyncOpenAI

client = None
aclient = None

def init_client():
    global client

    api_key = os.getenv("moonshot_key")
    if api_key is None:
        raise ValueError("环境变量'MOONSHOT_API_KEY'未设置。请确保已经定义了API密钥。")
    
    base_url = os.getenv("MOONSHOT_API_BASE", "https://api.moonshot.cn/v1")
    client = OpenAI(base_url=base_url, api_key=api_key)

def get_response(message):
    if client is None:
        init_client()
    response = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=message,
        max_tokens=300,
        temperature=0.1
    )
    return response.choices[0].message.content

def init_aclient():
    global aclient

    api_key = os.getenv("moonshot_key")
    if api_key is None:
        raise ValueError("环境变量'MOONSHOT_API_KEY'未设置。请确保已经定义了API密钥。")
    
    base_url = os.getenv("MOONSHOT_API_BASE", "https://api.moonshot.cn/v1")
    aclient = AsyncOpenAI(base_url=base_url, api_key=api_key)

async def async_get_response(message):
    if aclient is None:
        init_aclient()
    response = await aclient.chat.completions.create(
        model="moonshot-v1-8k",
        messages=message,
        max_tokens=300,
        temperature=0.1
    )
    return response.choices[0].message.content