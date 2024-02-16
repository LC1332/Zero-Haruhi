import erniebot

aclient = None

client = None

import os

def normalize2uaua_ernie( message, if_replace_system = False ):
    new_message = []
    last_role = ""

    for msg in message:
        role = msg["role"]
        if if_replace_system and role == "system":
            role = "user"
            msg["role"] = role
        
        if last_role == role:
            new_message[-1]["content"] = new_message[-1]["content"] + "\n" + msg["content"]
        else:
            last_role = role
            new_message.append( msg )

    return new_message

def init_client():
    
    # 将client设置为全局变量
    global client
    
    # 将ERNIE_ACCESS_TOKEN作为参数值传递给OS
    api_key = os.getenv("ERNIE_ACCESS_TOKEN")
    if api_key is None:
        raise ValueError("环境变量'ERNIE_ACCESS_TOKEN'未设置，请确保已经定义了API密钥")
    erniebot.api_type = "aistudio"
    erniebot.access_token = api_key
    client = erniebot

def get_response( message, model_name = "ernie-4.0" ):
    if client is None:
        init_client()
    
    message_ua = normalize2uaua_ernie(message, if_replace_system = True)
    # print(message_ua)
    response = client.ChatCompletion.create(\
        model=model_name,\
        messages = message_ua, \
        temperature = 0.1 )
    return response.get_result()

import json
import asyncio
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import HumanMessage, AIMessage, SystemMessage, FunctionMessage

def init_aclient(model="ernie-4.0"):
    
    # 将aclient设置为全局变量
    global aclient

    api_key = os.getenv("ERNIE_ACCESS_TOKEN")
    if api_key is None:
        raise ValueError("环境变量'ERNIE_ACCESS_TOKEN'未设置。请确保已经定义了API密钥。")
    os.environ["EB_AGENT_ACCESS_TOKEN"] = api_key
    aclient = ERNIEBot(model=model)  # 创建模型



async def async_get_response( message, model="ernie-4.0" ):
    if aclient is None:
        init_aclient(model=model)
    
    messages = []
    system_message = None
    message_ua = normalize2uaua_ernie(message, if_replace_system = False)
    print(message_ua)
    for item in message_ua:
        if item["role"] == "user":
            messages.append(HumanMessage(item["content"]))
        elif item["role"] == "system":
            system_message = SystemMessage(item["content"])
        else:
            messages.append(AIMessage(item["content"])) 
    if system_message:
        ai_message = await aclient.chat(messages=messages, temperature = 0.1)
    else:
        ai_message = await aclient.chat(messages=messages, system=system_message.content, temperature = 0.1)  # 调用模型chat接口，非流式返回

    return ai_message.content