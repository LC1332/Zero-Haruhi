import requests

client = None

import os
import json
class Baichuan:
    def __init__(self, api_key):
        self.api_key = api_key

    def do_request(self, message, temperature = 0.3, top_p = 0.85, model = "Baichuan2-Turbo"):
        url = "https://api.baichuan-ai.com/v1/chat/completions"
        api_key = self.api_key

        data = {
            "model": model,
            "messages": message,
            "stream": False,
            "temperature": temperature,
            "top_p": top_p
        }

        json_data = json.dumps(data)

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key
        }

        response = requests.post(url, data=json_data, headers=headers, timeout=60)

        # if response.status_code == 200:
        #     print("请求成功！")
        #     print("响应body:", response.text)
        #     print("请求成功，X-BC-Request-Id:", response.headers.get("X-BC-Request-Id"))
        # else:
        #     print("请求失败，状态码:", response.status_code)
        #     print("请求失败，body:", response.text)
        #     print("请求失败，X-BC-Request-Id:", response.headers.get("X-BC-Request-Id"))
        return response.text

def normalize2uaua_baichuan( message, if_replace_system = False ):
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
    baichuan_api_key = os.getenv("BAICHUAN_API_KEY")

    if baichuan_api_key is None:
        raise ValueError("环境变量'BAICHUAN_API_KEY'未设置，请确保已经定义了API密钥")

    client = Baichuan(baichuan_api_key)

def get_response(message, model_name = "Baichuan2-Turbo"):
    if client is None:
        init_client()

    # print(message_ua)
    message_ua = normalize2uaua_baichuan( message, if_replace_system = True )
    response = client.do_request(message_ua, model = model_name)
    data = json.loads(response)

    content = data['choices'][0]['message']['content']

    return content

