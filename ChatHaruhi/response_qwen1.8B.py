from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

client = None

import os
class qwen_model:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained("silk-road/"+model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("silk-road/"+model_name, device_map="auto", trust_remote_code=True).eval()

    def get_response(self, message):
        from ChatHaruhi.utils import normalize2uaua
        message_ua = normalize2uaua(message, if_replace_system = True)
        import json
        message_tuples = []
        for i in range(0, len(message_ua)-1, 2):
            message_tuple = (message_ua[i]["content"], message_ua[i+1]["content"])
            message_tuples.append(message_tuple)
        response, _ = self.model.chat(self.tokenizer, message_ua[-1]["content"], history=message_tuples)
        return response
def init_client(model_name):

    # 将client设置为全局变量
    global client

    client = qwen_model(model_name = model_name)

def get_response(message, model_name = "Haruhi-Zero-1_8B"):
    if client is None:
        init_client(model_name)

    response = client.get_response(message)
    return response
