# coding: utf-8
import warnings
warnings.filterwarnings("ignore")
import os
import re
import json
import torch
import pickle
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

client = None

def get_prompt(message):
    #prompt = system_info.format(role_name=role_name, persona=persona)
    persona = ""
    for msg in message:
        if msg["role"] == "system":
            persona = persona + msg["content"]
    prompt = "<<SYS>>" + persona + "<</SYS>>"
    from ChatHaruhi.utils import normalize2uaua
    message_ua = normalize2uaua(message[1:], if_replace_system = True)

    for i in range(0, len(message_ua)-1, 2):
        prompt = prompt + "[INST]" + message_ua[i]["content"] + "[/INST]" + message_ua[i+1]["content"] + "<|im_end|>"
    prompt = prompt + "[INST]" + message_ua[-1]["content"] + "[/INST]"
    print(prompt)
    return prompt

import os
class qwen_model:
    def __init__(self, model_name):
        self.DEVICE = torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "silk-road/"+model_name,
            low_cpu_mem_usage=True,
            use_fast = False,
            padding_side="left",
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.eos_token_id = 151645
        # print(tokenizer.eos_token_id)
        # print(tokenizer.pad_token_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            "silk-road/"+model_name,
            load_in_8bit=False,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map='auto',
            trust_remote_code=True,
        ).eval()
        # model.to("cuda")
        # model.eval()
        # self.tokenizer = AutoTokenizer.from_pretrained("silk-road/"+model_name, trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained("silk-road/"+model_name, device_map="auto", trust_remote_code=True).eval()

    def get_response(self, message):
        with torch.inference_mode():
            prompt = get_prompt(message)
            batch = self.tokenizer(prompt, return_tensors="pt", padding=True)
            batch = self.tokenizer(prompt,
                             return_tensors="pt",
                             padding=True,
                             add_special_tokens=False)
            batch = {k: v.to(self.DEVICE) for k, v in batch.items()}
            generated = self.model.generate(input_ids=batch["input_ids"],
                                   max_new_tokens=1024,
                                   temperature=0.2,
                                   top_p=0.9,
                                   top_k=40,
                                   do_sample=False,
                                   num_beams=1,
                                   repetition_penalty=1.3,
                                   eos_token_id=self.tokenizer.eos_token_id,
                                   pad_token_id=self.tokenizer.pad_token_id)
            response = self.tokenizer.decode(generated[0][batch["input_ids"].shape[1]:]).strip().replace("<|im_end|>", "")
        return response


def init_client(model_name):

    # 将client设置为全局变量
    global client

    client = qwen_model(model_name = model_name)

def get_response(message, model_name = "Haruhi-Zero-1_8B-0_4"):
    if client is None:
        init_client(model_name)

    response = client.get_response(message)
    return response

