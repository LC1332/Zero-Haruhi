import os
from string import Template
from typing import List, Dict

import torch.cuda
from transformers import AutoTokenizer, AutoModelForCausalLM

from ChatHaruhi.utils import pretrained_model_download

aclient = None

client = None
tokenizer = None


def init_client(model_name: str, verbose: bool) -> None:
    """
        初始化模型，通过可用的设备进行模型加载推理。

        Params:
            model_name (`str`)
                HuggingFace中的模型项目名，例如"THUDM/chatglm3-6b"
    """

    # 将client设置为全局变量
    global client
    global tokenizer

    # 判断 使用MPS、CUDA、CPU运行模型
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if verbose:
        print("Using device: ", device)

    # TODO 考虑支持deepspeed 进行多gpu推理，以及zero

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=True)
        client = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=True)
    except Exception as e:
        if verbose:
            print(e)
        if pretrained_model_download(model_name, verbose=verbose):
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, local_files_only=True)
            client = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True, local_files_only=True)

    client = client.to(device).eval()


def message2query(messages: List[Dict[str, str]]) -> str:
    # [{'role': 'user', 'content': '老师: 同学请自我介绍一下'}]
    # <|system|>
    # You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.
    # <|user|>
    # Hello
    # <|assistant|>
    # Hello, I'm ChatGLM3. What can I assist you today?
    template = Template("<|$role|>\n$content\n")

    return "".join([template.substitute(message) for message in messages])


def get_response(message, model_name: str = "silk-road/Haruhi-Zero-GLM3-6B-0_4", verbose: bool = True):
    global client
    global tokenizer

    if client is None:
        init_client(model_name, verbose=verbose)

    if verbose:
        print(message)
        print(message2query(message))

    response, history = client.chat(tokenizer, message2query(message))

    return response
