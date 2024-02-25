import os
from string import Template
from typing import List, Dict

import torch.cuda
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM, PeftConfig, get_peft_model

from ChatHaruhi.utils import message2query4GLM

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

    if verbose:
        print("进入init_client")

    # 判断 使用MPS、CUDA、CPU运行模型
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if verbose:
        print("Using device: ", device)

    # TODO 直接使用peft_config加载模型
    # 加载peft_config
    peft_config = PeftConfig.from_pretrained(
        model_name, trust_remote_code=True)

    base_model_name = peft_config.base_model_name_or_path

    # 加载模型
    client = AutoModelForCausalLM.from_pretrained(
        base_model_name, trust_remote_code=True)
    client = get_peft_model(client, peft_config)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True)

    client = client.to(device).eval()


def get_response(message, model_name: str = "silk-road/Haruhi-Zero-GLM3-6B-Lora-0_4", verbose: bool = True):
    global client
    global tokenizer

    if client is None:
        init_client(model_name, verbose=verbose)

    if verbose:
        print(message)
        print(message2query4GLM(message))

    response, history = client.chat(tokenizer, message2query4GLM(message))
    if verbose:
        print((response, history))

    return response
