import os
from string import Template
from typing import List, Dict

import torch.cuda
from transformers import AutoTokenizer, AutoModelForCausalLM

from ChatHaruhi.utils import message2query4Gemma, pretrained_model_download

aclient = None

client = None
tokenizer = None

END_POINT = "https://hf-mirror.com"


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


def get_response(message, model_name: str = "silk-road/Haruhi-Zero-Gemma-2B-0_5", verbose: bool = True):
    global client
    global tokenizer

    if client is None:
        init_client(model_name, verbose=verbose)

    if verbose:
        # print(message)
        print(f"message2query:{message2query4Gemma(message,tokenizer)}")

    inputs = tokenizer.encode(
        message2query4Gemma(message, tokenizer), return_tensors="pt")
    response = client.generate(input_ids=inputs.to(
        client.device), max_new_tokens=1024)

    response = tokenizer.decode(
        response[0], skip_special_tokens=True).split("model\n")[-1]

    return response
