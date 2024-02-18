import os
from string import Template
from typing import List, Dict

import torch.cuda
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM, PeftConfig, get_peft_model

from ChatHaruhi.response_GLM_local import pretrained_model_download


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

    # TODO 上传模型后，更改为从huggingface获取模型
    # client = AutoPeftModelForCausalLM.from_pretrained(
    #     model_name, trust_remote_code=True)
    # tokenizer_dir = client.peft_config['default'].base_model_name_or_path
    # if verbose:
    #     print(tokenizer_dir)
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_name, trust_remote_code=True)

    # pretrained_model_download(model_name, verbose=verbose)
    # peft_config=AutoPeftConfig
    # client = AutoPeftModelForCausalLM.from_pretrained(
    #     model_name, trust_remote_code=True, local_files_only=True)
    # tokenizer_dir = client.peft_config['default'].base_model_name_or_path
    # pretrained_model_download(tokenizer_dir, verbose=verbose)
    # tokenizer = AutoTokenizer.from_pretrained(
    #     tokenizer_dir, trust_remote_code=True, local_files_only=True)

    # 加载peft_config
    # try:
    #     if verbose:
    #         print(f"在try中加载模型{model_name}")
    #     peft_config = PeftConfig.from_pretrained(
    #         model_name, trust_remote_code=True, local_files_only=True)
    #     if verbose:
    #         print(f"在try中加载模型完成")
    # except OSError as e:
    #     if pretrained_model_download(model_name, verbose=verbose):
    #         if verbose:
    #             print(f"在except中加载模型{model_name}")
    #         peft_config = PeftConfig.from_pretrained(
    #             model_name, trust_remote_code=True, local_files_only=True)
    #         if verbose:
    #             print("在except中加载模型完成")
    #     else:
    #         raise (f"下载peft_config {model_name}失败", e)

    # # 加载模型
    # try:
    #     if verbose:
    #         print(f"在try中加载模型{base_model_name}")
    #     client = AutoModelForCausalLM.from_pretrained(
    #         base_model_name, trust_remote_code=True, local_files_only=True)
    #     if verbose:
    #         print(f"在try中加载模型完成")
    # except OSError as e:
    #     if pretrained_model_download(base_model_name, verbose=verbose):
    #         if verbose:
    #             print(f"在except中加载模型{base_model_name}")
    #         client = AutoPeftModelForCausalLM.from_pretrained(
    #             base_model_name, trust_remote_code=True, local_files_only=True)
    #         if verbose:
    #             print(f"在except中加载模型完成")
    #     else:
    #         raise (f"下载{base_model_name}模型失败", e)

    # TODO 直接使用peft_config加载模型
    # peft_config = PeftConfig.from_pretrained(
    #     model_name, trust_remote_code=True)

    # base_model_name = peft_config.base_model_name_or_path

    # client = get_peft_model(client, peft_config)
    # client = AutoModelForCausalLM.from_pretrained(
    #     base_model_name, trust_remote_code=True)
    # # 加载tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(
    #     base_model_name, trust_remote_code=True)

    # TODO
    client = AutoPeftModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True)
    tokenizer_dir = client.peft_config['default'].base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=True)

    # try:
    #     if verbose:
    #         print("在try中加载模型")
    #     client = AutoPeftModelForCausalLM.from_pretrained(
    #         model_name, trust_remote_code=True, local_files_only=True)
    #     if verbose:
    #         print("在try中加载模型完成")
    #     # tokenizer_dir = client.peft_config['default'].base_model_name_or_path
    #     # tokenizer = AutoTokenizer.from_pretrained(
    #     #     tokenizer_dir, trust_remote_code=True, local_files_only=True)
    # except Exception as e:
    #     if verbose:
    #         print("在except中加载模型，错误为", e)
    #     if pretrained_model_download(model_name, verbose=verbose):
    #         client = AutoPeftModelForCausalLM.from_pretrained(
    #             model_name, trust_remote_code=True, local_files_only=True)
    #     if verbose:
    #         print("在except中加载模型完成。")

    # if pretrained_model_download(model_name, verbose=verbose):
    # if not client:
    #     client = AutoPeftModelForCausalLM.from_pretrained(
    #         model_name, trust_remote_code=True, local_files_only=True)
    # if client:
    #     tokenizer_dir = client.peft_config['default'].base_model_name_or_path
    #     if pretrained_model_download(tokenizer_dir, verbose=verbose):
    #         tokenizer = AutoTokenizer.from_pretrained(
    #             tokenizer_dir, trust_remote_code=True, local_files_only=True)
    # if not client:
    #     raise ("模型加载失败")

    # if verbose:
    #     print(tokenizer_dir)
    # try:
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         tokenizer_dir, trust_remote_code=True, local_files_only=True)
    # except Exception:
    #     if pretrained_model_download(tokenizer_dir, verbose=verbose):
    #         tokenizer = AutoTokenizer.from_pretrained(
    #             tokenizer_dir, trust_remote_code=True, local_files_only=True)

    # client = client.to(device).eval()
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


def get_response(message, model_name: str = "silk-road/Haruhi-Zero-GLM3-6B-Lora-0_4", verbose: bool = True):
    global client
    global tokenizer

    if client is None:
        init_client(model_name, verbose=verbose)

    if verbose:
        print(message)
        print(message2query(message))

    response, history = client.chat(tokenizer, message2query(message))
    if verbose:
        print((response, history))

    return response
