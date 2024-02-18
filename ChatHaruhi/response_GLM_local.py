import os
from string import Template
from typing import List, Dict

import torch.cuda
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    except Exception:
        if pretrained_model_download(model_name, verbose=verbose):
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, local_files_only=True)
            client = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True, local_files_only=True)

    client = client.to(device).eval()


def pretrained_model_download(model_name_or_path: str, verbose: bool) -> bool:
    """
        使用huggingface_hub下载模型（model_name_or_path）。下载成功返回true，失败返回False。
        Params: 
            model_name_or_path (`str`): 模型的huggingface地址
        Returns:
            `bool` 是否下载成功
    """
    # TODO 使用hf镜像加速下载 未测试windows端
    # 尝试引入huggingface_hub
    try:
        import huggingface_hub
    except ImportError:
        print("Install huggingface_hub.")
        os.system("pip -q install huggingface_hub")
        import huggingface_hub

    # 使用huggingface_hub下载模型。
    try:
        print(f"downloading {model_name_or_path}")
        huggingface_hub.snapshot_download(
            repo_id=model_name_or_path, endpoint=END_POINT, resume_download=True, local_dir_use_symlinks=False, ignore_patterns=["pytorch_model*"])
    except Exception as e:
        raise e

    return True


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


def get_response(message, model_name: str = "THUDM/chatglm3-6b", verbose: bool = False):
    global client
    global tokenizer

    if client is None:
        init_client(model_name, verbose=verbose)

    if verbose:
        print(message)
        print(message2query(message))

    response, history = client.chat(tokenizer, message2query(message))

    return response
