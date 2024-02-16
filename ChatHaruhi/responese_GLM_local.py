import os
from string import Template
from typing import List, Dict

import torch.cuda
from huggingface_hub.utils import LocalEntryNotFoundError
from transformers import AutoTokenizer, AutoModelForCausalLM

aclient = None

client = None
tokenizer = None

END_POINT = "https://hf-mirror.com"


def init_client(model_name, verbose):
    """
        初始化模型，通过可用的设备进行模型加载推理。
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


def pretrained_model_download(model_name_or_path: str, verbose) -> bool:
    """
        使用huggingface_hub下载模型（model_name_or_path）。下载成功返回true，失败返回False。
    :param model_name_or_path: 模型的huggingface地址
    :return: bool
    """
    # TODO 使用hf镜像加速下载 未测试linux、windows端
    # 判断平台（windows 未测试安装hf_transfer）
    # m2 mac无法便捷安装hf_transfer，因此在mac上暂时不使用 hf_transfer
    import platform
    if os.getenv("HF_HUB_ENABLE_HF_TRANSFER") == 1:
        try:
            import hf_transfer
        except ImportError:
            print("Install hf_transfer.")
            os.system("pip -q install hf_transfer")
            import hf_transfer

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
            repo_id=model_name_or_path, endpoint=END_POINT, resume_download=True)
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


def get_response(message, model_name="THUDM/chatglm3-6b", verbose=False):
    global client
    global tokenizer

    if client is None:
        init_client(model_name, verbose=verbose)

    if verbose:
        print(message)
        print(message2query(message))

    response, history = client.chat(tokenizer, message2query(message))

    return response
