import os

import torch.cuda
from transformers import AutoTokenizer, AutoModelForCausalLM

aclient = None

client = None
tokenizer = None

END_POINT = "https://hf-mirror.com"


def init_client(model_name):
    # 将client设置为全局变量
    global client
    global tokenizer

    device = None

    # TODO 考虑支持deepspeed 进行多gpu推理，以及zero
    # 判断 使用MPS、CUDA、CPU运行模型
    if torch.cuda.is_available():
        device = torch.device("cuda")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    device = torch.device("cpu") if device is None else device

    print("Using device: ", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, mirror=END_POINT,
                                              output_loading_info=True)
    client = (AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, mirror=END_POINT,
                                                   output_loading_info=True).to(device).eval())


def pretrained_model_download(model_name_or_path):
    # TODO 使用hf加速下载 未测试
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    try:
        import huggingface_hub
    except ImportError:
        print("Install huggingface_hub.")
        os.system("pip -q install -U huggingface_hub")
    try:
        import hf_transfer
    except ImportError:
        print("Install hf_transfer.")
        os.system("pip -q install -U hf-transfer -i https://pypi.org/simple")
        # Enable hf-transfer if specified
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    print("export HF_HUB_ENABLE_HF_TRANSFER=", os.getenv("HF_HUB_ENABLE_HF_TRANSFER"))

    download_shell = f"huggingface-cli download --local-dir-use-symlinks False --resume-download {model_name_or_path}"
    os.system(download_shell)


def get_response(message, model_name="THUDM/chatglm3-6b"):
    global client
    global tokenizer

    if client is None:
        init_client(model_name)

    client.chat(tokenizer, prompt=message)
