from string import Template
import tiktoken
import os

import tqdm

END_POINT = "https://hf-mirror.com"

def package_role( system_prompt, texts_path , embedding ):
    datas = []

    # 暂时只有一种embedding 'luotuo_openai'
    # embed_name_1 = 'luotuo_openai'
    embed_name_2 = 'bge_zh_s15'

    datas.append({ 'text':system_prompt , embed_name_2:'system_prompt'})
    datas.append({ 'text':'Reserve Config Setting Here' , embed_name_2:'config'})
    

    # debug_count = 3

    # for file in os.listdir(texts_path):

    files = os.listdir(texts_path)

    for i in tqdm.tqdm(range(len(files))):
        file = files[i]
        # if file name end with txt
        if file.endswith(".txt"):
            file_path = os.path.join(texts_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                current_str = f.read()
                current_vec = embedding(current_str)
                encode_vec = float_array_to_base64(current_vec)
                datas.append({ 'text':current_str , embed_name_2:encode_vec})

                # debug_count -= 1
                # if debug_count == 0:
                #     break
    return datas


import struct

def get_model_name2funcs( locol_model_names = [] ):
    ans = {}

    # openai
    if "OPENAI_API_KEY" in os.environ and os.getenv("OPENAI_API_KEY").strip() != "":
        try:
            from .response_openai import get_response as get_response_openai
            ans["openai"] = get_response_openai
        except:
            print("OPENAI_API_KEY existed but failed to load response_openai.get_response, may need to pip install openai")

    # zhipu
    if "ZHIPUAI_API_KEY" in os.environ and os.getenv("ZHIPUAI_API_KEY").strip() != "":
        try:
            from .response_zhipu import get_response as get_response_zhipu
            ans["zhipu"] = get_response_zhipu
        except:
            print("ZHIPUAI_API_KEY existed but failed to load response_zhipu.get_response, may need to pip install zhipuai")

    # ernie
    if "ERNIE_ACCESS_TOKEN" in os.environ and os.getenv("ERNIE_ACCESS_TOKEN").strip() != "":
        try:
            from .response_erniebot import get_response as get_response_ernie
            ans["ernie"] = get_response_ernie
        except:
            print("ERNIE_ACCESS_TOKEN existed but failed to load response_erniebot.get_response, may need to pip install ernie")

    # spark
    if "SPARK_API_KEY" in os.environ and os.getenv("SPARK_API_KEY").strip() != "":
        try:
            from .response_spark import get_response as get_response_spark
            ans["spark"] = get_response_spark
        except:
            print("SPARK_API_KEY existed but failed to load response_spark.get_response")

    if "BAICHUAN_API_KEY" in os.environ and os.getenv("BAICHUAN_API_KEY").strip() != "":
        try:
            from .response_baichuan import get_response as get_response_baichuan
            ans["baichuan"] = get_response_baichuan
        except:
            print("BAICHUAN_API_KEY existed but failed to load response_baichuan.")

    for local_model_name in locol_model_names:
        if local_model_name.lower().strip() == "qwen1_8b":
            try:
                from .response_qwen1_8B import get_response as get_response_qwen1_8B
                ans["qwen1_8B"] = get_response_qwen1_8B
            except:
                print("Failed to load response_qwen1_8B.get_response")
            break
        elif local_model_name.lower().strip() == "glm":
            try:
                from .response_GLM_local import get_response as get_response_GLM
                ans["GLM"] = get_response_GLM
            except:
                print("Failed to load response_GLM.get_response")
            break
        elif local_model_name.lower().strip() == "glm_lora":
            try:
                from .response_GLM_lora import get_response as get_response_GLM_lora
                ans["GLM_lora"] = get_response_GLM_lora
            except:
                print("Failed to load response_GLM_lora.get_response")
            break

    if len(ans) == 0:
        ans["foo"] = lambda x: "No model is available"

    return ans

    

_enc_model = None

def normalize2uaua( message, if_replace_system = False ):
    new_message = []
    last_role = ""

    for msg in message:
        role = msg["role"]
        if if_replace_system and role == "system":
            role = "user"
        
        if last_role == role:
            new_message[-1]["content"] = new_message[-1]["content"] + "\n" + msg["content"]
        else:
            last_role = role
            new_message.append( msg )

    return new_message

def tiktoken_counter( text ):
    global _enc_model

    if _enc_model is None:
        _enc_model = tiktoken.get_encoding("cl100k_base")

    return len(_enc_model.encode(text))


def string_to_base64(text):
    import base64
    byte_array = b''
    for char in text:
        num_bytes = char.encode('utf-8')
        byte_array += num_bytes

    base64_data = base64.b64encode(byte_array)
    return base64_data.decode('utf-8')

def base64_to_string(base64_data):
    import base64
    byte_array = base64.b64decode(base64_data)
    text = byte_array.decode('utf-8')
    return text


def float_array_to_base64(float_arr):
    import struct
    import base64
    byte_array = b''
    
    for f in float_arr:
        # 将每个浮点数打包为4字节
        num_bytes = struct.pack('!f', f)  
        byte_array += num_bytes
    
    # 将字节数组进行base64编码    
    base64_data = base64.b64encode(byte_array)
    
    return base64_data.decode('utf-8')

def base64_to_float_array(base64_data):
    import struct
    import base64
    byte_array = base64.b64decode(base64_data)
    
    float_array = []
    
    # 每 4 个字节解析为一个浮点数
    for i in range(0, len(byte_array), 4):
        num = struct.unpack('!f', byte_array[i:i+4])[0] 
        float_array.append(num)

    return float_array

def load_datas_from_jsonl( file_path ):
    import json
    datas = []
    with open(file_path, 'r', encoding = 'utf-8') as f:
        for line in f:
            datas.append(json.loads(line))
    return datas

def save_datas_to_jsonl( file_path, datas ):
    import json
    with open(file_path, 'w', encoding = 'utf-8') as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

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

def message2query4GLM(messages) -> str:
    # [{'role': 'user', 'content': '老师: 同学请自我介绍一下'}]
    # <|system|>
    # You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.
    # <|user|>
    # Hello
    # <|assistant|>
    # Hello, I'm ChatGLM3. What can I assist you today?
    template = Template("<|$role|>\n$content\n")

    return "".join([template.substitute(message) for message in messages])

def message2query4Gemma(messages,tokenizer) -> str:
    # [{'role': 'user', 'content': '老师: 同学请自我介绍一下'}]
    # <start_of_turn>user
    # Write a hello world program<end_of_turn>
    # <start_of_turn>model

    prompt = messages[0]['content']
    messages[1]['content'] = f"{prompt}\n{messages[1]['content']}"

    conversation = tokenizer.apply_chat_template(
        messages[1:], tokenize=False, add_generation_prompt=True)

    return conversation