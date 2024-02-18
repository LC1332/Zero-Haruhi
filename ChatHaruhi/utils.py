import tiktoken
import os

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