import tiktoken

_enc_model = None


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