

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

