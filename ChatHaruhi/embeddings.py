import random

# elif embedding == 'bge_en':
#                 embed_name = 'bge_en_s15'
#             elif embedding == 'bge_zh':
#                 embed_name = 'bge_zh_s15'

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


embedshortname2model_name = {
    "bge_zh":"BAAI/bge-small-zh-v1.5",
}

embedname2columnname = {
    "luotuo_openai":"luotuo_openai",
    "openai":"luotuo_openai",
    "bge_zh":"bge_zh_s15",
    "bge_en":"bge_en_s15",
    "bce":"bce_base",
}

# 这是用来调试的foo embedding

def foo_embedding( text ):
    # whatever text input , output a 2 dim 0-1 random vects
    return [random.random(), random.random()]
    
# TODO: add bge-zh-small(or family)  BCE and openai embedding here 米唯实
# ======== add bge_zh mmodel
# by Weishi MI

def foo_bge_zh_15( text ):
    dim = 512
    model_name = "BAAI/bge-small-zh-v1.5"
    if isinstance(text, str):
        text_list = [text]
    else:
        get_general_embeddings_safe(text, model_name)
    
    global _model_pool
    global _tokenizer_pool

    if model_name not in _model_pool:
        from transformers import AutoTokenizer, AutoModel
        _tokenizer_pool[model_name] = AutoTokenizer.from_pretrained(model_name)
        _model_pool[model_name] = AutoModel.from_pretrained(model_name)

    _model_pool[model_name].eval()

    # Tokenize sentences
    encoded_input = _tokenizer_pool[model_name](text_list, padding=True, truncation=True, return_tensors='pt', max_length = 512)

    # Compute token embeddings
    with torch.no_grad():
        model_output = _model_pool[model_name](**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]

    # normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.cpu().tolist()[0]
    # return [random.random() for _ in range(dim)]

def foo_bce( text ):
    from transformers import AutoModel, AutoTokenizer
    if isinstance(text, str):
        text_list = [text]
    
    # init model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('maidalun1020/bce-embedding-base_v1')
    model = AutoModel.from_pretrained('maidalun1020/bce-embedding-base_v1')
    
    model.to(device)
    
    # get inputs
    inputs = tokenizer(text_list, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
    
    # get embeddings
    outputs = model(**inputs_on_device, return_dict=True)
    embeddings = outputs.last_hidden_state[:, 0]  # cls pooler
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  # normalize
    return embeddings
def download_models():
    print("正在下载Luotuo-Bert")
    # Import our models. The package will take care of downloading the models automatically
    model_args = Namespace(do_mlm=None, pooler_type="cls", temp=0.05, mlp_only_train=False,
                           init_embeddings_model=None)
    model = AutoModel.from_pretrained("silk-road/luotuo-bert-medium", trust_remote_code=True, model_args=model_args).to(
        device)
    print("Luotuo-Bert下载完毕")
    return model

def get_luotuo_model():
    global _luotuo_model
    if _luotuo_model is None:
        _luotuo_model = download_models()
    return _luotuo_model


def luotuo_embedding(model, texts):
    # Tokenize the texts_source
    tokenizer = AutoTokenizer.from_pretrained("silk-road/luotuo-bert-medium")
    inputs = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
    inputs = inputs.to(device)
    # Extract the embeddings
    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True, sent_emb=True).pooler_output
    return embeddings

def luotuo_en_embedding( texts ):
    # this function implemented by Cheng
    global _luotuo_model_en
    global _luotuo_en_tokenizer

    if _luotuo_model_en is None:
        _luotuo_en_tokenizer = AutoTokenizer.from_pretrained("silk-road/luotuo-bert-en")
        _luotuo_model_en = AutoModel.from_pretrained("silk-road/luotuo-bert-en").to(device)

    if _luotuo_en_tokenizer is None:
        _luotuo_en_tokenizer = AutoTokenizer.from_pretrained("silk-road/luotuo-bert-en")

    inputs = _luotuo_en_tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        embeddings = _luotuo_model_en(**inputs, output_hidden_states=True, return_dict=True, sent_emb=True).pooler_output
        
    return embeddings


def get_embedding_for_chinese(model, texts):
    model = model.to(device)
    # str or strList
    texts = texts if isinstance(texts, list) else [texts]
    # 截断
    for i in range(len(texts)):
        if len(texts[i]) > 510:
            texts[i] = texts[i][:510]
    if len(texts) >= 64:
        embeddings = []
        chunk_size = 64
        for i in range(0, len(texts), chunk_size):
            embeddings.append(luotuo_embedding(model, texts[i: i + chunk_size]))
        return torch.cat(embeddings, dim=0)
    else:
        return luotuo_embedding(model, texts)


def is_chinese_or_english(text):
    # no longer use online openai api
    return "chinese"

    text = list(text)
    is_chinese, is_english = 0, 0

    for char in text:
        # 判断字符的Unicode值是否在中文字符的Unicode范围内
        if '\u4e00' <= char <= '\u9fa5':
            is_chinese += 4
        # 判断字符是否为英文字符（包括大小写字母和常见标点符号）
        elif ('\u0041' <= char <= '\u005a') or ('\u0061' <= char <= '\u007a'):
            is_english += 1
    if is_chinese >= is_english:
        return "chinese"
    else:
        return "english"


def get_embedding_openai(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_embedding_for_english(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

import os

def foo_openai( text ):
    # dim = 1536

    openai_key = os.environ.get("OPENAI_API_KEY")

    if isinstance(texts, list):
        index = random.randint(0, len(texts) - 1)
        if openai_key is None or is_chinese_or_english(texts[index]) == "chinese":
            return [embed.cpu().tolist() for embed in get_embedding_for_chinese(get_luotuo_model(), texts)]
        else:
            return [get_embedding_for_english(text) for text in texts]
    else:
        if openai_key is None or is_chinese_or_english(texts) == "chinese":
            return get_embedding_for_chinese(get_luotuo_model(), texts)[0].cpu().tolist()
        else:
            return get_embedding_for_english(texts)


### BGE family


# ======== add bge_zh mmodel
# by Cheng Li
# 这一次我们试图一次性去适配更多的模型
import torch

_model_pool = {}
_tokenizer_pool = {}

# BAAI/bge-small-zh-v1.5

def get_general_embeddings( sentences , model_name = "BAAI/bge-small-zh-v1.5" ):

    global _model_pool
    global _tokenizer_pool

    if model_name not in _model_pool:
        from transformers import AutoTokenizer, AutoModel
        _tokenizer_pool[model_name] = AutoTokenizer.from_pretrained(model_name)
        _model_pool[model_name] = AutoModel.from_pretrained(model_name).to(device)

    _model_pool[model_name].eval()

    # Tokenize sentences
    encoded_input = _tokenizer_pool[model_name](sentences, padding=True, truncation=True, return_tensors='pt', max_length = 512).to(device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = _model_pool[model_name](**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]

    # normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.cpu().tolist()

def get_general_embedding( text_or_texts , model_name = "BAAI/bge-small-zh-v1.5" ):
    if isinstance(text_or_texts, str):
        return get_general_embeddings([text_or_texts], model_name)[0]
    else:
        return get_general_embeddings_safe(text_or_texts, model_name)
    
general_batch_size = 16

import math

def get_general_embeddings_safe(sentences, model_name = "BAAI/bge-small-zh-v1.5"):
    
    embeddings = []
    
    num_batches = math.ceil(len(sentences) / general_batch_size)

    from tqdm import tqdm
    
    for i in tqdm( range(num_batches) ):
        # print("run bge with batch ", i)
        start_index = i * general_batch_size
        end_index = min(len(sentences), start_index + general_batch_size)
        batch = sentences[start_index:end_index]
        embs = get_general_embeddings(batch, model_name)
        embeddings.extend(embs)
        
    return embeddings

def get_bge_zh_embedding( text_or_texts ):
    return get_general_embedding(text_or_texts, "BAAI/bge-small-zh-v1.5")

