import random

# elif embedding == 'bge_en':
#                 embed_name = 'bge_en_s15'
#             elif embedding == 'bge_zh':
#                 embed_name = 'bge_zh_s15'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

def foo_bge_zh_15( text ):
    dim = 512
    # text_list = [text]
    # model_name = "BAAI/bge-small-zh-v1.5"
    # global _model_pool
    # global _tokenizer_pool

    # if model_name not in _model_pool:
    #     from transformers import AutoTokenizer, AutoModel
    #     _tokenizer_pool[model_name] = AutoTokenizer.from_pretrained(model_name)
    #     _model_pool[model_name] = AutoModel.from_pretrained(model_name)

    # _model_pool[model_name].eval()

    # # Tokenize sentences
    # encoded_input = _tokenizer_pool[model_name](text_list, padding=True, truncation=True, return_tensors='pt', max_length = 512)

    # # Compute token embeddings
    # with torch.no_grad():
    #     model_output = _model_pool[model_name](**encoded_input)
    #     # Perform pooling. In this case, cls pooling.
    #     sentence_embeddings = model_output[0][:, 0]

    # # normalize embeddings
    # sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=dim)
    # return sentence_embeddings.cpu().tolist()
    print("warning! foo_bge_zh_15 is not implemented yet, 请催促 米唯实 进行实现" )
    return [random.random() for _ in range(dim)]

def foo_bce( text ):
    dim = 768
    print("warning! foo_bce is not implemented yet, 请催促 米唯实 进行实现" )
    return [random.random() for _ in range(dim)]

def foo_openai( text ):
    dim = 1536
    # model="text-embedding-ada-002"
    # text = text.replace("\n", " ")
    # return client.embeddings.create(input = [text], model=model).data[0].embedding
    print("warning! foo_openai is not implemented yet, 请催促 米唯实 进行实现" )
    return [random.random() for _ in range(dim)]


# TODO: add bge-zh-small(or family) and bce embedding here 米唯实

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

