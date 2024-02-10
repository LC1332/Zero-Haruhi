import random

# elif embedding == 'bge_en':
#                 embed_name = 'bge_en_s15'
#             elif embedding == 'bge_zh':
#                 embed_name = 'bge_zh_s15'

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
    model="text-embedding-ada-002"
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding
    print("warning! foo_openai is not implemented yet, 请催促 米唯实 进行实现" )
    return [random.random() for _ in range(dim)]


# TODO: add bge-zh-small(or family) and bce embedding here 米唯实
