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
    print("warning! foo_bge_zh_15 is not implemented yet, 请催促 米唯实 进行实现" )
    return [random.random() for _ in range(dim)]

def foo_bce( text ):
    dim = 768
    print("warning! foo_bce is not implemented yet, 请催促 米唯实 进行实现" )
    return [random.random() for _ in range(dim)]

def foo_openai( text ):
    dim = 1536
    print("warning! foo_openai is not implemented yet, 请催促 米唯实 进行实现" )
    return [random.random() for _ in range(dim)]


# TODO: add bge-zh-small(or family) and bce embedding here 米唯实