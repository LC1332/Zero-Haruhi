import random

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