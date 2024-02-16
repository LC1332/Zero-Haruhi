from . import SparkApi

aclient = None

client = None

import os

def init_client():

    # 将client设置为全局变量
    global client

    # 将ERNIE_ACCESS_TOKEN作为参数值传递给OS
    appid = os.getenv("SPARK_APPID")
    api_secret = os.getenv("SPARK_API_SECRET")
    api_key = os.getenv("SPARK_API_KEY")
    if appid is None:
        raise ValueError("环境变量'SPARK_APPID'未设置，请确保已经定义了API密钥")
    if api_secret is None:
        raise ValueError("环境变量'SPARK_API_SECRET'未设置，请确保已经定义了API密钥")
    if api_key is None:
        raise ValueError("环境变量'SPARK_API_KEY'未设置，请确保已经定义了API密钥")
    SparkApi.appid = appid
    SparkApi.api_secret = api_secret
    SparkApi.api_key = api_key
    client = SparkApi

def get_response(message, model_name = "Spark3.5"):
    if client is None:
        init_client()

    if model_name == "Spark2.0":
        domain = "generalv2"    # v2.0版本
        Spark_url = "ws://spark-api.xf-yun.com/v2.1/chat"  # v2.0环境的地址
    elif model_name == "Spark1.5":
        domain = "general"   # v1.5版本
        Spark_url = "ws://spark-api.xf-yun.com/v1.1/chat"  # v1.5环境的地址
    elif model_name == "Spark3.0":
        domain = "generalv3"   # v3.0版本
        Spark_url = "ws://spark-api.xf-yun.com/v3.1/chat"  # v3.0环境的地址
    elif model_name == "Spark3.5":
        domain = "generalv3.5"   # v3.5版本
        Spark_url = "ws://spark-api.xf-yun.com/v3.5/chat"  # v3.5环境的地址
    else:
        raise Exception("Unknown Spark model")
    # print(message_ua)
    client.answer = ""
    client.main(client.appid,client.api_key,client.api_secret,Spark_url,domain,message)
    return client.answer

