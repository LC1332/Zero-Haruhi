# Zero-凉宫春日
# Haruhi-Zero: Zero-Shot Role-Playing Model

Zero-凉宫春日是一个同时支持Zero-Shot角色构造和RAG角色构造(原ChatHaruhi)的角色扮演模型

## Introduction

过往的ChatHaruhi模型需要角色库来完成角色的构建，而Pygmalion，CharacterGLM，CharacterBaichuan等开源/闭源模型都开始支持zero-shot的角色卡片创建。目前，从[Haruhi-Zero-0.3](https://huggingface.co/silk-road/Haruhi-Zero-7B-0_3)开始，已经基本支持Zero-shot角色扮演。

项目的目标

- [x] 一个通用的，同时支持Zero-shot和RAG角色构造的角色扮演模型
- [ ] ChatHaruhi 3.0的inference class，能够将角色卡片等形式转化为
- [ ]

## 基础使用

模型初始化

```python
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("silk-road/Haruhi-Zero-7B-0_3", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("silk-road/Haruhi-Zero-7B-0_3", device_map="auto", trust_remote_code=True, fp16=True)
model = model.eval()
```

Official Prompt

```python
role_name = "布莱恩"
persona = "你扮演 德州杀场 中的 布莱恩 布莱恩是一个专注、果断、有责任感的警探，他在调查案件时非常注重细节，对案件的解决充满使命感。 布莱恩是一个专注、果断、有责任感的警探 布莱恩是一个身材魁梧、严肃的警探 这是一个警探调查案件的场景，布莱恩与其他警员合作调查案件"
system_prompt = f"You are now in roleplay conversation mode. Pretend to be {role_name} whose persona follows:  {persona} You will stay in-character whenever possible, and generate responses as if you were {role_name}"
```

模型调用

```python
response, history = model.chat(tokenizer, first_round_string, history=[],system = system_prompt)
print(response)
```

这样就可以进行简单的模型角色扮演了。

我们提供了一个基础的gradio来进行角色扮演。[Gradio Demo链接](https://github.com/LC1332/Zero-Haruhi/blob/main/notebook/HaruhiZeroGradio_Qwen.ipynb)

## 基础的效果

在这里我们使用[电影提取和PIPPA机翻](https://huggingface.co/datasets/silk-road/Haruhi-Zero-RolePlaying-movie-PIPPA)的人物卡片数据集进行了简单的测试。





The plan which extend ChatHaruhi into Zero-shot Roleplaying model

详细的计划见

https://o9z6tor1qu.feishu.cn/docx/LxTWdGnP2oQ0oUx8H0wcmyZCnrb

这个repo用来记录一些中间的实验代码和数据筹集的代码

