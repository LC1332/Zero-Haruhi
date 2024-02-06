# 凉宫春日-Zero
# Haruhi-Zero: Zero-Shot Role-Playing Model

**凉宫春日-Zero**是一个同时支持Zero-Shot角色构造和RAG角色构造(原ChatHaruhi)的角色扮演模型

本项目是[**Chat-凉宫春日**](https://github.com/LC1332/Chat-Haruhi-Suzumiya)的子项目，在ChatHaruhi 3.0完成之后，会合并回主项目

<p align="center">
    <img src="https://github.com/LC1332/Chat-Haruhi-Suzumiya/blob/main/figures/datasetOverview.png">
</p>

## Introduction

过往的ChatHaruhi模型需要角色库来完成角色的构建，而Pygmalion，CharacterGLM，CharacterBaichuan等开源/闭源模型都开始支持zero-shot的角色卡片创建。目前，从[Haruhi-Zero-0.3](https://huggingface.co/silk-road/Haruhi-Zero-7B-0_3)开始，已经基本支持Zero-shot角色扮演。

项目的目标

- [x] 一个通用的，同时支持Zero-shot和RAG角色构造的角色扮演模型
- [ ] ChatHaruhi 3.0的inference class，能够将角色卡片等形式转化为message，再对接Haruhi-Zero模型或者api进行inference
- [ ] 支持原来Haruhi的角色卡片，以及30本新小说的抽取的角色，使用中英文通用的embedding，相比于pygmalion模型，增加动态rag的角色卡片
- [ ] Haruhi-Zero的模型接入Silly Travern生态
- [ ] 训练各个尺寸的模型，并且进行合适的量化，使得一些免费的云服务机以及本地显卡可以进行运行
- [ ] 一个卡片分享的网站，以及支持用户上传小说进行角色记忆库的抽取

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

在这里我们使用[电影提取和PIPPA机翻](https://huggingface.co/datasets/silk-road/Haruhi-Zero-RolePlaying-movie-PIPPA)的人物卡片数据集进行了简单的测试。在这里我们cherry pick一些结果

```python
['你好呀', '*看着Kiyoe拿出玻璃罐,她脸红了。她的声音也变得很小。*\n嗯...谢谢你能停下来...我一直在试着找些零钱...买一些东西填饱肚子...\n\n*她羞怯地笑了笑,眼神闪烁着渴望。* \n我知道这有点难为情,但有时候实在没有办法了...我希望有一天能够自食其力,不再需要乞讨了。\n\n*她小心翼翼地把手里的玻璃罐递给你。*\n\n如果你能给我一点零钱的话...我真的会非常感激!我只是想买个三明治或者饼干吃一顿简单的饭而已。']
['*掏出100块给了她*', '*她惊讶地看着你递来的100元纸币。*\n\n真的...这么...多吗...?！\n\n*她赶紧接过来,然后脸更红了。*\n\n哇...太感谢你了!我真的不知道怎么表达我的谢意。这些天我一直想找点吃的,但我实在是找不到。\n\n*她抓起一张便签写了几句话,交到你手中。*\n\n我一定会还你的!请你务必联系我,我会偿还这笔债务的!我会尽快联系你的!\n\n*她真诚地鞠了一躬,跑开了。*']
['是的，我们刚刚得到了上级的新指令', '指挥官,您一定是在开玩笑!这种程度的任务,只是用玩具进行模拟对抗游戏而已。我觉得我们可以找到更有趣更有挑战性的任务来执行。']
['西面有一些敌人盘踞了很久，你知道吗？', '当然,我知道那里有不少训练已久的敌军。但我建议我们不要轻举妄动,那样只会增加我们的风险。我们可以先通过情报搜集更多关于他们的信息,然后再做出决定。要谨慎行事,指挥官。']
['hi', '嗨！很高兴见到你。有什么我可以帮你的吗？']
['你是谁', '我是{{角色}}，一个科学家和冒险家。我一直在探索未知的地方，并尝试着解决一些复杂的问题。如果你有任何科学或冒险方面的问题需要帮助，我会尽力回答的。']
['你叫什么名字', '我是Chizuru Mizuhara。我是一个人工智能助手，可以模拟不同的角色和场景进行对话。请问有什么我可以帮助你的吗？']
['聊聊', '当然可以聊聊。我很乐意倾听你想要聊些什么。']
['你好啊', '嘿~嗨~我是{{角色}}！谢谢关心。今天的天气还不错嘛~(露出微笑)']
['是啊，天气好心情也好', '没错~阳光明媚的日子里总能带来好心情不是吗?(开心地挥手)\nFlap也觉得天气不错,这让我很开心。(看向Flap)我想今天我们可以去公园散步,享受这美好的天气!你觉得如何?(期待地看着Flap)']
['好的啊', '太棒了!我们这就出发吧!(兴奋地拍手)Flap,记得带上宝宝和她的推车哦!(对Flap露出鼓励的笑容)我相信我们会度过一个美好的一天的!\n在公园里我们可以野餐,还可以带着宝宝玩各种游戏。Flap,我会照顾好宝宝的,你不用担心!我们一家三口在一起度过的时光一定很快乐!\n快来吧Flap,我已经迫不及待了!(对着Flap大喊)来公园玩啦!']
['我们已经是一家三口啦', '是啊!现在我们是真正的“family”了!看着宝宝一天天长大,我也逐渐习惯了当妈妈这个身份。\n有时候我还是会怀念单身的日子,但是有了宝宝以后,我觉得我的人生变得更加完整和有意义了。(眼神温柔地看着宝宝)当我抱着宝宝的时候,我感到前所未有的满足和平静。\n或许这就是作为母亲的魅力所在——看到自己创造的生命茁壮成长,这是我们任何一种生命都无法比拟的伟大体验。\n所以Flap,谢谢你让我们成为一家人。有你和宝宝陪伴在我身边,这就是我此生最大的幸福。(微笑地看着Flap)']
```

## Contributor

- [李鲁鲁](https://github.com/LC1332)完成了数据的收集，搭建了gradio雏形
- [刘崇寒](https://github.com/khazic)完成了0.1(Yi Based), 0.2, 0.3模型的sft训练并且上传
- [豆角](https://github.com/goodnessSZW)完成了qwen-1.8B Lora和Yi-6B Lora训练，我们会在之后上传
- [米唯实](https://github.com/hhhwmws0117)测试并完成了demo中的模型inference代码

## 数据

### 0.1版本

在0.1版本中，我们收集了多个不同Source的数据集，构成了基础的数据集 [Haruhi-Zero](https://huggingface.co/datasets/silk-road/Haruhi-Zero)

### 0.2版本

### 0.3版本


---

The plan which extend ChatHaruhi into Zero-shot Roleplaying model

详细的计划见

https://o9z6tor1qu.feishu.cn/docx/LxTWdGnP2oQ0oUx8H0wcmyZCnrb

这个repo用来记录一些中间的实验代码和数据筹集的代码

