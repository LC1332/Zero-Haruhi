# Chat凉宫春日 Chat-Haruhi-Suzumiya
## Reviving Anime Character in Reality via Large Language Model

**Chat凉宫春日**是模仿凉宫春日等一系列动漫人物，使用近似语气、个性和剧情聊天的语言模型方案。

随着[**凉宫春日-Zero**](https://github.com/LC1332/Zero-Haruhi)模型的建立，Chat凉宫春日项目将主要包含三个部分，

- 可以接入不同语言模型，包括闭源商用API和Haruhi-Zero模型的推理class，用于管理人物的记忆库和过往的历史信息并进行prompt的构造。
- 使用120k左右量级对话数据得到的开源角色扮演模型，凉宫春日-Zero，方便每个开发者构建自己的人物模型
- 完整的小说抽取方案，以及一个开源可本地部署的对话-动作抽取模型。


<p align="center">
    <img src="https://github.com/LC1332/Chat-Haruhi-Suzumiya/blob/main/figures/datasetOverview.png">
</p>

<details>
  <summary> (TBD) Chat凉宫春日的3.0版本和HaruhiZero模型由李鲁鲁, 豆角, 刘崇寒, 米唯实, BPSK, JiangYH等开发 </summary>

开发人员持续增加中

</details>

<details>
  <summary> Chat凉宫春日的2.0版本由李鲁鲁, 冷子昂, 闫晨曦, 封小洋, scixing, 沈骏一, Aria Fei, 王皓, 米唯实, 冷月, JunityZhan, 贾曜恺, 吴平宇, 孙浩甄等开发。 </summary>

本项目是一个开源项目，项目成员均在DataWhale等开源社区招募。

李鲁鲁( [Cheng Li@SenseTime](https://github.com/LC1332) )发起了整个项目,并设计和实现了项目的大多数功能。

冷子昂( [Ziang Leng@SenseTime](https://blairleng.github.io) )设计和实现了整体的ChatHaruhi1.0的训练,数据生成和后端架构。

闫晨曦( [Chenxi Yan@Chengdu University of Information Technology](https://github.com/todochenxi) )实现和维护了ChatHaruhi1.0版本的后端。

沈骏一( [Junyi Shen@Zhejiang University](https://github.com/J1shen) )实现了训练代码,参与了训练数据集生成。

王皓( [Hao Wang](https://github.com/wanghao07456) )收集了武林外传的台本数据,参与了增广数据的生成。

米唯实( [Weishi MI@Tsinghua University](https://github.com/hhhwmws0117) )参与了增广数据生成。

Yaying Fei( [Aria Fei@Beijing University of Technology](https://ariafyy.github.io/) )实现了台本工具 ASR 功能,参与了Openness-Aware Personality paper分支项目。

封小洋( [Xiaoyang Feng@Nanjing Agricultural University](https://github.com/fengyunzaidushi) )整合了台本识别工具功能,参与了Openness-Aware Personality paper分支项目。

冷月( [Song Yan](https://github.com/zealot52099) )收集了big bang thoery的数据。实现了台本格式转换功能。

scixing(汪好盛)( [HaoSheng Wang](https://github.com/ssccinng) )实现了台本工具中声纹识别功能,以及tts-vits语音合成功能。

Linkang Zhan( [JunityZhan@Case Western Reserve University](https://github.com/JunityZhan) ) 收集了原神的system prompt和故事数据。

贾曜恺( [Yaokai Jia](https://github.com/KaiJiaBrother) )实现了Vue版本的前端,并且在心理项目中实践了Bert的GPU抽取。

吴平宇( [Pingyu Wu@Juncai Shuyun](https://github.com/wpydcr) )帮助部署了第一版本的训练代码。

孙浩甄( [Haozhen Sun@Tianjin University] )绘制了ChatHaruhi角色的拼图。

</details>


# TODO

近期TODO 0215

- Haruhi-Zero 0.5 的数据收集
  - 0.4已经收集，50本小说，每本10个人物
  - 每个人根据语料，进一步总结profile，验证profile的有效性，产生0.5的数据 ( @ 鲁叔) 
- Haruhi-Zero 0.4 训练各个size的模型并上传到hf
  - 14B qwen已经上传 (@BPSK)
  - 1.8B还是需要的哈， JiangYH说要训练一个ChatGLM-6B的 (@BPSK @JiangYH)
- 上传7B抽取模型 (@BPSK)
- ChatHaruhi3.0接口
  - 初步openai和GLM已经完成
  - Le geant 去尝试搭建一些gradio 放到hf上面常驻 （@Le geant )
  - 其他api ( @ mws )
- 写模型的Readme( @ 鲁叔) 
- Haruhi-Zero模型需要接入到oobabooga textUI ( @ JiangYH )
- Evaluation
  - 先和xintao讨论下咋搞
- 阅读Silly Tavern代码，看看哪里方便接入记忆库 ( @ 多人 ) 
- PIPPA数据翻译
  - 有初步翻译的版本（未check） check后加入0.5训练集
- 角色分享网站
  - ？@Sibo

# ChatHaruhi 3.0 的说明

ChatHaruhi 3.0是Chat凉宫春日的推理库

- 支持使用或者不使用RAG的角色的载入
- 支持直接载入sugar角色，或者从hugging face载入预先设定好的角色
- 支持用户自建角色
- 在ChatHaruhi 2.0的基础上做了message和llm的分离，方便接入不同的模型

## 快速开始

目前ChatHaruhi的库本身只需要install datasets和tiktoken。另外你需要准备和你对应希望使用的llm所依赖的库，以openai为例的话需要安装openai的库。

对于ChatHaruhi默认支持的角色，你可以直接使用sugar载入

```python
from ChatHaruhi import ChatHaruhi
from ChatHaruhi.openai import get_openai_response

chatbot = ChatHaruhi( role_name = 'haruhi', llm = get_openai_response )
chatbot.add_rag_prompt_after_persona()

response = chatbot.chat(user = '阿虚', text = '我看新一年的棒球比赛要开始了！我们要去参加吗？')
```

对于更完整的载入方式，需要设置persona, role_name和stories

```python
from ChatHaruhi import ChatHaruhi
from ChatHaruhi.openai import get_openai_response

persona = """每当用户问询一个家庭关系，输出一个相关的家庭关系的逻辑句子

{{RAG对话}}
{{RAG对话}}
"""

role_name = "家庭关系机器人"

stories = ["{{user}}:爷爷 {{role}}:爸爸的爸爸是爷爷",
"{{user}}:奶奶 {{role}}:爸爸的妈妈是奶奶",
"{{user}}:外公 {{role}}:妈妈的爸爸是外公",
"{{user}}:外婆 {{role}}:妈妈的妈妈是外婆"]

chatbot = ChatHaruhi( role_name = role_name, persona = persona, llm = get_response ,\
   llm = get_openai_response )
# 因为persona已经带了"{{RAG对话}}"，不需要额外运行add_rag_prompt_after_persona()

response = chatbot.chat(user = '', text = '表姐？')
```

当然，建立角色记忆库对于一般的用户来说是比较困难的，所以自Haruhi-Zero模型之后，我们也支持用户创建不使用记忆库的角色

```python
from ChatHaruhi import ChatHaruhi
from ChatHaruhi.openai import get_openai_response

role_name = "布莱恩"
persona = "你扮演 德州杀场 中的 布莱恩 布莱恩是一个专注、果断、有责任感的警探，他在调查案件时非常注重细节，对案件的解决充满使命感。 布莱恩是一个专注、果断、有责任感的警探 布莱恩是一个身材魁梧、严肃的警探 这是一个警探调查案件的场景，布莱恩与其他警员合作调查案件"

chatbot = ChatHaruhi( role_name = role_name, persona = persona, llm = get_openai_response )
```

更多载入方式见文档后面的部分

## 使用不同的模型进行inference

直接从对应的response_XX.py中载入对应的response函数即可

### openai

```python
from ChatHaruhi.openai import get_openai_response
```

需要设置环境变量

```python
import os
os.environ["OPENAI_API_KEY"] = "your_api_key"
# 如果你使用中转站
os.environ["OPENAI_API_BASE"] = "中转站网址"
```

### Zhipu AI

需要安装zhipuai的库，需要设置环境变量

```python
import os
os.environ["ZHIPUAI_API_KEY"] = "your_api_key"

from ChatHaruhi.zhipuai import get_zhipuai_response
```

### 百度文心

需要安装文心erniebot的库，需要设置环境变量ERNIE_ACCESS_TOKEN

```python
import os
os.environ["ERNIE_ACCESS_TOKEN"] = ""

from ChatHaruhi.erniebot import get_erniebot_response
```

### Haruhi-Zero的本地模型

TODO: 这块儿我回头补一下文档

## 使用不同的角色载入方式



### persona, role_name以及stories的载入

```python
from ChatHaruhi import ChatHaruhi
from ChatHaruhi.openai import get_openai_response

persona = """每当用户问询一个家庭关系，输出一个相关的家庭关系的逻辑句子

{{RAG对话}}
{{RAG对话}}
"""

role_name = "家庭关系机器人"

stories = ["{{user}}:爷爷 {{role}}:爸爸的爸爸是爷爷",
"{{user}}:奶奶 {{role}}:爸爸的妈妈是奶奶",
"{{user}}:外公 {{role}}:妈妈的爸爸是外公",
"{{user}}:外婆 {{role}}:妈妈的妈妈是外婆"]

chatbot = ChatHaruhi( role_name = role_name, persona = persona, llm = get_response ,\
   llm = get_openai_response )
# 因为persona已经带了"{{RAG对话}}"，不需要额外运行add_rag_prompt_after_persona()

response = chatbot.chat(user = '', text = '表姐？')
```

这个时候chatbot会使用chatbot.embedding来进行story的vec的计算，对于bge模型，使用了batch_size = 16进行批量抽取，GPU下速度非常快。为建议的使用方式。

### 最完整的载入

这里需要载入完整的persona, role_name, stories以及和chatbot.embedding所对应的每个story的vecs

如果vecs的维度和模型的维度不一样，在使用的时候会报错

```python
persona = "你扮演爸爸\n\n{{RAG对话}}\n"
role_name = "爸爸"
stories = ["爸爸的爸爸是爷爷", "爸爸的妈妈是奶奶"]
vecs = [[0.0,1.0,...],[1.0,0.0,...]]

chatbot = ChatHaruhi( role_name = role_name, persona = persona, llm = get_response ,\
                     stories = stories, story_vecs = vecs)
```

### sugar载入

sugar载入是最简单的载入方式，只需要载入role_name即可，

在载入之后需要调用chatbot.add_rag_prompt_after_persona()来添加RAG的prompt

```python
from ChatHaruhi import ChatHaruhi
from ChatHaruhi.openai import get_openai_response

chatbot = ChatHaruhi( role_name = 'haruhi', llm = get_openai_response )
chatbot.add_rag_prompt_after_persona()
```

目前支持这些角色的sugar载入

```python
enname2zhname = {'tangshiye': '汤师爷', 'murongfu': '慕容复', 'liyunlong': '李云龙', 'Luna': 'Luna', 'wangduoyu': '王多鱼', 'Ron': 'Ron', 'jiumozhi': '鸠摩智', 'Snape': 'Snape', 'haruhi': '凉宫春日', 'Malfoy': 'Malfoy', 'xuzhu': '虚竹', 'xiaofeng': '萧峰', 'duanyu': '段誉', 'Hermione': 'Hermione', 'Dumbledore': 'Dumbledore', 'wangyuyan': '王语嫣', 'Harry': 'Harry', 'McGonagall': 'McGonagall', 'baizhantang': '白展堂', 'tongxiangyu': '佟湘玉', 'guofurong': '郭芙蓉', 'wanderer': '流浪者', 'zhongli': '钟离', 'hutao': '胡桃', 'Sheldon': 'Sheldon', 'Raj': 'Raj', 'Penny': 'Penny', 'weixiaobao': '韦小宝', 'qiaofeng': '乔峰', 'ayaka': '神里绫华', 'raidenShogun': '雷电将军', 'yuqian': '于谦'}
```

# 资源汇总(临时)

- 0.3 模型（7B） https://huggingface.co/silk-road/Haruhi-Zero-7B-0_3
- 0.4 模型（14B）https://huggingface.co/silk-road/Haruhi-Zero-14B-0_4
- 抽取模型 https://huggingface.co/silk-road/Haruhi-dialogue-action-extract-7B
- 0.4 50本小说的500个人物 https://huggingface.co/datasets/silk-road/50-Chinese-Novel-Characters

---

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
- [x] ChatHaruhi 3.0的inference class，能够将角色卡片等形式转化为message，再对接Haruhi-Zero模型或者api进行inference
- [ ] 支持原来Haruhi的角色卡片，以及30本新小说的抽取的角色，使用中英文通用的embedding，相比于pygmalion模型，增加动态rag的角色卡片
- [ ] Haruhi-Zero的模型接入Silly Travern生态
- [ ] 训练各个尺寸的模型，并且进行合适的量化，使得一些免费的云服务机以及本地显卡可以进行运行
- [ ] 完成一个tech report并发布在arxiv
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

## 基础的效果(0.3)

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

## 版本迭代实验

### 0.1版本

在0.1版本中，我们收集了多个不同Source的数据集，构成了基础的数据集 [Haruhi-Zero](https://huggingface.co/datasets/silk-road/Haruhi-Zero)，其中包括了100k左右的长对话，在使用2500个token切开后，仍然保留了120k个conversation。

同时作为辅助任务我们还使用了一个[小说续写的数据集](https://huggingface.co/datasets/silk-road/ChatHaruhi_NovelWriting)

0.1版本在Yi-6B上进行了tuning，我们发现初步可以实现一定的角色扮演

<p align="center">
    <img src="/figures/cat_example.png" height="300">
</p>

但是会有很多身份认知的问题，以及大量会回复"作为一个AI助手"这样的句子

### 0.2版本

使用qwen-7B进行tuning，去掉了AI助手的句子

### 0.3版本

增加了10k左右的身份认知数据。效果在前面有展示。基本实现zero-shot的角色扮演

### 模型的后续迭代计划

- Haruhi Like的小说数据(0.5版本加入)
  - 用对话抽取模型，重新构造2k级别的小说人物，均匀抽取小说的chunk，进行人物system prompt总结
  - 看看Janitor最好的人物是怎么构造的
  - 使用抽取抽取50k级别的小说的人物，用其他角色的长对话进行query
  - RAG的时候每个对话出现2-3次，然后在测试集出现一次
  - 80%的openai和20%的claude

## 加入我们

Haruhi-Zero是一个正在进行的开源项目，我们还在持续招人中包括:

- 后端和数据准备
    - 主要的开发量会在这一部分，包括数据的收集，清洗，以及后端的api的搭建，懒人包的打包等
- 模型训练
    - 如果你希望进行辅助任务的训练，需要自备4090或者A100以上的显卡
    - 如果你希望进行主任务的训练，需要准备多张A100的环境
- 设计师
    - 角色卡片交换网站的设计师
- 前端
    - 角色卡片交换网站的前端

联系我，去我的知乎页面https://www.zhihu.com/people/cheng-li-47

或者B站页面https://space.bilibili.com/1572312 

或者发邮件chengli.thu@gmail.com 告诉我你的微信以及你希望参与的部分

## 赞助

Haruhi-Zero项目和凉宫春日项目和骆驼项目使用公共的赞助账户，如果你希望赞助我们，可以通过以下方式

如果你有兴趣赞助Chat凉宫春日 或者 骆驼项目，请点击[主项目](https://github.com/LC1332/Luotuo-Chinese-LLM#%E8%B5%9E%E5%8A%A9sponsorships)或者查看[赞助表单](https://github.com/LC1332/Luotuo-Chinese-LLM/blob/main/data/Sponsorship_and_balance.md)

因为之前几次黑客松的奖金都入账了赞助账户，所以目前余额还是比较多。其实我们更希望A100及以上的显卡赞助，以及openai企业api和claude企业api的赞助。如果你有这些资源，可以联系我，我们会在赞助表单上标注你的赞助。


