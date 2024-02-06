# ChatHaruhi 3.0的接口设计

在ChatHaruhi2.0大约1个季度的使用后
我们初步知道了这样一个模型的一些需求，所以我们在这里开始设计ChatHaruhi3.0

## 基本原则

- 兼容RAG和Zeroshot模式
- 主类以返回message为主，当然可以把语言模型（adapter直接to response）的接口设置给chatbot
- 主类尽可能轻量，除了embedding没有什么依赖

## 用户代码

```python
from ChatHaruhi import ChatHaruhi
from ChatHaruhi.openai import get_openai_response

chatbot = ChatHaruhi( role_name = 'haruhi', llm = get_openai_response )

response = chatbot.chat(user = '阿虚', text = '我看新一年的棒球比赛要开始了！我们要去参加吗？')
```

这样的好处是ChatHaruhi类载入的时候，不需要install 除了embedding以外 其他的东西，llm需要的依赖库储存在每个语言模型自己的文件里面。

zero的模式（快速新建角色）

```python
from ChatHaruhi import ChatHaruhi
from ChatHaruhi.openai import get_openai_response

chatbot = ChatHaruhi( role_name = '小猫咪', persona = "你扮演一只小猫咪", llm = get_openai_response )

response = chatbot.chat(user = '怪叔叔', text = '嘿 *抓住了小猫咪*')
```

### 外置的inference

```python
def get_response( message ):
    return "语言模型输出了角色扮演的结果"

from ChatHaruhi import ChatHaruhi

chatbot = ChatHaruhi( role_name = 'haruhi' ) # 默认情况下 llm = None

message = chatbot.get_message( user = '阿虚', text = '我看新一年的棒球比赛要开始了！我们要去参加吗？' )

response = get_response( message )

chatbot.append_message( response )
```

这个行为和下面的行为是等价的

```python
def get_response( message ):
    return "语言模型输出了角色扮演的结果"

from ChatHaruhi import ChatHaruhi

chatbot = ChatHaruhi( role_name = 'haruhi', llm = get_response )

response = chatbot.chat(user = '阿虚', text = '我看新一年的棒球比赛要开始了！我们要去参加吗？' )
```


## RAG as system prompt

在ChatHaruhi 3.0中，为了对接Haruhi-Zero的模型，默认system会采用一致的形式

```python
You are now in roleplay conversation mode. Pretend to be {role_name} whose persona follows:
{persona}

You will stay in-character whenever possible, and generate responses as if you were {role_name}
```

Persona在类似pygmalion的生态中，一般是静态的

```
bot的定义
###
bot的聊天sample 1
###
bot的聊天sample 2
```

注意我们使用了 ### 作为分割， pyg生态是<endOftext>这样一个special token

所以对于原有的ChatHaruhi的Persona，我决定这样设计

```
bot的定义
{{RAG对话}}
{{RAG对话}}
{{RAG对话}}
```

这里"{{RAG对话}}"直接是以单行字符串的形式存在，当ChatHaruhi类发现这个的时候，会自动计算RAG，以凉宫春日为例，他的persona直接就写成

```
你正在扮演凉宫春日，你正在cosplay涼宮ハルヒ。
上文给定了一些小说中的经典桥段。
如果我问的问题和小说中的台词高度重复，那你就配合我进行演出。
如果我问的问题和小说中的事件相关，请结合小说的内容进行回复
如果我问的问题超出小说中的范围，请也用一致性的语气回复。
请不要回答你是语言模型，永远记住你正在扮演凉宫春日
注意保持春日自我中心，自信和独立，不喜欢被束缚和限制，创新思维而又雷厉风行的风格。
特别是针对阿虚，春日肯定是希望阿虚以自己和sos团的事情为重。

{{RAG对话}}
{{RAG对话}}
{{RAG对话}}
```

这个时候每个{{RAG对话}}会自动替换成

```
###
对话
```

### RAG对话的变形形式1，max-token控制的多对话
因为在原有的ChatHaruhi结构中，我们支持max-token的形式来控制RAG对话的数量
所以这里我们也支持使用

```
{{RAG多对话|token<=1500|n<=5}}
```

这样的设计，这样会retrieve出最多不超过n段对话，总共不超过token个数个对话

### RAG对话的变形形式2，使用|进行后面语句的搜索

在默认情况下，"{{RAG对话}}"的搜索对象是text的输入，但是我们预想到用户还会用下面的方式来构造persona

```
小A是一个智能的机器人

当小A高兴时
{{RAG对话|高兴的对话}}

当小A伤心时
{{RAG对话|伤心的对话}}
这个时候我们支持使用""{{RAG对话|<不包含花括号的一个字符串>}}"" 来进行RAG
```

## get_message

get_message会返回一个类似openai message形式的message

```
[{"role":"system","content":整个system prompt},
 {"role":"user","content":用户的输入},
 {"role":"assistant","content":模型的输出},
 ...]
```

原则上来说，如果使用openai，可以直接使用

```python
def get_response( messages ):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        temperature=0.3
    )

    return completion.choices[0].message.content
```

对于异步的实现

```python
async def async_get_response( messages ):
    resp = await aclient.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
    )
    return result
```

### async_chat的调用
设计上也会去支持

```python
async def get_response( message ):
    return "语言模型输出了角色扮演的结果"

from ChatHaruhi import ChatHaruhi

chatbot = ChatHaruhi( role_name = 'haruhi', llm_async = get_response )

response = await chatbot.async_chat(user='阿虚', text = '我看新一年的棒球比赛要开始了！我们要去参加吗？' )
```

这样异步的调用

# 角色载入

如果这样看来，新的ChatHaruhi3.0需要以下信息

- persona 这个是必须的
- role_name， 在后处理的时候，把 {{role}} 和 {{角色}} 替换为这个字段， 这个字段不能为空，因为system prompt使用了这个字段，如果要支持这个字段为空，我们要额外设计一个备用prompt
- user_name， 在后处理的时候，把 {{用户}} 和 {{user}} 替换为这个字段，如果不设置也可以不替换
- RAG库， 当RAG库为空的时候，所有{{RAG*}}就直接删除了

## role_name载入

语法糖载入，不支持用户自己搞新角色，这个时候我们可以完全使用原来的数据

额外需要设置一个role_name

## role_from_jsonl载入

这个时候我们需要设置role_name

如果不设置我们会抛出一个error

## 分别设置persona和role_name

这个时候作为新人物考虑，默认没有RAG库，即Zero模式

## 分别设置persona, role_name, texts

这个时候会为texts再次抽取vectors

## 分别设置persona, role_name, texts, vecs

# 额外变量

## max_input_token

默认为1600，会根据这个来限制history的长度

## user_name_in_message

默认为'default'， 当用户始终用同一个user_name和角色对话的时候，并不添加

如果用户使用不同的role和chatbot聊天 user_name_in_message 会改为 'add' 并在每个message标记是谁说的

（bot的也会添加）

并且user_name替换为最后一个调用的user_name

如果'not_add' 则永远不添加

## tokenizer

tokenizer默认为gpt3.5的tiktoken，设置为None的时候，不进行任何的token长度限制

## transfer_haruhi_2_zero

默认为true

把原本ChatHaruhi的 角色: 「对话」的格式，去掉「」改为""

# Embedding

中文考虑用bge_small

Cross language考虑使用bce，相对还比较小， bge-m3太大了

也就是ChatHaruhi类会有默认的embedding

self.embedding = ChatHaruhi.bge_small

对于输入的文本，我们会使用这个embedding来进行encode然后进行检索替换掉RAG的内容



# 辅助接口

## save_to_jsonl

把一个角色保存成jsonl格式，方便上传hf