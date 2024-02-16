from .utils import base64_to_float_array, base64_to_string

def get_text_from_data( data ):
    if "text" in data:
        return data['text']
    elif "enc_text" in data:
        # from .utils import base64_to_string
        return base64_to_string( data['enc_text'] )
    else:
        print("warning! failed to get text from data ", data)
        return ""

def parse_rag(text):
    lines = text.split("\n")
    ans = []

    for i, line in enumerate(lines):
        if "{{RAG对话}}" in line:
            ans.append({"n": 1, "max_token": -1, "query": "default", "lid": i})
        elif "{{RAG对话|" in line:
            query_info = line.split("|")[1].rstrip("}}")
            ans.append({"n": 1, "max_token": -1, "query": query_info, "lid": i})
        elif "{{RAG多对话|" in line:
            parts = line.split("|")
            max_token = int(parts[1].split("<=")[1])
            max_n = int(parts[2].split("<=")[1].rstrip("}}"))
            ans.append({"n": max_n, "max_token": max_token, "query": "default", "lid": i})
            
    return ans

class ChatHaruhi:
    def __init__(self,
                 role_name = None,
                 user_name = None,
                 persona = None,
                 stories = None,
                 story_vecs = None,
                 role_from_hf = None,
                 role_from_jsonl = None,
                 llm = None, # 默认的message2response的函数
                 llm_async = None, # 默认的message2response的async函数
                 user_name_in_message = "default",
                 verbose = None,
                 embed_name = None,
                 embedding = None,
                 db = None,
                 token_counter = "default",
                 max_input_token = 1800,
                 max_len_story_haruhi = 1000,
                 max_story_n_haruhi = 5
                 ):

        self.verbose = True if verbose is None or verbose else False

        self.db = db

        self.embed_name = embed_name

        self.max_len_story_haruhi = max_len_story_haruhi # 这个设置只对过往Haruhi的sugar角色有效
        self.max_story_n_haruhi = max_story_n_haruhi # 这个设置只对过往Haruhi的sugar角色有效

        self.last_query_msg = None

        if embedding is None:
            self.embedding = self.set_embedding_with_name( embed_name )

        if persona and role_name and stories and story_vecs and len(stories) == len(story_vecs):
            # 完全从外部设置，这个时候要求story_vecs和embedding的返回长度一致
            self.persona, self.role_name, self.user_name = persona, role_name, user_name
            self.build_db(stories, story_vecs)
        elif persona and role_name and stories:
            # 从stories中提取story_vecs，重新用self.embedding进行embedding
            story_vecs = self.extract_story_vecs(stories)
            self.persona, self.role_name, self.user_name = persona, role_name, user_name
            self.build_db(stories, story_vecs)
        elif role_from_hf:
            # 从hf加载role
            self.persona, new_role_name, self.stories, self.story_vecs = self.load_role_from_hf(role_from_hf)
            if new_role_name:
                self.role_name = new_role_name
            else:
                self.role_name = role_name
            self.user_name = user_name
            self.build_db(self.stories, self.story_vecs)
        elif role_from_jsonl:
            # 从jsonl加载role
            self.persona, new_role_name, self.stories, self.story_vecs = self.load_role_from_jsonl(role_from_jsonl)
            if new_role_name:
                self.role_name = new_role_name
            else:
                self.role_name = role_name
            self.user_name = user_name
            self.build_db(self.stories, self.story_vecs)
        elif persona and role_name:
            # 这个时候也就是说没有任何的RAG，
            self.persona, self.role_name, self.user_name = persona, role_name, user_name
            self.db = None
        elif role_name and self.check_sugar( role_name ):
            # 这个时候是sugar的role
            self.persona, self.role_name, self.stories, self.story_vecs = self.load_role_from_sugar( role_name )
            self.build_db(self.stories, self.story_vecs)
            self.add_rag_prompt_after_persona()
        else:
            raise ValueError("persona和role_name必须同时设置，或者role_name是ChatHaruhi的预设人物")

        self.llm, self.llm_async = llm, llm_async
        if not self.llm and self.verbose:
            print("warning, llm没有设置，仅get_message起作用，调用chat将回复idle message")

        self.user_name_in_message = user_name_in_message
        self.previous_user_pool = set([user_name]) if user_name else set()
        self.current_user_name_in_message = user_name_in_message.lower() == "add"

        self.idle_message = "idel message, you see this because self.llm has not been set."

        if token_counter.lower() == "default":
            # TODO change load from util
            from .utils import tiktoken_counter
            self.token_counter = tiktoken_counter
        elif token_counter == None:
            self.token_counter = lambda x: 0
        else:
            self.token_counter = token_counter
            if self.verbose:
                print("user set costomized token_counter")

        self.max_input_token = max_input_token

        self.history = []

    def check_sugar(self, role_name):
        from .sugar_map import sugar_role_names, enname2zhname
        return role_name in sugar_role_names

    def load_role_from_sugar(self, role_name):
        from .sugar_map import sugar_role_names, enname2zhname
        en_role_name = sugar_role_names[role_name]
        new_role_name = enname2zhname[en_role_name]
        role_from_hf = "silk-road/ChatHaruhi-RolePlaying/" + en_role_name
        persona, _, stories, story_vecs = self.load_role_from_hf(role_from_hf)

        return persona, new_role_name, stories, story_vecs

    def add_rag_prompt_after_persona( self ):
        rag_sentence = "{{RAG多对话|token<=" + str(self.max_len_story_haruhi) + "|n<=" + str(self.max_story_n_haruhi) + "}}"
        self.persona += "Classic scenes for the role are as follows:\n" + rag_sentence + "\n"

    def set_embedding_with_name(self, embed_name):
        if embed_name is None or embed_name == "bge_zh":
            from .embeddings import get_bge_zh_embedding
            self.embed_name = "bge_zh"
            return get_bge_zh_embedding
        elif embed_name == "foo":
            from .embeddings import foo_embedding
            return foo_embedding
        elif embed_name == "bce":
            from .embeddings import foo_bce
            return foo_bce
        elif embed_name == "openai" or embed_name == "luotuo_openai":
            from .embeddings import foo_openai
            return foo_openai

    def set_new_user(self, user):
        if len(self.previous_user_pool) > 0 and user not in self.previous_user_pool:
            if self.user_name_in_message.lower() == "default":
                if self.verbose:
                    print(f'new user {user} included in conversation')
                self.current_user_name_in_message = True
        self.user_name = user
        self.previous_user_pool.add(user)

    def chat(self, user, text):
        self.set_new_user(user)
        message = self.get_message(user, text)
        if self.llm:
            response = self.llm(message)
            self.append_message(response)
            return response
        return None

    async def async_chat(self, user, text):
        self.set_new_user(user)
        message = self.get_message(user, text)
        if self.llm_async:
            response = await self.llm_async(message)
            self.append_message(response)
            return response

    def parse_rag_from_persona(self, persona, text = None):
        #每个query_rag需要饱含
        # "n" 需要几个story
        # "max_token" 最多允许多少个token，如果-1则不限制
        # "query" 需要查询的内容，如果等同于"default"则替换为text
        # "lid" 需要替换的行，这里直接进行行替换，忽视行的其他内容

        query_rags = parse_rag( persona )

        if text is not None:
            for rag in query_rags:
                if rag['query'] == "default":
                    rag['query'] = text

        return query_rags, self.token_counter(persona)

    def append_message( self, response , speaker = None ):
        if self.last_query_msg is not None:
            self.history.append(self.last_query_msg)
            self.last_query_msg = None

        if speaker is None:
            # 如果role是none，则认为是本角色{{role}}输出的句子
            self.history.append({"speaker":"{{role}}","content":response})
            # 叫speaker是为了和role进行区分
        else:
            self.history.append({"speaker":speaker,"content":response})

    def check_recompute_stories_token(self):
        return len(self.db.metas) == len(self.db.stories)
    
    def recompute_stories_token(self):
        self.db.metas = [self.token_counter(story) for story in self.db.stories]

    def rag_retrieve( self, query, n, max_token, avoid_ids = [] ):
        # 返回一个rag_id的列表
        query_vec = self.embedding(query)

        self.db.clean_flag()
        self.db.disable_story_with_ids( avoid_ids )
        
        retrieved_ids = self.db.search( query_vec, n )

        if self.check_recompute_stories_token():
            self.recompute_stories_token()

        sum_token = 0

        ans = []

        for i in range(0, len(retrieved_ids)):
            if i == 0:
                sum_token += self.db.metas[retrieved_ids[i]]
                ans.append(retrieved_ids[i])
                continue
            else:
                sum_token += self.db.metas[retrieved_ids[i]]
                if sum_token <= max_token:
                    ans.append(retrieved_ids[i])
                else:
                    break
                
        return ans


    def rag_retrieve_all( self, query_rags, rest_limit ):
        # 返回一个rag_ids的列表
        retrieved_ids = []
        rag_ids = []

        for query_rag in query_rags:
            query = query_rag['query']
            n = query_rag['n']
            max_token = rest_limit
            if rest_limit > query_rag['max_token'] and query_rag['max_token'] > 0:
                max_token = query_rag['max_token']

            rag_id = self.rag_retrieve( query, n, max_token, avoid_ids = retrieved_ids )
            rag_ids.append( rag_id )
            retrieved_ids += rag_id

        return rag_ids

    def append_history_under_limit(self, message, rest_limit):
        # 返回一个messages的列表
        # print("call append history_under_limit")
        # 从后往前计算token，不超过rest limit，
        # 如果speaker是{{role}J,则message的role是assistant
        current_limit = rest_limit

        history_list = []

        for item in reversed(self.history):
            current_token = self.token_counter(item['content'])
            current_limit -= current_token
            if current_limit < 0:
                break
            else:
                history_list.append(item)

        history_list = list(reversed(history_list))

        # TODO: 之后为了解决多人对话，这了content还会额外增加speaker: content这样的信息

        for item in history_list:
            if item['speaker'] == "{{role}}":
                message.append({"role":"assistant","content":item['content']})
            else:
                message.append({"role":"user","content":item['content']})
        
        return message

    def get_message(self, user, text):
        query_token = self.token_counter(text)

        # 首先获取需要多少个rag story
        query_rags, persona_token = self.parse_rag_from_persona( self.persona, text )
        #每个query_rag需要饱含
        # "n" 需要几个story
        # "max_token" 最多允许多少个token，如果-1则不限制
        # "query" 需要查询的内容，如果等同于"default"则替换为text
        # "lid" 需要替换的行，这里直接进行行替换，忽视行的其他内容

        

        rest_limit = self.max_input_token - persona_token - query_token

        if self.verbose:
            print(f"query_rags: {query_rags} rest_limit = { rest_limit }")

        rag_ids = self.rag_retrieve_all( query_rags, rest_limit )

        # 将rag_ids对应的故事 替换到persona中
        augmented_persona = self.augment_persona( self.persona, rag_ids, query_rags )

        system_prompt = self.package_system_prompt( self.role_name, augmented_persona )

        token_for_system = self.token_counter( system_prompt )

        rest_limit = self.max_input_token - token_for_system - query_token

        message = [{"role":"system","content":system_prompt}]

        message = self.append_history_under_limit( message, rest_limit )

        # TODO: 之后为了解决多人对话，这了content还会额外增加speaker: content这样的信息

        message.append({"role":"user","content":text})

        self.last_query_msg = {"speaker":user,"content":text}

        return message

    def package_system_prompt(self, role_name, augmented_persona):
        bot_name = role_name
        return f"""You are now in roleplay conversation mode. Pretend to be {bot_name} whose persona follows:
{augmented_persona}

You will stay in-character whenever possible, and generate responses as if you were {bot_name}"""


    def augment_persona(self, persona, rag_ids, query_rags):
        lines = persona.split("\n")
        for rag_id, query_rag in zip(rag_ids, query_rags):
            lid = query_rag['lid']
            new_text = ""
            for id in rag_id:
                new_text += "###\n" + self.db.stories[id].strip() + "\n"
            new_text = new_text.strip()
            lines[lid] = new_text
        return "\n".join(lines)

    def load_role_from_jsonl( self, role_from_jsonl ):
        import json
        datas = []
        with open(role_from_jsonl, 'r') as f:
            for line in f:
                try:
                    datas.append(json.loads(line))
                except:
                    continue

        column_name = ""

        from .embeddings import embedname2columnname

        if self.embed_name in embedname2columnname:
            column_name = embedname2columnname[self.embed_name]
        else:
            print('warning! unkown embedding name ', self.embed_name ,' while loading role')
            column_name = 'luotuo_openai'

        stories, story_vecs, persona = self.extract_text_vec_from_datas(datas, column_name)

        return persona, None, stories, story_vecs


    def load_role_from_hf(self, role_from_hf):
        # 从hf加载role
        # self.persona, new_role_name, self.stories, self.story_vecs = self.load_role_from_hf(role_from_hf)

        from datasets import load_dataset

        if role_from_hf.count("/") == 1:
            dataset = load_dataset(role_from_hf)
            datas = dataset["train"]
        elif role_from_hf.count("/") >= 2:
            split_index = role_from_hf.index('/')
            second_split_index = role_from_hf.index('/', split_index+1)
            dataset_name = role_from_hf[:second_split_index]
            split_name = role_from_hf[second_split_index+1:]

            fname = split_name + '.jsonl'
            dataset = load_dataset(dataset_name,data_files={'train':fname})
            datas = dataset["train"]

        column_name = ""

        from .embeddings import embedname2columnname

        if self.embed_name in embedname2columnname:
            column_name = embedname2columnname[self.embed_name]
        else:
            print('warning! unkown embedding name ', self.embed_name ,' while loading role')
            column_name = 'luotuo_openai'

        stories, story_vecs, persona = self.extract_text_vec_from_datas(datas, column_name)

        return persona, None, stories, story_vecs

    def extract_text_vec_from_datas(self, datas, column_name):
        # 从datas中提取text和vec
        # extract text and vec from huggingface dataset
        # return texts, vecs
        # from .utils import base64_to_float_array

        texts = []
        vecs = []
        for data in datas:
            if data[column_name] == 'system_prompt':
                system_prompt = get_text_from_data( data )
            elif data[column_name] == 'config':
                pass
            else:
                vec = base64_to_float_array( data[column_name] )
                text = get_text_from_data( data )
                vecs.append( vec )
                texts.append( text )
        return texts, vecs, system_prompt

    def extract_story_vecs(self, stories):
        # 从stories中提取story_vecs

        if self.verbose:
            print(f"re-extract vector for {len(stories)} stories")
        
        story_vecs = []

        from .embeddings import embedshortname2model_name
        from .embeddings import device

        if device.type != "cpu" and self.embed_name in embedshortname2model_name:
            # model_name = "BAAI/bge-small-zh-v1.5"
            model_name = embedshortname2model_name[self.embed_name]

            from .utils import get_general_embeddings_safe
            story_vecs = get_general_embeddings_safe( stories, model_name = model_name )
            # 使用batch的方式进行embedding，非常快
        else:
            from tqdm import tqdm
            for story in tqdm(stories):
                story_vecs.append(self.embedding(story))

        return story_vecs

    def build_db(self, stories, story_vecs):
        # db的构造函数
        if self.db is None:
            from .NaiveDB import NaiveDB
            self.db = NaiveDB()
        self.db.build_db(stories, story_vecs)

