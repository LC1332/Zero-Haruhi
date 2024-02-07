def tiktoken_counter( text ):
    # TODO 把这个实现为tiktoken 然后放到util
    return len(text)

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
                 token_counter = "default",
                 max_input_token = 1800
                 ):
        
        self.verbose = True if verbose is None or verbose else False

        if persona and role_name and stories and story_vecs and len(stories) == len(story_vecs):
            # 完全从外部设置，这个时候要求story_vecs和embedding的返回长度一致
            self.persona, self.role_name, self.user_name = persona, role_name, user_name
            self.db = self.build_db(stories, story_vecs)
        elif persona and role_name and stories:
            # 从stories中提取story_vecs，重新用self.embedding进行embedding
            story_vecs = self.extract_story_vecs(stories)
            self.persona, self.role_name, self.user_name = persona, role_name, user_name
            self.db = self.build_db(stories, story_vecs)
        elif role_from_hf:
            # 从hf加载role
            self.persona, new_role_name, self.stories, self.story_vecs = self.load_role_from_hf(role_from_hf)
            if new_role_name:
                self.role_name = new_role_name
            else:
                self.role_name = role_name
            self.user_name = user_name
            self.db = self.build_db(self.stories, self.story_vecs)
        elif role_from_jsonl:
            # 从jsonl加载role
            self.persona, new_role_name, self.stories, self.story_vecs = self.load_role_from_jsonl(role_from_jsonl)
            if new_role_name:
                self.role_name = new_role_name
            else:
                self.role_name = role_name
            self.user_name = user_name
            self.db = self.build_db(self.stories, self.story_vecs)
        elif persona and role_name:
            # 这个时候也就是说没有任何的RAG，
            self.persona, self.role_name, self.user_name = persona, role_name, user_name
            self.db = None
        elif role_name and self.check_sugar( role_name ):
            # 这个时候是sugar的role
            self.persona, self.role_name, self.user_name, self.db = self.load_role_from_sugar( role_name )
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
            self.token_counter = tiktoken_counter
        elif token_counter == None:
            self.token_counter = lambda x: 0
        else:
            self.token_counter = token_counter
            if self.verbose:
                print("user set costomized token_counter")

        self.max_input_token = max_input_token

        self.history = []

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
            return self.llm(message)
        
    async def async_chat(self, user, text):
        self.set_new_user(user)
        message = self.get_message(user, text)
        if self.llm_async:
            response = await self.llm_async(message)
            self.append_message(response)
            return self.llm_async(message)
        
    def parse_rag_from_persona(self, persona):
        #每个query_rag需要饱含
        # "n" 需要几个story
        # "max_token" 最多允许多少个token，如果-1则不限制
        # "query" 需要查询的内容，如果等同于"default"则替换为text
        # "lid" 需要替换的行，这里直接进行行替换，忽视行的其他内容

        print("parse_rag_from_persona")
        return [], self.token_counter(persona)
    
    def append_message( self, response , speaker = None ):
        if speaker is None:
            # 如果role是none，则认为是本角色{{role}}输出的句子
            self.history.append({"speaker":"{{user}}","content":response})
            # 叫speaker是为了和role进行区分
        else:
            self.history.append({"speaker":speaker,"content":response})
    
    def rag_retrieve( self, query, n, max_token, avoid_ids = [] ):
        # 返回一个rag_id的列表
        print("call rag_retrieve")
        return []
    
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
        print("call append_history_under_limit")

        # 从后往前计算token，不超过rest_limit,
        # 如果speaker是{{role}},则message的role是assistant

        return message

    def get_message(self, user, text):
        query_token = self.token_counter(text)

        # 首先获取需要多少个rag story
        query_rags, persona_token = self.parse_rag_from_persona( self.persona )
        #每个query_rag需要饱含
        # "n" 需要几个story
        # "max_token" 最多允许多少个token，如果-1则不限制
        # "query" 需要查询的内容，如果等同于"default"则替换为text
        # "lid" 需要替换的行，这里直接进行行替换，忽视行的其他内容

        rest_limit = self.max_input_token - persona_token - query_token

        rag_ids = self.rag_retrieve_all( query_rags, rest_limit )

        # 将rag_ids对应的故事 替换到persona中
        augmented_persona = self.augment_persona( self.persona, rag_ids, query_rags )

        system_prompt = self.package_system_prompt( self.role_name, augmented_persona )

        token_for_system = self.token_counter( system_prompt )

        rest_limit = self.max_input_token - token_for_system - query_token

        message = [{"role":"system","content":system_prompt}]

        message = self.append_history_under_limit( message, rest_limit )

        message.append({"role":"user","content":text})

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
                new_text += "###\n" + self.db.get_text(id) + "\n"
            new_text = new_text.strip()
            lines[lid] = new_text
        return "\n".join(lines)


    def load_role_from_hf(self, role_from_hf):
        # 从hf加载role
        return None
    
    def load_role_from_jsonl(self, role_from_jsonl):
        # 从jsonl加载role
        return None

    def extract_story_vecs(self, stories):
        # 从stories中提取story_vecs
        return None
    
    def build_db(self, stories, story_vecs):
        # db的构造函数
        return None