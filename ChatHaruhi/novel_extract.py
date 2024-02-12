



import re

def extract_speaker(text):
    # 使用正则表达式匹配文本开头的 "<name> :" 格式，并捕获冒号后面的内容
    match = re.match(r'^([^:]+) :(.*)', text)
    if match:
        return (match.group(1), match.group(2).strip())  # 返回匹配到的name部分和冒号后面的内容作为元组
    else:
        return None, text  # 如果不匹配，返回None和原始文本


def get_line_recall(query, line):
    # 获得query中每个汉字在 line 中的recall
    if not query or not line:
        return 0
    line_set = set(line)
    return sum(char in line_set for char in query) / len(query)


def get_max_recall_in_lines(query, lines):
    recall_values = [(get_line_recall(query, line), i) for i, line in enumerate(lines)]
    return max(recall_values, default=(-1, -1), key=lambda x: x[0])

def extract_dialogues_from_response(text):
    # Split the text into lines
    lines = text.split('\n')

    # Initialize an empty list to store the extracted dialogues
    extracted_dialogues = []

    valid_said_by = ["said by", "thought by", "described by", "from"]

    # Iterate through each line
    for line in lines:
        # Split the line by '|' and strip whitespace from each part
        parts = [part.strip() for part in line.split('|')]

        # Check if the line has 4 parts and the third part is 'said by'
        if len(parts) == 3:
            # Extract the dialogue and speaker, and add to the list
            if parts[2] == "speaker":
                continue

            if parts[1].strip().lower() not in valid_said_by:
                continue

            dialogue_dict = {
                'dialogue': parts[0],
                'speaker': parts[2],
                "said_by": parts[1]
            }
            extracted_dialogues.append(dialogue_dict)

    return extracted_dialogues


def extract_dialogues_from_glm_response(text):
    # Split the text into lines
    lines = text.split('\n')

    # Initialize an empty list to store the extracted dialogues
    extracted_dialogues = []

    valid_said_by = ["said by", "thought by", "described by", "from"]

    # Iterate through each line
    for line in lines:
        # Split the line by '|' and strip whitespace from each part
        parts = [part.strip() for part in line.split('|')]

        # Check if the line has 4 parts and the third part is 'said by'
        if len(parts) == 4:
            # Extract the dialogue and speaker, and add to the list
            if parts[3] == "speaker":
                continue

            if parts[2].strip().lower() not in valid_said_by:
                continue

            try:
                id_num = int(parts[0])
            except ValueError:
                id_num = id

            dialogue_dict = {
                'id': id_num,
                'dialogue': parts[1],
                'speaker': parts[3],
                "said_by": parts[2]
            }
            extracted_dialogues.append(dialogue_dict)

    return extracted_dialogues


def has_dialogue_sentences(text: str) -> int:
    # 定义成对的引号
    paired_quotes = [
        ("“", "”"),
        ("‘", "’"),
        ("「", "」")
    ]
    # 定义符号列表（包括全角和半角的逗号和句号）
    symbols = ['。', '!', '?', '*', '.', '？', '！', '"', '”', ',', '~', ')', '）', '…', ']', '♪','，']

    # 检查成对引号内的内容
    for start_quote, end_quote in paired_quotes:
        start_index = text.find(start_quote)
        while start_index != -1:
            end_index = text.find(end_quote, start_index + 1)
            if end_index != -1:
                quote_content = text[start_index + 1:end_index]
                # 检查引号内的内容是否符合条件
                if any(symbol in quote_content for symbol in symbols) or len(quote_content) >= 10:
                    return 2  # 成对引号内有符号或长度>=10
                start_index = text.find(start_quote, end_index + 1)
            else:
                break

    # 检查双引号'"'
    double_quotes_indices = [i for i, char in enumerate(text) if char == '"']
    if len(double_quotes_indices) % 2 == 0:  # 必须是偶数个双引号
        for i in range(0, len(double_quotes_indices), 2):
            start_index, end_index = double_quotes_indices[i], double_quotes_indices[i+1]
            quote_content = text[start_index+1:end_index]
            # 检查引号内的内容是否含有符号
            if any(symbol in quote_content for symbol in symbols):
                return 1  # 双引号内有符号

    return 0  # 没有符合条件的对话型句子

def replace_recalled_dialogue( raw_text, response_text ):
    dialogues = extract_dialogues_from_response( response_text )

    lines = raw_text.split("\n")

    lines = [line.strip().strip("\u3000") for line in lines]

    recall_flag = [ False for line in lines ]
    line2ids = [ [] for line in lines ]

    for id, dialogue in enumerate(dialogues):
        dialogue_text = dialogue['dialogue']
        remove_symbol_text = dialogue_text.replace("*","").replace('"',"")

        recall, lid = get_max_recall_in_lines( remove_symbol_text, lines )

        if recall > 0.3:
            recall_flag[lid] = True
            line2ids[lid].append(id)

    new_text = ""

    for lid, line in enumerate(lines):
        if recall_flag[lid]:
            if len(line2ids[lid]) == 1 and ("未知" in dialogues[0]['speaker'] or dialogues[0]['speaker'].strip() == ""):
                new_text += line + "\n"
                continue

            for dia_id in line2ids[lid]:
                speaker = dialogues[dia_id]['speaker']
                dialogue = dialogues[dia_id]['dialogue']
                dialogue = dialogue.replace('"',"").replace('“',"").replace('”',"")
                new_text += speaker + " : " + dialogue + "\n"
        else:
            new_text += line + "\n"

    return new_text.strip()



            