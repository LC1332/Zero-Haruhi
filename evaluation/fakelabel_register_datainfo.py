import os
import sys
import json


path = sys.argv[1]
paths = path.split(".")
file  = paths[-2].split("/")[-1] #文件名
paths[-2] +='_fakelabel'
path1 = '.'.join(paths) #tofl


f = open(path, 'r')

results = []

for data in f:
    content = json.loads(data)
    conversations = content['conversations'] 
    if conversations[-1]['from'] != 'gpt':
        content['conversations'].append({'from':'gpt', 'value':'1&&&&'})

    results.append(content)



with open(path1,'w') as to_f:
    for result in results:
        to_f.write(json.dumps(result, ensure_ascii=False)+'\n')



#register datainfo
datainfo = './data/dataset_info.json'

with open(datainfo, 'r') as f:
    data = json.load(f)

# 添加新信息
data[f'{file}'] = {
    "file_name": file+'_fakelabel.jsonl',
    "formatting": "sharegpt",
    "columns": {
        "messages": "conversations",
        "system": "system",
    },
    "tags": {
        "role_tag": "from",
        "content_tag": "value",
        "user_tag": "human",
        "assistant_tag": "gpt",
    }
}

# 将修改后的数据写回文件
with open(datainfo, 'w') as f:
    json.dump(data, f, indent=2)

print('Data copied and new info appended to', datainfo)



