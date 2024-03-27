import os
import sys
import json


model_outputs = open(sys.argv[1],'r')
metrics       = json.load(open(sys.argv[2],'r'))
rag_chaevals  = open(sys.argv[3],'r')
test_datas    = json.load(open(sys.argv[4], 'r')) #获取真正的context


predicts = []
for output in model_outputs:
    predicts.append(json.loads(output)['predict'])


contexts = {}
for output in test_datas:
    contexts[output['id']] = output['context']
    


datas = []
for info in rag_chaevals:

    data_dict = {}
    contents = json.loads(info) 
    conversations = contents['conversations']
    infos = contents['id'].split('_')

    
    data_dict['context'] = contexts[int(infos[0])]
    data_dict['id'] = int(infos[0])
    data_dict['role'] = infos[-1]
    data_dict['novel_name'] = infos[1]

    datas.append(data_dict)


assert len(datas) == len(predicts), 'len(datas) must be same as len(predicts)!!'


#merge predict
for i, (predict, data) in enumerate(zip(predicts, datas)):

    datas[i]['model_output'] = predict.strip()



#convert characterEval format
import copy
data_trans = []
for i, data in enumerate(datas):

    try:
        mtcs = metrics[str(data['id'])]
        model_output = data['model_output']
        role = data['role']
        idx = model_output.find(f"{role}：")
        
        #添加user：
        if idx != 0:
            model_output = role+'：'+model_output

        #修改action为（）
        strs = ''
        idx = 0
        for st in model_output:
            if st == '*':
                if idx % 2 == 0:
                    st = '（'
                else:
                    st = '）'
                idx += 1
                
            strs += st

        data['model_output'] = strs
        
        for mt in mtcs:
            data_ = copy.deepcopy(data)
            data_['metric_en'] = mt[0]
            data_['metric_zh'] = mt[1]
            data_trans.append(data_)
    except:
        pass

tofl = open(sys.argv[5], 'w') 

json.dump(data_trans, tofl, ensure_ascii=False)
    
     
    
    
    
    


     
    
        
        

     


