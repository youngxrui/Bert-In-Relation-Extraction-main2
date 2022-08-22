import json
from transformers import BertTokenizer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
def setup_seed(seed):
     torch.manual_seed(seed)#设置CPU生成随机数的种子，方便下次复现实验结果
     torch.cuda.manual_seed_all(seed)##为当前GPU设置随机种子；
     np.random.seed(seed)
     # np.random.sseed()中的参数被设置了之后，np.random.seed()可以按顺序产生一组固定的数组，
     # 如果使用相同的seed()值，则每次生成的随机数都相同。如果不设置这个值，那么每次生成的随机数不同。
     # 但是，只在调用的时候seed()一下并不能使生成的随机数相同，需要每次调用都seed()一下，表示种子相同，从而生成的随机数相同。
     random.seed(seed)#同上

setup_seed(44)
def prepare_data():
    print("---Regenerate Data---")
    print("---Regenerate Train Data---")
    with open("train_data.json", 'r', encoding='utf-8') as load_f:
        info=[]
        import random
        for line in load_f.readlines():
            dic = json.loads(line)
            for j in dic['spo_list']:
                single_data={}
                single_data['rel']=j["predicate"]
                single_data['ent1']=j["object"]
                single_data['ent2'] = j["subject"]
                single_data['text']=dic['text']
                info.append(single_data)
        sub_train = info
    with open("train.json", "w",encoding='utf-8') as dump_f:
        for i in sub_train:
            a = json.dumps(i, ensure_ascii=False)#json.dumps将一个Python数据结构转换为JSON
            dump_f.write(a)
            dump_f.write("\n")
    print("训练集准备完成")
    print("---Regenerate test Data---")
    with open("dev_data.json", 'r', encoding='utf-8') as load_f:
        info=[]
        import random
        for line in load_f.readlines():
            dic = json.loads(line)
            for j in dic['spo_list']:
                single_data={}
                single_data['rel']=j["predicate"]
                single_data['ent1']=j["object"]
                single_data['ent2'] = j["subject"]
                single_data['text']=dic['text']
                info.append(single_data)
            
        sub_train = info
    with open("dev.json", "w",encoding='utf-8') as dump_f:
        for i in sub_train:
            a = json.dumps(i, ensure_ascii=False)
            dump_f.write(a)
            dump_f.write("\n")
    print("测试集准备完成")
#prepare_data()




#main.py中使用
def map_id_rel():
    # id2rel={0: 'UNK', 1: '主演', 2: '歌手', 3: '简称', 4: '总部地点', 5: '导演', 6: '出生地', 7: '目', 8: '出生日期', 9: '占地面积', 10: '上映时间', 11: '出版社', 12: '作者', 13: '号', 14: '父亲', 15: '毕业院校', 16: '成立日期', 17: '改编自', 18: '主持人', 19: '所属专辑', 20: '连载网站', 21: '作词', 22: '作曲', 23: '创始人', 24: '丈夫', 25: '妻子', 26: '朝代', 27: '民族', 28: '国籍', 29: '身高', 30: '出品公司', 31: '母亲', 32: '编剧', 33: '首都', 34: '面积', 35: '祖籍', 36: '嘉宾', 37: '字', 38: '海拔', 39: '注册资本', 40: '制片人', 41: '董事长', 42: '所在城市', 43: '气候', 44: '人口数量', 45: '邮政编码', 46: '主角', 47: '官方语言', 48: '修业年限'}
    id2rel={0:'包含',1:'采样方',2:'常住地',3:'出行时间',4:'到达时间',5:'发生地点',6:'发现方式',7:'防范区',
            8:'封控区',9:'隔离方式',10:'隔离治疗点',11:'管控区',12:'国籍',13:'行为',14:'籍贯',15:'交通方式',
            16:'开始隔离时间',17:'来源',18:'年龄',19:'入境地点',20:'入境时间',21:'涉疫地区',22:'下一班',23:'下一程',
            24:'性别',25:'诊断时间',26:'职业'}
    rel2id={}
    for i in id2rel:
        rel2id[id2rel[i]]=i
    return rel2id,id2rel


#main.py中使用
def load_train():
    rel2id,id2rel=map_id_rel()
    # max_length=128   #定义了句子的最大长度，要注意
    max_length = 128
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_data = {}
    train_data['label'] = []
    train_data['mask'] = []
    train_data['text'] = []

    # with open("train.json", 'r', encoding='utf-8') as load_f:
    with open("data\核查train.json", 'r', encoding='utf-8') as load_f:
        temp = load_f.readlines()
        temp = temp[:18600]#原来的用部分数据即200条数据作为训练，现在用200->18600
        for line in temp:
            dic = json.loads(line)
            if dic['rel'] not in rel2id:
                train_data['label'].append(0) #append方法是浅拷贝，什么是浅拷贝呢？官方一点的解释是，
                # 在python中，对象赋值实际上是对象的引用，当创建一个对象，然后把它赋值给另一个变量的时候，python并没有拷贝这个对象，而只是拷贝了这个对象的引用
            else:
                train_data['label'].append(rel2id[dic['rel']])
            sent=dic['ent1']+dic['ent2']+dic['text']
            indexed_tokens = tokenizer.encode(sent, add_special_tokens=True)
            #encode方法可以一步到位地生成对应模型的输入。结果如：# tensor([7592, 1010, 2026, 2365, 2003, 3013, 2075, 1012])
            # 相比之下，tokenize只是用于分词，可以分成WordPiece的类型，并且在分词之后还要手动使用convert_tokens_to_ids方法，比较麻烦。
            # encode方法中调用了tokenize方法，所以在使用的过程中，我们可以通过设置encode方法中的参数，达到转化数据到可训练格式一步到位的目的，
            #add_special_tokens: bool = True            将句子转化成对应模型的输入形式，默认开启

            avai_len = len(indexed_tokens)
            while len(indexed_tokens) <  max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[: max_length]
            indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)在第一维增加一个维度

            # Attention mask
            att_mask = torch.zeros(indexed_tokens.size()).long()
            # (1, L)函数torch.zeros()返回一个由标量值0填充的张量，其形状由变量参数size定义。
            att_mask[0, :avai_len] = 1#表示第 0 维 取0个元素 ，第 1 维取前avai_len个元素
            train_data['text'].append(indexed_tokens)
            train_data['mask'].append(att_mask)
    return train_data
#main.py中使用
def load_dev():
    rel2id,id2rel=map_id_rel()
    # max_length=128
    max_length = 256
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_data = {}
    train_data['label'] = []
    train_data['mask'] = []
    train_data['text'] = []

    with open("data\核查dev.json", 'r', encoding='utf-8') as load_f:
        for line in load_f.readlines():
            dic = json.loads(line)
            if dic['rel'] not in rel2id:
                train_data['label'].append(0)
            else:
                train_data['label'].append(rel2id[dic['rel']])
            sent=dic['ent1']+dic['ent2']+dic['text']
            indexed_tokens = tokenizer.encode(sent, add_special_tokens=True)
            avai_len = len(indexed_tokens)
            while len(indexed_tokens) <  max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[: max_length]
            indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

            # Attention mask
            att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
            att_mask[0, :avai_len] = 1
            train_data['text'].append(indexed_tokens)
            train_data['mask'].append(att_mask)
    return train_data


# prepare_data()
