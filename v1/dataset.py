import json, glob, torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import re
import unicodedata

def get_tags(cinnamon_path):
    tags = set()
    files = glob.glob(f'{cinnamon_path}/ca_data/*')
    for file in files:
        dataframe = pd.read_excel(file, encoding="utf8")
        label_str = filter(lambda i:(type(i) is str), dataframe['Tag'])
        def split(strings):
            out = list()
            for string in strings: 
                out += string.split(";")
            out = [unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tag)) for tag in out]
            return out
        items = split(label_str)
        tags.update(items)
    return tuple(sorted(list(tags)))

tags = ('仕様書交付期限',
         '入札件名',
         '入札書締切日時',
         '入札書送付先',
         '入札書送付先部署/担当者名',
         '公告日',
         '施設名',
         '調達年度',
         '調達終了日',
         '調達開始日',
         '資格申請締切日時',
         '資格申請送付先',
         '資格申請送付先部署/担当者名',
         '質問票締切日時',
         '質問箇所TEL/FAX',
         '質問箇所所属/担当者',
         '都道府県',
         '開札場所',
         '開札日時',
         '需要場所(住所)')

########################################################
##################  Cinnamon Dataset  ##################
class Cinnamon_Dataset(Dataset):
    def __init__(self, cinnamon_path, tokenizer, delta=11, tags=None):        
        def get_samples(cinnamon_path, delta):
            groups = []
            files = glob.glob(f'{cinnamon_path}/ca_data/*')
            for file in files:
                doc_id = file[file.find('ca_data/')+8:file.find('.pdf.xlsx')]
                
                dataframe = pd.read_excel(file, encoding="utf8")
                dataframe['doc_id'] = doc_id
                '''
                for i in range(10):
                    if not isinstance(dataframe['Parent Index'][i], int):
                        dataframe['Parent Index'][i] = 0 # index是nan的補 0
                '''
                delta = delta 
                for i in range(0, len(dataframe), delta):
                    sample = dataframe.loc[i:i+delta-1]
                    groups.append(sample)
                    
                    text = ''.join(sample['Text']) 
                    if len(tokenizer.encode(text))>512:
                        print(len(tokenizer.encode(text)))#, text)
                    
            return groups
        
        self.tokenizer = tokenizer
        self.samples = get_samples(cinnamon_path, delta)
        self.tags = get_tags(cinnamon_path) if tags is None else tags

        print(f'\t[Info] Load Cannon_Dataset complete !! len:{self.__len__()}')    
        
    def __len__(self):
        return len(self.samples) 
    
    def __getitem__(self, idx):
        return self.samples[idx]
            
    def collate_fn(self, samples):        
        tokenizer, tags = self.tokenizer, self.tags
            
        CLS, SEP, PAD = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
        
        def zero_vec(): 
            return [0 for i in range(len(tags))]
        
        def sub_idx_finder(list1, list2, t=None):            
            if t=='入札件名':
                for i in range(len(list1)-len(list2)+1):
                    find = True
                    hit, miss = 0, 0
                    for j in range(len(list2)):
                        if list1[i+j] != list2[j]: 
                            find = False
                            miss += 1
                        else:
                            hit += 1
                    if miss < len(list2)/4:
                        find = True
                    if find:
                        return i
            elif t=='需要場所(住所)': #反過來找
                for i in range(len(list1)-len(list2), -1, -1):
                    find = True
                    hit, miss = 0, 0
                    for j in range(len(list2)):
                        if list1[i+j] != list2[j]: 
                            find = False
                            miss += 1
                        else:
                            hit += 1
                    if miss < len(list2)/6:
                        find = True
                    if find:
                        return i          
            elif t=='質問箇所所属/担当者': #反過來找
                for i in range(len(list1)-len(list2), -1, -1):
                    find = True
                    hit, miss = 0, 0
                    for j in range(len(list2)):
                        if list1[i+j] != list2[j]: 
                            find = False
                            miss += 1
                        else:
                            hit += 1
                    if miss < len(list2)/4:
                        find = True
                    if find:
                        return i           
            elif t=='質問箇所TEL/FAX': #反過來找
                for i in range(len(list1)-len(list2), -1, -1):
                    find = True
                    hit, miss = 0, 0
                    for j in range(len(list2)):
                        if list1[i+j] != list2[j]: 
                            find = False
                            miss += 1
                        else:
                            hit += 1
                    if miss < len(list2)/3:
                        find = True
                    if find:
                        return i    
            else: #正向找
                for i in range(len(list1)-len(list2)+1):
                    find = True
                    hit, miss = 0, 0
                    for j in range(len(list2)):
                        if list1[i+j] != list2[j]: 
                            find = False
                            miss += 1
                        else:
                            hit += 1
                    if find:
                        return i                
            return None
        
        ## text tokenized, label vectoized
        b_token_ids, b_output = [], []
        for sample in samples:
            token_ids = [CLS]
            output = [zero_vec()]
            
            for text, tag, value in zip(sample['Text'],sample['Tag'],sample['Value']):
                # 全形半形問題
                text = str(text).replace('イ．','').replace('ア．','').replace('．','').replace(' ','')
                
                tag = str(unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tag))) if tag is not np.nan else tag
                value = str(value).replace('イ．','').replace('ア．','').replace('．','').replace(' ','')
                    
                ###
                ids = tokenizer.encode(text)[1:-1]# + [SEP]
                
                labels = [[0 for i in range(len(tags))] for j in range(len(ids)) ] # + [zero_vec()]  #512*20
                
                                
                if isinstance(tag, str):
                    for t,v in zip(tag.split(';'), str(value).split(';')):   
                        t = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', t))
                        #v = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', v))
                        
                        ids_v = tokenizer.encode(v)[1:-1]
                        pivote = sub_idx_finder(ids, ids_v, t)                        
                        if pivote is not None:
                            for k in range(len(ids_v)):
                                labels[pivote+k][tags.index(t)] = 1
                        else:
                            print(text, ids)
                            print(v, ids_v)
                            print(pivote)
                            print(t)
                            input("")
                token_ids += ids
                output += labels
            b_token_ids.append(token_ids)
            b_output.append(output)

        ## pad to same lenght
        max_len = min([max([len(s) for s in b_token_ids]), 512])
        for idx,(token_ids, output) in enumerate(zip(b_token_ids, b_output)):            
            token_ids = token_ids[:max_len]
            token_ids += [PAD]*(max_len-len(token_ids))
            b_token_ids[idx] = token_ids
            
            output = output[:max_len]
            output += [zero_vec()]*(max_len-len(output))
            b_output[idx] = output

        return torch.tensor(b_token_ids), torch.tensor(b_output)
    
    
########################################################
##############  Cinnamon Dataset Testing  ##############
class Cinnamon_Dataset_Testing(Dataset):
    def __init__(self, cinnamon_path, tokenizer, delta=11, tags=None):
        def get_samples(cinnamon_path, delta):
            groups = []
            files = glob.glob(f'{cinnamon_path}/ca_data/*')
            for file in files:
                doc_id = file[file.find('ca_data/')+8:file.find('.pdf.xlsx')]
                
                dataframe = pd.read_excel(file, encoding="utf8")
                dataframe['doc_id'] = [doc_id]*len(dataframe)
                dataframe['ID'] = dataframe['Index'].apply(lambda x: "{}-{}".format(doc_id,x))
                dataframe['id'] = int(doc_id)
                '''
                for i in range(10):
                    if not isinstance(dataframe['Parent Index'][i], int):
                        dataframe['Parent Index'][i] = 0 # index是nan的補 0
                '''
                delta = delta
                for i in range(0,len(dataframe),delta):
                    sample = dataframe.loc[i:i+delta-1]
                    groups.append(sample)
                    
                    text = ''.join(sample['Text']) 
                    if len(tokenizer.encode(text))>512:
                        print(len(tokenizer.encode(text)), text)
                    
            return groups
        
        self.tokenizer = tokenizer
        self.samples = get_samples(cinnamon_path, delta)
        self.tags = get_tags(cinnamon_path) if tags is None else tags

        print(f'\t[Info] Load Cannon_Dataset_Testing complete !! len:{self.__len__()}')    
        
    def __len__(self):
        return len(self.samples) 
    
    def __getitem__(self, idx):
        return self.samples[idx]
            
    def collate_fn(self, samples):        
        tokenizer, tags = self.tokenizer, self.tags
        
        CLS, SEP, PAD = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
                        
        ## text tokenized, label vectoized
        b_token_ids, b_token_indexs, b_doc_id = [], [], []
        for sample in samples:
            doc_id = list(sample['doc_id'])[0]
            
            token_ids = [CLS]
            token_indexs = [-1]
            
            for text, index in zip(sample['Text'], sample['Index']):
                # 全形半形問題
                text = str(text).replace('イ．','').replace('ア．','').replace('．','')
                
                ids = tokenizer.encode(text)[1:-1]# + [SEP]
                
                token_ids += ids
                token_indexs += [index for jj in range(len(ids))]
                
            assert len(token_ids)==len(token_indexs)
            b_token_ids.append(token_ids)
            b_token_indexs.append(token_indexs)
            b_doc_id.append(doc_id)

        ## pad to same lenght
        max_len = min([max([len(s) for s in b_token_ids]), 512])
        for idx,(token_ids, token_indexs) in enumerate(zip(b_token_ids, b_token_indexs)):            
            token_ids = token_ids[:max_len]
            token_ids += [PAD]*(max_len-len(token_ids))
            b_token_ids[idx] = token_ids
            
            token_indexs = token_indexs[:max_len]
            token_indexs += [-1]*(max_len-len(token_indexs))
            b_token_indexs[idx] = token_indexs

        return torch.tensor(b_token_ids), None, b_token_indexs, b_doc_id, samples[0]
    
    
'''
class Cinnamon_Dataset(Dataset):
    def __init__(self, cinnamon_path, tokenizer):
        def get_tags(cinnamon_path):
            tags = set()
            files = glob.glob(f'{cinnamon_path}/ca_data/*')
            for file in files:
                dataframe = pd.read_excel(file, encoding="utf8")
                label_str = filter(lambda i:(type(i) is str), dataframe['Tag'])
                def split(strings):
                    out = list()
                    for string in strings: 
                        out += string.split(";")
                    out = [unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tag)) for tag in out]
                    return out
                items = split(label_str)
                tags.update(items)
            return tuple(sorted(list(tags)))
        
        def get_samples(cinnamon_path):
            groups = []
            files = glob.glob(f'{cinnamon_path}/ca_data/*')
            for file in files:
                dataframe = pd.read_excel(file, encoding="utf8")
                dataframe['Parent Index'][0] = 0 #第一行的index是nan，補 0

                p_index = dataframe.groupby('Parent Index')
                for g in list(p_index.groups.keys()):
                    groups.append(p_index.get_group(g))
            return groups
        
        self.tokenizer = tokenizer
        self.samples = get_samples(cinnamon_path)
        self.tags = get_tags(cinnamon_path)

        print(f'\t[Info] Load Cannon_Dataset complete !! len:{self.__len__()}')    
        
    def __len__(self):
        return len(self.samples) 
    
    def __getitem__(self, idx):
        return self.samples[idx]
            
    def collate_fn(self, samples):        
        tokenizer, tags = self.tokenizer, self.tags
            
        CLS, SEP, PAD = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
        
        def zero_vec(): 
            return [0]*len(tags)
        
        def sub_idx_finder(list1, list2):            
            for i in range(len(list1)-len(list2)):
                find = True
                hit, miss = 0, 0
                for j in range(len(list2)):
                    if list1[i+j] != list2[j]: 
                        find = False
                        miss += 1
                    else:
                        hit += 1
                if miss < len(list2)/5:
                    find = True
                if find:
                    return i
            #print('yeh')
        
        ## text tokenized, label vectoized
        b_token_ids, b_output = [], []
        for sample in samples:
            token_ids = [CLS]
            output = [zero_vec()]
            for text, tag, value in zip(sample['Text'],sample['Tag'],sample['Value']):
                # 全形半形問題
                text = str(unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', text)))
                tag = str(unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tag))) if tag is not np.nan else tag
                value = str(unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', value))) if value is not np.nan else value
                    
                ###
                ids = tokenizer.encode(text)[1:-1] + [SEP]
                labels = [zero_vec()]*(len(ids)-1) + [zero_vec()]
                
                if isinstance(tag, str):
                    for t,v in zip(tag.split(';'), str(value).split(';')):
                        t = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', t))
                        v = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', v))
                        
                        ids_v = tokenizer.encode(v)[1:-1]
                        pivote = sub_idx_finder(ids, ids_v)
                        for k in range(len(ids_v)):
                            if pivote is not None:
                                labels[pivote+k][tags.index(t)] = 1
                token_ids += ids
                output += labels
            b_token_ids.append(token_ids)
            b_output.append(output)

        ## pad to same lenght
        max_len = min([max([len(s) for s in b_token_ids]), 512])
        for idx,(token_ids, output) in enumerate(zip(b_token_ids, b_output)):            
            token_ids = token_ids[:max_len]
            token_ids += [PAD]*(max_len-len(token_ids))
            b_token_ids[idx] = token_ids
            
            output = output[:max_len]
            output += [zero_vec()]*(max_len-len(output))
            b_output[idx] = output

        return torch.tensor(b_token_ids), torch.tensor(b_output)
    
## testing  
class Cinnamon_Dataset_Testing(Dataset):
    def __init__(self, cinnamon_path, tokenizer):
        def get_tags(cinnamon_path):
            tags = set()
            files = glob.glob(f'{cinnamon_path}/ca_data/*')
            for file in files:
                dataframe = pd.read_excel(file, encoding="utf8")
                label_str = filter(lambda i:(type(i) is str), dataframe['Tag'])
                def split(strings):
                    out = list()
                    for string in strings: 
                        out += string.split(";")
                    out = [unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tag)) for tag in out]
                    return out
                items = split(label_str)
                tags.update(items)
            return tuple(sorted(list(tags)))
        
        def get_samples(cinnamon_path):
            groups = []
            files = glob.glob(f'{cinnamon_path}/ca_data/*')
            for file in files:
                doc_id = file[file.find('ca_data/')+8:file.find('.pdf.xlsx')]
                dataframe = pd.read_excel(file, encoding="utf8")
                for i in range(5):
                    if not isinstance(dataframe['Parent Index'][i], int):
                        dataframe['Parent Index'][i] = 0 # index是nan的補 0
                #dataframe['Parent Index'][0] = 0 # index是nan的補 0
                dataframe['ID'] = dataframe['Index'].apply(lambda x: "{}-{}".format(doc_id,x))
                dataframe['id'] = int(doc_id)
                
                p_index = dataframe.groupby('Parent Index')
                for g in list(p_index.groups.keys()):
                    groups.append({'doc_id':doc_id,'sample':p_index.get_group(g)})
            return groups
        
        self.tokenizer = tokenizer
        self.samples = get_samples(cinnamon_path)
        self.tags = get_tags(cinnamon_path)

        print(f'\t[Info] Load Cannon_Dataset complete !! len:{self.__len__()}')    
        
    def __len__(self):
        return len(self.samples) 
    
    def __getitem__(self, idx):
        return self.samples[idx]
            
    def collate_fn(self, samples):        
        tokenizer, tags = self.tokenizer, self.tags
            
        CLS, SEP, PAD = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
        
        def zero_vec(): 
            return [0]*len(tags)
        
        def sub_idx_finder(list1, list2):            
            for i in range(len(list1)-len(list2)):
                find = True
                hit, miss = 0, 0
                for j in range(len(list2)):
                    if list1[i+j] != list2[j]: 
                        find = False
                        miss += 1
                    else:
                        hit += 1
                if miss < len(list2)/5:
                    find = True
                if find:
                    return i
        
        ## text tokenized, label vectoized
        b_doc_id, b_token_ids, b_token_indexs = [], [], []
        for sample in samples:
            doc_id = sample['doc_id']
            sample = sample['sample']
            
            token_ids = [CLS]
            token_indexs = [-1]
            for index, text, tag, value in zip(sample['Index'],sample['Text'],sample['Tag'],sample['Value']):
                # 全形半形問題
                #text = str(unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', text)))
                text = str(text)

                ###
                ids = tokenizer.encode(text)[1:-1] + [SEP]
                labels = [zero_vec()]*(len(ids)-1) + [zero_vec()]
                indexs = [index]*(len(ids)-1) + [-1]
                
                token_ids += ids
                token_indexs += indexs
            b_doc_id.append(doc_id)
            b_token_ids.append(token_ids)
            b_token_indexs.append(token_indexs)

        ## pad to same lenght
        max_len = min([max([len(s) for s in b_token_ids]), 512])
        for idx,(token_ids, token_indexs) in enumerate(zip(b_token_ids, b_token_indexs)):            
            token_ids = token_ids[:max_len]
            token_ids += [PAD]*(max_len-len(token_ids))
            b_token_ids[idx] = token_ids
            
            token_indexs = token_indexs[:max_len]
            token_indexs += [-1]*(max_len-len(token_indexs))
            b_token_indexs[idx] = token_indexs

        return torch.tensor(b_token_ids), None, b_token_indexs, b_doc_id, sample
'''
    
