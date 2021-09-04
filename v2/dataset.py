try:
    from utils.utils import get_tags, clean_str, sub_idx_finder
except Exception as inst:
    from v2.utils.utils import get_tags, clean_str, sub_idx_finder 

import json, glob, torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import re
import unicodedata

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
class Cinnamon_Dataset_v2(Dataset):
    def __init__(self, cinnamon_path, tokenizer, tags=None):        
        def get_samples(cinnamon_path):
            datas = []
            files = glob.glob(f'{cinnamon_path}/ca_data/*')
            for file in files:
                doc_id = file[file.find('ca_data/')+8:file.find('.pdf.xlsx')]
                
                dataframe = pd.read_excel(file, encoding="utf8")
                dataframe['Parent Index'] = dataframe['Parent Index'].fillna(1)
                
                for item in dataframe.iterrows(): 
                    #item:(Page No, Text, Index, Parent Index, Is Title, Is Table, Tag, Value)
                    item = item[1]
                    
                    doc, index = doc_id, item['Index']
                    text, p_text = item['Text'], dataframe.loc[dataframe['Index']==item['Parent Index'],'Text'].item()
                    tags, values = item['Tag'], item['Value']
                    
                    datas.append({'doc':doc_id,'index':index,
                                  'text':text, 'p_text':p_text,
                                  'tags':tags,'values':values})
                #print(datas)
            return datas
        
        self.tokenizer = tokenizer
        self.samples = get_samples(cinnamon_path)
        self.tags = get_tags(cinnamon_path) if tags is None else tags

        print(f'\t[Info] Load Cannon_Dataset_v2 complete !! len:{self.__len__()}')    
        
    def __len__(self):
        return len(self.samples) 
    
    def __getitem__(self, idx):
        return self.samples[idx]
            
    def collate_fn(self, samples):        
        tokenizer, TAGS = self.tokenizer, self.tags
            
        CLS, SEP, PAD = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
                
        ## text tokenized, label vectoized
        b_ids, b_labels, b_masks = [], [], []
        for sample in samples:            
            text, p_text, tags, values = sample['text'],sample['p_text'],sample['tags'],sample['values']
            
            # string cleaning
            text = clean_str(text)
            p_text = clean_str(p_text)
            tags = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tags)) if tags is not np.nan else tags
            values = clean_str(values)
                    
            # text to tokens
            text_ids = tokenizer.encode(text)[1:-1]
            p_text_ids = tokenizer.encode(p_text)[1:-1]
                
            # input, output, mask
            ids = [CLS] + text_ids + [SEP] + p_text_ids + [SEP]
            labels = [[0 for i in range(len(TAGS))] for j in range(len(ids)) ]
            masks = [[False]] + [[True] for i in range(len(text_ids))] + [[False] for i in range(len(ids)-len(text_ids)-1)]
                                
            # assign label 
            if isinstance(tags, str):
                for tag,value in zip(tags.split(';'), str(values).split(';')):   
                    tag = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tag))
                        
                    value_ids = tokenizer.encode(value)[1:-1]
                    pivote = sub_idx_finder(text_ids, value_ids, tag)                        
                    if pivote is not None:
                        for k in range(len(value_ids)):
                            labels[1+pivote+k][TAGS.index(tag)] = 1
                    else:
                        print("\t[ERROR] pivote not found ")
            b_ids.append(ids)
            b_labels.append(labels)
            b_masks.append(masks)

        ## pad to same lenght
        max_len = min([max([len(s) for s in b_ids]), 512])
        for i,(ids, labels, masks) in enumerate(zip(b_ids, b_labels, b_masks)):            
            ids = ids[:max_len]
            ids += [PAD]*(max_len-len(ids))
            b_ids[i] = ids
            
            labels = labels[:max_len]
            labels += [[0 for j in range(len(TAGS))] for k in range(max_len-len(labels))]
            b_labels[i] = labels
            
            masks = masks[:max_len]
            masks += [[False]]*(max_len-len(masks))
            b_masks[i] = masks

        return torch.tensor(b_ids), torch.tensor(b_labels), torch.tensor(b_masks)

########################################################
##############  Cinnamon Dataset Testing  ##############
class Cinnamon_Dataset_Testing_v2(Dataset):
    def __init__(self, cinnamon_path, tokenizer, tags=None):        
        def get_samples(cinnamon_path):
            datas = []
            files = glob.glob(f'{cinnamon_path}/ca_data/*')
            for file in files:
                doc_id = file[file.find('ca_data/')+8:file.find('.pdf.xlsx')]
                
                dataframe = pd.read_excel(file, encoding="utf8")
                dataframe['Parent Index'] = dataframe['Parent Index'].fillna(1)
                
                for item in dataframe.iterrows(): 
                    #item:(Page No, Text, Index, Parent Index, Is Title, Is Table, Tag, Value)
                    item = item[1]
                    
                    doc, index = doc_id, item['Index']
                    text, p_text = item['Text'], dataframe.loc[dataframe['Index']==item['Parent Index'],'Text'].item()
                    #tags, values = item['Tag'], item['Value']
                    
                    datas.append({'doc':doc_id,'index':index,
                                  'text':text, 'p_text':p_text})
                                  #'tags':tags,'values':values})
                #print(datas)
            return datas
        
        self.tokenizer = tokenizer
        self.samples = get_samples(cinnamon_path)
        self.tags = get_tags(cinnamon_path) if tags is None else tags

        print(f'\t[Info] Load Cannon_Dataset_v2 complete !! len:{self.__len__()}')    
        
    def __len__(self):
        return len(self.samples) 
    
    def __getitem__(self, idx):
        return self.samples[idx]
            
    def collate_fn(self, samples):        
        tokenizer, TAGS = self.tokenizer, self.tags
            
        CLS, SEP, PAD = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
                
        ## text tokenized, label vectoized
        b_docs, b_indexs, b_ids, _, b_masks = [], [], [], [], []
        for sample in samples:            
            #text, p_text, tags, values = sample['text'],sample['p_text'],sample['tags'],sample['values']
            doc, index = sample['doc'], sample['index']
            text, p_text = sample['text'],sample['p_text']
            
            # string cleaning
            text = clean_str(text)
            p_text = clean_str(p_text)
            #tags = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tags)) if tags is not np.nan else tags
            #values = clean_str(values)
                    
            # text to tokens
            text_ids = tokenizer.encode(text)[1:-1]
            p_text_ids = tokenizer.encode(p_text)[1:-1]
                
            # input, output, mask
            ids = [CLS] + text_ids + [SEP] + p_text_ids + [SEP]
            masks = [[False]] + [[True] for i in range(len(text_ids))] + [[False] for i in range(len(ids)-len(text_ids)-1)]
                                
            b_docs.append(doc)
            b_indexs.append(index)
            b_ids.append(ids)
            b_masks.append(masks)

        ## pad to same lenght
        max_len = min([max([len(s) for s in b_ids]), 512])
        for i,(ids, masks) in enumerate(zip(b_ids, b_masks)):            
            ids = ids[:max_len]
            ids += [PAD]*(max_len-len(ids))
            b_ids[i] = ids
            
#             labels = labels[:max_len]
#             labels += [[0 for j in range(len(TAGS))] for k in range(max_len-len(labels))]
#             b_labels[i] = labels
            
            masks = masks[:max_len]
            masks += [[False]]*(max_len-len(masks))
            b_masks[i] = masks

        return b_docs, b_indexs, torch.tensor(b_ids), _, torch.tensor(b_masks), samples
