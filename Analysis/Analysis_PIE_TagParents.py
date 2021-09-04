import torch
import torch.nn as nn
import torch.nn.functional as F

import glob
import numpy as np
import pandas as pd

from transformers import BertJapaneseTokenizer, BertModel

import re
import unicodedata

from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

#####################################################################
TAGs = ['調達年度', '都道府県', '入札件名', '施設名', '需要場所(住所)', 
        '調達開始日', '調達終了日', '公告日','仕様書交付期限', '質問票締切日時', 
        '資格申請締切日時', '入札書締切日時', '開札日時', '質問箇所所属/担当者', '質問箇所TEL/FAX',
        '資格申請送付先', '資格申請送付先部署/担当者名', '入札書送付先', '入札書送付先部署/担当者名', '開札場所']
        
TAGs = [unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', t)) for t in TAGs]
TAGs = tuple(TAGs)

cinnamon_path = '/home/md531/ADL/Final/data/release/'
######################################################################

def remove_no(text):
    # handling 3 differ case - ex. 3工業用水取水場, (2)実施場所, 1.入札に付する事項
    bidx = text.find(')')
    didx = text.find('.')

    _text = text

    if bidx != -1 and bidx < 5:
        _text = text[bidx+1:]
    elif didx != -1 and bidx < 5:
        _text = text[didx+1:]
    else:
        remove = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        for r in remove:
            _text = _text.lstrip(r)

    return _text

######################################################################
tags_values = [[] for i in range(len(TAGs))]
tags_pText = [[] for i in range(len(TAGs))]

data_type = ['train', 'dev']

for mode in data_type:
    files = glob.glob(f'{cinnamon_path}/{mode}/ca_data/*')
    for f_idx, file in enumerate(files):
        dataframe = pd.read_excel(file, encoding="utf8")

        for tag, value, parent_idx in zip(dataframe['Tag'], dataframe['Value'], dataframe['Parent Index']):
            tag = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tag)) if isinstance(tag, str) else tag
            value = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', value)) if isinstance(value, str) else value
            
            if (not isinstance(tag, str)) or (not isinstance(value, str)):
                pass
            else:
                # parent:
                parent_Text = dataframe['Text'].loc[dataframe['Index'] == parent_idx].values[0]
                parent_Text = remove_no( unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', parent_Text)) if isinstance(parent_Text, str) else parent_Text )
                

                _tags = [t for t in tag.split(';')]
                _values = [v for v in value.split(';')]

                if len(_tags) == len(_values):
                    for t, v in zip(_tags, _values):
                        tags_values[ TAGs.index(t) ].append(v)
                        tags_pText[ TAGs.index(t) ].append(parent_Text)
                        #if(v=='契約締結日' and TAGs.index(t) == 0):
                        #    print('???')
                elif len(_tags) > len(_values):
                    assert len(_values) == 1, "the condition: diff. tags -> same value ?"
                    for idx, t in enumerate(_tags):
                        tags_values[ TAGs.index(t) ].append(_values[0])
                        tags_pText[ TAGs.index(t) ].append(parent_Text)
                        #if(_values[0]=='契約締結日' and TAGs.index(t) == 0):
                        #    print('???')
                elif len(_tags) < len(_values):
                    assert len(_tags) == 1, "the condition: diff. values -> same tag ?"
                    for idx, v in enumerate(_values):
                        tags_values[ TAGs.index(_tags[0]) ].append(v)
                        tags_pText[ TAGs.index(_tags[0]) ].append(parent_Text)
                        #if(v=='契約締結日' and TAGs.index(_tags[0]) == 0):
                        #    print('???')
        print("\r{}:[{}/{}]".format(mode, f_idx, len(files)), end='   \r')
# tags_values: group the values having same tag
print("Finish Collecting")

# tags_ParentText Statistics
statistics_dic = {}
for i in range(len(TAGs)):
    # tags_pText[i]
    dic = {}
    for p_Text in tags_pText[i]:
        if p_Text in dic:
            dic[p_Text] += 1
        else:
            dic.update({p_Text:1})
    
    dic = sorted(dic.items(), key=lambda v:v[1])
    statistics_dic[ TAGs[i] ] = dic

statistics_dic = sorted(statistics_dic.items(), key=lambda v:len(v[0]))
print("End Analysis")

## color
color1 = plt.cm.Set1(range(9))
color2 = plt.cm.Set2(range(8))
color3 = plt.cm.Set3(range(12))

color = np.concatenate((color1, color2, color3[:20-len(color1)-len(color2)]), axis=0)

matplotlib.rc('font', family='TakaoPGothic')

for i in range(len(TAGs)):
    tag_name = statistics_dic[i][0]
    tag_analy = statistics_dic[i][1]
    
    _num = [t[0] for t in tag_analy]
    _val = [t[1] for t in tag_analy]

    plt.figure(figsize=(8, 8), dpi=600)

    wedges, texts, autotexts = plt.pie(_val, colors=color[:len(_num)], labels=_num, autopct='%2.2f%%')
    plt.setp(autotexts, size=8, weight='bold')

    plt.title(TAGs[i])
    plt.savefig("figure/PIE/PIE_{}.png".format(re.sub('/', '_', TAGs[i])))
    plt.close()
