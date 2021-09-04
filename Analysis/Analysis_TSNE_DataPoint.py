import torch
import torch.nn as nn
import torch.nn.functional as F

import glob
import numpy as np
import pandas as pd

from transformers import BertJapaneseTokenizer, BertModel

import re
import unicodedata

from model import *

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

###########################SETTING###################################
pretrained_weights = 'cl-tohoku/bert-base-japanese-whole-word-masking'

TAGs = ['調達年度', '都道府県', '入札件名', '施設名', '需要場所(住所)', 
        '調達開始日', '調達終了日', '公告日','仕様書交付期限', '質問票締切日時', 
        '資格申請締切日時', '入札書締切日時', '開札日時', '質問箇所所属/担当者', '質問箇所TEL/FAX',
        '資格申請送付先', '資格申請送付先部署/担当者名', '入札書送付先', '入札書送付先部署/担当者名', '開札場所']
        
TAGs = [unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', t)) for t in TAGs]
TAGs = tuple(TAGs)

cinnamon_path = '/home/md531/ADL/Final/data/release/'
FINE_TUNE_BERT_MODEL_PATH = './epoch_34.pt'
######################################################################

tags_values = [[] for i in range(len(TAGs))]

data_type = ['train', 'dev']

for mode in data_type:
    files = glob.glob(f'{cinnamon_path}/{mode}/ca_data/*')
    for f_idx, file in enumerate(files):
        dataframe = pd.read_excel(file, encoding="utf8")

        for tag, value in zip(dataframe['Tag'], dataframe['Value']):
            tag = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tag)) if isinstance(tag, str) else tag
            value = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', value)) if isinstance(value, str) else value
            
            if (not isinstance(tag, str)) or (not isinstance(value, str)):
                pass
            else:
                _tags = [t for t in tag.split(';')]
                _values = [v for v in value.split(';')]

                if len(_tags) == len(_values):
                    for t, v in zip(_tags, _values):
                        tags_values[ TAGs.index(t) ].append(v)

                elif len(_tags) > len(_values):
                    assert len(_values) == 1, "the condition: diff. tags -> same value ?"
                    for idx, t in enumerate(_tags):
                        tags_values[ TAGs.index(t) ].append(_values[0])

                elif len(_tags) < len(_values):
                    assert len(_tags) == 1, "the condition: diff. values -> same tag ?"
                    for idx, v in enumerate(_values):
                        tags_values[ TAGs.index(_tags[0]) ].append(v)

        print("\r{}:[{}/{}]".format(mode, f_idx, len(files)), end='   \r')
# tags_values: group the values having same tag
print("Finish Collecting")

tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_weights, do_lower_case=True)

fine_tune_model = Model()
fine_tune_model.load(FINE_TUNE_BERT_MODEL_PATH)

model = fine_tune_model.bert_embedd
#model = BertModel.from_pretrained(pretrained_weights)
#model.load_state_dict(torch.load(FINE_TUNE_BERT_MODEL_PATH)['state_dict'])

#tags_tokens_ids = [[] for i in range(len(TAGs))]
tsne = TSNE(n_components=2, init='pca', random_state=501)
X, y = torch.tensor([]), torch.LongTensor([])

for t_idx in range(len(TAGs)):
    #tags_values[t_idx]
    for idx, sent in enumerate(tags_values[t_idx]):
        sent_ids = torch.tensor(tokenizer.encode(sent))
        outputs = model(input_ids=sent_ids.view(1, -1))
        last_hidden_states = outputs[0]
        # Ver.1: Take [CLS] embedding as the sentence embedding
        #sent_embedding = last_hidden_states.squeeze(0)[0, :] # shape = [768]
        # Ver.2: Average
        sent_embedding = torch.mean(last_hidden_states.squeeze(0), dim=0) # shape = [768]

        # prepare data for TSNE
        X = torch.cat((X, sent_embedding.view(1, -1)), dim=0)
        y = torch.cat((y, torch.tensor([t_idx])), dim=0)

        print("\r{}({}/{}):[{}/{}]".format(TAGs[t_idx], t_idx+1, len(TAGs), idx+1, len(tags_values[t_idx])), end='   \r')

X = X.cpu().detach().numpy()
y = y.cpu().detach().numpy()

X_tsne = tsne.fit_transform(X) # dim from '768' reduce to '2' | X_tsne.shape = (2504, 2)

# Data Visualization
x_min, x_max = X_tsne.min(0), X_tsne.max(0)

# Normalization
X_norm = (X_tsne - x_min) / (x_max - x_min) 

## color
color1 = plt.cm.Set1(range(9))
color2 = plt.cm.Set2(range(8))
color3 = plt.cm.Set3(range(12))

color = np.concatenate((color1, color2, color3[:20-len(color1)-len(color2)]), axis=0)
## 

# 1. LABELS CENTERED ON CLUSTER MEANS
plt.figure(figsize=(8, 8))

for i in range(20):
    # add data points
    plt.scatter(x=X_norm[y==i,0],y=X_norm[y==i,1],c='C' + str(i), s=7, label=i+TAGs[i], marker='o', alpha=0.2)

    # add label
    plt.annotate(i, X_norm[y==i].mean(axis=0), 
                horizontalalignment='center', 
                verticalalignment='center', 
                size = 20, weight='bold',
                color='C'+str(i) )

    #plt.annotate(i, X_norm[y==i].mean(axis=0), 
    #            horizontalalignment='center', 
    #            verticalalignment='center', 
    #            size = 20, weight='bold',
    #            color='white', backgroundcolor='C'+str(i) )

plt.legend(loc='upper right')
plt.savefig("figure/TSNE/TSNE_[TRAIN_DEV]_point.png")
plt.close()

# 2. TEXT Markers
plt.figure(figsize=(8, 8), dpi=600)

for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=color[ y[i] ],fontdict={'weight': 'normal', 'size': 5})

plt.savefig('figure/TSNE/TSNE_[TRAIN_DEV]_text.png')
plt.close()