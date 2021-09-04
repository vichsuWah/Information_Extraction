import torch
import torch.nn as nn
import torch.nn.functional as F

import glob
import numpy as np
import pandas as pd

from transformers import BertJapaneseTokenizer, BertModel, BertConfig

import re
import unicodedata

from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

import seaborn as sns

from model import *
#####################################################################
pretrained_weights = 'cl-tohoku/bert-base-japanese-whole-word-masking'
FINE_TUNE_BERT_MODEL_PATH = 'epoch_34.pt'

TAGs = ['調達年度', '都道府県', '入札件名', '施設名', '需要場所(住所)', 
        '調達開始日', '調達終了日', '公告日','仕様書交付期限', '質問票締切日時', 
        '資格申請締切日時', '入札書締切日時', '開札日時', '質問箇所所属/担当者', '質問箇所TEL/FAX',
        '資格申請送付先', '資格申請送付先部署/担当者名', '入札書送付先', '入札書送付先部署/担当者名', '開札場所']
        
TAGs = [unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', t)) for t in TAGs]
TAGs = tuple(TAGs)

cinnamon_path = '/home/md531/ADL/Final/data/release/'
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


config = BertConfig.from_pretrained(pretrained_weights, output_hidden_states=True)

fine_tune_model = Model(config)
fine_tune_model.load(FINE_TUNE_BERT_MODEL_PATH)

model = fine_tune_model.bert_embedd

#config = BertConfig.from_pretrained(pretrained_weights, output_hidden_states=True)
#model = BertModel.from_pretrained(pretrained_weights, config=config)
#model.load_state_dict(torch.load(FINE_TUNE_BERT_MODEL_PATH))

Clusters = []
ALL_LAYERS = 12
for t_idx in range(len(TAGs)):
    inter_Cluster = [torch.tensor([]) for i in range(ALL_LAYERS)]

    for idx, sent in enumerate(tags_values[t_idx]):
        sent_ids = torch.tensor(tokenizer.encode(sent))
        outputs = model(input_ids=sent_ids.view(1, -1))
        last_hidden_states = outputs[0]
        hidden_states = outputs[2]

        embedding_output = hidden_states[0]
        attention_hidden_states = hidden_states[1:]

        
        # Ver.1: Take [CLS] embedding as the sentence embedding
        #sent_embedding = last_hidden_states.squeeze(0)[0, :] # shape = [768]
        # Ver.2: Average
        sent_embedding = torch.mean(last_hidden_states.squeeze(0), dim=0) # shape = [768]

        for j in range(len(attention_hidden_states)):
            hidden_sent_embedding = torch.mean(attention_hidden_states[j].squeeze(0), dim=0)
            inter_Cluster[j] = torch.cat((inter_Cluster[j], hidden_sent_embedding.unsqueeze(0)), dim=0)

        print("\r{}({}/{}):[{}/{}]".format(TAGs[t_idx], t_idx+1, len(TAGs), idx+1, len(tags_values[t_idx])), end='   \r')

    Clusters.append(inter_Cluster)

#############################################################################################
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

# Intra Cluster Similarity 群內
Layers = {}
for layer in range(ALL_LAYERS):
    # layer 1 ~ 12 [all layers]
    Inter_similarity = []
    Cluster_center = []
    for t_idx in range(len(TAGs)):
        cluster = Clusters[t_idx][layer]
        
        center = cluster.mean(dim=0)
        Cluster_center.append(center)

        cos_sim = []
        for sent in range(cluster.shape[0]):
            # count L2 Norm Distance between cluster center & cluster inter-point
            # dist += torch.norm((center - cluster[sent, :]), 2, -1)
            cos_sim.append( cos(center.unsqueeze(0), cluster[sent, :].unsqueeze(0)).detach().cpu().numpy() )

        Inter_similarity.append({'mean':np.mean(cos_sim), 'var':np.var(cos_sim)})

    # Intra Cluster Similarity 群間
    Intra_similarity = np.zeros((len(TAGs), len(TAGs)))
    for t_idx in range(len(TAGs)):
        center = Cluster_center[t_idx]
        
        for i in range(1, len(TAGs)):
            tar_idx = ( t_idx + i ) % len(TAGs)
            tar_center = Cluster_center[tar_idx]

            #dist = torch.norm((center - tar_center), 2, -1)
            Intra_similarity[t_idx, tar_idx] = cos(center.unsqueeze(0), tar_center.unsqueeze(0))
            
    Layers[layer] = {'Inter_similarity': Inter_similarity, 'Intra_similarity': Intra_similarity}

print("test")
#############################################################################################
# plot figure
color1 = plt.cm.Set1(range(9))
color2 = plt.cm.Set2(range(8))
color3 = plt.cm.Set3(range(12))

color = np.concatenate((color1, color2, color3[:20-len(color1)-len(color2)]), axis=0)

matplotlib.rc('font', family='TakaoPGothic')

# 1. [Inter] - line chart
plt.figure(figsize=(32,16), dpi=300)

_layer_InterSim = [[ (Layers[l]['Inter_similarity'][t]['mean']) for l in range(len(Layers)) ] for t in range(len(TAGs))]
for t_idx, t in enumerate(TAGs):
    plt.plot(range(1, len(Layers)+1), _layer_InterSim[t_idx], 'o-', color=color[t_idx], label=t)

plt.xticks(np.arange(1, 13))
plt.xlabel('Layers', fontsize=15, labelpad = 15)
plt.ylabel('Intra Similarity', fontsize=15, labelpad = 20)
plt.legend(loc='upper right')
plt.savefig('figure/SIMILARITY/Inter_similarity.png')
plt.title('Cluster Inter Similarity')
plt.close()

#2. [Inter] - radar chart
from math import pi

angles = [n / float(len(TAGs)-1) * 2 * pi for n in range(len(TAGs)-1)]
angles += angles[:1]
    
for t_idx, tag in enumerate(TAGs):
    # Initialise the spider plot
    plt.figure(figsize=(12,12), dpi=300)
    ax = plt.subplot(111, polar=True)

    # Draw one axe per variable + add labels labels yet
    categories = [TAGs[_t%20] for _t in range(t_idx+1, t_idx+20)]
    plt.xticks(angles[:-1], categories, color='grey', size=15)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    
    for l in range(len(Layers)):
        # Layers[l]['Intra_similarity']: matrix (shape: [20, 20])
        mat = Layers[l]['Intra_similarity'][t_idx]
        
        values = []
        for _t in range(t_idx+1, t_idx+20):
            values.append( -np.log( mat[ _t%20 ] ) )

        #values = [ mat[_t % 20] for _t in range(t_idx+1, t_idx+20) ]
        
        values += values[:1]

        # Plot data
        ax.plot(angles, values, linewidth=1, linestyle='solid', label='Layer_{}'.format(l+1))
        
        # Fill area
        ax.fill(angles, values, color=color[l], alpha=0.005)

    plt.legend(loc='best', bbox_to_anchor=(0.6, 0.65, 0.5, 0.5), fontsize=12)
    plt.title(tag, fontsize=25)
    plt.savefig("figure/SIMILARITY/INTRA_{}.png".format(re.sub('/', '_', tag)))
    plt.close()
    
    print('\rIntra_Radar_{}-({}/{})'.format(tag, t_idx+1, len(TAGs)))

