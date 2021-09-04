import os 
import json
import torch
import torch.nn as nn

from transformers import BertTokenizer, BertJapaneseTokenizer, BertModel

pretrained_weights = 'cl-tohoku/bert-base-japanese-whole-word-masking'
#pretrained_weights = 'bert-base-japanese-whole-word-masking'

class MM(nn.Module):
    def save(self, epoch, t_log, v_log, path='ckpt/'):
        self.step_loss.append({'epoch':epoch,
                            'train_acc':t_log['acc'], 'train_f1':t_log['f1'],
                            'valid_acc':v_log['acc'], 'valid_f1':v_log['f1']}) 
        with open(f'{path}/step_loss.json', 'w', encoding='utf-8') as f:
            json.dump(self.step_loss, f, indent=4)
        torch.save({'epoch': epoch, 'state_dict': self.state_dict()},
                  f'{path}/epoch_{epoch}.pt')
        print(f'\t[Info] save weight, {path}/epoch_{epoch}.pt')

    def load(self, load_file):
        if os.path.isfile(load_file):
            self.load_state_dict(torch.load(load_file)['state_dict'])
            print(f'\t[Info] load weight, {load_file}')
        else:
            print(f'\t[ERROR] {load_file} not exist !')
        return self
        
class Model_BLSTM(MM):
    def __init__(self):
        super(Model_BLSTM, self).__init__()
        
        self.bert_embedd = BertModel.from_pretrained(pretrained_weights)
        for param in self.bert_embedd.parameters():
            #param.requires_grad = False
            continue 

        self.hidden_dim = 768
        
        self.lstm = nn.LSTM(768, self.hidden_dim//2, num_layers=1, 
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0)
        self.fc = nn.Linear(self.hidden_dim, 20)

        self.step_loss = []
    
    def forward(self, input_ids):
        last_hidden_states, cls_hidden = self.bert_embedd(input_ids)
        output = self.lstm(last_hidden_states)
        output = self.dropout(output[0])
        output = self.fc(output)
        return output

    def init_hidden(self):
        return (torch.zeros((2,1,self.hidden_dim)),
                torch.zeros((2,1,self.hidden_dim)))
   
 
class Model(MM):
    def __init__(self):
        super(Model, self).__init__()
        
        self.bert_embedd = BertModel.from_pretrained(pretrained_weights)
        for param in self.bert_embedd.parameters():
            #param.requires_grad = False
            continue 

        #self.dropout = nn.Dropout(0.3)

        hidden_dim = 768
        self.fc = nn.Linear(hidden_dim, 20)

        self.step_loss = []
    
    def forward(self, input_ids):
        last_hidden_states, cls_hidden = self.bert_embedd(input_ids)
        output = self.fc(last_hidden_states)
        return output


