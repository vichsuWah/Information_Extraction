import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel

torch.manual_seed(1)
START_TAG = '[CLS]'
STOP_TAG = '[SEP]'

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

class BERT_BiLSTM_CRF(nn.Module):
    def __init__(self, tag_to_ix, hidden_dim=768):
        super(BERT_BiLSTM_CRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim // 2,
                            num_layers=2, bidirectional=True, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        # Using BertModel to replace nn.Embedding
        self.bert = BertModel.from_pretrained('./models/bert-base-japanese-whole-word-masking')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def _viterbi_decode(self, feats):
        batch_size = feats.shape[0]
        T = feats.shape[1]

        log_delta = torch.Tensor(batch_size, 1, self.tagset_size).fill_(-10000.).to(self.device)
        log_delta[:, 0, self.tag_to_ix[START_TAG]] = 0

        psi = torch.zeros((batch_size, T, self.tagset_size), dtype=torch.long)
        for t in range(1, T):
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        path = torch.zeros((batch_size, T), dtype=torch.long)

        max_logP, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T-2, -1, -1):
            path[:, t] = psi[:, t+1].gather(-1, path[:, t+1].view(-1, 1)).squeeze()
        
        return max_logP, path

    def _get_lstm_features(self, sentence):
        # sentence: [batch_size, sent_len]
        encoded, _ = self.bert(sentence)
        encoded = encoded[-1] # [batch_size, sent_len, hidden_dim]

        lstm_out, _ = self.lstm(encoded)
        
        lstm_feats = self.hidden2tag(lstm_out)
        # lstm_feats: [batch_size, sent_len, tagset_size]
        return lstm_feats

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq