import torch.nn as nn
import torch
from config import *
from torchcrf import CRF


class DeepPunctuation(nn.Module):
    def __init__(self, pretrained_model, freeze_bert=False, lstm_dim=-1,punctuation_dict = {'O': 0, 'COMMA': 1, 'PERIOD': 2, 'QUESTION': 3,'EXCLAMATION': 4}):
        super(DeepPunctuation, self).__init__()
        self.output_dim = len(punctuation_dict)
        #self.bert_layer = MODELS[pretrained_model][0].from_pretrained(pretrained_model)
        #self.bert_layer = MODELS[pretrained_model][0].from_pretrained("../../../xlm-roberta-large")   
        self.bert_layer = MODELS[pretrained_model][0].from_pretrained("../../../"+pretrained_model)         
        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        bert_dim = MODELS[pretrained_model][2]
        if lstm_dim == -1:
            hidden_size = bert_dim
        else:
            hidden_size = lstm_dim
        self.lstm = nn.LSTM(input_size=bert_dim, hidden_size=hidden_size, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(in_features=hidden_size*2, out_features=len(punctuation_dict))

    def forward(self, x, attn_masks):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample
        # (B, N, E) -> (B, N, E)
        x = self.bert_layer(x, attention_mask=attn_masks)[0]
        # (B, N, E) -> (N, B, E)
        x = torch.transpose(x, 0, 1)
        x, (_, _) = self.lstm(x)
        # (N, B, E) -> (B, N, E)
        x = torch.transpose(x, 0, 1)
        x = self.linear(x)
        return x


class DeepPunctuationCRF(nn.Module):
    def __init__(self, pretrained_model, freeze_bert=False, lstm_dim=-1,punctuation_dict = {'O': 0, 'COMMA': 1, 'PERIOD': 2, 'QUESTION': 3,'EXCLAMATION': 4}):
        super(DeepPunctuationCRF, self).__init__()
        self.bert_lstm = DeepPunctuation(pretrained_model, freeze_bert, lstm_dim)
        self.crf = CRF(len(punctuation_dict), batch_first=True)

    def log_likelihood(self, x, attn_masks, y):
        x = self.bert_lstm(x, attn_masks)
        attn_masks = attn_masks.byte()
        return -self.crf(x, y, mask=attn_masks, reduction='token_mean')

    def forward(self, x, attn_masks, y):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample
        x = self.bert_lstm(x, attn_masks)
        attn_masks = attn_masks.byte()
        dec_out = self.crf.decode(x, mask=attn_masks)
        y_pred = torch.zeros(y.shape).long().to(y.device)
        for i in range(len(dec_out)):
            y_pred[i, :len(dec_out[i])] = torch.tensor(dec_out[i]).to(y.device)
        return y_pred

class DeepPunctuationCRF_jyupin(nn.Module):
    def __init__(self, pretrained_model, freeze_bert=False, lstm_dim=-1,punctuation_dict = {'O': 0, 'COMMA': 1, 'PERIOD': 2, 'QUESTION': 3,'EXCLAMATION': 4}):
        super(DeepPunctuationCRF_jyupin, self).__init__()
        self.bert_lstm = DeepPunctuation_jyupin(pretrained_model, freeze_bert, lstm_dim)
        self.crf = CRF(len(punctuation_dict), batch_first=True)

    def log_likelihood(self, x, attn_masks, y, jyupin):
        x = self.bert_lstm(x, attn_masks, jyupin)
        attn_masks = attn_masks.byte()
        return -self.crf(x, y, mask=attn_masks, reduction='token_mean')

    def forward(self, x, attn_masks, y, jyupin):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample
        x = self.bert_lstm(x, attn_masks, jyupin)
        attn_masks = attn_masks.byte()
        dec_out = self.crf.decode(x, mask=attn_masks)
        y_pred = torch.zeros(y.shape).long().to(y.device)
        for i in range(len(dec_out)):
            y_pred[i, :len(dec_out[i])] = torch.tensor(dec_out[i]).to(y.device)
        return y_pred


class DeepPunctuation_multitask_parallel(nn.Module):
    def __init__(self, pretrained_model, freeze_bert=False, lstm_dim=-1,punctuation_dict = {'O': 0, 'COMMA': 1, 'PERIOD': 2, 'QUESTION': 3,'EXCLAMATION': 4}):
        super(DeepPunctuation_multitask, self).__init__()
        self.output_dim = len(punctuation_dict)
        self.output_dim_multitask = 2   #change
        #self.bert_layer = MODELS[pretrained_model][0].from_pretrained(pretrained_model)
        #self.bert_layer = MODELS[pretrained_model][0].from_pretrained("../../../xlm-roberta-large")
        self.bert_layer = MODELS[pretrained_model][0].from_pretrained("../../../"+pretrained_model)   
        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        bert_dim = MODELS[pretrained_model][2]
        if lstm_dim == -1:
            hidden_size = bert_dim
        else:
            hidden_size = lstm_dim
        self.lstm = nn.LSTM(input_size=bert_dim, hidden_size=hidden_size, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(in_features=hidden_size*2, out_features=len(punctuation_dict))
        self.linear_multitask = nn.Linear(in_features=hidden_size*2, out_features=2)

    def forward(self, x, attn_masks):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample
        # (B, N, E) -> (B, N, E)
        x = self.bert_layer(x, attention_mask=attn_masks)[0]
        # (B, N, E) -> (N, B, E)
        x = torch.transpose(x, 0, 1)
        x, (_, _) = self.lstm(x)
        # (N, B, E) -> (B, N, E)
        x = torch.transpose(x, 0, 1)
        '''
        print(x)
        print(x.size())
        '''
        multitask_input=x[:,0,:]
        x_multitask = self.linear_multitask(multitask_input)
        #print(x[:,0,:].size())
        #print(x.size())
        x = self.linear(x)

        #print(x.size())
        return x,x_multitask


class DeepPunctuation_jyupin(nn.Module):
    def __init__(self, pretrained_model, freeze_bert=False, lstm_dim=-1,punctuation_dict = {'O': 0, 'COMMA': 1, 'PERIOD': 2, 'QUESTION': 3,'EXCLAMATION': 4}):
        super(DeepPunctuation_jyupin, self).__init__()
        self.output_dim = len(punctuation_dict)
        #self.bert_layer = MODELS[pretrained_model][0].from_pretrained(pretrained_model)
        #self.bert_layer = MODELS[pretrained_model][0].from_pretrained("../../../xlm-roberta-large")   
        self.bert_layer = MODELS[pretrained_model][0].from_pretrained("../../../"+pretrained_model)         
        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        bert_dim = MODELS[pretrained_model][2]
        self.embedding = nn.Embedding(num_embeddings=31855,embedding_dim=768,padding_idx=31854)
        if lstm_dim == -1:
            hidden_size = bert_dim
        else:
            hidden_size = lstm_dim
        self.lstm = nn.LSTM(input_size=bert_dim, hidden_size=hidden_size, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(in_features=hidden_size*2, out_features=len(punctuation_dict))

    def forward(self, x, attn_masks, jyupin):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample
        # (B, N, E) -> (B, N, E)

        x = self.bert_layer(x, attention_mask=attn_masks)[0]
        jyupin_embedding = self.embedding(jyupin)

        x += jyupin_embedding
        # (B, N, E) -> (N, B, E)
        x = torch.transpose(x, 0, 1)
        x, (_, _) = self.lstm(x)
        # (N, B, E) -> (B, N, E)
        x = torch.transpose(x, 0, 1)
        x = self.linear(x)
        return x


