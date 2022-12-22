import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np

from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model

device = torch.device("cpu")

bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return self.classifier(out)


class BERTDatasetEval(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i])

    def __len__(self):
        return (len(self.sentences))

model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
model.load_state_dict(torch.load("./weights/model_state_dict16.pt"))
model.eval()
torch.no_grad()

def _initialize_dict():
    return {
        "Positive" : 0,
        "Negative" : 0,
        "Neutral" : 0
    }

def sentiment_list_score_ko(sentence):
    sentence_analysis_result = _initialize_dict()
    max_len = 64
    batch_size = 16
    sentence = BERTDatasetEval(sentence, 0, 1, tok, max_len, True, False)
    sentence_dataloader = torch.utils.data.DataLoader(sentence, batch_size=batch_size, num_workers=0)
    
    for token_ids, valid_length, segment_ids in sentence_dataloader:
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        out = model(token_ids, valid_length, segment_ids)
        for neg, pos in out:
            print(neg,pos)
            if max(neg,pos)<1.5 or neg*pos>0:
                sentence_analysis_result['Neutral']+=1
            elif neg < pos:
                sentence_analysis_result['Positive']+=1
            else:
                sentence_analysis_result['Negative']+=1
    return sentence_analysis_result