from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch import nn
import torch.nn.functional as F
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
MAX_INPUT_LENGTH = 510 #self.bert.config.max_position_embeddings
class BertMidi(nn.Module):

  def __init__(self, n_classes = 2):
    super().__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    # self.l1 = [nn.Linear(self.bert.config.hidden_size , 1) for _ in range(MAX_INPUT_LENGTH)]
    self.l1 = nn.Linear(self.bert.config.hidden_size*MAX_INPUT_LENGTH , MAX_INPUT_LENGTH) 


  def forward(self, input_ids, attention_mask):
    bert_out = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    x = self.drop(bert_out.last_hidden_state)
    x = x.view(-1 , self.bert.config.hidden_size*MAX_INPUT_LENGTH)
    # res = [ F.sigmoid(self.l1[_](x[:,_])) for _ in range(MAX_INPUT_LENGTH)  ]
    res = self.l1(x)
    res = F.sigmoid(res)

    return res