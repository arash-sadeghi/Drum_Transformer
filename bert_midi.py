from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch import nn

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

class BertMidi(nn.Module):

  def __init__(self, n_classes = 2):
    super().__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size * self.bert.config.max_position_embeddings , self.bert.config.max_position_embeddings)

  def forward(self, input_ids, attention_mask):
    bert_out = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    hidden_states_flatten = bert_out.last_hidden_state.view(-1,self.bert.config.hidden_size * self.bert.config.max_position_embeddings)
    x = self.drop(hidden_states_flatten)
    return self.out(x)