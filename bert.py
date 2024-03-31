import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math

MAX_POS_EMBEDDINGS = 150
D_MODEL = 200
NUM_HIDDEN_LAYERS = 1
NUM_ATTN_HEADS = 2
CNN_OUTPUT_DIM = 4096 

model_config = transformers.BertConfig(
            hidden_size=D_MODEL,
            num_attention_heads=NUM_ATTN_HEADS,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
        )

class PositionEmbedding(nn.Module):
# Reference : https://github.com/codertimo/BERT-pytorch/blob/d10dc4f9d5a6f2ca74380f62039526eb7277c671/bert_pytorch/model/embedding/position.py
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        position = torch.arange(MAX_POS_EMBEDDINGS).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D_MODEL, 2) * (-math.log(10000.0) / D_MODEL))
        pe = torch.zeros(MAX_POS_EMBEDDINGS, 1, D_MODEL)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return x

class BERT(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.l1 = nn.Linear(in_features=CNN_OUTPUT_DIM, out_features=model_config.hidden_size)
        self.embedding = PositionEmbedding()
        self.bert_layer = transformers.BertLayer(model_config)
        self.l2 = nn.Linear(in_features=model_config.hidden_size, out_features=n_classes)

    def forward(self, x):
        x = self.l1(x)
        x = self.embedding(x)
        x = self.bert_layer(x)[0]
        x = torch.max(x, dim=1).values
        x = self.l2(x)
        return x

if __name__=='__main__':
    # Some testing snippets
    pass
