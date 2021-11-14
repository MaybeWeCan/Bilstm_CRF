import torch.nn as nn
from src.torchcrf import CRF


class Bilstm_CRF(nn.Module):
    '''
        model设计:
            embedding = look_up_embedding(ids)

            bi-lstm_embedding = Bi_lstm(embedding)

            CRF_embedding = CRF(embedding)
    '''
    def __init__(self, config, is_layernorm = False):
        super(Bilstm_CRF,self).__init__()

        self.vocab_embedding = nn.Embedding(num_embeddings = config.vocab_size, embedding_dim = config.embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(input_size=config.embedding_dim,
                            hidden_size=config.lstm_hidden_size,
                            num_layers = config.lstm_num_layers,
                            batch_first = True,
                            dropout = config.dropout,
                            bidirectional=True)

        self.dropout = nn.Dropout(config.dropout)

        self.hidden2label = nn.Linear(config.lstm_hidden_size * 2, config.label_size)
        self.is_layernorm = is_layernorm
        
        if self.is_layernorm:

            self.layernorm1 = nn.LayerNorm(config.lstm_hidden_size * 2)
            self.layernorm2 = nn.LayerNorm(config.label_size)

        self.crf = CRF(config.label_size, batch_first=True)



    def forward(self, datas, mask=None, labels=None):
        '''
            datas:     id: [batch, pad_seq_length]
            labels:    id: [batch, pad_seq_length]
        '''

        embed_batch = self.vocab_embedding(datas)

        # [batch, seq_length, n_hidden]
        output, (_, _) = self.lstm(embed_batch)

        output = self.dropout(output)

        if self.is_layernorm:
            output = self.layernorm1(output)

        output = self.hidden2label(output)

        if self.is_layernorm:
            output = self.layernorm2(output)


        if labels != None:
            return - self.crf(output, labels, mask)
        else:
            # 预测得到解码路径
            return self.crf.decode(output, mask)








