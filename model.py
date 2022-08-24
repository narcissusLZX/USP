import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, feature_dim=3, encoder_embedding_dim = 8, decoder_embedding_dim = 8, hidden_dim = 16,  n_layers = 2, target_len = 3):
        super().__init__()
        self.target_len = target_len
        self.out_dim = feature_dim
        self.encoder = self.EncoderLSTM(feature_dim, encoder_embedding_dim, hidden_dim, n_layers)
        self.decoder = self.AttenDecoderLSTM(feature_dim, decoder_embedding_dim, hidden_dim, n_layers)

    def forward(self, x):
        batch_size = x.shape[1]
        out = torch.zeros((self.target_len, batch_size, self.out_dim))
        output, hidden = self.encoder(x)
        #print(output.shape, output[-1])
        #print(hidden[0][-1])
        #hidden = hidden[0]
        #print(hidden[0].shape, hidden[1].shape, batch_size)
        
        for i in range(self.target_len):
            de_out, hidden = self.decoder(hidden, output)
           # print(de_out.shape)
            out[i] = de_out
        return out
        '''
        return self.decoder(de_in, output)
        '''


    class EncoderLSTM(nn.Module):
        def __init__(self, input_size=5, embedding_dim=8, hidden_size=16, nlayers=2):
            super().__init__()
            self.embedding = nn.Linear(input_size, embedding_dim)
            self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=nlayers, bias=True, dropout=0)

        def forward(self, x):
            embedded = self.embedding(x)
            out, hidden = self.lstm(embedded)
            return out, hidden
    
    class AttenDecoderLSTM(nn.Module):
        def __init__(self, output_size=5, embedding_dim=8, hidden_size=16, nlayers=2):
            super().__init__()
            self.embedding = nn.Linear(hidden_size, embedding_dim)
            self.lstm = nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=nlayers, bias=True, dropout=0)
            self.linear = nn.Linear(hidden_size, output_size)
        
        def forward(self, hidden, encoder_hidden):
            hidden_unit = hidden[0][-1]
            atten = torch.sum(hidden_unit*encoder_hidden,dim=-1)
            atten = torch.softmax(atten,dim=0)
            batch_size = atten.shape[1]
            input = torch.zeros(batch_size, encoder_hidden.shape[-1])
            for i in range(batch_size):
                input[i] = torch.matmul(atten[:,i].T, encoder_hidden[:,i])
            
            input = torch.cat([hidden_unit, input], dim=-1)
            #input = torch.cat([embedded, input], dim=-1)
            input = input.unsqueeze(0)
            y, hidden = self.lstm(input, hidden)

            ret = self.linear(y.squeeze(0))
            #print("ret", ret.shape)
            return ret, hidden

model = EncoderDecoder()
print(model(x))