import torch.nn as nn
import torch

class Transformer_Encoder(nn.Module):
    def __init__(self, vocab_size_en, embedding_dim, model_dim, nhead):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size_en, embedding_dim)
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=model_dim,
                                                              nhead=nhead,
                                                              dim_feedforward=6,
                                                              dropout=0.0,
                                                              batch_first=True)

    # src = [batch_size, seq_length]
    def forward(self, src):
        embedded = self.embedding(src)                # [batch_size, seq_length, d]
        context = self.transformer_encoder(embedded)  # [batch_size, seq_length, d]
        return context
    
class Transformer_Decoder(nn.Module):
    def __init__(self, vocab_size_vn, embedding_dim, model_dim, nhead, sequence_length_vn):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size_vn, embedding_dim)
        self.mask = torch.triu(torch.ones(sequence_length_vn, sequence_length_vn), diagonal=1).bool()
        self.transformer_decoder = nn.TransformerDecoderLayer(d_model=model_dim,
                                                              nhead=nhead,
                                                              dim_feedforward=6,
                                                              dropout=0.0,
                                                              batch_first=True)
        self.fc_out = nn.Linear(model_dim, vocab_size_vn)

    # input: [batch_size, seq_length_vn]
    # context: [batch_size, seq_length_en, d]
    def forward(self, input, context):
        embedded = self.embedding(input)                                           # [batch_size, seq_length_vn, d]
        output = self.transformer_decoder(embedded, context, tgt_mask=self.mask)   # [batch_size, seq_length_vn, d]
        prediction = self.fc_out(output)                                           # [batch_size, seq_length_vn, vocab_size_vn]

        return prediction.unsqueeze(1)                                 # [batch_size, vocab_size_vn, seq_length_vn]
    

class Transformer_Seq2Seq_Model(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, sequence_en, sequence_vn):
        context = self.encoder(sequence_en)
        outputs = self.decoder(sequence_vn, context)
        return outputs