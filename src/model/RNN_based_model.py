from torch import nn


class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, src):
        embeddings = self.embedding(src)
        outputs, hidden = self.rnn(embeddings)
        return outputs, hidden
    
class RNNDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, context, hidden):
        # Ensure hidden state has the correct shape
        hidden = hidden.unsqueeze(0)  # Add an extra layer dimension
        embeddings = self.embedding(input)
        output, hidden = self.rnn(embeddings, hidden.squeeze(0))
        prediction = self.fc_out(output)
        return prediction, hidden.squeeze(0)  # Remove the extra layer dimension
    

class RNN_Seq2Seq_Model(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, sequence_en, sequence_vn):
        outputs, hidden = self.encoder(sequence_en)
        decoder_outputs, _ = self.decoder(sequence_vn, outputs, hidden)
        return decoder_outputs