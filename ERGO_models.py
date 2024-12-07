import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DoubleLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, dropout, device):
        super(DoubleLSTMClassifier, self).__init__()
        # GPU
        self.device = device
        # Dimensions
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.dropout_val = dropout
        # Embedding matrices - 20 amino acids + padding
        self.tcr_embedding = nn.Embedding(20 + 1, embedding_dim, padding_idx=0)
        self.pep_embedding = nn.Embedding(20 + 1, embedding_dim, padding_idx=0)
        # Bi-directional LSTMs (num_layers=2, bidirectional=True)
        self.tcr_lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers=2, batch_first=True, dropout=dropout, bidirectional=True)
        self.pep_lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers=2, batch_first=True, dropout=dropout, bidirectional=True)
        # Since each LSTM is bidirectional, output dimension = 2*lstm_dim for each sequence (TCR and peptide)
        # After concatenation of TCR and peptide: 2*lstm_dim (TCR) + 2*lstm_dim (pep) = 4*lstm_dim
        self.hidden_layer = nn.Linear(lstm_dim * 4, lstm_dim)
        self.relu = torch.nn.LeakyReLU()
        self.output_layer = nn.Linear(lstm_dim, 1)
        self.dropout = nn.Dropout(p=dropout)

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(2*2, batch_size, self.lstm_dim)).to(self.device),
                autograd.Variable(torch.zeros(2*2, batch_size, self.lstm_dim)).to(self.device))
        # 2*2 because num_layers=2 and bidirectional=True

    def lstm_pass(self, lstm, padded_embeds, lengths):
        # Sort sequences by length
        lengths, perm_idx = lengths.sort(0, descending=True)
        padded_embeds = padded_embeds[perm_idx]
        # Pack
        # packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(padded_embeds, lengths, batch_first=True)
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(padded_embeds, lengths.cpu().to(torch.int64), batch_first=True)
        # Initialize hidden
        batch_size = len(lengths)
        hidden = self.init_hidden(batch_size)
        # Forward pass through LSTM
        lstm_out, hidden = lstm(packed_embeds, hidden)
        # Unpack
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # Restore original order
        _, unperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unperm_idx]
        lengths = lengths[unperm_idx]
        return lstm_out, lengths

    def forward(self, tcrs, tcr_lens, peps, pep_lens):
        # TCR LSTM
        tcr_embeds = self.tcr_embedding(tcrs)
        tcr_lstm_out, tcr_lens = self.lstm_pass(self.tcr_lstm, tcr_embeds, tcr_lens)
        # Extract forward last and backward last hidden states for TCR
        # Forward last hidden state: tcr_lstm_out[i, tcr_lens[i]-1, :lstm_dim]
        # Backward last hidden state: tcr_lstm_out[i, 0, lstm_dim:]
        tcr_last_cell_fwd = torch.cat([tcr_lstm_out[i, j.data - 1, :self.lstm_dim].unsqueeze(0) for i, j in enumerate(tcr_lens)], dim=0)
        tcr_last_cell_bwd = torch.cat([tcr_lstm_out[i, 0, self.lstm_dim:].unsqueeze(0) for i, j in enumerate(tcr_lens)], dim=0)
        tcr_last_cell = torch.cat([tcr_last_cell_fwd, tcr_last_cell_bwd], dim=1)  # shape: [batch, 2*lstm_dim]

        # Peptide LSTM
        pep_embeds = self.pep_embedding(peps)
        pep_lstm_out, pep_lens = self.lstm_pass(self.pep_lstm, pep_embeds, pep_lens)
        # Extract forward last and backward last hidden states for peptide
        pep_last_cell_fwd = torch.cat([pep_lstm_out[i, j.data - 1, :self.lstm_dim].unsqueeze(0) for i, j in enumerate(pep_lens)], dim=0)
        pep_last_cell_bwd = torch.cat([pep_lstm_out[i, 0, self.lstm_dim:].unsqueeze(0) for i, j in enumerate(pep_lens)], dim=0)
        pep_last_cell = torch.cat([pep_last_cell_fwd, pep_last_cell_bwd], dim=1)  # shape: [batch, 2*lstm_dim]

        # Concatenate TCR and peptide representations: [batch, 4*lstm_dim]
        tcr_pep_concat = torch.cat([tcr_last_cell, pep_last_cell], 1)

        # MLP
        hidden_output = self.dropout(self.relu(self.hidden_layer(tcr_pep_concat)))
        mlp_output = self.output_layer(hidden_output)
        output = torch.sigmoid(mlp_output)
        return output


class PaddingAutoencoder(nn.Module):
    def __init__(self, input_len, input_dim, encoding_dim):
        super(PaddingAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.input_len = input_len
        self.encoding_dim = encoding_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_len * self.input_dim, 300),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(300, 100),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(100, self.encoding_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.encoding_dim, 100),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(100, 300),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(300, self.input_len * self.input_dim)
        )

    def forward(self, batch_size, padded_input):
        concat = padded_input.view(batch_size, self.input_len * self.input_dim)
        encoded = self.encoder(concat)
        decoded = self.decoder(encoded)
        decoding = decoded.view(batch_size, self.input_len, self.input_dim)
        decoding = F.softmax(decoding, dim=2)
        return decoding


class AutoencoderLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, device, max_len, input_dim, encoding_dim, batch_size, ae_file, train_ae):
        super(AutoencoderLSTMClassifier, self).__init__()
        # GPU
        self.device = device
        # Dimensions
        self.embedding_dim = embedding_dim
        self.lstm_dim = encoding_dim
        self.max_len = max_len
        self.input_dim = input_dim
        self.batch_size = batch_size
        # TCR Autoencoder
        self.autoencoder = PaddingAutoencoder(max_len, input_dim, encoding_dim)
        checkpoint = torch.load(ae_file, map_location=device)
        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        if train_ae is False:
            for param in self.autoencoder.parameters():
                param.requires_grad = False
        self.autoencoder.eval()
        # Embedding for peptide
        self.pep_embedding = nn.Embedding(20 + 1, embedding_dim, padding_idx=0)
        # Bi-directional LSTM for peptide
        self.pep_lstm = nn.LSTM(embedding_dim, self.lstm_dim, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
        # With bidirection, peptide output dim = 2*lstm_dim
        # mlp_dim = encoding_dim (from TCR AE) + 2*lstm_dim (from pep)
        self.mlp_dim = encoding_dim + 2 * encoding_dim
        self.hidden_layer = nn.Linear(self.mlp_dim, self.mlp_dim // 2)
        self.relu = torch.nn.LeakyReLU()
        self.output_layer = nn.Linear(self.mlp_dim // 2, 1)
        self.dropout = nn.Dropout(p=0.1)

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(2*2, batch_size, self.lstm_dim)).to(self.device),
                autograd.Variable(torch.zeros(2*2, batch_size, self.lstm_dim)).to(self.device))

    def lstm_pass(self, lstm, padded_embeds, lengths):
        lengths, perm_idx = lengths.sort(0, descending=True)
        padded_embeds = padded_embeds[perm_idx]
        # packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(padded_embeds, lengths, batch_first=True)
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(padded_embeds, lengths.cpu().to(torch.int64), batch_first=True)
        batch_size = len(lengths)
        hidden = self.init_hidden(batch_size)
        lstm_out, hidden = lstm(packed_embeds, hidden)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        _, unperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unperm_idx]
        lengths = lengths[unperm_idx]
        return lstm_out, lengths

    def forward(self, padded_tcrs, peps, pep_lens):
        # TCR encoding from AE
        concat = padded_tcrs.view(self.batch_size, self.max_len * self.input_dim)
        encoded_tcrs = self.autoencoder.encoder(concat)  # shape: [batch, encoding_dim]

        # Peptide encoder
        pep_embeds = self.pep_embedding(peps)
        pep_lstm_out, pep_lens = self.lstm_pass(self.pep_lstm, pep_embeds, pep_lens)

        # Forward and backward extraction
        pep_last_cell_fwd = torch.cat([pep_lstm_out[i, j.data - 1, :self.lstm_dim].unsqueeze(0) for i, j in enumerate(pep_lens)], dim=0)
        pep_last_cell_bwd = torch.cat([pep_lstm_out[i, 0, self.lstm_dim:].unsqueeze(0) for i, j in enumerate(pep_lens)], dim=0)
        pep_last_cell = torch.cat([pep_last_cell_fwd, pep_last_cell_bwd], dim=1)  # shape: [batch, 2*lstm_dim]

        # Concatenate TCR encoding and peptide encoding
        # encoded_tcrs: [batch, encoding_dim]
        # pep_last_cell: [batch, 2*lstm_dim] but lstm_dim == encoding_dim
        # so final dimension = encoding_dim + 2*encoding_dim = 3*encoding_dim
        tcr_pep_concat = torch.cat([encoded_tcrs, pep_last_cell], 1)
        hidden_output = self.dropout(self.relu(self.hidden_layer(tcr_pep_concat)))
        mlp_output = self.output_layer(hidden_output)
        output = torch.sigmoid(mlp_output)
        return output



# python3 ERGO.py train lstm mcpas specific cuda:0 --model_file=model_run_double_lstm_1.pt --train_data_file=/home/nskane/ERGO/ERGO/data/data/BAP/tcr_split/train.csv --test_data_file=/home/nskane/ERGO/ERGO/data/data/BAP/tcr_split/test.csv
