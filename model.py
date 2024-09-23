import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *


class RepAs(nn.Module):
    def __init__(self, args, vocab_size, max_len, V_size):
        super(RepAs, self).__init__()
        self.hidden_size = args.hidden_size
        self.max_len = max_len
        self.lamda = args.lamda

        self.encoder = Encoder(args, vocab_size, max_len)
        self.V_linear = nn.Linear(V_size, args.hidden_size)

        self.linear2 = nn.Linear(args.hidden_size, args.final_linear_size)
        self.linear3 = nn.Linear(args.final_linear_size, 1)

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.kaiming_uniform_(param)


    def forward(self, data):
        x, y = data.x, data.y
        loss = 0

        x = [torch.tensor(pas) for pas in x]
        x = torch.stack(x)

        K_true = x[:, :self.max_len].int().cuda()
        V_true = x[:, self.max_len:].cuda()

        V_true = self.V_linear(V_true)
        V = V_true[0, :].unsqueeze(0)

        for t in range(len(K_true)):
            K = K_true[:t+1, :]
            V = self.encoder(V, K)
            X = F.relu(self.linear2(V))
            X = F.sigmoid(self.linear3(X))
            loss = loss + mae_loss(y[:t+1].cuda(), X[:t+1].flatten())

            if t < (len(K_true)-1):
                V = torch.cat((V, V_true[t+1, :].unsqueeze(0)), dim=0)

        loss = loss + dev_loss(X.flatten(), self.lamda)
        return loss

    def predict(self, data):
        x = data.x
        x = [torch.tensor(pas) for pas in x]
        x = torch.stack(x)

        K_true = x[:, :self.max_len].int().cuda()
        V_true = x[:, self.max_len:].cuda()

        V_true = self.V_linear(V_true)
        V = V_true[0, :].unsqueeze(0)

        for t in range(len(K_true)):
            K = K_true[:t+1, :]
            V = self.encoder(V, K)
            if t < (len(K_true) - 1):
                V = torch.cat((V, V_true[t + 1, :].unsqueeze(0)), dim=0)

        X = F.relu(self.linear2(V))
        X = F.sigmoid(self.linear3(X))

        return X.flatten()


class Encoder(nn.Module):
    def __init__(self, args, vocab_size, max_len):
        super(Encoder, self).__init__()
        conv_size = args.conv_size
        after_seq_size = (((max_len-args.kernel_conv_size+1)//args.kernel_max1_size)-args.kernel_conv_size+1)//args.kernel_max2_size

        self.embed = nn.Embedding(vocab_size, args.embedding_size, padding_idx=0)
        self.Seq_encoder = nn.Sequential(
            nn.Conv1d(args.embedding_size, conv_size, kernel_size=args.kernel_conv_size),
            nn.BatchNorm1d(conv_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=args.kernel_max1_size),
            nn.Conv1d(conv_size, conv_size, kernel_size=args.kernel_conv_size),
            nn.BatchNorm1d(conv_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=args.kernel_max2_size),
        )
        self.lstm = nn.LSTM(conv_size*after_seq_size, args.hidden_size)

        self.att = self.attention_module(args.hidden_size)
        self.refine = Refinement(args.hidden_size, dropout=args.dropout)

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.kaiming_uniform_(param)


    def forward(self, V, K):
        H_K = self.embed(K)
        H_K = [torch.t(pas) for pas in H_K]
        H_K = torch.stack(H_K)

        H_K = self.Seq_encoder(H_K)
        H_K = H_K.view(H_K.shape[0], H_K.shape[1]*H_K.shape[2])
        H_K, _ = self.lstm(H_K)
        H_K = F.relu(H_K)

        VH_K = torch.cat([V, H_K], dim=-1)
        Vmask = torch.t(self.att(torch.t(V)))
        H_v = self.refine(V, VH_K, Vmask)

        return H_v

    def attention_module(self, hidden_dim):
        att_block = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        return att_block


class Refinement(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(Refinement, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = Normalize(hidden_size)
        self.Linear_module = nn.Sequential(
                nn.Linear(2*hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
        )


    def forward(self, V, VH_K, Vmask):
        message = self.Linear_module(VH_K) * Vmask
        mes_mean = torch.mean(message, dim=-2)
        H_V = self.norm(V + self.dropout(mes_mean))

        return H_V