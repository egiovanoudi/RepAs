import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from utils import *


def get_all_features(seq):
    features=[]
    features.extend(polya_signal_features(seq))
    features.extend(aue_features(seq))
    features.extend(cue_features(seq))
    features.extend(cde_features(seq))
    features.extend(ade_features(seq))
    features.extend(rbp_features(seq))
    features.extend(mer_features(seq,'one'))
    features.extend(mer_features(seq,'two'))
    features.extend(mer_features(seq,'three'))
    features.extend(mer_features(seq,'four'))

    return features

def generate_kmers(sequence, k):    #Convert DNA sequences into k-mers
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def create_kmer_vocab(sequences, k):    #Assign a unique integer to each k-mer
    kmer_set = set()
    for seq in sequences:
        kmers = generate_kmers(seq, k)
        kmer_set.update(kmers)
    kmer_to_index = {kmer: idx+1 for idx, kmer in enumerate(kmer_set)}
    return kmer_to_index

def encode_kmers(sequence, k, kmer_to_index):   #Convert sequences into numerical representations
    kmers = generate_kmers(sequence, k)
    return [kmer_to_index[kmer] for kmer in kmers]

def prepare_data(sequences, k, kmer_to_index):
    features = [torch.tensor(get_all_features(seq), dtype=torch.float) for seq in sequences]
    features = torch.stack(features)
    features = (features - features.mean()) / features.std()
    feature_size = features.shape[1]

    encoded_seq = [torch.tensor(encode_kmers(seq, k, kmer_to_index)) for seq in sequences]
    encoded_seq = pad_sequence(encoded_seq, batch_first=True, padding_value=0)
    max_len = encoded_seq.shape[1]
    x = torch.cat((encoded_seq, features), dim=1)

    return x.tolist(), max_len, feature_size

def create_dataloader(data_grouped, batch_size, shuffle):
    data_list = []
    for _, batch in data_grouped:
        pas = batch['pas']
        x = batch['x']
        y = torch.tensor(batch['usage'].tolist(), dtype=torch.float)
        data = Data(x=x, edge_index=pas, y=y, num_nodes=0)
        data_list.append(data)

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)
    return loader