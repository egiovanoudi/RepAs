import argparse
import pandas as pd
import torch
import os

from data import *
from model import *


if __name__ == '__main__':
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   parser = argparse.ArgumentParser()
   parser.add_argument('--train_path', default='data/train_sample.txt')
   parser.add_argument('--val_path', default='data/val_sample.txt')
   parser.add_argument('--test_path', default='data/test_sample.txt')
   parser.add_argument('--model_path', default='ckpts')

   parser.add_argument('--hidden_size', type=int, default=256)
   parser.add_argument('--embedding_size', type=int, default=10)
   parser.add_argument('--conv_size', type=int, default=40)
   parser.add_argument('--kernel_conv_size', type=int, default=12)
   parser.add_argument('--kernel_max1_size', type=int, default=3)
   parser.add_argument('--kernel_max2_size', type=int, default=4)
   parser.add_argument('--final_linear_size', type=int, default=32)

   parser.add_argument('--k', type=int, default=4)
   parser.add_argument('--epochs', type=int, default=6)
   parser.add_argument('--batch_size', type=int, default=16)
   parser.add_argument('--lr', type=float, default=0.001)
   parser.add_argument('--dropout', type=float, default=0.1)
   parser.add_argument('--lamda', type=float, default=0.1)
   args = parser.parse_args()

   train_data = pd.read_csv(args.train_path, sep='\t')
   val_data = pd.read_csv(args.val_path, sep='\t')
   test_data = pd.read_csv(args.test_path, sep='\t')

   len_train = len(train_data)
   len_val = len(val_data)

   dataset = pd.concat([train_data, val_data, test_data])
   dataset['sequence'] = dataset['sequence'].str.upper()

   print('Data preprocessing...')

   vocab = create_kmer_vocab(dataset['sequence'], args.k)
   dataset['x'], max_len, feature_size = prepare_data(dataset['sequence'], args.k, vocab)

   train_data = dataset.iloc[:len_train]
   val_data = dataset.iloc[len_train:len_train + len_val]
   test_data = dataset.iloc[len_train + len_val:]

   train_data_grouped = train_data.groupby('gene')
   val_data_grouped = val_data.groupby('gene')
   test_data_grouped = test_data.groupby('gene')

   train_loader = create_dataloader(train_data_grouped, args.batch_size, True)
   val_loader = create_dataloader(val_data_grouped, args.batch_size, False)
   test_loader = create_dataloader(test_data_grouped, args.batch_size, False)

   # Initialize the model
   my_model = RepAs(args, len(vocab) + 1, max_len, feature_size)
   my_model.to(device)
   optimizer = torch.optim.Adam(my_model.parameters(), lr=args.lr)

   print('Model training...')
   best_mae = 100
   best_epoch = 0
   for i in range(args.epochs):
       total_loss = 0
       train_mae = []
       for batch in train_loader:
           gene_loss = []
           loss = 0
           my_model.train()
           optimizer.zero_grad()
           for gene in range(len(batch)):
               gene_loss.append(my_model(batch[gene]))
           loss = sum(gene_loss) / len(gene_loss)
           loss.backward()
           optimizer.step()
           total_loss += loss.item()

       print(f'Epoch {i + 1}: Training Loss =', total_loss)
       eval_mae = evaluate(my_model, val_loader, False)
       ckpt = (my_model.state_dict(), optimizer.state_dict())
       torch.save(ckpt, os.path.join(args.model_path, f"model{i + 1}"))
       print(f'Epoch {i + 1}: Validation MAE =', eval_mae)
       if eval_mae < best_mae:
           best_mae = eval_mae
           best_epoch = i + 1

   best_ckpt = os.path.join(args.model_path, f"model{best_epoch}")
   my_model.load_state_dict(torch.load(best_ckpt)[0])
   evaluate(my_model, test_loader, True)