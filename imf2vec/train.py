from model import Imf2Vec
from embedding import Item2Vec
from utils import log_transform, linear_transform, get_ratings
import pandas as pd
import pickle
import scipy.sparse as sp
import argparse
import numpy as np
import wandb
import os
# os.environ['WANDB_API_KEY'] = # INSERT weights&biases key here for experiment tracking

parser = argparse.ArgumentParser(description='Imf2Vec')
parser.add_argument('-d','--dataset', choices=['bets','lastfm'], required=True,
                    help='dataset to run experiment on (either lastfm or bets)')
parser.add_argument('-e','--embedding-size', type=int, default=32,
                    help='item2vec embedding dimension size')    
parser.add_argument('-f','--factors', type=int, default=128, metavar='F',
                    help='latent dimension size')    
parser.add_argument('--epochs', type=int, default=750, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('-lr', type=float, default=0.01,
                    help='learning rate to apply for user updates')                      
parser.add_argument('--l2', type=float, default=0.1, metavar='l2',
                    help='L2 regularisation parameter')
parser.add_argument('--eta', type=float, default=0.0,
                    help='eta momentum param')
parser.add_argument('-a', type=float, default=0.5,
                    help='similarity parameter, adjusts item2vec regularisation strength')
parser.add_argument('-k', type=int, default=4,
                    help='neighbours to compare similarity to')
parser.add_argument('-t', '--transform', choices=['log', 'linear', 'ratings', None], default=None,
                    help='choose data preprocessing')
parser.add_argument('--alpha', type=float, default=0.1,
                    help='linear scaling parameter in transformation')
parser.add_argument('--eps', type=float, default=10e-04,
                    help='log scaling parameter in transformation')
parser.add_argument('--top-n', type=int, default=5, metavar='top_N',
                    help='number of recommendations to make per user - to be evaluated used MAP@N and Recall')
parser.add_argument('--project', type=str, required=True,
                    help='wandb project name')
args = parser.parse_args()


path = 'data/'+args.dataset+'/train_val/item2vec_train_ex_val.pkl'
sequences_df = pd.read_pickle(path)

# load train data
raw_data = {
    'train': sp.load_npz('data/'+args.dataset+'/train_val/train_ex_val.npz'),
    'val': sp.load_npz('data/'+args.dataset+'/train_val/val_unmasked.npz'),
    'val_masked': sp.load_npz('data/'+args.dataset+'/train_val/val_masked.npz'),
    'test': sp.load_npz('data/'+args.dataset+'/full_train/test_unmasked.npz'),
    'test_masked': sp.load_npz('data/'+args.dataset+'/full_train/test_masked.npz')
}

# full train
# path = 'data/'+args.dataset+'/full_train/item2vec_train_full.pkl'
# sequences_df = pd.read_pickle(path)

# load items list
with open('data/'+args.dataset+'/full_train/test_items.txt', 'rb') as f:
    # read the data as binary data stream
    items = pickle.load(f)


# set specific transforms
if args.dataset == 'lastfm':
    # args.num_items = 26307 # full train
    args.num_items = 26775
    args.t = 'log'
    args.alpha = 40
    args.eps = 10e-04
    lr_dict = {'u':args.lr, 'i':args.lr}
    column_name = 'artist_per_day'
else:
    raise ValueError('invalid dataset selected')

wandb.init(project=args.project)
wandb.config.update(args)

# transform data
if args.transform == 'log':
    data = {k: log_transform(v, args.alpha, args.eps) for k, v in raw_data.items()}
elif args.transform == 'linear':
    data = {k: linear_transform(v, args.alpha) for k, v in raw_data.items()}
else:
    data = raw_data
    
item2vec = Item2Vec(df=sequences_df, 
                    column_name=column_name, 
                    window_size=5, 
                    embedding_size=args.embedding_size)

item2vec.similarity(train_items=items)

model = Imf2Vec(counts=data['train'],
                Item2Vec=item2vec,
                full_items=items,
                k=args.k,
                num_factors=args.factors,
                num_iterations=args.epochs,
                learning_rate=lr_dict,
                alpha_sim_param=args.a, 
                reg_param=args.l2,
                eta=args.eta)

model.train_model(data['val'], data['val_masked'])


