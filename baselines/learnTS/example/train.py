import os
import torch
from   torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from   pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from   pytorch_lightning.loggers import CSVLogger
import sys
sys.path.append('../learnts')
from db_ts1x import symbol2number, num_images
from model import MyModel

if __name__=="__main__":
    import os
    import pickle
    import numpy as np
    from db_ts1x import DB, my_collate_fn
    from torch.utils.data import DataLoader
    from functools import partial
    from argparse import ArgumentParser
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import seed_everything
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    import socket
    from itertools import product
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)


    parser = ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_pair_embedding", type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_gru_layers', type=int, default=2)
    parser.add_argument('--gru_dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--factor1', type=float, default=1000.0) #description="This value correspond c in Equation 5 in the reference paper")
    parser.add_argument('--factor2', type=float, default=0.0) #description="This value correspond c' in Equation 5 in the reference paper")
    parser.add_argument('--factor3', type=float, default=0.0)
    parser.add_argument('--factor4', type=float, default=0.0)
    parser.add_argument('--factor5', type=float, default=0.0) #description="This value correspond c' in Equation 5 in the reference paper")
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()    

    grid = np.mgrid[0.0:30.0:301j]
    print(f'start learning :   {args.factor1}, {args.factor2}')

    db = pickle.load(open("../learnts/train-db-distance_rgdy_10.pkl",'rb')) ############
    #db = pickle.load(open('./train-db-fast.pkl','rb'))
    train_db_length =  len(db)
    #loader1 = DataLoader(db, batch_size = args.batch_size, shuffle=True,  num_workers=32)
    loader1 = DataLoader(db, batch_size = args.batch_size, shuffle=True,  collate_fn=partial(my_collate_fn, num_images=num_images, grid = grid, alpha=10.0, duplicate=True),  num_workers=4)
    
    #db = pickle.load(open('./valid-db-fast.pkl','rb'))
    db = pickle.load(open("../learnts/valid-db-distance_rgdy_10.pkl",'rb')) #############
    #db, _ = torch.utils.data.random_split(db, [100, len(db)-100])
    valid_db_length =  len(db)
    #loader2 = DataLoader(db,  batch_size = 2*args.batch_size, shuffle=False, num_workers=16)
    loader2 = DataLoader(db, batch_size = 2*args.batch_size, shuffle=False, collate_fn=partial(my_collate_fn, num_images=num_images, grid = grid, alpha=10.0, duplicate=False), num_workers=4)
    
    print(f'db lengths: {train_db_length}, {valid_db_length}')
    for dropout_d, dropout_e\
        in product( [ 0.5],[ 0.5,]) :
        #in product([256,128, 64],[256,128,64],[256],[5,3,1], [ 0.3],[ 0.3],[0.5,],[ 1e-4] ) :

        print(args.num_pair_embedding, args.num_layers, args.gru_dropout, dropout_d, dropout_e, args.learning_rate, args.precision)
        checkpoint_callback = ModelCheckpoint( 
                                               monitor='ts_edm_loss_v', 
                                               save_top_k = 5, save_last=True)
        args.callbacks=[checkpoint_callback]
    
        #tensorboard_logger = TensorBoardLogger('lightning_logs', default_hp_metric=True, name=f'{args.num_pair_embedding}-{args.num_layers}-{args.num_gru_layers}-{gru_dropout}-{dropout_d}-{dropout_e}-{args.learning_rate:.5E}')
        csv_logger = CSVLogger('lightning_csv_logs', name=f'{args.num_pair_embedding}-{args.num_layers}-{args.num_gru_layers}-{args.gru_dropout}-{dropout_d}-{dropout_e}-{args.factor1}-{args.factor2}-{args.factor3}-{args.factor4}-{args.factor5}-{args.learning_rate:.5E}')
    
        #model = MyModel(num_pair_embedding, len(grid), forward, update, gru, transformer_encoder, dropout_d=dropout_d, dropout_e=dropout_e, learning_rate=learning_rate, precision=precision, factor1=factor1, factor2=factor2)
        model = MyModel(args.num_pair_embedding, len(grid),   args.num_layers, num_gru_layers=args.num_gru_layers, gru_dropout = args.gru_dropout, dropout_d=dropout_d, dropout_e=dropout_e, learning_rate=args.learning_rate, precision=args.precision, factor=[args.factor1, args.factor2, args.factor3, args.factor4, args.factor5])
#        if (i>0):
#            os.system('cp lightning_eigvals_v4_logs/150.183.238.215/version_'+str(i-1)+'/checkpoints/epo* model.ckpt')
#        else:
#            os.system('cp model_original.ckpt model.ckpt')
#        os.system('ls model.ckpt -l')
#        model = model.load_from_checkpoint(checkpoint_path='model.ckpt')
        trainer = pl.Trainer.from_argparse_args(
    args,
    logger=csv_logger,
    callbacks=[checkpoint_callback],
)
        #trainer.logger=[tensorboard_logger, csv_logger]
        # trainer.logger=csv_logger
        trainer.fit(model, loader1, loader2)
    
