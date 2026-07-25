import os
import torch
from   torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from   pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import sys
sys.path.append('../learnts')
from   db_ts1x import symbol2number, DB, my_collate_fn, num_images
from   model import MyModel


if __name__=="__main__":
    import os
    import pickle
    import numpy as np
    from torch.utils.data import DataLoader
    from functools import partial
    from argparse import ArgumentParser
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import seed_everything
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from itertools import product
    from glob import glob 
    import re 


    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    grid = np.mgrid[0.0:30.0:301j]
    db = pickle.load(open('../learnts/test-db-distance_rgdy.pkl','rb'))
    db_length =  len(db)
    print(f'LENGTH of DATABASE: {db_length}')
    loader = DataLoader(db, batch_size = args.batch_size, shuffle=False, collate_fn=partial(my_collate_fn, num_images=num_images, grid = grid, alpha=10.0, duplicate=False), num_workers=32)

    # use 6 different combinations of coefficients and three random seeds.
    # folder = []
    # for i, j,v in product([500.0,1000.0,2000.0],[1.0,0.0], [0,1,2]):
    #     folder+= ['lightning_csv_logs/128-2-2-0.4-0.5-0.5-'+str(i)+'-'+str(j)+'-0.0-'+str(j)+'-'+str(j)+'-2.00000E-04/version_'+str(v)+'/checkpoints/']
    # print(folder)
    
    # list_filenames=[]
    # for f in folder:
    #     list_filenames+=glob(f+'epoch*ckpt')
    # print(list_filenames)
    # print(len(list_filenames))
    # for filename in list_filenames:
    #     words = re.split('-',re.split('/',filename)[1])
    #     learning_rate = float(words[-2]+'-'+words[-1])
    #     factor1 = float(words[-7])
    #     factor2 = float(words[-6])
    #     num_layers = int(words[-12])
    #     num_gru_layers = int(words[-11])
    #     gru_dropout = float(words[-10])
    #     dropout_d = float(words[-9])
    #     dropout_e = float(words[-8])
    #     num_pair_embedding = int(words[-13])
    #     epoch = re.split('-',re.split('/',filename)[-1])[0]
    #     version = re.split('_',re.split('/',filename)[2])[-1]


    #     model = MyModel(num_pair_embedding, len(grid), num_layers, num_gru_layers=num_gru_layers, gru_dropout = gru_dropout, dropout_d=dropout_d, dropout_e=dropout_e, learning_rate=learning_rate, precision=args.precision, factor=[ factor1, factor2, 0.0, factor2, factor2 ])

    #     model = model.load_from_checkpoint(checkpoint_path=filename )
    #     model.filename = name+"-result-" +f'{num_pair_embedding}-{num_layers}-{factor1}-{factor2}-{learning_rate:.5E}-{epoch}-{version}_v1.pkl'
    #     model.TTA = True # or False

    #     print(model.filename)
    #     trainer = pl.Trainer.from_argparse_args(args)
    #     print( trainer.test(model, dataloaders=loader) )

    learning_rate = 0.0002
    factor1 = 1000.0
    factor2 = 0.0
    num_layers = 2
    num_gru_layers = 2
    gru_dropout = 0.4
    dropout_d = 0.5
    dropout_e = 0.5
    num_pair_embedding = 128

    # Trained checkpoint, e.g. './lightning_csv_logs/<run>/version_0/checkpoints/<name>.ckpt'
    filename = ''
    model = MyModel(num_pair_embedding, len(grid), num_layers, num_gru_layers=num_gru_layers, gru_dropout = gru_dropout, dropout_d=dropout_d, dropout_e=dropout_e, learning_rate=learning_rate, precision=args.precision, factor=[ factor1, factor2, 0.0, factor2, factor2 ])
    model = model.load_from_checkpoint(checkpoint_path=filename )
    model.filename = 'train_rgdy10_ep565_test_rgdy100.pkl'
    model.TTA = False # or False

    print(model.filename)
    trainer = pl.Trainer.from_argparse_args(args)
    print( trainer.test(model, dataloaders=loader) )
