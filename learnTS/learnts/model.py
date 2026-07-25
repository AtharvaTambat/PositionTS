import os
import torch
from   torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from   pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from   db_ts1x import symbol2number, num_images

class MyModel(pl.LightningModule):
#class MyModel(nn.Module):
    def __init__(self, num_pair_embedding, num_grid, num_layers=2, num_gru_layers=1, num_encoder_layers=2, gru_dropout=0.1, dropout_d=0.1, dropout_e=0.5, dropout_h=0.5, learning_rate = 1e-4, precision=32, factor=[1,1,1,1,1], filename='output.pkl', TTA=False):
        super(MyModel, self).__init__()
        assert len(factor)==5, "length of factor should be 4"
        self.save_hyperparameters()

        self.node_embedding = nn.Embedding(len(symbol2number), num_pair_embedding//4)
        self.edge_embedding = nn.Linear(num_grid, num_pair_embedding//2, bias=False )

        self.update              = nn.ModuleList( [ nn.TransformerEncoder( nn.TransformerEncoderLayer(d_model=(2**i)*num_pair_embedding, nhead=8, dropout=gru_dropout), 2 ) for i in range(num_layers)] )         
        self.gru                 = nn.ModuleList( [ nn.GRU( (2**i)*num_pair_embedding, (2**i)*num_pair_embedding, num_gru_layers, bias=True, batch_first=False, dropout=gru_dropout, bidirectional=True)  for i in range(num_layers)] )         

        self.linear9             = nn.Linear( (2**num_layers)*num_pair_embedding, 9*num_pair_embedding//16) 

        self.readout_d = nn.Sequential(
#                                       nn.LayerNorm(9*num_pair_embedding//16),
#                                       nn.Dropout(dropout_d),
#                                       nn.Linear(9*num_pair_embedding//16, num_pair_embedding//16),
#                                       nn.ReLU(),
                                       nn.Linear(9*num_pair_embedding//16, 1),
                                       nn.ELU(),
                                      )

        self.readout_e = nn.Sequential(
#                                       nn.LayerNorm( 9*num_pair_embedding//16),
#                                       nn.Dropout(dropout_e),
#                                       nn.Linear(9*num_pair_embedding//16, num_pair_embedding//16),
#                                       nn.ReLU(),
                                       nn.Linear( 9*num_pair_embedding//16, 3),
                                      )
#        self.readout_e = nn.Sequential(
#                                       nn.LayerNorm(num_pair_embedding//4 + 9*num_pair_embedding//16),
#                                       nn.Dropout(dropout_e),
#                                       nn.Linear(num_pair_embedding//4 + 9*num_pair_embedding//16, num_pair_embedding//32),
#                                       nn.ReLU(),
#                                       nn.Linear( num_pair_embedding//32, 1),
#                                      )
        self.readout_h = nn.Sequential(
                                       nn.Linear(num_pair_embedding//16, num_pair_embedding//16),
                                       nn.ReLU(),
                                       nn.Linear(num_pair_embedding//16, num_pair_embedding//16),
                                       nn.ReLU(),
                                       nn.Linear( num_pair_embedding//16, 1),
                                      )

        self.learning_rate = learning_rate

        #self.norm = nn.LayerNorm(num_pair_embedding)
        #self.d_std = 7.570663340431402
        self.e_mean, self.e_std = 68.54006539613046, 32.42880103339008
        self.s1_mean,self.s1_std = -2.98733666, 5.74321139
        self.s2_mean,self.s2_std = -0.13244184, 0.46543787

        self.factor1=factor[0]
        self.factor2=factor[1]
        self.factor3=factor[2]
        self.factor4=factor[3]
        self.factor5=factor[4]

        self.num_layers = num_layers
        self.filename = filename
        self.TTA = TTA

    def update_pair(self, update, pair_f, mask):
        batch_size = pair_f.size(0)
        num_images_all  = mask.size(1) # 2 num_images +1
        num_edges  = pair_f.size(-2)
        pair_f = update(
                        pair_f.reshape(batch_size*num_images_all,num_edges, -1).transpose(0,1), 
                        src_key_padding_mask= (mask.reshape(batch_size*num_images_all,num_edges)==False)
                       ).transpose(0,1).reshape(batch_size, num_images_all, num_edges, -1)
        return pair_f

    def step(self,  atomic_numbers, edge_f, linear_edm, edm_mask, edm_diag_mask):

        batch_size, num_atoms = atomic_numbers.size()
        num_images_all  = edm_mask.size(1) # 2 num_images +1
        num_edges   = edge_f.size(-2)

        edge_f = self.edge_embedding(edge_f)

        triu_indices = torch.triu_indices(num_atoms, num_atoms, device=edm_mask.device)
        node_f = self.node_embedding(atomic_numbers)
        pair_f = torch.cat([node_f.unsqueeze(-2).repeat(1,1,num_atoms,1) , node_f.unsqueeze(-3).repeat(1,num_atoms,1,1) ], -1 )[:,triu_indices[0], triu_indices[1]]

        pair_f = pair_f.unsqueeze(1).repeat(1,num_images_all,1,1)

        pair_f0 = torch.cat( [pair_f, edge_f], -1 )
        pair_f  = torch.clone(pair_f0)

        #####  PSI layers  ######
        for i in range(self.num_layers):
            pair_f = self.update_pair( self.update[i], pair_f, edm_mask)
            pair_f = self.gru[i]( pair_f.transpose(0,1).reshape(num_images_all, batch_size*num_edges, -1) )[0].reshape(num_images_all, batch_size, num_edges, -1).transpose(0,1)

        pair_f = self.linear9(pair_f)
        pred_edm = (1+ self.readout_d( pair_f ).squeeze(-1) ) * linear_edm

        ##### Calculate Residue ######
        res = self.calculate_residue(pred_edm[:,num_images_all//2 ], num_atoms)

        ##### Update pair_f  #######

        ##### Hessian prediction from pair_f #####
        # To make 3N-by-3N matrix, build matrix form of pair_f for reactant, product and ts 
        mat_pair_f = torch.zeros( (batch_size, 3, num_atoms, num_atoms, pair_f.size(-1) ), dtype=pair_f.dtype, device=pair_f.device)
        mat_pair_f[:, 0, triu_indices[0], triu_indices[1]] = pair_f[:,0]
        mat_pair_f[:, 0, triu_indices[1], triu_indices[0]] = pair_f[:,0]
        mat_pair_f[:,-1, triu_indices[0], triu_indices[1]] = pair_f[:,-1]
        mat_pair_f[:,-1, triu_indices[1], triu_indices[0]] = pair_f[:,-1]
        mat_pair_f[:, 1, triu_indices[0], triu_indices[1]] = pair_f[:,num_images_all//2]
        mat_pair_f[:, 1, triu_indices[1], triu_indices[0]] = pair_f[:,num_images_all//2]

        mat_pair_f_mask = torch.zeros( (batch_size, 3, num_atoms, num_atoms ), dtype=edm_mask.dtype, device=edm_mask.device)
        mat_pair_f_mask[:, 0, triu_indices[0], triu_indices[1]] = edm_mask[:,0]
        mat_pair_f_mask[:, 0, triu_indices[1], triu_indices[0]] = edm_mask[:,0]
        mat_pair_f_mask[:,-1, triu_indices[0], triu_indices[1]] = edm_mask[:,-1]
        mat_pair_f_mask[:,-1, triu_indices[1], triu_indices[0]] = edm_mask[:,-1]
        mat_pair_f_mask[:, 1, triu_indices[0], triu_indices[1]] = edm_mask[:,num_images_all//2]
        mat_pair_f_mask[:, 1, triu_indices[1], triu_indices[0]] = edm_mask[:,num_images_all//2]

        num_mat_elements = torch.sum(mat_pair_f_mask, dim=[-1,-2])

        energy = self.readout_e( 
                                torch.sum( mat_pair_f.masked_fill(mat_pair_f_mask.unsqueeze(-1)==False, 
                                                                  0.0 ), 
                                            -2)
                               )
        energy = energy.masked_fill(mat_pair_f_mask.unsqueeze(-1)[:,:,0]==False, 0.0)                               
        energy = torch.sum( energy, -1)

        e_r =self.e_mean + self.e_std*( energy[:,1,0]-energy[:,0,0] )
        e_p =self.e_mean + self.e_std*( energy[:,1,0]-energy[:,2,0] )

        s1_r =self.s1_mean + self.s1_std*( energy[:,1,1]-energy[:,0,1] )
        s1_p =self.s1_mean + self.s1_std*( energy[:,1,1]-energy[:,2,1] )

        s2_r =self.s2_mean + self.s2_std*( energy[:,1,2]-energy[:,0,2] )
        s2_p =self.s2_mean + self.s2_std*( energy[:,1,2]-energy[:,2,2] )

        # reshape mat_pair_f as (batch_size, 3, 3*num_atoms, 3*num_atoms, feature_size) 
        mat_pair_f = mat_pair_f.reshape(batch_size, 3, num_atoms, num_atoms, 3, 3, -1 ).transpose(3,4).reshape(batch_size, 3, 3*num_atoms, 3*num_atoms, -1 )

        # readout and symmetrize
        pred_hess       = self.readout_h( mat_pair_f ).squeeze(-1)

        return  pred_edm, e_r, e_p, s1_r, s1_p, s2_r, s2_p, pred_hess, res, num_mat_elements # clear out meaningless numbers

    def calculate_residue(self, pred_edm, dim ):
        batch_size = pred_edm.size(0)
        target = torch.zeros( (batch_size, dim, dim) , dtype=pred_edm.dtype, device=pred_edm.device)
        triu_indices = torch.triu_indices(dim, dim, device=pred_edm.device)
        target[:, triu_indices[0], triu_indices[1]] = (pred_edm*pred_edm) #.masked_fill(m==False, 0.0)
        target[:, triu_indices[1], triu_indices[0]] = target[:, triu_indices[0], triu_indices[1]]


        # shift; because of numerical stability 
        I = torch.eye(target.size(-1),dtype=target.dtype, device=target.device) 
        try:
            eigvals = torch.linalg.eigvalsh( (target+I.unsqueeze(0)).float() ).type(I.dtype) -1.0
        except RuntimeError:
            with torch.no_grad():
                largest_eig = torch.lobpcg(target.float(), largest=False, method='ortho')[0]
                print(largest_eig)
                largest_eig = (largest_eig+1).reshape(batch_size, 1, 1).to(target.dtype)
            eigvals = torch.linalg.eigvalsh( target - largest_eig*I.unsqueeze(0) ) + largest_eig.squeeze(-1)

        sort_index = torch.sort( torch.abs(eigvals), -1, True)[1]
        eigvals = torch.gather(eigvals, 1, sort_index ) 
        return torch.linalg.vector_norm(eigvals[:,5:], ord=1) + torch.linalg.vector_norm(torch.sum(eigvals[:,:5], dim=-1 ), ord=1)
        #eigvals = torch.sort( torch.abs(eigvals), -1, True)[0][:,5:]
        #return torch.norm(eigvals, p=1)

    def calculate_loss(self, pred_edm, pred_e_r, pred_e_p, pred_s1_r, pred_s1_p, pred_s2_r, pred_s2_p, pred_hess, num_mat_elements, batch):
        atomic_numbers, true_edm, linear_edm, edge_f, true_e_r, true_e_p, true_s_r, true_s_p, B_matrices, true_hess, inv_weights, edm_mask, edm_diag_mask = batch 
        
        num_images = linear_edm.size(1)//2

        m = edm_mask[:,num_images]
        num_mat_e = num_mat_elements[:,num_images]

        ts_edm_diff = true_edm[:,num_images] - pred_edm[:,num_images]
        
        ref_loss = torch.sum( 2*ts_edm_diff**2, dim=[-1] ) / num_mat_e  # PCCP (2020)
        
        r_e_loss = torch.abs(pred_e_r-true_e_r )
        p_e_loss = torch.abs(pred_e_p-true_e_p )

        r_s1_loss = torch.abs(pred_s1_r - true_s_r[:,0])
        p_s1_loss = torch.abs(pred_s1_p - true_s_p[:,0])

        r_s2_loss = torch.abs(pred_s2_r - true_s_r[:,1])
        p_s2_loss = torch.abs(pred_s2_p - true_s_p[:,1])

        ts_edm_loss = 2*torch.linalg.vector_norm( ts_edm_diff, ord=1, dim=[-1]) / num_mat_e # sum of abs

        # averaging MAPE loss (bond axis) 
        # nansum exclude values where true distance ==0; 
        ts_edm_mape = 2*torch.nansum( torch.abs(ts_edm_diff) / true_edm[:,num_images], dim=-1) / num_mat_e 

        all_hess_loss = torch.linalg.norm(pred_hess - true_hess, ord='fro', dim=[1,2])
        # wrong!
        #all_hess_mape = torch.nansum( all_hess_mape, dim=-1) / torch.nansum( torch.logical_not(torch.isnan(all_hess_mape)).type(true_hess.dtype) ,  dim=-1)
        return  torch.sum(ts_edm_mape), torch.sum(ts_edm_loss), torch.sum(r_e_loss), torch.sum(p_e_loss), torch.sum(r_s1_loss), torch.sum(p_s1_loss), torch.sum(r_s2_loss), torch.sum(p_s2_loss), torch.sum( all_hess_loss ), torch.sum( ref_loss)

    def training_step(self,batch, batch_idx):
        atomic_numbers, true_edm, linear_edm, edge_f, true_e_r, true_e_p, true_s_r, true_s_p, B_matrices, true_hess, inv_weights, edm_mask, edm_diag_mask = batch 

        batch_size = atomic_numbers.size(0)
        num_atoms  = atomic_numbers.size(1)
         
        pred_edm, pred_e_r, pred_e_p, pred_s1_r, pred_s1_p, pred_s2_r, pred_s2_p, pred_hess, res, num_mat_elements= self.step(atomic_numbers, edge_f, linear_edm, edm_mask, edm_diag_mask)
        pred_hess = torch.einsum('niab,niac,nicd->nibd', B_matrices, pred_hess*inv_weights.unsqueeze(1), B_matrices) # clear out trans/rot modes (3N-6, 3N-6)
        triu_indices = torch.triu_indices(3*num_atoms-6, 3*num_atoms-6, device=pred_hess.device)
        pred_hess = pred_hess[:,:,triu_indices[0], triu_indices[1]]

        ts_edm_mape, ts_edm_loss, r_e_loss, p_e_loss, r_s1_loss, p_s1_loss, r_s2_loss, p_s2_loss, all_hess_loss, ref_loss = self.calculate_loss(pred_edm, pred_e_r, pred_e_p, pred_s1_r, pred_s1_p, pred_s2_r, pred_s2_p, pred_hess, num_mat_elements, batch)
        
        self.log('ts_edm_mape_t',  ts_edm_mape  /batch_size, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('ts_edm_loss_t',  ts_edm_loss  /batch_size, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        self.log('e_loss_t',(r_e_loss+p_e_loss)/batch_size,  prog_bar=True,logger=True, on_step=False, on_epoch=True)
        self.log('s_loss_t',(r_s1_loss+p_s1_loss+r_s2_loss+p_s2_loss)/batch_size,  prog_bar=True,logger=True, on_step=False, on_epoch=True)

        self.log('r_e_loss_t',     r_e_loss  /batch_size,    prog_bar=False,logger=True, on_step=False, on_epoch=True)
        self.log('p_e_loss_t',     p_e_loss  /batch_size,    prog_bar=False,logger=True, on_step=False, on_epoch=True)
        self.log('r_s1_loss_t',    r_s1_loss /batch_size,    prog_bar=False,logger=True, on_step=False, on_epoch=True)
        self.log('p_s1_loss_t',    p_s1_loss /batch_size,    prog_bar=False,logger=True, on_step=False, on_epoch=True)
        self.log('r_s2_loss_t',    r_s2_loss /batch_size,    prog_bar=False,logger=True, on_step=False, on_epoch=True)
        self.log('p_s2_loss_t',    p_s2_loss /batch_size,    prog_bar=False,logger=True, on_step=False, on_epoch=True)

        self.log('all_hess_loss_t',all_hess_loss/batch_size, prog_bar=True,logger=True, on_step=False, on_epoch=True)
        self.log('res_t',      res     /batch_size, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('ref_t',      ref_loss/batch_size, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return self.factor1*ts_edm_loss + self.factor2*(r_e_loss + p_e_loss) + self.factor5*(r_s1_loss + p_s1_loss + r_s2_loss + p_s2_loss) + self.factor3*all_hess_loss + self.factor4*res
        #return self.factor1*torch.sum( ts_loss )+e_r_loss+e_p_loss+self.factor1*res*self.current_epoch/args.max_epochs

    def validation_step(self,batch, batch_idx):
        atomic_numbers, true_edm, linear_edm, edge_f, true_e_r, true_e_p, true_s_r, true_s_p, B_matrices, true_hess, inv_weights, edm_mask, edm_diag_mask = batch 

        batch_size = atomic_numbers.size(0)
        num_atoms  = atomic_numbers.size(1)

        pred_edm, pred_e_r, pred_e_p, pred_s1_r, pred_s1_p, pred_s2_r, pred_s2_p, pred_hess, res, num_mat_elements= self.step(atomic_numbers, edge_f, linear_edm, edm_mask, edm_diag_mask)
        pred_hess = torch.einsum('niab,niac,nicd->nibd', B_matrices, pred_hess*inv_weights.unsqueeze(1), B_matrices) # clear out trans/rot modes (3N-6, 3N-6)
        triu_indices = torch.triu_indices(3*num_atoms-6, 3*num_atoms-6, device=pred_hess.device)
        pred_hess = pred_hess[:,:,triu_indices[0], triu_indices[1]]

        ts_edm_mape, ts_edm_loss, r_e_loss, p_e_loss, r_s1_loss, p_s1_loss, r_s2_loss, p_s2_loss, all_hess_loss, ref_loss = self.calculate_loss(pred_edm, pred_e_r, pred_e_p, pred_s1_r, pred_s1_p, pred_s2_r, pred_s2_p, pred_hess, num_mat_elements, batch)
        
        self.log('ts_edm_mape_v',  ts_edm_mape  /batch_size, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('ts_edm_loss_v',  ts_edm_loss  /batch_size, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        self.log('e_loss_v',(r_e_loss+p_e_loss)/batch_size,  prog_bar=True,logger=True, on_step=False, on_epoch=True)
        self.log('s_loss_v',(r_s1_loss+p_s1_loss+r_s2_loss+p_s2_loss)/batch_size,  prog_bar=True,logger=True, on_step=False, on_epoch=True)
        
        self.log('r_e_loss_v',     r_e_loss  /batch_size, prog_bar=False,logger=True, on_step=False, on_epoch=True)
        self.log('p_e_loss_v',     p_e_loss  /batch_size, prog_bar=False,logger=True, on_step=False, on_epoch=True)
        self.log('r_s1_loss_v',    r_s1_loss /batch_size, prog_bar=False,logger=True, on_step=False, on_epoch=True)
        self.log('p_s1_loss_v',    p_s1_loss /batch_size, prog_bar=False,logger=True, on_step=False, on_epoch=True)
        self.log('r_s2_loss_v',    r_s2_loss /batch_size, prog_bar=False,logger=True, on_step=False, on_epoch=True)
        self.log('p_s2_loss_v',    p_s2_loss /batch_size, prog_bar=False,logger=True, on_step=False, on_epoch=True)
        
        self.log('all_hess_loss_v',all_hess_loss/batch_size, prog_bar=True,logger=True, on_step=False, on_epoch=True)
        self.log('res_v',      res     /batch_size, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('ref_v',      ref_loss/batch_size, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return self.factor1*ts_edm_loss + self.factor2*(r_e_loss + p_e_loss) + self.factor5*(r_s1_loss + p_s1_loss + r_s2_loss + p_s2_loss) + self.factor3*all_hess_loss + self.factor4*res
        #return self.factor1*torch.sum( ts_loss )+e_r_loss+e_p_loss+self.factor1*res*self.current_epoch/args.max_epochs


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-6) } }
        #return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=2e-6, max_lr=self.learning_rate, step_size_up=500, step_size_down=50, mode='triangular', cycle_momentum=False) } }

        #return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5, patience=10, threshold_mode='abs',  cooldown=10, min_lr=1e-6,verbose=True) , 'monitor': 'MAPE_v'} }
#                "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1 if (epoch<50) else 0.1)} } 

    def test_step(self,batch, batch_idx):
        if (len(batch)==13):
            atomic_numbers, true_edm, linear_edm, edge_f, true_e_r, true_e_p, true_s_r, true_s_p, B_matrices, true_hess, inv_weights, edm_mask, edm_diag_mask = batch
        elif(len(batch)==4):
            atomic_numbers,  linear_edm,  edge_f, mask = batch
        else:
            raise ValueError
        batch_size = linear_edm.size(0)
        num_atoms  = atomic_numbers.size(1)
        num_images = linear_edm.size(1)//2

        pred_edm, pred_e_r, pred_e_p, pred_s1_r, pred_s1_p, pred_s2_r, pred_s2_p, pred_hess, res, num_mat_elements= self.step(atomic_numbers, edge_f, linear_edm, edm_mask, edm_diag_mask)
        pred_hess = torch.einsum('niab,niac,nicd->nibd', B_matrices, pred_hess*inv_weights.unsqueeze(1), B_matrices) # clear out trans/rot modes (3N-6, 3N-6)
        triu_indices = torch.triu_indices(3*num_atoms-6, 3*num_atoms-6, device=pred_hess.device)
        pred_hess = pred_hess[:,:,triu_indices[0], triu_indices[1]]


        if (len(batch)==13):
            ts_edm_mape, ts_edm_loss, r_e_loss, p_e_loss, r_s1_loss, p_s1_loss, r_s2_loss, p_s2_loss, all_hess_loss, ref_loss = self.calculate_loss(pred_edm, pred_e_r, pred_e_p, pred_s1_r, pred_s1_p, pred_s2_r, pred_s2_p, pred_hess, num_mat_elements, batch)

            self.log('ts_edm_mape_v',  ts_edm_mape  /batch_size, prog_bar=True, logger=True, on_step=False, on_epoch=True)
            self.log('ts_edm_loss_v',  ts_edm_loss  /batch_size, prog_bar=True, logger=True, on_step=False, on_epoch=True)

            self.log('e_loss_v',(r_e_loss+p_e_loss)/batch_size,  prog_bar=True,logger=True, on_step=False, on_epoch=True)
            self.log('s_loss_v',(r_s1_loss+p_s1_loss+r_s2_loss+p_s2_loss)/batch_size,  prog_bar=True,logger=True, on_step=False, on_epoch=True)

            self.log('r_e_loss_v',     r_e_loss  /batch_size, prog_bar=False,logger=True, on_step=False, on_epoch=True)
            self.log('p_e_loss_v',     p_e_loss  /batch_size, prog_bar=False,logger=True, on_step=False, on_epoch=True)
            self.log('r_s1_loss_v',    r_s1_loss /batch_size, prog_bar=False,logger=True, on_step=False, on_epoch=True)
            self.log('p_s1_loss_v',    p_s1_loss /batch_size, prog_bar=False,logger=True, on_step=False, on_epoch=True)
            self.log('r_s2_loss_v',    r_s2_loss /batch_size, prog_bar=False,logger=True, on_step=False, on_epoch=True)
            self.log('p_s2_loss_v',    p_s2_loss /batch_size, prog_bar=False,logger=True, on_step=False, on_epoch=True)


        if(self.TTA):
            original_batch_size = batch_size//2
            for i in range(original_batch_size):
                pred_edm[i] = (pred_edm[i+original_batch_size] + pred_edm[i])/2
                pred_e_r[i] = (pred_e_r[i+original_batch_size] + pred_e_r[i])/2
                pred_e_p[i] = (pred_e_p[i+original_batch_size] + pred_e_p[i])/2
                pred_s1_r[i] = (pred_s1_r[i+original_batch_size] + pred_s1_r[i])/2
                pred_s1_p[i] = (pred_s1_p[i+original_batch_size] + pred_s1_p[i])/2
                pred_s2_r[i] = (pred_s2_r[i+original_batch_size] + pred_s2_r[i])/2
                pred_s2_p[i] = (pred_s2_p[i+original_batch_size] + pred_s2_p[i])/2
            pred_edm = pred_edm[:original_batch_size]
            pred_e_r = pred_e_r[:original_batch_size]
            pred_e_p = pred_e_p[:original_batch_size]
            pred_s1_r = pred_s1_r[:original_batch_size]
            pred_s1_p = pred_s1_p[:original_batch_size]
            pred_s2_r = pred_s2_r[:original_batch_size]
            pred_s2_p = pred_s2_p[:original_batch_size]
            edm_mask = edm_mask[:original_batch_size]
        return [ o[m] for o,m in zip(pred_edm[:,num_images],edm_mask[:,num_images]) ],\
               pred_e_r, pred_e_p, pred_s1_r, pred_s1_p, pred_s2_r, pred_s2_p
    def test_epoch_end(self, outputs):
        from functools import reduce
        import pickle
        pred_e_r = torch.cat([output[1].detach().cpu() for output in outputs ])
        pred_e_p = torch.cat([output[2].detach().cpu() for output in outputs ])
        pred_s1_r= torch.cat([output[3].detach().cpu() for output in outputs ])
        pred_s1_p= torch.cat([output[4].detach().cpu() for output in outputs ])
        pred_s2_r= torch.cat([output[5].detach().cpu() for output in outputs ])
        pred_s2_p= torch.cat([output[6].detach().cpu() for output in outputs ])
        outputs = [output[0] for output in outputs ]
        output = reduce (lambda x,y: x+y, outputs, [])
        #print(output)
        list_dim =  list(  map( lambda c: int( (-1+(1+8*c)**0.5 )/2) , [ len(o) for o in output ] ) )
        #print(list_dim)

        return_val=[]
        for i, dim in enumerate(list_dim):
            return_val.append( torch.zeros(dim, dim, device=output[i].device, dtype=output[i].dtype) )
            triu_indices = torch.triu_indices(dim,dim, device=output[i].device)
            return_val[-1][ triu_indices[0], triu_indices[1] ] = output[i]
            return_val[-1]=return_val[-1]+return_val[-1].transpose(0,1) # symmetrize
            return_val[-1]=return_val[-1].detach().cpu()

        return_val = (return_val, pred_e_r, pred_e_p, pred_s1_r, pred_s1_p, pred_s2_r, pred_s2_p)
        with open(self.filename, 'wb') as f:
            pickle.dump(return_val, f)
