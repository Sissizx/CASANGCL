import os
import shutil
import sys
import torch
import yaml
import numpy as np
from datetime import datetime

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from nt_xent import NT_Xent
apex_support = False



class pretrain_cl(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self.device_vail()
        
        dir_name = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('ckpt', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.dataset = dataset
        self.nt_xent_criterion = NT_Xent(self.device, config['batch_size'], **config['loss'])

    def device_vail(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _train(self, model, x_i, x_j, n_iter):
        r_i, z_i = model(x_i)
        r_j, z_j = model(x_j)

        zis = F.normalize(z_i, dim=1)
        zjs = F.normalize(z_j, dim=1)

        loss = self.nt_xent_criterion(z_i, z_j)
        return loss

    def train(self):
        train_loader, valid_loader = self.dataset.data_load()
        if self.config['model_type'] == 'casanet':
            from model_pre_CASANET import LGEncoder
            model = LGEncoder(**self.config["model"]).to(self.device)
            model = self.Trained_parameters(model)

        optimizer = torch.optim.Adam(model.parameters(), self.config['init_lr'],weight_decay=eval(self.config['weight_decay'])
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config['epochs']-self.config['warm_up'], eta_min=0, last_epoch=-1)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        save_para(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0              
        best_valid_loss = np.inf

        for ctx in range(self.config['epochs']):
            for bn, (x_i, x_j) in enumerate(train_loader):
                optimizer.zero_grad()

                x_i = x_i.to(self.device)
                x_j = x_j.to(self.device)

                loss = self._train(model, x_i, x_j, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                    print(ctx, bn, loss.item())


                optimizer.step()
                n_iter += 1

            if ctx % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._vail(model, valid_loader)
                print(ctx, bn, valid_loss, '(validation)')
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'pretrained_model.pth'))
            
                self.writer.add_scalar('vail_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
            
            if (ctx+1) % self.config['save_every_n_epochs'] == 0:
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(str(ctx))))

            if ctx >= self.config['warm_up']:
                scheduler.step()


    def Trained_parameters(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['load_model'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'pretrained_model.pth'))
            model.load_state_dict(state_dict)
            print("The pre-trained model was successfully loaded.")
        except FileNotFoundError:
            print("Pre-trained weights failed to load. Training from scratch.")

        return model

    def _vail(self, model, valid_loader):
        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (x_i, x_j) in valid_loader:
                x_i = x_i.to(self.device)
                x_j = x_j.to(self.device)

                loss = self._train(model, x_i, x_j, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        
        model.train()
        return valid_loss

def save_para(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'pre_config.yaml'))

def main():
    config = yaml.load(open("config.yaml", "r",encoding='utf-8'), Loader=yaml.FullLoader)

    if config['aug'] == 'node':
        from pretrain_masking import Mol_node_aug
        dataset = Mol_node_aug(config['batch_size'], **config['dataset'])
    elif config['aug'] == 'edge':
        from pretrain_edgepred import Mol_edge_aug
        dataset = Mol_edge_aug(config['batch_size'], **config['dataset'])
    elif config['aug'] == 'subgraph':
        from pretrain_subextract import Mol_sub_aug
        dataset = Mol_sub_aug(config['batch_size'], **config['dataset'])
    else:
        raise ValueError('Not defined molecule augmentation!')
    pretrain_cl = pretrain_cl(dataset, config)
    pretrain_cl.train()


if __name__ == "__main__":
    main()

