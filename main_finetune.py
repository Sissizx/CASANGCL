import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from Supclloss import SupCLLoss

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from data_trans import Mol_graph_dataset



def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./fin_config.yaml', os.path.join(model_checkpoints_folder, 'fin_config.yaml'))


class Normalization(object):       
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

class FineTune(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self.device_vail()

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time + '_' + config['task_name'] + '_' + config['dataset']['target']
        log_dir = os.path.join('fine_log', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset


        self.criterion = nn.CrossEntropyLoss()
        self.criterion_scl = SupCLLoss(temperature=0.1)
    def device_vail(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _train(self, model, data, n_iter):

        __, pred = model(data)

        features = pred
        labels = data.y



        loss_CR = self.criterion(pred, data.y.flatten())
        loss_SCL = self.criterion_scl(features, labels)
        loss = loss_CR+loss_SCL
        return loss


    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.data_load()

        self.Normalization = None


        if self.config['model_type'] == 'casanet':
            from model_fin_CASANET import LGEncoder
            model = LGEncoder(self.config['dataset']['task'], **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        layer_list = []
        for name, param in model.named_parameters():
            if 'pred_lin' in name:
                print(name, param.requires_grad)
                layer_list.append(name)

        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        optimizer = torch.optim.Adam(
            [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
            self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')


        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_cls = 0


        for ctx in range(self.config['epochs']):
            for bn, data in enumerate(train_loader):
                optimizer.zero_grad()
                data = data.to(self.device)
                loss = self._train(model, data, n_iter)
                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print(ctx, bn, loss.item())

                optimizer.step()
                n_iter += 1


            if ctx % self.config['eval_every_n_epochs'] == 0:
                    valid_loss, valid_cls = self.validation(model, valid_loader)
                    if valid_cls > best_valid_cls:
                        best_valid_cls = valid_cls
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'pretrained_model.pth'))
            self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
            valid_n_iter += 1
        
            self.get_fin_test(model, test_loader)



    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['pre_trained_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'pretrained_model.pth'), map_location=self.device)
            model.load_my_state_dict(state_dict)
        except FileNotFoundError:
            print("Pre-trained weights failed to load. Training from scratch.")
        return model

    def validation(self, model, valid_loader):
        predictions = []
        labels = []
        global roc_auc
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._train(model, data, bn)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.Normalization:
                    pred = self.Normalization.denorm(pred)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

                if num_data == 0:
                    print('error: devision=0')
                    continue

            valid_loss /= num_data
        
        model.train()


        predictions = np.array(predictions)
        labels = np.array(labels)
        roc_auc = roc_auc_score(labels, predictions[:,1])
        print('Validation loss:', valid_loss, 'ROC AUC:', roc_auc)
        return valid_loss, roc_auc

    def get_fin_test(self, model, test_loader):
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("The pre-trained model was successfully loaded.")
        global roc_auc

        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._train(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.Normalization:
                    pred = self.Normalization.denorm(pred)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())


        
        model.train()

        predictions = np.array(predictions)
        labels = np.array(labels)

        self.predictions=predictions
        self.labels=labels

        self.roc_auc = roc_auc_score(labels, predictions[:,1])
        test_loss /= num_data

        print('Test loss:', test_loss, 'Test ROC AUC:', self.roc_auc)




def main(conf):
    dataset = Mol_graph_dataset(conf['batch_size'], **conf['dataset'])

    fine_tune = FineTune(dataset, conf)
    fine_tune.train()
    return fine_tune.roc_auc

if __name__ == "__main__":
    conf = yaml.load(open("fin_config.yaml", "r"), Loader=yaml.FullLoader)

    if conf['task_name'] == 'BBBP':

        conf['dataset']['data_path'] = 'data/bbbp/BBBP.csv'
        target_list = ["p_np"]

    elif conf['task_name'] == 'Tox21':

        conf['dataset']['data_path'] = 'data/tox21/tox21.csv'
        target_list = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", 
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]

    elif conf['task_name'] == 'ClinTox':

        conf['dataset']['data_path'] = 'data/clintox/clintox.csv'
        target_list = ['CT_TOX', 'FDA_APPROVED']

    elif conf['task_name'] == 'HIV':

        conf['dataset']['data_path'] = 'data/hiv/HIV.csv'
        target_list = ["HIV_active"]

    elif conf['task_name'] == 'BACE':

        conf['dataset']['data_path'] = 'data/bace/bace.csv'
        target_list = ["Class"]

    elif conf['task_name'] == 'SIDER':

        conf['dataset']['data_path'] = 'data/sider/sider.csv'
        target_list = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", 
            "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", 
            "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", 
            "Reproductive system and breast disorders", 
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
            "General disorders and administration site conditions", "Endocrine disorders", 
            "Surgical and medical procedures", "Vascular disorders", 
            "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", 
            "Congenital, familial and genetic disorders", "Infections and infestations", 
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", 
            "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", 
            "Ear and labyrinth disorders", "Cardiac disorders", 
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]
    results_list = []
    for target in target_list:
        conf['dataset']['target'] = target
        result = main(conf)
        results_list.append([target, result])

    os.makedirs('down_tasks', exist_ok=True)
    df = pd.DataFrame(results_list)
    df.to_csv(
        'down_tasks/{}_{}_results.csv'.format(conf['pre_trained_from'], conf['task_name']),
        mode='a', index=False, header=False
    )