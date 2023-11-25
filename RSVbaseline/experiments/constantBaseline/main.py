import sys
sys.path.insert(0, '/home/jupyter/datasphere/project/RSVbaseline/')
import torch
from correction.config import cfg
from correction.models.model2 import ConstantBias
from torch.optim import lr_scheduler
from correction.models.loss import TurbulentMSE

import os
from correction.models.changeToERA5 import MeanToERA5
from correction.data.train_test_split import split_train_val_test
from correction.data.my_dataloader import WRFNPDataset
from correction.data.logger import WRFLogger
from correction.data.scalers import StandardScaler
from torch.utils.data import DataLoader
from correction.train import train
#from correction.train import train_epoch


if __name__ == "__main__":
    print('Device is:', cfg.GLOBAL.DEVICE)
    batch_size = 24
    max_epochs = 2

    LR = 1e-4

    wrf_folder = '/home/jupyter/datasphere/project/train/wrf'
    era_folder = '/home/jupyter/datasphere/project/train/era5'

    print('Splitting train val test...')
    train_files, val_files, test_files = split_train_val_test(wrf_folder, era_folder, 0.7, 0.1, 0.2)
    print('Split completed!')

    folder_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1]
    logger = WRFLogger(cfg.GLOBAL.MODEL_SAVE_DIR, folder_name)

    era_scaler = StandardScaler()
    wrf_scaler = StandardScaler()
    era_scaler.apply_scaler_channel_params(torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'era_means')),
                                           torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'era_stds')))
    wrf_scaler.apply_scaler_channel_params(torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'wrf_means')),
                                           torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'wrf_stds')))

    train_dataset = WRFNPDataset(train_files[0], train_files[1])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    valid_dataset = WRFNPDataset(val_files[0], val_files[1])
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    meaner = MeanToERA5(os.path.join(cfg.GLOBAL.BASE_DIR, 'wrferaMapping.npy'))
    criterion = TurbulentMSE(meaner, beta=0, logger=logger).to(cfg.GLOBAL.DEVICE)

    model = ConstantBias(3).to(cfg.GLOBAL.DEVICE)
    #model = LSTMModel().to(cfg.GLOBAL.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    mult_step_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
    
    #train_epoch(train_dataloader, model, criterion, optimizer, wrf_scaler, era_scaler)

    train(train_dataloader, valid_dataloader, model, optimizer, wrf_scaler, era_scaler, criterion,
          mult_step_scheduler, logger, max_epochs)