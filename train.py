import os
import gc
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from albumentations import *
from sklearn.model_selection import GroupKFold, StratifiedKFold

# from segmentation_models_pytorch.unet import Unet
from segmentation_models_pytorch.deeplabv3 import DeepLabV3Plus
from segmentation_models_pytorch.encoders import get_preprocessing_fn

import logging
import sys


class CFG:
    lr = 0.01
    batch_size= 4
    epochs = 30
    seed = 2021
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_csv = './data/data.csv'
    save_dir = '/content/drive/MyDrive/Machine_Learning/droplet_detection/checkpoints/effb4-DeepLabV3Plus'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class DropletDataset(Dataset):
    def __init__(self, 
                 data_df, 
                 preprocess_input = None,
                 transforms = None):
        self.data_df = data_df
        self.preprocess_input = preprocess_input
        self.transforms = transforms
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        img_path = self.data_df['image'][idx]
        mask_path = self.data_df['mask'][idx]
        
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask==38] = 1
        
        if self.transforms:
            # Applying augmentations if any. 
            sample = self.transforms(image = img, 
                                     mask = mask)
            
            img, mask = sample['image'], sample['mask']
            
        if self.preprocess_input:
            # Normalizing the image with the given mean and
            # std corresponding to each channel.
            img = self.preprocess_input(image = img)['image']
            
        # PyTorch assumes images in channels-first format. 
        # Hence, bringing the channel at the first place.
        img = img.transpose((2, 0, 1))
        
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
            
        return img, mask


class DropletModel(nn.Module):
    def __init__(self):
        super(DropletModel, self).__init__()
        self.model = DeepLabV3Plus(encoder_name = ENCODER_NAME, 
                          encoder_weights = 'imagenet',
                          classes = 1,
                          activation = None)
        
    def forward(self, images):
        img_masks = self.model(images)
        return img_masks


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


def get_dice_coeff(pred, targs):
    '''
    Calculates the dice coeff of a single or batch of predicted mask and true masks.
    
    Args:
        pred : Batch of Predicted masks (b, w, h) or single predicted mask (w, h)
        targs : Batch of true masks (b, w, h) or single true mask (w, h)
  
    Returns: Dice coeff over a batch or over a single pair.
    '''
    
    pred = (pred>0).float()
    return 2.0 * (pred*targs).sum() / ((pred+targs).sum() + 1.0)


def train_one_epoch(epoch_no, data_loader, model, optimizer, device, scheduler = None):

    model.train()
    losses = []
    dice_coeffs = []

    train_process = tqdm(enumerate(data_loader), total=len(data_loader))
    for i, batch in train_process:
        img_btch, mask_btch = batch
        print(img_btch[0].shape)
        img_btch = img_btch.to(device)
        mask_btch = mask_btch.to(device)

        pred_mask_btch = model(img_btch.float())

        loss = loss_fn(pred_mask_btch, mask_btch.float())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        losses.append(loss.item())
  
        dice_coeff = get_dice_coeff(torch.squeeze(pred_mask_btch), 
                                    mask_btch.float())
        dice_coeffs.append(dice_coeff.cpu().detach().numpy())

        # print(f'train loss: {loss.item():.4f}, train dice coeff: {dice_coeff:.4f}')
        train_process.set_description(f'train loss: {np.mean(losses):.4f}, train dice coeff: {np.mean(dice_coeffs):.4f}')
        
        del img_btch, pred_mask_btch, mask_btch
        gc.collect()
        
    scheduler.step()

    return np.mean(losses), np.mean(dice_coeffs)


def eval_one_epoch(data_loader, model, device):
    '''
    Calculates metrics on the validation data.
    
    Returns: returns calculated metrics
    '''
    model.eval()
    
    dice_coeffs = []
    losses = []

    eval_process = tqdm(enumerate(data_loader), total=len(data_loader))
    for i, batch in eval_process:
        img_btch, mask_btch = batch
        
        img_btch = img_btch.to(device)
        mask_btch = mask_btch.to(device)
        
        pred_mask_btch = model(img_btch.float())
        
        loss = loss_fn(pred_mask_btch, 
                       mask_btch.float())
        # print('loss', f'{loss.item()}')
        losses.append(loss.item())
        
        dice_coeff = get_dice_coeff(torch.squeeze(pred_mask_btch), 
                                    mask_btch.float())
        dice_coeffs.append(dice_coeff.cpu().detach().numpy())
        
        eval_process.set_description(f'val loss: {np.mean(losses):.4f}, val dice coeff: {np.mean(dice_coeffs):.4f}')
    
    return np.mean(losses), np.mean(dice_coeffs)


def setup_logger(out_file=None, stderr=True, stderr_level=logging.INFO, file_level=logging.DEBUG):
    LOGGER.handlers = []
    LOGGER.setLevel(min(stderr_level, file_level))

    if stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(FORMATTER)
        handler.setLevel(stderr_level)
        LOGGER.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(FORMATTER)
        handler.setLevel(file_level)
        LOGGER.addHandler(handler)

    LOGGER.info("logger set up")
    return LOGGER



if __name__ == '__main__':
    set_all_seeds(CFG.seed)

    ENCODER_NAME = 'efficientnet-b4'
    # ENCODER_NAME = 'se_resnext50_32x4d'
    
    LOGGER = logging.getLogger()
    FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    setup_logger(out_file=CFG.save_dir+'/log.txt')
    
    preprocessing_fn = Lambda(image = get_preprocessing_fn(encoder_name = ENCODER_NAME,
                                                            pretrained = 'imagenet'))
    transforms = Compose([
                    CenterCrop(480, 480),
                    HorizontalFlip(),
                    VerticalFlip(),
                    RandomRotate90(),
                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9, 
                                    border_mode=cv2.BORDER_REFLECT),
                    GaussianBlur(blur_limit=(15, 25), p=0.5),
                    RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.9),
                ], p=1.0)

    data = pd.read_csv(CFG.data_csv)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.seed).split(np.arange(data.shape[0]), np.zeros(data.shape[0]))

    device = CFG.device
    loss_fn = DiceLoss()

    for fold, (t_idx, v_idx) in enumerate(folds):
        
        if fold > 0:
            break

        LOGGER.info(f'Fold: {fold}')
        LOGGER.info('-' * 40)
        
        train_df = data.loc[t_idx,:].reset_index(drop=True)
        val_df = data.loc[v_idx,:].reset_index(drop=True)

        train_ds = DropletDataset(data_df = train_df,
                                preprocess_input = preprocessing_fn,
                                transforms = transforms)
        val_ds = DropletDataset(data_df = val_df,
                            preprocess_input = preprocessing_fn,
                            transforms = None)
        train_dl = DataLoader(dataset = train_ds,
                            batch_size = CFG.batch_size,
                            num_workers = 0)
        val_dl = DataLoader(dataset = val_ds,
                                    batch_size = CFG.batch_size,
                                    num_workers = 0)
        
        LOGGER.info(f'train: {len(t_idx)}, val: {len(v_idx)}')
        
        # model 
        model = DropletModel()
        model.float()
        model.to(device)
        
        # optimizer
        optimizer = optim.AdamW(model.parameters(), lr = CFG.lr, weight_decay=1e-3)
        
        # scheduler 
        scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                                T_0 = CFG.epochs//2, 
                                                T_mult=1, 
                                                eta_min=1e-5, 
                                                last_epoch=-1, 
                                                verbose=False)
        
        LOGGER.info('Training now...')
        for e_no, epoch in enumerate(range(CFG.epochs)):
            
            train_loss, train_dice_coeff = train_one_epoch(e_no, train_dl, model, optimizer, device, scheduler)
            LOGGER.info(f'epoch: {e_no}, train loss: {train_loss: .4f}, train dice: {train_dice_coeff: .4f}')
            # LOGGER.info('Validating now...')
            with torch.no_grad():    
                val_loss, val_dice_coeff = eval_one_epoch(val_dl, model, device)
            LOGGER.info(f'epoch: {e_no}, val loss: {val_loss: .4f}, val dice: {val_dice_coeff: .4f}')
            torch.save(model.state_dict(), f"{CFG.save_dir}/fold{fold}_epoch{e_no}.pth")
            
        del train_ds, val_ds, model