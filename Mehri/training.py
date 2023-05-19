import re
import sys
import cv2
import torch
import functions
import numpy as np
import __dataset__
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
import Levenshtein
import warnings

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
from torch import nn
from network import Network
from PIL import Image
from tqdm import tqdm
from __parser__ import args_m01
from __syslog__ import EventlogHandler, ExecTime
from keras.preprocessing.image import img_to_array
from skimage.metrics import structural_similarity

@EventlogHandler
def save_model(model, path, epoch, optimizer, loss_history, file_name):
    print('Saving' + file_name)
    
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    
    torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_history
                }, path / Path(file_name))

@EventlogHandler
def load_model(model, path, device_ids):
    load = torch.load(path, map_location=torch.device('cpu'))
    epoch = load['epoch']
    model.load_state_dict(load['model_state_dict'])
    best_loss, train_loss, val_loss = load['loss']
    
    model = modelAllocation(model, True, device_ids)
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(load['optimizer_state_dict'])
    early_stopping = EarlyStopping(best_loss)
    
    return epoch, model, train_loss, val_loss, optimizer, early_stopping

class EarlyStopping():
    def __init__(self, patience=5, min_delta=0.00001, best_loss = None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = best_loss
        self.early_stop = False
        
    def __call__(self, save, model, loss, epoch, optimizer):
        if self.best_loss == None:
            self.best_loss = loss[0]
            
        elif self.best_loss - loss[0] > self.min_delta:
            self.best_loss = loss[0]
            self.counter = 0
            
            save_model(model, save, epoch, optimizer, loss, 'bestmodel.pt')
            
        elif self.best_loss - loss[0] < self.min_delta:
            self.counter += 1
            print(f'WARNING: Stop counter {self.counter} of {self.patience}')
            if self.counter >= self.patience:
                print('WARNING: Early stopping')
                self.early_stop = True
                
@EventlogHandler
class SSIMLoss(nn.Module):
    def __init__(self, OCR_path=None):
        super().__init__()
        self.criterion = nn.MSELoss(reduction ='none')
        self.SSIM_param = {
            'gaussian_weights':True,
            'multichannel':True,
            'sigma':1.5,
            'use_sample_covariance':False,
            'data_range':1.0
            }
        self.to_numpy = transforms.ToPILImage()
        if OCR_path is not None:
            self.OCR_path = OCR_path
            self.OCR = functions.load_model(self.OCR_path.as_posix())
            self.IMAGE_DIMS = self.OCR.layers[0].input_shape[0][1:]
            self.parameters = np.load(self.OCR_path.as_posix() + '/parameters.npy', allow_pickle=True).item()
            self.tasks = self.parameters['tasks']
            self.ocr_classes = self.parameters['ocr_classes']
            self.num_classes = self.parameters['num_classes']
            self.padding = True
            self.aspect_ratio = self.IMAGE_DIMS[1]/self.IMAGE_DIMS[0]
            self.min_ratio = self.aspect_ratio - 0.15
            self.max_ratio = self.aspect_ratio + 0.15
            self.OCR.compile()
        else:
            self.OCR_path = None

    def compare_string(self, strA, strB):
        if (strA is None) or (strB is None):
            return 0
        
        size = min(len(strA), len(strB))
        counter = 0
        
        for i in range(size):
            if strA[i] == strB[i]:
                counter+=1
                
        return counter
    
    def OCR_pred(self, img, fl = None, convert_to_bgr=True):
        if convert_to_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        if self.padding:
            img, _, _ = functions.padding(img, self.min_ratio, self.max_ratio, color = (127, 127, 127))        
        
        img = cv2.resize(img, (self.IMAGE_DIMS[1], self.IMAGE_DIMS[0]))        
        img = img_to_array(img)        
        img = img.reshape(1, self.IMAGE_DIMS[0], self.IMAGE_DIMS[1], self.IMAGE_DIMS[2])
        
        img = (img/255.0).astype('float')
        predictions = self.OCR.predict(img)

        plates = [''] * 1
        for task_idx, pp in enumerate(predictions):
            idx_aux = task_idx + 1
            task = self.tasks[task_idx]

            if re.match(r'^char[1-9]$', task):
                for aux, p in enumerate(pp):
                    plates[aux] += self.ocr_classes['char{}'.format(idx_aux)][np.argmax(p)]
            else:
                raise Exception('unknown task \'{}\'!'.format(task))
                
        return plates
    
    def levenshtein(self, a, b):
        return Levenshtein.distance(a, b)
    
    def number_zeros(self, zeros):
        total_zeros =  len(re.search('\d+\.(0*)', str(zeros.item())).group(1))
        if total_zeros <= 0:
            total_zeros += 1
        else:
            total_zeros += 2

        return total_zeros     
   
    def forward(self, SR, HR):
        count = 0
        diff = SR-HR
        MSE = diff*diff
        MSE_zeros = self.number_zeros(MSE.sum()/(diff.numel()))
        #print('MSE_zeros', MSE_zeros)
        ssim = 0
        distHRSR = 0

        if self.OCR_path != False:
            for imgSR, imgHR in zip(SR, HR):
                imgSR = np.array(self.to_numpy(imgSR.to('cpu'))).astype('uint8')
                imgHR = np.array(self.to_numpy(imgHR.to('cpu'))).astype('uint8')

                if self.OCR is not None:
                    predSR = self.OCR_pred(imgSR)[0]
                    predHR = self.OCR_pred(imgHR)[0]
                    _distHRSR = torch.as_tensor(self.levenshtein(predHR, predSR))

                _ssim = 1 - structural_similarity(imgSR, imgHR, **self.SSIM_param)
                _ssim = _ssim if _ssim > 0 else _ssim*(-1)

                ssim += _ssim
                distHRSR += _distHRSR

                #print('MSE before', MSE[count].sum()/(diff[count].numel()),'zeros_mse',  _mse_zeros)
                #print('in loop', _ssim, _distHRSR, ssim, distHRSR)

                #print('MSE after', MSE[count].sum()/(diff[count].numel()), '_ssim', _ssim, '_distHRSR', _distHRSR)
                count+=1
        #print('ssim', ssim/count, 'distHRSR', distHRSR/count)
        ssim = (1/(10**(MSE_zeros)))*(ssim/count)
        distHRSR = (1/(10**(MSE_zeros)))*(distHRSR/count)
        #print('AFTER=>ssim', ssim/count, 'distHRSR', distHRSR/count)
        MSE = MSE.sum()/diff.numel()
        #print('before MSE', MSE)
        #print('sum', MSE, ssim, distHRSR)
        MSE = MSE + ssim + distHRSR

        Image.fromarray(np.array(self.to_numpy(SR[0].to('cpu'))).astype('uint8')).save('TEST_newloss_1_2.jpg')
        #print('MSE', MSE)
        return MSE    


@EventlogHandler
def modelAllocation(model, parallel, device_ids):
    if isinstance(device_ids, list):
        if (torch.cuda.device_count() > 1) and parallel:
            print('Running training on', torch.cuda.device_count(),'GPU(s)')
            model = nn.DataParallel(model, device_ids=device_ids)
            
            return model.to(device_ids[0])
        
        else:            
            return model.to(device_ids[0])
        
    else:
        raise Exception('Invalid type for: {}.'.format(device_ids), ' List required.')
        
def GPU_devices(device_ids):
    if torch.cuda.is_available():    
        if torch.cuda.device_count() > 1:
            return device_ids
        else:
            return [device_ids[0]]
    else:
        print('No gpu available for training. Aborting...')
        sys.exit(1)
        
@EventlogHandler
def fit(model, train_dataloader, optimizer, criterion, device):
    print('TRAINING')
    model.train()
    train_running_loss = 0.0
    counter = 0
    
    with tqdm(enumerate(train_dataloader), total=len(train_dataloader)) as prog_bar:
        for i, batch in enumerate(prog_bar):
            counter += 1
            _, image = batch
            imageSR = model(image['LR'].to(device, non_blocking=True)) 
            loss = criterion(imageSR, image['HR'].to(device, non_blocking=True))
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.detach().item()

            postfix={
                'loss': loss.detach().item(),
                'running_loss': train_running_loss/counter
            }

            prog_bar.set_postfix(postfix)           
            
    train_loss = train_running_loss/counter
    
    
    return train_loss

@EventlogHandler
def validate(model, test_dataloader, criterion, device):
    print('VALIDATING')
    model.eval()
    val_running_loss = 0.0
    counter = 0

    with torch.no_grad():
        with tqdm(enumerate(test_dataloader), total=len(test_dataloader)) as prog_bar:
            for i, batch in enumerate(prog_bar):
                counter +=1
                _, image = batch            
                imageSR = model(image['LR'].to(device, non_blocking=True))            
                
                loss = criterion(imageSR, image['HR'].to(device, non_blocking=True))
                
                val_running_loss += loss.detach().item()
                
                postfix={
                'loss': loss.detach().item(),
                'running_loss': val_running_loss/counter
                }

                prog_bar.set_postfix(postfix)
                
        val_loss = val_running_loss/counter

        return val_loss    

@ExecTime
def main():    
    args = args_m01()
    train_dataloader, val_dataloader = __dataset__.load_dataset(args.samples, args.batch, args.mode, pin_memory=True, num_workers=1)
    device_ids = GPU_devices([0, 1])
    model = Network(3, 5)
    
    path_ocr = Path('2023-02-02-exp-016-br-paper-valfride-cg-ocr-goncalves2018realtime-original-120-60-adam-batch64-pat7')
    criterion = nn.L1Loss()
    
    if args.mode == 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        train_loss = []
        val_loss = []
        early_stopping = EarlyStopping()
        model = modelAllocation(model, True, device_ids)
        current_epoch = 0

    elif args.mode == 1:
        current_epoch, model, train_loss, val_loss, optimizer, early_stopping = load_model(model, args.model, device_ids)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor = 0.3, min_lr=1e-7, verbose=True)
    
    for epoch in range(current_epoch, 100):
        print(f"Epoch {epoch} of 100")
        
        train_epoch_loss = fit(model, train_dataloader, optimizer, criterion, device_ids[0])
        train_loss.append(train_epoch_loss)

        val_epoch_loss = validate(model, val_dataloader, criterion, device_ids[0])
        val_loss.append(val_epoch_loss)

        scheduler.step(val_epoch_loss)
        early_stopping(args.save, model, [val_epoch_loss, train_loss, val_loss], epoch, optimizer)
        
        if early_stopping.early_stop:
            break
    
        save_model(model, args.save, epoch, optimizer, [val_epoch_loss, train_loss, val_loss], 'backup.pt')
        print('Validation Loss: ', val_epoch_loss)
        print('Training Loss: ', train_epoch_loss)
        
    return 0

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
    
    
