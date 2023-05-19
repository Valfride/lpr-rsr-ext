import re
import sys
import cv2
import torch
import functions
import numpy as np
import __dataset__
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import warnings
from keras.models import Model
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
from torch import nn
from network import Network
from keras.models import Model

from PIL import Image
from tqdm import tqdm
from __parser__ import args_m01
from __syslog__ import EventlogHandler, ExecTime
from keras.preprocessing.image import img_to_array
from skimage.metrics import structural_similarity
from torchvision import datasets, models, transforms
from math import floor, log10
from NetSr_v1 import MobileNetV2
# from network import FeatureExtractor


initial_G_lr = 0.0001
initial_D_lr = 0.0001
device_list = [0, 0]


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

@EventlogHandler
def save_model(model_G, path, epoch, optimizer_G, loss_history, file_name):
    print('Saving: ' + file_name)
    
    state_dicts = model_G.state_dict()
    optimizer_dicts = optimizer_G.state_dict()
    
    torch.save({
                'epoch': epoch,
                'model_state_dict': state_dicts,
                'optimizer_state_dict': optimizer_dicts,
                'loss': loss_history
                }, path / Path(file_name))

@EventlogHandler
def load_model(model_G, path):
    load = torch.load(path, map_location=torch.device('cpu'))    
    model_G.to(device_list[1])   
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=initial_G_lr, betas=(0.1, 0.111))
    epoch = load['epoch']
    model_G.load_state_dict(load['model_state_dict'])    
    train_loss_g, val_loss_g = load['loss']    
    optimizer_G = torch.optim.Adam(model_G.parameters())    
    optimizer_G.load_state_dict(load['optimizer_state_dict'])    
    early_stopping = EarlyStopping(best_loss=val_loss_g[-1])    
    training_summary = """TRAINING SUMMARY:\nBest loss: {}\nLast epoch: {}\nLearning Rate G: {}\nPatience: {}\n""".format(val_loss_g[-1], epoch, optimizer_G.param_groups[0]['lr'], early_stopping.patience)
    print(training_summary)
    
    return epoch, model_G, optimizer_G, train_loss_g, val_loss_g, early_stopping 


class EarlyStopping():
    def __init__(self, patience=20, min_delta=0.0001, best_loss = None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = best_loss
        self.early_stop = False
        
    def __call__(self, model_G, save, epoch, optimizer_G, loss_history):
        if self.best_loss == None:
            self.best_loss = loss_history[1][-1]
            
        elif self.best_loss - loss_history[1][-1] > self.min_delta:
            self.best_loss = loss_history[1][-1]
            self.counter = 0
            
            save_model(model_G, save, epoch, optimizer_G, loss_history, 'bestmodel.pt')
            
        elif self.best_loss - loss_history[1][-1] < self.min_delta:
            self.counter += 1
            print(f'WARNING: Stop counter {self.counter} of {self.patience}')
            if self.counter >= self.patience:
                print('WARNING: Early stopping')
                self.early_stop = True


@EventlogHandler
class SSIMLoss(nn.Module):
    def __init__(self, OCR_path=None):
        super().__init__()
        self.to_numpy = transforms.ToPILImage()
        if OCR_path is not None:
            self.OCR_path = OCR_path
            self.criterion = nn.MSELoss()
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
        if not a: 
            return len(b)

        if not b: 
            return len(a)

        return min(self.levenshtein(a[1:], b[1:])+(a[0] != b[0]), self.levenshtein(a[1:], b)+1, self.levenshtein(a, b[1:])+1)

    def forward(self, SR, HR):
        dists = 0
        count = 0

        if self.OCR_path != False:
            for imgSR, imgHR in zip(SR, HR):
                imgSR = np.array(self.to_numpy(imgSR.to('cpu'))).astype('uint8')
                imgHR = np.array(self.to_numpy(imgHR.to('cpu'))).astype('uint8')
                
                if self.OCR is not None:
                    predSR = self.OCR_pred(imgSR)[0]
                    predHR = self.OCR_pred(imgHR)[0]
                    dists += torch.as_tensor(self.levenshtein(predHR, predSR)/7)
                count+=1
            dists/=count
        
        return (dists + self.criterion(SR, HR))/2
    

class OCRFeatureExtractor(nn.Module):
    def __init__(self, OCR_path=None):
        super().__init__()
        self.to_numpy = transforms.ToPILImage()
        if OCR_path is not None:
            self.OCR_path = OCR_path
            self.OCR = functions.load_model(self.OCR_path.as_posix())
            self.OCR = Model(self.OCR.input, self.OCR.layers[-41].output)
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
                
        return predictions

    def forward(self, images):
        batch = []
        if self.OCR_path != False:
            for img in images:
                img = np.array(self.to_numpy(img.to('cpu'))).astype('uint8')

                if self.OCR is not None:
                    features = self.OCR_pred(img)
                batch.append(features)
        logits = Variable(torch.as_tensor(batch), requires_grad=True)
        # logits = logits.view(logits.size(0), -1)
        
        return logits
    

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = models.vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:-1])

    def forward(self, img):
        return self.feature_extractor(img)


def validate_gan (generator, val_dataloader, feature_extractor, L1_loss, MSE_loss):
    print('VALIDATION')
    generator.eval()
    
    val_running_loss_G = 0.0
    
    steps = 0
    with torch.no_grad():
        with tqdm(enumerate(val_dataloader), total=len(val_dataloader)) as prog_bar:
    
            for i, batch in enumerate(prog_bar):           
                _, image = batch
                imgs_LR = Variable(image['LR']).cuda()
                imgs_HR = Variable(image['HR']).cuda()
            
                #  ---------------------
                #   VALIDATE GENERATOR 
                #  ---------------------
            
                # generate SR image
                fake_output = generator(imgs_LR)
            
                # adversarial loss
                gen_features = feature_extractor(fake_output)
                real_features = feature_extractor(imgs_HR)
                features = L1_loss(gen_features, real_features)
                mse_loss = MSE_loss(fake_output, imgs_HR)
                loss = mse_loss + features
            
                val_running_loss_G+=loss.detach().item()
                Image.fromarray(np.array(feature_extractor.to_numpy(fake_output[0].to('cpu'))).astype('uint8')).save('TEST_08.jpg')
            
                steps+=1
                postfix={
                    'Running_loss': (val_running_loss_G)/steps,
                    'Current_loss': loss.detach().item(),
                    'mse_loss': mse_loss.detach().item(),
                    'features': features.detach().item()
                    }
            
                prog_bar.set_postfix(postfix)
            return val_running_loss_G/steps

def fit_gan(generator, train_dataloader, optimizer_G, feature_extractor, L1_loss, MSE_loss):
    print('TRAINING')
    generator.train()    
    train_running_loss_G = 0.0
    steps = 0
    
    with tqdm(enumerate(train_dataloader), total=len(train_dataloader)) as prog_bar:
        for i, batch in enumerate(prog_bar):           
            _, image = batch
            imgs_LR = Variable(image['LR']).cuda()
            imgs_HR = Variable(image['HR']).cuda()
            
            #  ---------------------
            #   TRAIN GENERATOR 
            #  ---------------------
            optimizer_G.zero_grad()
            
            # generate SR image
            fake_output = generator(imgs_LR)
            
            # loss
            gen_features = feature_extractor(fake_output)
            real_features = feature_extractor(imgs_HR)
            features = L1_loss(gen_features, real_features)
            mse_loss = MSE_loss(fake_output, imgs_HR)
            loss = mse_loss + features
            
            loss.backward()
            optimizer_G.step()
            
            train_running_loss_G+=loss.detach().item()
            Image.fromarray(np.array(feature_extractor.to_numpy(fake_output[0].to('cpu'))).astype('uint8')).save('TEST_08.jpg')
            
            steps+=1
            postfix={
                'Running_loss': (train_running_loss_G)/steps,
                'Current_loss': loss.detach().item(),
                'mse_loss': mse_loss.detach().item(),
                'features': features.detach().item()
                }
            
            prog_bar.set_postfix(postfix)
        return train_running_loss_G/steps
    
  
@ExecTime
def main(): 
    args = args_m01()
    train_dataloader, val_dataloader = __dataset__.load_dataset(args.samples, args.batch, args.mode, pin_memory=True, num_workers=0)
    path_ocr = Path('./saved_models/2023-02-02-exp-016-br-paper-valfride-cg-ocr-goncalves2018realtime-original-120-60-adam-batch64-pat7')
    
    generator = Network(3, 3)
    feature_extractor = OCRFeatureExtractor(path_ocr)
    feature_extractor.cuda()
    L1_loss =  nn.L1Loss()
    MSE_loss = nn.MSELoss()
     
    
    
    if args.mode == 0:
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=initial_G_lr, betas=(0.5, 0.555))
        
        train_loss_g = []
        
        val_loss_g = []
        
        early_stopping = EarlyStopping()
        current_epoch = 0
        generator.to(device_list[1])    
        

    elif args.mode == 1:
        current_epoch, generator, optimizer_G, train_loss_g, val_loss_g, early_stopping = load_model(generator, args.model)
    
    #scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(200, 2, 2).step, verbose=True, last_epoch=-1)
    scheduler_G = lr_scheduler.ReduceLROnPlateau(optimizer_G, factor=0.8, patience=2, verbose=True)
 
#    for _ in range(0, current_epoch):
#        scheduler_G.step()

    for epoch in range(current_epoch, 200):
        print(f"Epoch {epoch} of 200:")
         
        train_loss_G = fit_gan(generator,      
                               train_dataloader, 
                               optimizer_G, 
                               feature_extractor,
                               L1_loss,
                               MSE_loss)
        
        train_loss_g.append(train_loss_G)

        val_loss_G = validate_gan(generator,
                               val_dataloader, 
                               feature_extractor,
                               L1_loss,
                               MSE_loss)
        
        val_loss_g.append(val_loss_G)
        
        scheduler_G.step(val_loss_G)
        
        early_stopping(generator, args.save, epoch, optimizer_G, [train_loss_g, val_loss_g])
        
        if early_stopping.early_stop:
            break
         
        save_model(generator, args.save, epoch, optimizer_G, [train_loss_g, val_loss_g], 'backup.pt')
        print('G validation Loss: ', val_loss_G)
        print('G Training Loss: ', train_loss_G)

        
        
    return 0

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
    
    
