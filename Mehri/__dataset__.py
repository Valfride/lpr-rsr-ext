import os
import re
import cv2
import shutil
import argparse
import functions
import numpy as np
import albumentations as A
import torchvision.transforms as transforms
from __syslog__ import EventlogHandler, ExecTime
from functools import partial
import matplotlib.pyplot as plt


from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity

IMG_LR = (40, 20)
IMG_HR = (160, 80)


def __getPaths__(path, sep=';'):
    if os.path.isfile(path) and path.endswith('.txt'):
        try:
            with open(path, '+r')  as f:
               _training = [line for line in f if re.search('\Atraining', line.split(sep)[1]) != None]
               f.seek(0)
               _validation = [line for line in f if re.search('\Avalidation', line.split(sep)[1]) != None]
               f.seek(0)
               _testing = [line for line in f if re.search('\Atesting', line.split(sep)[1]) != None]
               f.seek(0)
               
        except IOError:
            print("File not accessible!")
            
        return [_training, _validation, _testing]

def convertScale(img, alpha, beta):
    """Add bias and gain to an image with saturation arithmetics. Unlike
    cv2.convertScaleAbs, it does not take an absolute value, which would lead to
    nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
    becomes 78 with OpenCV, when in fact it should become 0).
    """

    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    
    return new_img.astype(np.uint8)

def automatic_brightness_and_contrast(image, clip_hist_percent=25):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # image = cv2.addWeighted(image, 4, cv2.blur(image, (30, 30)), -4, 128)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = convertScale(image, alpha=alpha, beta=beta)
    auto_result = cv2.cvtColor(auto_result, cv2.COLOR_BGR2RGB)
    return (auto_result, alpha, beta)

def gauss_noise(imgPath, ddegree):
    txt_path = Path(imgPath)
    txt_path = txt_path.with_suffix('.txt')
    img, pts = customDataset.open_image(customDataset, imgPath), customDataset.get_pts(customDataset, txt_path, 3)
    
    if len(pts) > 0:
        img = customDataset.rectify_img(customDataset, img, pts)
        
    imgHR = np.array(img, copy=True)
    
    SSIM = 1
    limit = 1
    _filter = 3
    
    while SSIM > ddegree:
        imgLR = np.array(img, copy=True)
        for i in range(0, limit):
            imgLR = cv2.GaussianBlur(imgLR, (_filter, _filter), 0)
        
        BSSIM = SSIM
        SSIM = structural_similarity(img, imgLR,
                                     gaussian_weights=True, multichannel=True, sigma=1.5,
                                     use_sample_covariance=False, data_range=1.0)
        limit+=1
        if BSSIM-SSIM < 0.001:
            _filter = _filter+2
            limit = 1
        
    
    return imgHR, imgLR
    
def train_test_split(path):
    train = []
    test = []
    validate = []
    
    with open(path, 'r+', encoding='utf8') as fp:
        for line in fp.readlines():
            line = line.split(';')
            
            imgHR = line[0]
            if 'jpg' in line[0]:
                ocrTxt = line[0].replace('jpg', 'txt')
            else:
                ocrTxt = line[0].replace('png', 'txt')
            imgLR = line[1]
            
            with open(ocrTxt, encoding='utf8') as fp:
                fp.seek(0)
                tp = fp.readlines()[0].split(':')[1].replace('\n', '').replace(' ', '')
                fp.seek(0)
                plate = fp.readlines()[1].split(':')[1].replace('\n', '').replace(' ', '')
                fp.seek(0)
                layout = fp.readlines()[2].split(':')[1].replace('\n', '').replace(' ', '')
            
            if 'training' in line[2]:
                train.append({
                    'HR': imgHR,
                    'LR': imgLR,
                    'plate': plate,
                    'layout': layout, 
                    'type': tp}
                    )
                    
            elif 'test' in line[2]:
                test.append({
                    'HR': imgHR,
                    'LR': imgLR,
                    'plate': plate,
                    'layout': layout, 
                    'type': tp}
                    )
            else:
                validate.append({
                    'HR': imgHR,
                    'LR': imgLR,
                    'plate': plate,
                    'layout': layout, 
                    'type': tp}
                    )
                
    return (train, test, validate)

class customDataset(Dataset):
    def __init__(self, x_tensor, augmentation = True):
        self.x = x_tensor
        self.to_tensor = transforms.ToTensor()
        self.augmentation = True
        
        self.background_color = (127, 127, 127)
        self.aspect_ratio = 2.0
        self.min_ratio = self.aspect_ratio - 0.15
        self.max_ratio = self.aspect_ratio + 0.15
        self.transformHR = np.array([
                            A.HueSaturationValue (hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=True),
                            A.RandomBrightnessContrast (brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=True),
                            A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=True),
                            None
                            ])
        
        self.transformLR = np.array([
                            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=True),
                           # A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False, elementwise=False, always_apply=True),
                           # A.PixelDropout(dropout_prob=0.01, per_channel=False, drop_value=0, mask_drop_value=None, always_apply=True),
                           # A.ImageCompression(quality_lower=50, quality_upper=95, compression_type=0, always_apply=True), #0 = JPEG
                           # A.ImageCompression(quality_lower=50, quality_upper=95, compression_type=1, always_apply=True), #1 = WEBP
                            A.RandomBrightnessContrast (brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=True),
                            A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=True),
                            None])
        
        print('Dataset: ', self.aspect_ratio, self.min_ratio, self.max_ratio) 
        
    def open_image(self, path):
        img = Image.open(path)
        img = np.array(img)
        
        return img
        
    def get_pts(self, path, pos):
        with open(path, 'r') as fp:
            lines = fp.readlines()[pos]
            pts = re.findall(r'\d+', lines)
            pts = [[int(pts[i]), int(pts[i+1])] for i in range(0, len(pts), 2)]
            
        return np.array(pts).astype('float32')
    
    def rectify_img(self, img, pts, margin=2):
    	# obtain a consistent order of the points and unpack them individually
    	# rect = order_points(pts)
    	(tl, tr, br, bl) = pts
     
    	# compute the width of the new image, which will be the maximum distance between bottom-right and bottom-left x-coordiates or the top-right and top-left x-coordinates
    	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    	maxWidth = max(int(widthA), int(widthB))
     
    	# compute the height of the new image, which will be the maximum distance between the top-right and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
    	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    	maxHeight = max(int(heightA), int(heightB))

    	maxWidth += margin*2
    	maxHeight += margin*2
     
    	# now that we have the dimensions of the new image, construct the set of destination points to obtain a "birds eye view", (i.e. top-down view) of the image, again specifying points in the top-left, top-right, bottom-right, and bottom-left order
    	ww = maxWidth - 1 - margin
    	hh = maxHeight - 1 - margin
    	c1 = [margin, margin]
    	c2 = [ww, margin]
    	c3 = [ww, hh]
    	c4 = [margin, hh]

    	dst = np.array([c1, c2, c3, c4], dtype = 'float32')

    	# compute the perspective transform matrix and then apply it
    	M = cv2.getPerspectiveTransform(pts, dst)
    	warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
     
    	return warped
    
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        txt_path = Path(self.x[index]['HR'])
        file_name = txt_path.as_posix().split('/')[-1]
        txt_path = txt_path.with_suffix('.txt')
        
        imgHR = self.open_image(self.x[index]['HR'])
        
        imgLR = self.open_image(self.x[index]['LR'])
        imgLR = np.array(imgLR)
        
        augLR = None
        augHR = None
        
        if self.augmentation:
            augLR = np.random.choice(self.transformLR, replace = True)
            augHR = np.random.choice(self.transformHR, replace = True)
        
        if self.augmentation and (augHR is not None):
            imgHR = augHR(image=imgHR)["image"]
        
        if self.augmentation and (augLR is not None):
            imgLR = augLR(image=imgLR)["image"]
        
        plate, layout, tp = self.x[index]['plate'], self.x[index]['layout'], self.x[index]['type']
          
        imgLR, _, _ = functions.padding(imgLR, self.min_ratio, self.max_ratio, color = self.background_color)
        imgHR, _, _ = functions.padding(imgHR, self.min_ratio, self.max_ratio, color = self.background_color)
        
        # imgLR = cv2.cvtColor(imgLR, cv2.COLOR_BGR2GRAY)
        # imgHR = cv2.cvtColor(imgHR, cv2.COLOR_BGR2GRAY)
        
        imgLR = cv2.resize(imgLR, IMG_LR, interpolation=cv2.INTER_CUBIC)
        imgHR = cv2.resize(imgHR, IMG_HR, interpolation=cv2.INTER_CUBIC)

        imgHR = self.to_tensor(imgHR)
        imgLR = self.to_tensor(imgLR)

        return {
                'HR': imgHR,
                'LR': imgLR,
                'plate': plate,
                'layout': layout,
                'type': tp,
                'file': file_name
                }

@EventlogHandler
def load_dataset(path, batch_size, mode, pin_memory, num_workers):
    if mode == 0 or mode == 1:
        train_dataloader = DataLoader(customDataset(train_test_split(path)[0], augmentation=True), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_dataloader = DataLoader(customDataset(train_test_split(path)[2], augmentation=True), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        return train_dataloader, val_dataloader
            
    elif mode == 2:
        test_dataloader = DataLoader(customDataset(train_test_split(path)[1], augmentation=False), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return test_dataloader

@EventlogHandler    
def getArgs():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--samples", type=str, required=True, help="Txt containing the samples paths")
    args = ap.parse_args()
    my_file = Path(args.samples)
    my_file = my_file.resolve(strict=True)    
	
    return args



def rodosol_dataset(path, dest):
    dest = Path(dest)
    destHR = dest / Path('HR')
    destHR.mkdir(parents=True, exist_ok=True)
    destLR = dest / Path('LR')
    destLR.mkdir(parents=True, exist_ok=True)
    
    split_text = Path('./images/splits/split.txt')
    
    with open(split_text, 'r') as tp:
        with open('./dataset/split_all.txt', 'w+') as f1:
            count = 0
            # prog_bar = tqdm(enumerate(data), total=len(data))
            for line in tqdm(iter(partial(tp.readline, 1024), ''), total=20000):
                file, _set = line.replace('\n', '').split(';')
                if 'motorcycles' in file:
                    pass
                else:                
                    with open(Path(file).with_suffix('.txt'), 'r')as fp:
                        pts = fp.readlines()[3].replace('\n', '').split(': ')[1].replace(',', ' ').split(' ')
                        pts = np.array([np.array([int(x), int(y)], dtype='float32') for x, y in zip(*[iter(pts)]*2)])
                        imgHR = cv2.imread(file)
                        imgHR = customDataset.rectify_img(customDataset, imgHR, pts)
                        shutil.copy2(Path(file).with_suffix('.txt'), destHR)
                        
                    
                    imgLR = cv2.resize(imgHR, None, fx=1/3, fy=1/3, interpolation=cv2.INTER_CUBIC)
                    # imgLR = cv2.resize(imgHR, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                    # imgLR = cv2.resize(imgHR, None, fx=1/3, fy=1/3, interpolation=cv2.INTER_CUBIC)
                    imgLR = cv2.GaussianBlur(imgLR, (7, 7), 0)
                    
                    cv2.imwrite((destHR / file.split('/')[-1]).as_posix(), imgHR)
                    cv2.imwrite((destLR / file.split('/')[-1]).as_posix(), imgLR)
                    
                    f1.write((destHR / file.split('/')[-1]).as_posix()+';'+(destLR / file.split('/')[-1]).as_posix()+';'+_set+'\n')
    
    print(count)
    

@ExecTime
@EventlogHandler
def main():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    args = getArgs()
    dataset = __getPaths__(args.samples)    
    degradation_levels = np.array([0.1, 0.25, 0.50, 0.75])  
    
    saveHR = Path('./dataset/HR')
    saveHR.mkdir(parents=True, exist_ok=True)
    
    saveLR = Path('./dataset/LR')
    saveLR.mkdir(parents=True, exist_ok=True)    
    
    saveSplits = Path('./dataset/splits')
    saveSplits.mkdir(parents=True, exist_ok=True)
    
    with open('./dataset/splits/split_all.txt', 'w+') as f1:
        
        for dlevel in degradation_levels:
            print('Generating {} SSIM'.format(dlevel))
            split_path = saveSplits / Path('split'+str(dlevel).replace('.', '').replace('0', '')+'.txt')
            with open(split_path.resolve(), 'w+') as f2:
                for data in dataset:
                    prog_bar = tqdm(enumerate(data), total=len(data))
                    for i, item in prog_bar:
                        img_path, _set = item.split(';')
                        _set = _set.replace('\n', '')
                                                               
                        img_HR, img_LR = gauss_noise(img_path, dlevel)
                        img_HR, img_LR = Image.fromarray(img_HR), Image.fromarray(img_LR)
                        
                        HR_name = img_path.split('/')[-1]
                        HR_name = (saveHR / Path(HR_name)).as_posix()
                        
                        
                        if not Path(HR_name).is_file():
                            img_HR.save(HR_name)
                            shutil.copy2(Path(img_path).with_suffix('.txt'), saveHR )
                            
                        
                        LR_name, ext = img_path.split('/')[-1].split('.')
                        LR_name = Path(LR_name+'_SSIM'+str(dlevel).replace('.', '')+'.'+ext)
                        LR_name = (saveLR / LR_name).as_posix()
                        img_LR.save(LR_name)
                        
                        f1.write(HR_name + ';' + LR_name + ';' + _set + '\n')
                        f2.write(HR_name + ';' + LR_name + ';' + _set + '\n')
                        
if __name__ == '__main__':
    # print(load_dataset('a', 6, 0, True, 4))
    rodosol_dataset('./images', './dataset')
    
        
        

        
        
       
