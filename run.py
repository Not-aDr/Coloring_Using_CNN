import argparse
import torch

from PIL import Image
from coloring_models import Resnet_ColorNet
from utilities import convert_show_rgb, convert_frame_rgb

from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage.transform import resize

import numpy as np
import matplotlib.pyplot as plt

import cv2

parser = argparse.ArgumentParser(description="Testing CNN_ColorNet")
parser.add_argument('--weight_pth', required = True, help = 'Loading Pretrained Weights')
parser.add_argument('--image_pth', default= '', help = 'Image for Evaluation')
parser.add_argument('--video_pth', default= '', help = 'Video for Evaluation')

args = parser.parse_args()
print('Arguments: {}'.format(args))

use_gpu = torch.cuda.is_available()

model = Resnet_ColorNet(train_mode = False)

if use_gpu:
    torch.cuda.empty_cache()
    model.cuda()
    print("Model moved to GPU.")

try:
    model.load_state_dict(torch.load(args.weight_pth)['state_dict'])
    print("Model Weights Loaded")
    model.eval()
except:
    print("Error Occured")

image_pth = args.image_pth
video_pth = args.video_pth
    
if image_pth:
    img = Image.open(args.image_pth)
    img = np.array(img)
    img = resize(img, (224, 224, 3), anti_aliasing=True)     #rescales pixel values due to anti-aliasing
    img_input = rgb2gray(img)
    img_input = torch.from_numpy(img_input).unsqueeze(0).float()
    if use_gpu:
        img_input = img_input.unsqueeze(0).cuda()                       #Convert to 4D tensor
    
    output_ab = model(img_input)
    convert_show_rgb(img_input,output_ab,img)

elif video_pth:
    cap = cv2.VideoCapture(video_pth)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_resize = cv2.resize(image,(224,224), interpolation=cv2.INTER_AREA)
        img_input = rgb2gray(image_resize)
        img_input = torch.from_numpy(img_input).unsqueeze(0).float()
        

        if use_gpu:
            img_input = img_input.unsqueeze(0).cuda()                       #Convert to 4D tensor
        
        
        output_ab = model(img_input)
        colored_frame = convert_frame_rgb(img_input,output_ab)
        colored_frame = colored_frame[...,::-1]                             #reversing order of numpy array (np.flip())
        colored_frame = cv2.resize(colored_frame,(640,360), interpolation=cv2.INTER_CUBIC)
        
    
        cv2.imshow('frame',frame)
        cv2.imshow('colored',colored_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
else :
    print("Enter a valid Image or Video Path")