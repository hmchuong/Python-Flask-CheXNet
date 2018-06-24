import os
import numpy as np
import time
import sys
from PIL import Image

import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from chexnet.DensenetModels import DenseNet121
from chexnet.DensenetModels import DenseNet169
from chexnet.DensenetModels import DenseNet201

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

#--------------------------------------------------------------------------------
#---- Class to generate heatmaps (CAM)

class HeatmapGenerator ():

    #---- Initialize heatmap generator
    #---- pathModel - path to the trained densenet model
    #---- nnArchitecture - architecture name DENSE-NET121, DENSE-NET169, DENSE-NET201
    #---- nnClassCount - class count, 14 for chxray-14


    def __init__ (self, pathModel, nnArchitecture, nnClassCount, transCrop):

        #---- Initialize the network
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, True)#.cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, True)#.cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, True)#.cuda()

        model = torch.nn.DataParallel(model)#.cuda()

        modelCheckpoint = torch.load(pathModel, map_location='cpu')
        model.load_state_dict(modelCheckpoint['state_dict'])

        self.eval_model = model
        self.eval_model.eval()

        self.features_model = model.module.densenet121.features
        self.features_model.eval()

        #---- Initialize the weights
        self.weights = list(self.features_model.parameters())[-2]

        #---- Initialize the image transform - resize + normalize
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize(transCrop))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)

        self.transformSequence = transforms.Compose(transformList)

    #--------------------------------------------------------------------------------

    def generate (self, pathImageFile, pathOutputFile, transCrop):

        #---- Load image, transform, convert
        imageData = Image.open(pathImageFile).convert('RGB')
        imageData = self.transformSequence(imageData)
        imageData = imageData.unsqueeze_(0)

        input = torch.autograd.Variable(imageData)

        #self.model.cuda()
        output = self.features_model(input)#.cuda())
        eval_output = self.eval_model(input)
        outMean = eval_output.view(1, 1, -1).mean(1)

        #---- Generate heatmap
        heatmap = None
        for i in range (0, len(self.weights)):
            map = output[0,i,:,:]
            if i == 0: heatmap = self.weights[i] * map
            else: heatmap += self.weights[i] * map

        #---- Blend original and heatmap
        npHeatmap = heatmap.cpu().data.numpy()

        imgOriginal = cv2.imread(pathImageFile, 1)
        img = imgOriginal.copy()
        try:
            imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))
        except:
            print("Error on file ", pathImageFile)
            return
        cam = npHeatmap / np.max(npHeatmap)

        cam = cv2.resize(cam, (1024, 1024))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

        img = heatmap * 0.5 + img

        cv2.imwrite(pathOutputFile, img)
        return outMean.cpu().data.numpy()

#--------------------------------------------------------------------------------
