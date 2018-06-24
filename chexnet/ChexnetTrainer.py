import os
import numpy as np
import time
import sys
import shutil
import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func

from sklearn.metrics.ranking import roc_auc_score
from sklearn.metrics import roc_curve

from DensenetModels import DenseNet121
from DensenetModels import DenseNet169
from DensenetModels import DenseNet201
from DensenetModels import ResNet152
from DatasetGenerator import DatasetGenerator


#--------------------------------------------------------------------------------

class ChexnetTrainer ():

    #---- Train the densenet network
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training

    def train (pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, transResize, transCrop, launchTimestamp, checkpoint):


        #-------------------- SETTINGS: NETWORK ARCHITECTURE
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'RES-NET-152': model = ResNet152(nnClassCount, nnIsTrained).cuda()

        model = torch.nn.DataParallel(model).cuda()

        #-------------------- SETTINGS: DATA TRANSFORMS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        transformList = []
        transformList.append(transforms.RandomResizedCrop(transCrop))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        transformSequence=transforms.Compose(transformList)

        #-------------------- SETTINGS: DATASET BUILDERS
        datasetTrain = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain, transform=transformSequence)
        datasetVal =   DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileVal, transform=transformSequence)

        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=24, pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=24, pin_memory=True)

        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam (model.parameters(), lr=0.0000001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')

        #-------------------- SETTINGS: LOSS
        loss = torch.nn.BCELoss(size_average = True)

        #---- Load checkpoint
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])


        #---- TRAIN THE NETWORK

        lossMIN = 1000

        for epochID in range (0, trMaxEpoch):

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime

            ChexnetTrainer.epochTrain (model, dataLoaderTrain, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            lossVal, losstensor = ChexnetTrainer.epochVal (model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime

            scheduler.step(lossVal)

            if lossVal < lossMIN:
                lossMIN = lossVal
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 'm-' + launchTimestamp + '.pth.tar')
                print ('\nEpoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
            else:
                print ('\nEpoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))

    #--------------------------------------------------------------------------------

    def epochTrain (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        print("\nEpoch training...")
        model.train()
        total = len(dataLoader)
        for batchID, (input, target) in enumerate (dataLoader):
            sys.stdout.write('\r')
            sys.stdout.write("\rSteps: {}/{}".format(batchID+1,total))
            sys.stdout.flush()
            target = target.cuda(async = True)

            varInput = torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(target)
            varOutput = model(varInput)

            lossvalue = loss(varOutput, varTarget)

            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()

    #--------------------------------------------------------------------------------

    def epochVal (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):

        model.eval ()

        lossVal = 0
        lossValNorm = 0

        losstensorMean = 0
        print("\nEpoch evaluate...")
        total = len(dataLoader)
        for i, (input, target) in enumerate (dataLoader):
            sys.stdout.write('\r')
            sys.stdout.write("\rSteps: {}/{}".format(i+1,total))
            sys.stdout.flush()
            torch.cuda.empty_cache()
            target = target.cuda(async=True)
            with torch.no_grad():
                varInput = torch.autograd.Variable(input)
                varTarget = torch.autograd.Variable(target)
                varOutput = model(varInput)
                losstensor = loss(varOutput, varTarget)
                losstensorMean += losstensor.item()
                lossVal += losstensor.item()#losstensor.data[0]
                lossValNorm += 1

        outLoss = lossVal / lossValNorm
        losstensorMean = losstensorMean / lossValNorm

        return outLoss, losstensorMean

    #--------------------------------------------------------------------------------

    #---- Computes area under ROC curve
    #---- dataGT - ground truth data
    #---- dataPRED - predicted data
    #---- classCount - number of classes

    def computeAUROC (dataGT, dataPRED, classCount):

        outAUROC = []

        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()

        for i in range(classCount):
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))

        return outAUROC

    def splitResult(dataGT, dataPRED, imagePaths):
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        root_path = '/home/nthieuitus/bse_analyse_1'
        analyse_dict = {}
        class_thresholds = []

        for i in range(datanpGT.shape[1]):
            fpr, tpr, thresholds = roc_curve(datanpGT[:, i], datanpPRED[:, i], pos_label=1)

            chosen_index = 0
            min_distance = 1000000
            for j in range(len(fpr)):
                if ((1-tpr[0])**2 + fpr[j]**2) <= min_distance**2 and thresholds[j] < 1:
                    chosen = j
                    min_distance = math.sqrt((1-tpr[0])**2 + (fpr[j])**2)

            class_thresholds.append(thresholds[chosen])

        print("Thresholds for all classes")
        print(class_thresholds)

        for i in range(datanpGT.shape[0]):
            for j in range(datanpGT.shape[1]):
                if (datanpPRED[i,j] > class_thresholds[j] and datanpGT[i,j] == 1) or (datanpPRED[i,j] <= class_thresholds[j] and datanpGT[i,j] == 0):
                    # Copy to right label
                    folder_path = os.path.join(root_path, str(j), 'right')
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    counter = analyse_dict.get(folder_path, 0)
                    counter += 1
                    analyse_dict[folder_path] = counter
                    if counter > 20:
                        continue
                    shutil.copy2(imagePaths[i], folder_path)
                else:
                    # Copy to wrong label
                    #print("Wrong prediction {}".format(imagePaths[i]))
                    folder_path = os.path.join(root_path, str(j), 'wrong')
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    counter = analyse_dict.get(folder_path, 0)
                    counter += 1
                    analyse_dict[folder_path] = counter
                    if counter > 20:
                        continue
                    shutil.copy2(imagePaths[i], folder_path)

    def predict(dataGT, dataPRED, imagePaths, filename):
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        with open(filename, "w") as f:
            lines = []
            for i in range(datanpPRED.shape[0]):
                line = [os.path.basename(imagePaths[i])]
                for element in datanpPRED[i]:
                    line.append(str(element))
                lines.append(" ".join(line))
            f.write('\n'.join(lines))
        # class_thresholds = []
        # for i in range(datanpGT.shape[1]):
        #     fpr, tpr, thresholds = roc_curve(datanpGT[:, i], datanpPRED[:, i], pos_label=1)
        #
        #     chosen_index = 0
        #     min_distance = 1000000
        #     for j in range(len(fpr)):
        #         if ((1-tpr[0])**2 + fpr[j]**2) <= min_distance**2 and thresholds[j] < 1:
        #             chosen = j
        #             min_distance = math.sqrt((1-tpr[0])**2 + (fpr[j])**2)
        #
        #     class_thresholds.append(thresholds[chosen])
        # print("Thresholds", class_thresholds)
        #
        # lines = []
        #
        # for i in range(datanpGT.shape[0]):
        #     line = [os.path.basename(imagePaths[i])]
        #     for j in range(datanpGT.shape[1]):
        #         if (datanpPRED[i,j] > class_thresholds[j] and datanpGT[i,j] == 1) or (datanpPRED[i,j] <= class_thresholds[j] and datanpGT[i,j] == 0):
        #             # Right label
        #             line.append("1")
        #         else:
        #             # Wrong label
        #             line.append("0")
        #     lines.append(" ".join(line))
        # with open(filename, "w") as f:
        #     f.write('\n'.join(lines))



    #--------------------------------------------------------------------------------

    #---- Test the trained network
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training

    def test (pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimeStamp, predict_output=''):


        CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        #CLASS_NAMES = ['WORST', 'BEST']

        cudnn.benchmark = True

        #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()

        model = torch.nn.DataParallel(model).cuda()

        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])

        #-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        #-------------------- SETTINGS: DATASET BUILDERS
        transformList = []
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transformSequence=transforms.Compose(transformList)

        datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=8, shuffle=False, pin_memory=True)

        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()

        model.eval()
        print("\nTesting...\n")
        total = len(dataLoaderTest)
        for i, (input, target) in enumerate(dataLoaderTest):
            sys.stdout.write('\r')
            sys.stdout.write("\rSteps: {}/{}".format(i+1,total))
            sys.stdout.flush()
            target = target.cuda()
            outGT = torch.cat((outGT, target), 0)

            bs, n_crops, c, h, w = input.size()
            with torch.no_grad():
                varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda())

                out = model(varInput)
                outMean = out.view(bs, n_crops, -1).mean(1)

                outPRED = torch.cat((outPRED, outMean.data), 0)

        #ChexnetTrainer.splitResult(outGT, outPRED, datasetTest.listImagePaths)
        ChexnetTrainer.predict(outGT, outPRED, datasetTest.listImagePaths, predict_output)
        aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, nnClassCount)

        aurocMean = np.array(aurocIndividual).mean()

        print ('\nAUROC mean ', aurocMean)

        for i in range (0, len(aurocIndividual)):
            print (CLASS_NAMES[i], ' ', aurocIndividual[i])

        return
#--------------------------------------------------------------------------------
