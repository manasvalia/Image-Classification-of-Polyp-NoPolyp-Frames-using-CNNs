import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import torchvision.models as models
import os
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn import metrics

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from google.colab import drive
drive.mount('/content/drive/')

class QAPolypDataset(Dataset):

    def __init__(self, images_path, file_path, augment=None):
        self.img_list = []
        self.img_label = []
        self.augment = augment

        with open(file_path, "r") as fileDescriptor:
            line = True

            while line:
                line = fileDescriptor.readline()

                if line:
                    lineItems = line.split()
                    imagePath = os.path.join(images_path, lineItems[0])
                    imageLabel = [int(lineItems[1])]

                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.img_label[index])

        if self.augment != None:
            imageData = self.augment(imageData)

        return imageData, imageLabel

    def __len__(self):

        return len(self.img_list)

def build_transform_classification(normalize, crop_size=224, resize=256, mode="train"):
    transformations_list = []

    if normalize.lower() == "imagenet":
        normalize = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        normalize = None

    if mode == "train":
        transformations_list.append(transforms.RandomResizedCrop(crop_size))
        transformations_list.append(transforms.RandomHorizontalFlip())
        transformations_list.append(transforms.RandomRotation(7))
        transformations_list.append(transforms.ToTensor())
        if normalize is not None:
            transformations_list.append(normalize)
    elif mode == "test":
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.TenCrop(crop_size))
        transformations_list.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if normalize is not None:
            transformations_list.append(transforms.Lambda(
                lambda crops: torch.stack([normalize(crop) for crop in crops])))
    transformSequence = transforms.Compose(transformations_list)

    return transformSequence

def classification_engine(model_type, train_set, model_path, num_class, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    cudnn.benchmark = True

    save_model_path = os.path.join(model_path, model_type+"_model.pth")

    data_loader_train = DataLoader(
        dataset=train_set, batch_size=8, shuffle=True, pin_memory=True)

    print("start training.....")
    # model = models.resnet50(pretrained=True)

    # for param in model.parameters():
    #     param.requires_grad = False

    # kernelCount = model.fc.in_features
    # model.fc = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())

    if model_type=='resnet-50':
       model = models.resnet50(pretrained=True)
       for param in model.parameters():
          param.requires_grad = False
       kernelCount = model.fc.in_features
       model.fc = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())
    elif model_type=='resnet-18':
       model = models.resnet18(pretrained=True)
       for param in model.parameters():
          param.requires_grad = False
       kernelCount = model.fc.in_features
       model.fc = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())
    elif model_type=='resnet-34':
       model = models.resnet34(pretrained=True)
       for param in model.parameters():
          param.requires_grad = False
       kernelCount = model.fc.in_features
       model.fc = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())
    elif model_type=='resnet-101':
       model = models.resnet101(pretrained=True)
       for param in model.parameters():
          param.requires_grad = False
       kernelCount = model.fc.in_features
       model.fc = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())
    elif model_type=='resnet-18':
       model = models.resnet18(pretrained=True)
       for param in model.parameters():
          param.requires_grad = False
       kernelCount = model.fc.in_features
       model.fc = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())
    elif model_type=='densenet-161':
       model = models.densenet161(pretrained=True)
       for param in model.parameters():
          param.requires_grad = False
       kernelCount = model.classifier.in_features
       model.classifier = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())
    elif model_type=='efficientnet_b0':
       model = models.efficientnet_b0(pretrained=True)
       for param in model.parameters():
          param.requires_grad = False
       kernelCount = model.classifier[1].in_features
       model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(kernelCount, num_class), nn.Sigmoid())
    elif model_type=='efficientnet_b2':
       model = models.efficientnet_b2(pretrained=True)
       for param in model.parameters():
          param.requires_grad = False
       kernelCount = model.classifier[1].in_features
       model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(kernelCount, num_class), nn.Sigmoid())
    elif model_type=='efficientnet_b4':
       model = models.efficientnet_b4(pretrained=True)
       for param in model.parameters():
          param.requires_grad = False
       kernelCount = model.classifier[1].in_features
       model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(kernelCount, num_class), nn.Sigmoid())
    elif model_type=='efficientnet_b6':
       model = models.efficientnet_b6(pretrained=True)
       for param in model.parameters():
          param.requires_grad = False
       kernelCount = model.classifier[1].in_features
       model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(kernelCount, num_class), nn.Sigmoid())


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    model.to(device)

    if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)

    # for param in model.parameters():
    #   print(param)

    for epoch in range(start_epoch, end_epoch):
        print(f"epoch: {epoch}")
        model.train()
        running_loss = 0
        for i, (samples, targets) in enumerate(data_loader_train):
            samples, targets = samples.float().to(device), targets.float().to(device)
            optimizer.zero_grad()
            outputs = model.forward(samples)
            # print(f"outputs: {outputs}")
            # print(f"targets: {targets}")
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 5 == 0:
                print(f"batch[{i}]: loss={loss.item()}")

        print(f"epoch[{epoch}]: Total loss={running_loss}")
    torch.save(model.state_dict(), save_model_path)

def test_classification(model_type, saved_model, data_loader_test, num_class, learning_rate):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    cudnn.benchmark = True

    if model_type=='resnet-50':
       model = models.resnet50(pretrained=True)
       for param in model.parameters():
          param.requires_grad = False
       kernelCount = model.fc.in_features
       model.fc = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())
    elif model_type=='resnet-18':
       model = models.resnet18(pretrained=True)
       for param in model.parameters():
          param.requires_grad = False
       kernelCount = model.fc.in_features
       model.fc = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())
    elif model_type=='resnet-34':
       model = models.resnet34(pretrained=True)
       for param in model.parameters():
          param.requires_grad = False
       kernelCount = model.fc.in_features
       model.fc = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())
    elif model_type=='resnet-101':
       model = models.resnet101(pretrained=True)
       for param in model.parameters():
          param.requires_grad = False
       kernelCount = model.fc.in_features
       model.fc = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())
    elif model_type=='resnet-18':
       model = models.resnet18(pretrained=True)
       for param in model.parameters():
          param.requires_grad = False
       kernelCount = model.fc.in_features
       model.fc = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())
    elif model_type=='densenet-161':
       model = models.densenet161(pretrained=True)
       for param in model.parameters():
          param.requires_grad = False
       kernelCount = model.classifier.in_features
       model.classifier = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())
    elif model_type=='efficientnet_b0':
       model = models.efficientnet_b0(pretrained=True)
       for param in model.parameters():
          param.requires_grad = False
       kernelCount = model.classifier[1].in_features
       model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(kernelCount, num_class), nn.Sigmoid())
    elif model_type=='efficientnet_b2':
       model = models.efficientnet_b2(pretrained=True)
       for param in model.parameters():
          param.requires_grad = False
       kernelCount = model.classifier[1].in_features
       model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(kernelCount, num_class), nn.Sigmoid())
    elif model_type=='efficientnet_b4':
       model = models.efficientnet_b4(pretrained=True)
       for param in model.parameters():
          param.requires_grad = False
       kernelCount = model.classifier[1].in_features
       model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(kernelCount, num_class), nn.Sigmoid())
    elif model_type=='efficientnet_b6':
       model = models.efficientnet_b6(pretrained=True)
       for param in model.parameters():
          param.requires_grad = False
       kernelCount = model.classifier[1].in_features
       model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(kernelCount, num_class), nn.Sigmoid())


    # for param in model.parameters():
    #     param.requires_grad = False


    # optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    model.load_state_dict(torch.load(saved_model))

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)
    model.eval()

    y_test = torch.FloatTensor().cuda()
    p_test = torch.FloatTensor().cuda()

    with torch.no_grad():
        for i, (samples, targets) in enumerate(tqdm(data_loader_test)):
            print(f"{i}th batch")
            targets = targets.cuda()
            y_test = torch.cat((y_test, targets), 0)

            if len(samples.size()) == 4:
                bs, c, h, w = samples.size()
                n_crops = 1
            elif len(samples.size()) == 5:
                bs, n_crops, c, h, w = samples.size()

            varInput = torch.autograd.Variable(samples.view(-1, c, h, w))

            outputs = model.forward(varInput.cuda())
            outMean = outputs.view(bs, n_crops, -1).mean(1)
            p_test = torch.cat((p_test, outMean.data), 0)
    return y_test, p_test

def metric_AUROC(target, output, num_class):
    outAUROC = []

    target = target.cpu().numpy()
    output = output.cpu().numpy()

    for i in range(num_class):
        outAUROC.append(roc_auc_score(target[:, i], output[:, i]))

    return outAUROC

image_path = r'../QA-Polyp/train/'
train_list = r'../QA-Polyp/train/split_train.txt'
test_list = r'../QA-Polyp/train/split_test.txt'
model_path = r'../QA-Polyp/finaloutput_pretrained'
output_path = r'../QA-Polyp/finaloutput_pretrained'

num_class = 1
learning_rate = 0.001
start_epoch = 0
end_epoch = 10
init_loss = 100000

train_set = QAPolypDataset(images_path=image_path, file_path=train_list, augment=build_transform_classification(normalize="imagenet", mode="train"))
test_set = QAPolypDataset(images_path=image_path, file_path=test_list, augment=build_transform_classification(normalize="imagenet", mode="test"))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


model_list = ['resnet-18', 'resnet-34', 'resnet-50', 'resnet-101', 'efficientnet_b0', 'efficientnet_b2', 'efficientnet_b4', 'efficientnet_b6']

#  model_list = ['resnet-18', 'resnet-34', 'resnet-50', 'resnet-101', 
     #'densenet-161', 'efficientnet_b0', 'efficientnet_b2', 'efficientnet_b4', 'efficientnet_b6']

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])


for model_type in model_list:
  print(model_type)
  classification_engine(model_type, train_set, model_path, num_class, learning_rate)
  data_loader_test = DataLoader(dataset=test_set, batch_size=8, shuffle=False, pin_memory=True)
  saved_model = os.path.join(model_path, model_type+"_model.pth")
  y_test, p_test = test_classification(model_type, saved_model, data_loader_test, num_class, learning_rate)
  AUC_value = metric_AUROC(y_test.cpu(), p_test.cpu(), num_class)
  auc = roc_auc_score(y_test.cpu(), p_test.cpu())

  print(f"AUC: {AUC_value}")

  model_name=str(model_type)


  result_table = result_table.append({'classifiers':model_type,
                                        'fpr':y_test.cpu(), 
                                        'tpr':p_test.cpu(),
                                        'auc':AUC_value}, ignore_index=True)
  
# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)


for i in result_table.index:
  plt.plot(result_table.loc[i]['fpr'], 
           result_table.loc[i]['tpr'])

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
model_dict = {'resnet-18':'ResNet-18',
              'resnet-34':'ResNet-34',
              'resnet-50':'ResNet-50',
              'resnet-101':'ResNet-101 ',
              'efficientnet_b0':'EfficientNet-b0',
              'efficientnet_b2':'EfficientNet-b2',
              'efficientnet_b4':'EfficientNet-b4',
              'efficientnet_b6':'EfficientNet-b6',}
model_list = [ 'efficientnet_b0', 'efficientnet_b2', 'efficientnet_b4', 'efficientnet_b6']
#'resnet-18', 'resnet-34', 'resnet-50', 'resnet-101'
  #   'densenet-161', 'efficientnet_b0', 'efficientnet_b2', 'efficientnet_b4', 'efficientnet_b6']

data_loader_test = DataLoader(dataset=test_set, batch_size=16, shuffle=True, pin_memory=True)
for i in model_list:
    saved_model = os.path.join(model_path, i+"_model.pth")
    y_test, p_test = test_classification(i, saved_model, data_loader_test, num_class, learning_rate)
    AUR = metric_AUROC(y_test.cpu(), p_test.cpu(), num_class)
    print(i,' ', AUR)
    AUR = round(AUR[0],4)
    fpr, tpr, _ = roc_curve(y_test.cpu(), p_test.cpu())
    plt.plot(fpr,tpr,lw=2,label="{}, AUC={}".format(model_dict[i], AUR))

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('Pretrained models for training', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')
plt.show()
plt.savefig("efficientnet")
