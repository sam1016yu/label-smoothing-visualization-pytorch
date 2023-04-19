import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import resnet as RN
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix
import os,sys

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--ce', action='store_true', help='Cross Entropy use')
parser.add_argument('--sim', action='store_true', help='modified label smoothing with similarity matrix')
parser.add_argument('--epoch',type=int,required=True,help='which epoch to laod')
args = parser.parse_args()

model = RN.ResNet18()
epoch_num = args.epoch

if args.ce == True:
    path = f'./checkpoint_epoch/CrossEntropy@epoch_{epoch_num}.bin'
    npy_path = './CE.npy'
    npy_target = './CE_tar.npy'
    title = f'Confusion_CrossEntropy@epoch_{epoch_num}'
elif args.sim == True:
    path = f'./checkpoint_epoch/SimLabelSmoothing@epoch_{epoch_num}.bin'
    npy_path = './LS_sim.npy'
    npy_target = './LS_sim_tar.npy'
    title = f'Confusion_enhanced_LabelSmoothing@epoch_{epoch_num}'
else:
    path = f'./checkpoint_epoch/LabelSmoothing@epoch_{epoch_num}.bin'
    npy_path = './LS.npy'
    npy_target = './LS_tar.npy'
    title = f'Confusion_LabelSmoothing@epoch_{epoch_num}'


if not os.path.exists(path):
    sys.exit(1)

states = torch.load(path)
model.load_state_dict(states)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

extract = model
extract.cuda()
extract.eval()

out_target = []
out_output = []

for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs, targets = inputs.cuda(), targets.cuda()
    outputs = extract(inputs)
    output_np = outputs.data.cpu().numpy()
    target_np = targets.data.cpu().numpy()
    out_output.append(output_np)
    out_target.append(target_np[:,np.newaxis])

output_array = np.concatenate(out_output, axis=0)
target_array = np.concatenate(out_target, axis=0)

y_pred = np.argmax(output_array,axis=1)
y_true = np.squeeze(target_array)
cm_data = confusion_matrix(y_true, y_pred)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
df_cm = pd.DataFrame(cm_data,columns=classes,index=classes)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
fig, ax = plt.subplots(1,1,figsize=(10,10))
sns.set(font_scale=2)
sns.heatmap(df_cm, cmap="Blues",annot=True,annot_kws={"size": 16},fmt="d",ax=ax,mask=np.eye(len(df_cm)))
ax.set_title(title)
fig.savefig('./plots_epoch/'+title+'.png', bbox_inches='tight')


