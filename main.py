'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import resnet as RN
from utils import progress_bar, LabelSmoothingCrossEntropy, SimLabelSmoothingCrossEntropy,save_model


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--ce', action='store_true', help='Cross entropy use')
parser.add_argument('--sim', action='store_true', help='modified label smoothing with similarity matrix')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=30)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = RN.ResNet18()
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.ce == True:
    criterion = nn.CrossEntropyLoss()
    save_path = 'CrossEntropy'
    print("Use CrossEntropy")
elif args.sim == True:
    criterion = SimLabelSmoothingCrossEntropy()
    save_path = 'SimLabelSmoothing'
    print("Use Modified Label Smooting")
else:
    criterion = LabelSmoothingCrossEntropy()
    save_path = 'LabelSmoothing'
    print("Use Label Smooting")

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001, nesterov= True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90])

checkpoint_base_path = "checkpoint_epoch"
if not os.path.isdir(checkpoint_base_path):
    os.mkdir(checkpoint_base_path)

log_path_train = f"./{checkpoint_base_path}/{save_path}_train.csv"
log_path_test = f"./{checkpoint_base_path}/{save_path}_test.csv"

with open(log_path_train,"w") as f:
    print("Epoch,Loss,Acc",file=f)

with open(log_path_test,"w") as f:
    print("Epoch,Loss,Acc",file=f)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    scheduler.step()
    
    with open(log_path_train,"a") as f:
        print(f"{epoch},{train_loss/(batch_idx+1)},{100.*correct/total}",file=f)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    with open(log_path_test,"a") as f:
        print(f"{epoch},{test_loss/(batch_idx+1)},{100.*correct/total}",file=f)
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        save_path_epoch = f"./{checkpoint_base_path}/{save_path}@epoch_{epoch}.bin"
        save_model(net, save_path_epoch)
        best_acc = acc


for epoch in range(start_epoch, start_epoch+120):
    train(epoch)
    test(epoch)
