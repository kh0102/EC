import joblib
import pandas as pd
import torch.nn.functional as F
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import confusion_matrix

import copy
from Dataset import MddData
from torch import nn

import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)

class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''
    def __init__(self, in_planes, planes, nrois, bias=False):
        super(E2EBlock, self).__init__()
        self.d = nrois
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)


class BrainNetCNN(torch.nn.Module):
    def __init__(self, nrois, num_classes=2):
        super(BrainNetCNN, self).__init__()
        self.in_planes = 1
        self.d = nrois

        self.e2econv1 = E2EBlock(1, 32, nrois, bias=True)
        self.e2econv2 = E2EBlock(32, 64, nrois, bias=True)
        self.E2N = torch.nn.Conv2d(64, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, 256, (self.d, 1))

        self.batchNorm = torch.nn.BatchNorm1d(256 )
        self.layer1 = nn.Linear(256 , 128)
        self.layer2 = nn.Linear(128, 64)
        self.drop = nn.Dropout(p=0.5)
        self.classify = nn.Linear(64, num_classes)

    def forward(self, fc, ec):
        fc = fc.float().to(device)
        ec = fc
        # ec = ec.float().to(device)

        out = F.leaky_relu(self.e2econv1(ec), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)

        out = self.batchNorm(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.classify(out)
        out = self.drop(out)
        out = nn.functional.softmax(out,dim=1)
        return out

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=5):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    #if epoch % lr_decay_epoch == 0:
        #print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def train_model(train_loader, val_loader, nrois, epochs):
    # Create training and testing set.

    val_loss = []
    best_val_acc = None
    best_model = None
    labels = np.empty([], dtype=int)
    predictions = np.empty([], dtype=int)

    model = BrainNetCNN(nrois).to(device)

    loss_func = nn.CrossEntropyLoss().to(device)
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)


    model.train()
    es = 10
    es_counter = 0
    print('Training....')
    epoch_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for iter, (fc,ec, label) in enumerate(train_loader):

            label = Variable(label).type(torch.cuda.LongTensor).to(device)
            prediction = model(fc,ec).to(device)

            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            labels = np.append(labels, label.cpu().numpy())
            predictions = np.append(predictions, torch.argmax(prediction, 1).cpu().numpy())

            epoch_loss += loss.detach().item()
        epoch_loss = epoch_loss / iter
        optimizer = exp_lr_scheduler(optimizer, epoch, init_lr=lr, lr_decay_epoch=10)

        epoch_val_acc = validate_model(model, val_loader)
        val_loss.append(epoch_val_acc)

        if not best_val_acc or epoch_val_acc > best_val_acc:
            best_model = copy.deepcopy(model)
            best_val_acc = epoch_val_acc
            es_counter = 0
        if es_counter > es:
            break
        if epoch > 30:
            es_counter += 1

        y_test = labels[1:]
        y_pred = predictions[1:]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        # print("%d %d %d %d",tn, fp, fn, tp)
        epoch_sen = tp / (tp + fn)
        epoch_spe = tn / (tn + fp)
        epoch_acc = (tn + tp) / (tn + tp + fn + fp)
        print("Training ... acc is : ", round(epoch_acc, 5), "sen is:", round(epoch_sen, 5), "spe is:",
              round(epoch_spe, 5))

        print('Training ... Epoch {}, train_loss {:.3f}, val_loss {:.3f}'.format(epoch, epoch_loss, epoch_val_acc))
        epoch_losses.append(epoch_loss / (iter + 1))

    return best_model


def validate_model(model, val_loader):
    model.eval()
    labels = np.empty([], dtype=int)
    predictions = np.empty([], dtype=int)
    val_loss = 0
    loss_func = nn.CrossEntropyLoss().cuda()

    for iter, (fc,ec,label) in enumerate(val_loader):
        label = Variable(label).type(torch.cuda.LongTensor).to(device)
        prediction = model(fc,ec).to(device)
        loss = loss_func(prediction, label)
        val_loss += loss.detach().item()
        labels = np.append(labels, label.cpu().numpy())
        predictions = np.append(predictions, torch.argmax(prediction, 1).cpu().numpy())

    y_test = labels[1:]
    y_pred = predictions[1:]

    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    # print("%d %d %d %d",tn, fp, fn, tp)
    epoch_sen = tp / (tp + fn)
    epoch_spe = tn / (tn + fp)
    epoch_acc = (tn + tp) / (tn + tp + fn + fp)
    print("Validating ... acc is : ",round(epoch_acc,5),"sen is:",round(epoch_sen,5),"spe is:",round(epoch_spe,5))

    return round(epoch_acc, 5)

    # return val_loss / iter


def test_model(model, test_loader):
    model.eval()
    labels = np.empty([], dtype=int)
    predictions = np.empty([], dtype=int)
    predss = []
    print('Testing...')
    for iter, (fc,ec, label) in enumerate(test_loader):

        label = Variable(label).type(torch.cuda.LongTensor).to(device)
        prediction = model(fc,ec).to(device)
        predss.append(prediction)

        labels = np.append(labels, label.cpu().numpy())
        predictions = np.append(predictions, torch.argmax(prediction, 1).cpu().numpy())

    y_test = labels[1:]
    y_pred = predictions[1:]
    predss = torch.cat(predss, dim=0).cpu().detach().numpy()[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    # print("%d %d %d %d",tn, fp, fn, tp)
    epoch_sen = tp / (tp + fn)
    epoch_spe = tn / (tn + fp)
    epoch_acc = (tn + tp) / (tn + tp + fn + fp)

    print("Testing ... acc is : ",round(epoch_acc,5),"sen is:",round(epoch_sen,5),"spe is:",round(epoch_spe,5))
    return round(epoch_acc, 5), round(epoch_sen, 5), round(epoch_spe, 5), y_test, predss


def run_kfold(epochs=1000, bs=32):
    print('Preparing Data ...')
    dataset = MddData()
    nrois = dataset.nrois

    kf = StratifiedKFold(n_splits=5, shuffle=True,random_state= 131)
    kf.get_n_splits(dataset)
    accs = []
    senss = []
    specs = []
    fprs, tprs, aucs = [], [], []
    k = 1

    for train_index, valtest_index in kf.split(dataset,dataset.labels):

        train_sampler = SubsetRandomSampler(train_index)
        train_loader = DataLoader(dataset, batch_size=bs, sampler=train_sampler)

        val_index, test_index, _, _ = train_test_split(valtest_index, dataset[valtest_index][2], test_size=0.5, shuffle=True,stratify=dataset[valtest_index][2])

        val_loader = DataLoader(dataset, batch_size=bs, sampler=val_index)
        test_loader = DataLoader(dataset, batch_size=bs, sampler=test_index)


        print('Training Fold : {}'.format(k))
        best_model = train_model(train_loader, val_loader, nrois, epochs)

        print('Saving best model ...')
        torch.save(best_model.state_dict(), '../visualizationAnalysis/GCN_weight/best_model_%02i.pth' % (k))

        acc, sens, spec, y_test, y_predss = test_model(best_model, test_loader)
        print("Test Accuracy for fold {} = {}".format(k, acc))
        print("Test sens for fold {} = {}".format(k, sens))
        print("Test spec for fold {} = {}".format(k, spec))
        accs.append(acc)
        senss.append(sens)
        specs.append(spec)

   
        fpr, tpr, threshold = metrics.roc_curve(y_test, y_predss)
        fprs.append(fpr)
        tprs.append(tpr)
        auc = metrics.auc(fpr, tpr)
        aucs.append(auc)

        k += 1
    print('-'*30)
    print("5 fold Test Accuracy: mean = {} ,std = {}".format(np.mean(accs), np.std(accs)))
    print("5 fold Test Sens: mean = {} ,std = {}".format(np.mean(senss), np.std(senss)))
    print("5 fold Test Specs: mean = {} ,std = {}".format(np.mean(specs), np.std(specs)))

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  


    for i in range(5):
        axs[0].plot(fprs[i], tprs[i], label='ROC fold %d (AUC = %0.2f)' % (i, aucs[i]))

    axs[0].plot([0, 1], [0, 1], 'k--')
    axs[0].set_xlim([0.0, 1.0])
    axs[0].set_ylim([0.0, 1.05])
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].set_title('ROC Curves for 5-fold Cross Validation')
    axs[0].legend(loc="lower right")


    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fprs, tprs)], axis=0)
    mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
    mean_auc = np.mean(aucs, axis=0)
    axs[1].plot(mean_fpr, mean_tpr, label='Mean ROC (AUC = %0.2f)' % mean_auc)
    axs[1].plot([0, 1], [0, 1], 'k--')
    axs[1].set_xlim([0.0, 1.0])
    axs[1].set_ylim([0.0, 1.05])
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].set_title('Mean ROC Curve')
    axs[1].legend(loc="lower right")


    plt.tight_layout()
    plt.savefig(f'../../data/BrainNetCNN_roc.tif', dpi=500, format='tif')
    plt.show()

    dict = {"fpr": mean_fpr, "tpr": mean_tpr, "auc": mean_auc}
    df = pd.DataFrame(dict)
    df.to_csv("../../data/BrainNetCNN_fpr_tpr.csv", index=False)

    print("Average AUC: ", np.mean(aucs))

if __name__ == "__main__":

    run_kfold()


