import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
from torch.utils.data.sampler import SubsetRandomSampler
import copy
from torch.autograd import Variable
import torch
import torch.nn as nn
import os

# model
from model import TSTransformerEncoderClassiregressor

# dataset
import load_data

# utils
import utils

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def exp_lr_scheduler(optimizer, epoch, init_lr=0.0001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

# todo: set the model params
feat_dim = 116
max_seq_len = 128
d_model = 128
num_heads = 8
num_layers = 3
dim_feedforward = 256
num_labels = 2
dropout = 0.1
pos_encoding = 'learnable'
activation = 'gelu'
normalization_layer = 'BatchNorm'
freeze = False

# model training， input：
def train_model(train_loader, val_loader, epochs, k_fold, store_name):
    # Create training and testing set.

    val_loss = []  # Store the loss of the validation set
    best_val_acc = None
    best_model = None

    # todo: set the model
    model = TSTransformerEncoderClassiregressor(feat_dim, max_seq_len, d_model,
                                                num_heads,
                                                num_layers, dim_feedforward,
                                                num_classes=num_labels,
                                                dropout=dropout, pos_encoding=pos_encoding,
                                                activation=activation,
                                                norm=normalization_layer, freeze=freeze).to(device)

    # loss and optimizer
    loss_func = nn.CrossEntropyLoss().to(device)
    lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    es = 100
    es_counter = 0
    print('Training....')
    epoch_losses = []  # store the train loss
    epoch_val_accs = []  # store the val acc
    for epoch in range(epochs):
        epoch_loss = 0
        labels = np.empty([], dtype=int)
        predictions = np.empty([], dtype=int)

        for iter, batch in enumerate(train_loader):
            tc, label, padding_masks = batch
            tc = tc.to(device)
            label = Variable(label).type(torch.LongTensor).to(device)

            padding_masks = padding_masks.to(device)

            prediction = model(tc.to(device), padding_masks.to(device)).to(device)
            loss = loss_func(prediction.to(device), torch.max(label, 1)[1].to(device))

            labels = np.append(labels, torch.argmax(label, 1).cpu().numpy())
            predictions = np.append(predictions, torch.argmax(prediction, 1).cpu().numpy())

            optimizer.zero_grad()
            # backward and update
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss = epoch_loss / iter


        optimizer = exp_lr_scheduler(optimizer, epoch, init_lr=lr, lr_decay_epoch=40)

        epoch_val_loss, epoch_val_acc, _, _ = validate_model(model, val_loader)

        val_loss.append(epoch_val_loss)
        epoch_val_accs.append(epoch_val_acc)


        print('Epoch {}, train_loss {:.8f}, val_loss {:.8f}'.format(epoch, epoch_loss, epoch_val_loss))
        epoch_losses.append(epoch_loss / (iter + 1))

        if not best_val_acc or epoch_val_acc > best_val_acc:
            best_model = copy.deepcopy(model)
            best_val_acc = epoch_val_acc
            es_counter = 0
        if es_counter > es:
            break
        if epoch > 5:
            es_counter += 1

        # plot loss and acc pictures
        # train loss plot
        utils.plot_loss(epoch_losses, save_path=r'../transformer_code/logs/'+store_name+'/train_loss'+str(k_fold)+'.jpg')
        # val loss plot
        utils.plot_loss(val_loss, save_path=r'../transformer_code/logs/'+store_name+'/val_loss'+str(k_fold)+'.jpg', label_name='val_loss')
        # val acc plot
        utils.plot_loss(epoch_val_accs,
                        save_path=r'../transformer_code/logs/' + store_name + '/val_acc' + str(k_fold) + '.jpg',
                        label_name='val_acc')

    print('Done.')
    # wandb.finish()
    return best_model


def validate_model(model, val_loader):
    model.eval()
    val_loss = 0
    loss_func = nn.CrossEntropyLoss().to(device)
    labels = np.empty([], dtype=int)
    predictions = np.empty([], dtype=int)
    predss = []

    for iter, batch in enumerate(val_loader):

        tc, label, padding_masks = batch
        padding_masks = padding_masks.to(device)

        label = Variable(label).type(torch.LongTensor)
        prediction = model(tc.to(device), padding_masks.to(device))
        predss.append(prediction)
        loss = loss_func(prediction.to(device), torch.max(label, 1)[1].to(device))
        val_loss += loss.detach().item()

        labels = np.append(labels, torch.argmax(label, 1).cpu().numpy())
        predictions = np.append(predictions, torch.argmax(prediction, 1).cpu().numpy())

    y_test = labels[1:]
    y_pred = predictions[1:]
    predss = torch.cat(predss, dim=0).cpu().detach().numpy()[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    epoch_sen = tp/(tp+fn)
    epoch_spe = tn/(tn+fp)
    epoch_acc = (tn+tp)/(tn+tp+fn+fp)


    return val_loss / iter, round(epoch_acc, 5), y_test, predss


def test_model(model, test_loader):
    model.eval()
    labels = np.empty([], dtype=int)
    predictions = np.empty([], dtype=int)
    predss = []
    print('Testing...')
    for iter, batch in enumerate(test_loader):
        tc, label, padding_masks = batch
        tc = tc.to(device)
        padding_masks = padding_masks.to(device)

        label = Variable(label).type(torch.LongTensor)
        prediction = model(tc.to(device), padding_masks.to(device))
        predss.append(prediction)

        labels = np.append(labels, torch.argmax(label, 1).cpu().numpy())
        predictions = np.append(predictions, torch.argmax(prediction, 1).cpu().numpy())

    y_test = labels[1:]
    y_pred = predictions[1:]
    predss = torch.cat(predss, dim=0).cpu().detach().numpy()[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    epoch_sen = tp/(tp+fn)
    epoch_spe = tn/(tn+fp)
    epoch_acc = (tn+tp)/(tn+tp+fn+fp)

    print(f'val... acc:{epoch_acc},sen:{epoch_sen},spe:{epoch_spe}')
    return round(epoch_acc, 5), round(epoch_sen, 5), round(epoch_spe, 5), y_test, predss






def run_kfold(store_name, dataset_type='mdd_all_fc', epochs=400, bs=32):
    print('Preparing Data ...')

    # todo: change the data file and dataset

    if dataset_type == 'mdd_all_ec':
        print('the data is mdd all ec')
        file_path = r'../../data/ECMatrix.npy'
        dataset = load_data.Mdd_EC(Mdd_EC_npz_file_path=file_path)

    elif dataset_type == 'mdd_all_fc':
        print('the data is mdd all fc')
        file_path = r'../../data/FCMatrix.npy'
        dataset = load_data.Mdd_FC(Mdd_FC_npz_file_path=file_path)


    kf = StratifiedKFold(n_splits=5, shuffle=True)
    kf.get_n_splits(dataset)
    accs = []
    senss = []
    specs = []
    fprs, tprs, aucs = [], [], []
    k = 1

    for train_index, valtest_index in kf.split(dataset.fcs, dataset.labels[:, 1]):
        # Creating PT data samplers and loaders:

        val_index, test_index, _, _ = train_test_split(valtest_index, dataset[valtest_index][1], test_size=0.5,
                                                       shuffle=True, stratify=dataset[valtest_index][1])

        train_sampler = SubsetRandomSampler(train_index)
        valid_sampler = SubsetRandomSampler(val_index)
        test_sampler = SubsetRandomSampler(test_index)

        train_loader = DataLoader(dataset, batch_size=bs,
                                  sampler=train_sampler,
                                  collate_fn=lambda x: utils.collate_superv(x, max_len=max_seq_len))
        val_loader = DataLoader(dataset, batch_size=bs,
                                sampler=valid_sampler,
                                collate_fn=lambda x: utils.collate_superv(x, max_len=max_seq_len))
        test_loader = DataLoader(dataset, batch_size=bs,
                                 sampler=test_sampler,
                                 collate_fn=lambda x: utils.collate_superv(x, max_len=max_seq_len))
        print('Training Fold : {}'.format(k))

        best_model = train_model(train_loader, val_loader, epochs, k, store_name)

        # 保存最好的模型到指定文件夹
        print('Saving best model ...')
        joblib.dump(best_model, "../visualizationAnalysis/Transformer_weight/transformer"+str(k)+".pkl")

        acc, sens, spec, y_test, y_predss = test_model(best_model, test_loader)
        print("Test Accuracy for fold {} = {}".format(k, acc))
        print("Test sens for fold {} = {}".format(k, sens))
        print("Test spec for fold {} = {}".format(k, spec))
        accs.append(acc)
        senss.append(sens)
        specs.append(spec)

        # 计算fpr与tpr
        fpr, tpr, threshold = metrics.roc_curve(y_test, y_predss)
        fprs.append(fpr)
        tprs.append(tpr)
        auc = metrics.auc(fpr, tpr)
        aucs.append(auc)

        k += 1
    print('---------------------------------')
    print("5 fold Test Accuracy: mean = {} ,std = {}".format(np.mean(accs), np.std(accs)))
    print("5 fold Test Sens: mean = {} ,std = {}".format(np.mean(senss), np.std(senss)))
    print("5 fold Test Specs: mean = {} ,std = {}".format(np.mean(specs), np.std(specs)))

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 创建包含两个子图的图

    # 在第一个子图中绘制每一折的 ROC 曲线
    for i in range(5):
        axs[0].plot(fprs[i], tprs[i], label='ROC fold %d (AUC = %0.2f)' % (i, aucs[i]))

    axs[0].plot([0, 1], [0, 1], 'k--')
    axs[0].set_xlim([0.0, 1.0])
    axs[0].set_ylim([0.0, 1.05])
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].set_title('ROC Curves for 5-fold Cross Validation')
    axs[0].legend(loc="lower right")

    # 在最后一个子图中绘制平均 ROC 曲线
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

    # 显示与保存图
    plt.tight_layout()
    plt.savefig(f'../../data/transformer_roc.tif', dpi=500, format='tif')
    plt.show()

    # 保存当前模型得到平均fpr与tpr
    dict = {"fpr": mean_fpr, "tpr": mean_tpr, "auc": mean_auc}
    df = pd.DataFrame(dict)
    df.to_csv("../../data/transformer_fpr_tpr.csv", index=False)

    print("Average AUC: ", np.mean(aucs))


if __name__ == "__main__":

    store_name = 'mdd_all_fc'
    # store_name = 'mdd_all_ec'
    store_file_path = r'../transformer_code/logs/'+store_name
    is_exist = os.path.exists(store_file_path)
    if not is_exist:
        os.makedirs(store_file_path)
        print('create log dir ok')

    run_kfold(store_name, dataset_type="mdd_all_fc")
