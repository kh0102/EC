import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.nn.functional as func
from matplotlib import pyplot as plt
from sklearn import metrics
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix

from Model import GCN
from Dataset import ConnectivityData
from scipy.io import loadmat
from torch.utils.data.sampler import SubsetRandomSampler

def GCN_train(loader):
    model.train()

    loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = func.cross_entropy(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    # return loss_all / len(train_dataset)
    return loss_all / len(train_index)


def GCN_test(loader):
    model.eval()

    pred = []
    predss = []
    label = []
    loss_all = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        loss = func.cross_entropy(output, data.y)
        loss_all += data.num_graphs * loss.item()
        predss.append(output)
        pred.append(func.softmax(output, dim=1).max(dim=1)[1])
        label.append(data.y)

    y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
    y_predss = torch.cat(predss, dim=0).cpu().detach().numpy()[:, 1]
    y_true = torch.cat(label, dim=0).cpu().detach().numpy()
    tn, fp, fn, tp = confusion_matrix(y_pred, y_true).ravel()
    epoch_sen = tp / (tp + fn)
    epoch_spe = tn / (tn + fp)
    epoch_acc = (tn + tp) / (tn + tp + fn + fp)
    sklearn.metrics.accuracy_score(y_true, y_pred)
    # return epoch_sen, epoch_spe, epoch_acc, loss_all / len(val_dataset)
    return epoch_sen, epoch_spe, epoch_acc, loss_all / len(val_index), y_predss, y_true


labels = loadmat(r"../../").get('')[0]
dataset = ConnectivityData('../../')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
eval_metrics = np.zeros((skf.n_splits, 3))



aucs = []
fprs, tprs = [], []
# for n_fold, (train_val, test) in enumerate(skf.split(labels, labels)):
for n_fold, (train_index, valtest_index) in enumerate(skf.split(labels, labels)):

    # train_val_dataset, test_dataset = dataset[train_val.tolist()], dataset[test.tolist()]
    #
    # train_val_labels = labels[train_val]
    # train_val_index = np.arange(len(train_val_dataset))
    #
    # train, val, _, _ = train_test_split(train_val_index, train_val_labels, test_size=0.11, shuffle=True, stratify=train_val_labels)
    # train_dataset, val_dataset = train_val_dataset[train.tolist()], train_val_dataset[val.tolist()]
    #
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    bs = 32
    train_sampler = SubsetRandomSampler(train_index)
    train_loader = DataLoader(dataset, batch_size=bs, sampler=train_sampler)

    val_index, test_index, _, _ = train_test_split(valtest_index, dataset.data['y'][valtest_index], test_size=0.5,
                                                   shuffle=True, stratify=dataset.data['y'][valtest_index])

    val_loader = DataLoader(dataset, batch_size=bs, sampler=val_index)
    test_loader = DataLoader(dataset, batch_size=bs, sampler=test_index)


    model = GCN(dataset.num_features, dataset.num_classes, 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # min_v_loss = np.inf
    best_val_acc = None

    y_predss, y_true = 0.0, 0.0
    for epoch in range(50):
        t_loss = GCN_train(train_loader)
        val_sen, val_spe, val_acc, v_loss, _, _ = GCN_test(val_loader)
        test_sen, test_spe, test_acc, _, predss_y, true_y = GCN_test(test_loader)

        if not best_val_acc or val_acc > best_val_acc:

            best_val_acc = val_acc
            best_test_sen, best_test_spe, best_test_acc = test_sen, test_spe, test_acc
            y_predss, y_true = predss_y, true_y
            print("saving best model...")
            torch.save(model.state_dict(), '../visualizationAnalysis/GCN_weight/best_model_%02i.pth' % (n_fold + 1))
            print('CV: {:03d}, Epoch: {:03d}, Val Loss: {:.5f}, Val ACC: {:.5f}, Test ACC: {:.5f}, TEST SEN: {:.5f}, '
                  'TEST SPE: {:.5f}'.format(n_fold + 1, epoch + 1, v_loss, best_val_acc, best_test_acc,
                                            best_test_sen, best_test_spe))

    eval_metrics[n_fold, 0] = best_test_sen
    eval_metrics[n_fold, 1] = best_test_spe
    eval_metrics[n_fold, 2] = best_test_acc

    # 计算fpr与tpr
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_predss)
    fprs.append(fpr)
    tprs.append(tpr)
    auc = metrics.auc(fpr, tpr)
    aucs.append(auc)

eval_df = pd.DataFrame(eval_metrics)
eval_df.columns = ['SEN', 'SPE', 'ACC']
eval_df.index = ['Fold_%02i' % (i + 1) for i in range(skf.n_splits)]
print(eval_df)
print('Average Sensitivity: %.4f±%.4f' % (eval_metrics[:, 0].mean(), eval_metrics[:, 0].std()))
print('Average Specificity: %.4f±%.4f' % (eval_metrics[:, 1].mean(), eval_metrics[:, 1].std()))
print('Average Accuracy: %.4f±%.4f' % (eval_metrics[:, 2].mean(), eval_metrics[:, 2].std()))


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
plt.savefig(f'../../data/GCN_roc.tif', dpi=500, format='tif')
plt.show()


dict = {"fpr": mean_fpr, "tpr": mean_tpr, "auc": mean_auc}
df = pd.DataFrame(dict)
df.to_csv("../../data/GCN_fpr_tpr.csv", index=False)

print("Average AUC: ", np.mean(aucs))
