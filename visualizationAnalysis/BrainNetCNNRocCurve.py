import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PowerTransformer



global data ,label, importantLightgbmFeat,lightgbm_feats_col_name

data= None
label = None
importantLightgbmFeat = None
lightgbm_feats_col_name = None

def init():

    global data, label, importantLightgbmFeat, lightgbm_feats_col_name

    FeatureName = []
    k=0
    for i in range(1,116):
        for j in range(i+1,117):
            FeatureName.append(str(i)+"to"+str(j))
            k=k+1

    for i in range(1,116):
        for j in range(i+1,117):
            FeatureName.append(str(j)+"to"+str(i))
            k=k+1

    FeatureName = np.array(FeatureName)
    FeatureName = FeatureName.flatten()


    data = loadmat(r"../../../")
    data = data.get('')
    label=loadmat(r"../../../").get('')
    scaler = PowerTransformer()
    data = scaler.fit_transform(data)
    label=label.T
    label=label.flatten()
    label=np.array(label)

    map_a = dict()  #Hash mapping of feature name to feature index
    for i in range(13340):
        map_a[FeatureName[i]] = i

    importantLightgbmFeat = loadmat(r"../../../")
    importantLightgbmFeat  = importantLightgbmFeat.get(" ")
    importantLightgbmFeat = importantLightgbmFeat[0]

    lightgbm_feats_col_name = []
    for i in importantLightgbmFeat:
        lightgbm_feats_col_name.append(FeatureName[i])

def drow_roc():
    global data, label, importantLightgbmFeat, lightgbm_feats_col_name

    cnt = 1
    # random_state is consistent with the training model
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=58)

    aucs = []
    fprs, tprs = [], []
    for train_index, test_index in kfold.split(data, label):
        train_x, test_x = data[train_index], data[test_index]
        train_y, test_y = label[train_index], label[test_index]

        T_train = []
        for i in importantLightgbmFeat:
            T_train.append(train_x[:, i])
        T_train = np.asarray(T_train)
        T_train = T_train.T

        T_test = []
        for i in importantLightgbmFeat:
            T_test.append(test_x[:, i])
        T_test = np.asarray(T_test)
        T_test = T_test.T

        model = joblib.load(str(r"./BrainNetCNN" + str(cnt) + ".pkl"))
        pred_y = model(None, T_test)
        fpr, tpr, threshold = metrics.roc_curve(test_y, pred_y)
        fprs.append(fpr)
        tprs.append(tpr)
        auc = metrics.auc(fpr, tpr)
        aucs.append(auc)

        cnt = cnt + 1

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
    # plt.savefig(f'../../data/BrainNetCNN_roc.tif', dpi=500, format='tif')
    plt.show()

    # dict = {"fpr": mean_fpr, "tpr": mean_tpr, "auc": mean_auc}
    # df = pd.DataFrame(dict)
    # df.to_csv("../../data/BrainNetCNN_fpr_tpr.csv", index=False)

    print("Average AUC: ", np.mean(aucs))


if __name__ == '__main__':
    init()
    drow_roc()
