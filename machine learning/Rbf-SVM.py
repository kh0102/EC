import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm, metrics
from sklearn.metrics import classification_report
from scipy import stats
import numpy as np
import sklearn
from scipy.io import loadmat
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import os



global data,label

data = None
label = None


def init():
    global data, label, featureNum

    # EC feature
    # data = loadmat(r"../../").get('')

    # FC feature
    data = loadmat(r"../../").get('').T

    label=loadmat(r"../../").get('')

    scaler = PowerTransformer()
    data = scaler.fit_transform(data)

    label=label.T
    label=label.flatten()
    label=np.array(label)
    # featureNum = 13340
    featureNum = 6670


def run():

    global data,label,featureNum

    RbfSVM_Accuracy=[]
    RbfSVM_Specificity = []
    RbfSVM_Sensitivity = []

    cnt = 1
    kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=131)

    aucs = []
    fprs, tprs = [], []
    for train_index, test_index in kfold.split(data, label):

        train_x, test_x = data[train_index], data[test_index]
        train_y, test_y = label[train_index], label[test_index]
        p=0.05

        index = []
        """
        Get feature index according to the specified p-value
        """
        for j in range(featureNum):
            feat = train_x[:, j]
            data1 = []
            data0 = []
            for i in range(train_x.shape[0]):
                if(train_y[i]==1):
                    data1.append(feat[i])
                else:
                    data0.append(feat[i])
            lev = stats.levene(data1, data0)
            if (lev.pvalue < p):
                ttest = stats.ttest_ind(data1, data0, equal_var=False)
            else:
                ttest = stats.ttest_ind(data1, data0, equal_var=True)
            if (ttest.pvalue < p):
                index.append(j)

        print("The p-value is %f, and there are %d features that satisfy the requirement" % (p, index.__len__()))

        T_train=[]
        for i in index:
            T_train.append(train_x[:,i])
        T_train=np.asarray(T_train)
        T_train=T_train.T

        T_test=[]
        for i in index:
            T_test.append(test_x[:,i])
        T_test = np.asarray(T_test)
        T_test = T_test.T

        classifier = svm.SVC(kernel='rbf', probability=True)

        smo = BorderlineSMOTE(kind='borderline-2', random_state=131)  # kind='borderline-2'

        T_train, train_y = smo.fit_resample(T_train, train_y)
        print(sorted(Counter(train_y).items()))

        classifier.fit(T_train, train_y)
        y_pred = classifier.predict(T_test)
        pred_labels = np.rint(y_pred)
        accuracy = sklearn.metrics.accuracy_score(test_y, pred_labels)
        print("RbfSVM testing set accuracy:%.2f" % accuracy)
        dictScore = classification_report(test_y, y_pred, output_dict=True)
        print(classification_report(test_y,y_pred))
        RbfSVM_Accuracy.append(dictScore['weighted avg']['recall'])
        RbfSVM_Sensitivity.append(dictScore['1']['recall'])
        RbfSVM_Specificity.append(dictScore['0']['recall'])


        joblib.dump(classifier,"../visualizationAnalysis/Rbf-SVM"+str(cnt)+".pkl")
        y_predss = classifier.predict_proba(T_test)
        fpr, tpr, threshold = metrics.roc_curve(test_y, y_predss[:, 1])
        fprs.append(fpr)
        tprs.append(tpr)
        auc = metrics.auc(fpr, tpr)
        aucs.append(auc)

        cnt = cnt + 1

    print("Current running file path" + os.getcwd())
    print("REST_meta_MDD RbfSVM five-fold cross-validation average Accuracy:%.4f"%(np.mean(RbfSVM_Accuracy)))
    print("REST_meta_MDD RbfSVM five-fold cross-validation average Sensitivity:%.4f"%(np.mean(RbfSVM_Sensitivity)))
    print("REST_meta_MDD RbfSVM five-fold cross-validation average Specificity:%.4f"%(np.mean(RbfSVM_Specificity)))

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
    plt.savefig(f'../../data/RbfSVM_roc.tif', dpi=500, format='tif')
    plt.show()


    dict = {"fpr": mean_fpr, "tpr": mean_tpr, "auc": mean_auc}
    df = pd.DataFrame(dict)
    df.to_csv("../../data/RbfSVM_fpr_tpr.csv", index=False)

    print("Average AUC: ", np.mean(aucs))


if __name__ == "__main__":

    init()
    run()



