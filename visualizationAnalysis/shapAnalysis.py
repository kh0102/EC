import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.preprocessing import PowerTransformer

shap.initjs()

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


    data = loadmat(r"../../")
    data = data.get('')
    label=loadmat(r"../..").get('')
    scaler = PowerTransformer()
    data = scaler.fit_transform(data)
    label=label.T
    label=label.flatten()
    label=np.array(label)

    map_a = dict()  #Hash mapping of feature name to feature index
    for i in range():
        map_a[FeatureName[i]] = i

    importantLightgbmFeat = loadmat(r"../../")
    importantLightgbmFeat  = importantLightgbmFeat.get("")
    importantLightgbmFeat = importantLightgbmFeat[0]

    lightgbm_feats_col_name = []
    for i in importantLightgbmFeat:
        lightgbm_feats_col_name.append(FeatureName[i])


def getPicture():

    global data, label, importantLightgbmFeat, lightgbm_feats_col_name

    cnt = 1
    # random_state is consistent with the training model
    kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=58)

    for train_index, test_index in kfold.split(data, label):
        train_x, test_x = data[train_index], data[test_index]
        train_y, test_y = label[train_index], label[test_index]

        T_train=[]
        for i in importantLightgbmFeat:
            T_train.append(train_x[:,i])
        T_train=np.asarray(T_train)
        T_train=T_train.T

        T_test=[]
        for i in importantLightgbmFeat:
            T_test.append(test_x[:,i])
        T_test = np.asarray(T_test)
        T_test = T_test.T

        gbm = joblib.load(str(r"./lightgbm"+str(cnt)+".pkl"))

        T_train = pd.DataFrame(T_train)
        T_train.columns = lightgbm_feats_col_name

        T_test = pd.DataFrame(T_test)
        T_test.columns = lightgbm_feats_col_name


        explainer = shap.Explainer(gbm, T_test)
        shap_values = explainer.shap_values(T_test)

        plt.figure(figsize=(10,10),dpi=500)
        shap.summary_plot(shap_values, T_test[lightgbm_feats_col_name], show=False, plot_type="dot",max_display=20, plot_size=(10, 10))
        plt.tight_layout()
        plt.savefig(r"C:/shapPicture/Figure A.{}.(a).tif".format(cnt),dpi=500,format ='tif')
        plt.show()

        plt.figure(figsize=(10,10),dpi=500)
        shap.summary_plot(shap_values, T_test[lightgbm_feats_col_name], show=False, plot_type="bar", plot_size=(10,10))
        plt.tight_layout()
        plt.savefig(r"C:/shapPicture/Figure A.{}.(b).tif".format(cnt),dpi=500,format ='tif')
        plt.show()


        cnt = cnt + 1

if __name__ == "__main__":

    init()
    getPicture()