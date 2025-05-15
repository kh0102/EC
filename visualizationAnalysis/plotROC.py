import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc


data_type = " "


lgb_ROC_data = pd.read_csv("../../data/" + data_type + "/lgb_fpr_tpr.csv")
GCN_ROC_data = pd.read_csv("../../data/" + data_type + "/GCN_fpr_tpr.csv")
BrainNetCNN_ROC_data = pd.read_csv("../../data/" + data_type + "/BrainNetCNN_fpr_tpr.csv")
transformer_ROC_data = pd.read_csv("../../data/" + data_type + "/transformer_fpr_tpr.csv")
RbfSVM_ROC_data = pd.read_csv("../../data/" + data_type + "/RbfSVM_fpr_tpr.csv")
xgb_ROC_data = pd.read_csv("../../data/" + data_type + "/xgb_fpr_tpr.csv")
RF_ROC_data = pd.read_csv("../../data/" + data_type + "/RF_fpr_tpr.csv")


lgb_fpr, lgb_tpr, lgb_auc = lgb_ROC_data["fpr"].values, lgb_ROC_data["tpr"].values, lgb_ROC_data["auc"].values[0]
lgb_fpr[0], lgb_tpr[0], lgb_fpr[-1], lgb_tpr[-1] = 0, 0, 1, 1

GCN_fpr, GCN_tpr, GCN_auc = GCN_ROC_data["fpr"].values, GCN_ROC_data["tpr"].values, GCN_ROC_data["auc"].values[0]
GCN_fpr[0], GCN_tpr[0], GCN_fpr[-1], GCN_tpr[-1] = 0, 0, 1, 1

BrainNetCNN_fpr, BrainNetCNN_tpr, BrainNetCNN_auc = BrainNetCNN_ROC_data["fpr"].values, BrainNetCNN_ROC_data["tpr"].values, BrainNetCNN_ROC_data["auc"].values[0]
BrainNetCNN_fpr[0], BrainNetCNN_tpr[0], BrainNetCNN_fpr[-1], BrainNetCNN_tpr[-1] = 0, 0, 1, 1

transformer_fpr, transformer_tpr, transformer_auc = transformer_ROC_data["fpr"].values, transformer_ROC_data["tpr"].values, transformer_ROC_data["auc"].values[0]
transformer_fpr[0], transformer_tpr[0], transformer_fpr[-1], transformer_tpr[-1] = 0, 0, 1, 1

RbfSVM_fpr, RbfSVM_tpr, RbfSVM_auc = RbfSVM_ROC_data["fpr"].values, RbfSVM_ROC_data["tpr"].values, RbfSVM_ROC_data["auc"].values[0]
RbfSVM_fpr[0], RbfSVM_tpr[0], RbfSVM_fpr[-1], RbfSVM_tpr[-1] = 0, 0, 1, 1

xgb_fpr, xgb_tpr, xgb_auc = xgb_ROC_data["fpr"].values, xgb_ROC_data["tpr"].values, xgb_ROC_data["auc"].values[0]
xgb_fpr[0], xgb_tpr[0], xgb_fpr[-1], xgb_tpr[-1] = 0, 0, 1, 1

RF_fpr, RF_tpr, RF_auc = RF_ROC_data["fpr"].values, RF_ROC_data["tpr"].values, RF_ROC_data["auc"].values[0]
RF_fpr[0], RF_tpr[0], RF_fpr[-1], RF_tpr[-1] = 0, 0, 1, 1


fig = plt.figure(figsize = (8,6))


plt.plot(RbfSVM_fpr,RbfSVM_tpr,color='#f03b43',label=r'RbfSVM (AUC=%0.3f)'%RbfSVM_auc,lw=1.5,alpha=1)
plt.plot(RF_fpr,RF_tpr,color='#14d164',label=r'Random Forest (AUC=%0.3f)'%RF_auc,lw=1.5,alpha=1)
plt.plot(xgb_fpr,xgb_tpr,color='#ff9900',label=r'XGBoost (AUC=%0.3f)'%xgb_auc,lw=1.5,alpha=1)
plt.plot(GCN_fpr,GCN_tpr,color='#33a02c',label=r'GCN (AUC=%0.3f)'%GCN_auc,lw=1.5,alpha=1)
plt.plot(BrainNetCNN_fpr,BrainNetCNN_tpr,color='#fcc006',label=r'BrainNetCNN (AUC=%0.3f)'%BrainNetCNN_auc,lw=1.5,alpha=1)
plt.plot(transformer_fpr,transformer_tpr,color='#FF69B4',label=r'Transformer (AUC=%0.3f)'%transformer_auc,lw=1.5,alpha=1)
plt.plot(lgb_fpr,lgb_tpr,color='#1f77b4',label=r'Lightgbm (AUC=%0.3f)'%lgb_auc,lw=1.5,alpha=1)

# plt.plot(SVM_fpr,SVM_tpr,color='#e41a1c',label=r'SVM (AUC=%0.3f)'%SVM_auc,lw=1.5,alpha=1, linestyle = 'dashdot')
#
# plt.plot(RF_fpr,RF_tpr,color='#9467bd',label=r'RF (AUC=%0.3f)'%RF_auc,lw=1.5,alpha=1, linestyle = 'dotted')

plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.title('')
plt.show()
plt.savefig(r"../../data/LgbCompareWithOthers(" + data_type + ").tif", dpi = 500, format = 'tif')
