import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import sklearn
from scipy.io import loadmat, savemat
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold,RepeatedStratifiedKFold
from collections import Counter
import joblib
import os
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import BorderlineSMOTE
import optuna
import lightgbm as lgb
from sklearn.metrics import accuracy_score



global data,label

data = None
label = None


def init():
    global data, label, importantLightgbmFeat

    data = loadmat(r"../../").get('')
    label=loadmat(r"../../").get('')

    scaler = PowerTransformer()
    data = scaler.fit_transform(data)

    label=label.T
    label=label.flatten()
    label=np.array(label)

    importantLightgbmFeat = loadmat(r"../../")
    importantLightgbmFeat = importantLightgbmFeat.get('')





def run():

    global data, label, importantLightgbmFeat

    LinearSVM_Accuracy = []
    LinearSVM_Specificity = []
    LinearSVM_Sensitivity = []

    LightGBM_Accuracy = []
    LightGBM_Specificity = []
    LightGBM_Sensitivity = []

    cnt = 1
    kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=131)

    lgb_fprs, lgb_tprs, lgb_aucs = [], [], []
    svm_fprs, svm_tprs, svm_aucs = [], [], []
    for train_index, test_index in kfold.split(data, label):

        train_x, test_x = data[train_index], data[test_index]
        train_y, test_y = label[train_index], label[test_index]



        T_train=[]
        for i in importantLightgbmFeat:
            T_train.append(train_x[:,i])
        T_train=np.asarray(T_train)
        # T_train=T_train.T
        T_train=T_train.reshape(T_train.shape[1], T_train.shape[2])


        T_test=[]
        for i in importantLightgbmFeat:
            T_test.append(test_x[:, i])
        T_test = np.asarray(T_test)
        # T_test = T_test.T
        T_test = T_test.reshape(T_test.shape[1], T_test.shape[2])

        # svm = joblib.load(str("linearSvm"+str(cnt)+".m"))
        # gbm = joblib.load(str("lightgbm"+str(cnt)+".pkl"))

        """
        LinearSVM tuning parameters
        """
        def objective(trial):
            param = {
                "penalty": trial.suggest_categorical('penalty', ['l1', 'l2']),
                "tol": trial.suggest_float("tol", 1e-5, 1),
                "C": trial.suggest_float('C', 1e-3, 1e2),
                "loss": "squared_hinge",
                "class_weight": "balanced",
            }
            # Network attribute features use max_iter = 1e7, all others are 1e6
            if (param["penalty"] == 'l1'):
                svm = LinearSVC(penalty=param["penalty"], loss=param["loss"], C=param["C"], tol=param["tol"],
                               max_iter=10000000,
                                       class_weight=param["class_weight"], dual=False)
            else:
                svm = LinearSVC(penalty=param["penalty"], loss=param["loss"], C=param["C"], tol=param["tol"],
                                       max_iter=10000000,
                                       class_weight=param["class_weight"], dual=True)

            kfold = StratifiedKFold(n_splits=5)
            X=T_train
            Y=train_y

            scores = []
            for train_index, valid_index in kfold.split(X, Y):
                x_train, x_valid = X[train_index], X[valid_index]
                y_train, y_valid = Y[train_index], Y[valid_index]

                smo = BorderlineSMOTE(kind='borderline-2', random_state=131)  # kind='borderline-2'
                x_train, y_train = smo.fit_resample(x_train, y_train)

                svm.fit(x_train, y_train)
                y_pred = svm.predict(x_valid)
                pred_labels = np.rint(y_pred)
                accuracy = accuracy_score(y_valid, pred_labels)
                scores.append(accuracy)
            print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
            print(param)
            return np.mean(scores)

        # Optuna for LinearSVM with 100 iterations and a time limit of 600 seconds
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, timeout=600)


        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        if (study.best_params["penalty"] == 'l1'):
            svm = LinearSVC(penalty=study.best_params["penalty"], loss="squared_hinge", C=study.best_params["C"],
                                   tol=study.best_params["tol"], max_iter=10000000,
                                   class_weight='balanced', dual=False)
        else:
            svm = LinearSVC(penalty=study.best_params["penalty"], loss="squared_hinge", C=study.best_params["C"],
                                   tol=study.best_params["tol"], max_iter=10000000,
                                   class_weight='balanced', dual=True)

        calibrated_svc = CalibratedClassifierCV(svm, method='sigmoid')
        calibrated_svc.fit(T_train, train_y)
        smo = BorderlineSMOTE(kind='borderline-2', random_state=131)

        T_traincopy = T_train
        train_ycopy = train_y

        T_train, train_y = smo.fit_resample(T_train, train_y)
        print(sorted(Counter(train_y).items()))
        svm.fit(T_train, train_y)
        joblib.dump(svm, "../visualizationAnalysis/LinearSVM" + str(cnt) + ".pkl")

        y_pred = svm.predict(T_test)
        pred_labels = np.rint(y_pred)
        accuracy = sklearn.metrics.accuracy_score(test_y, pred_labels)
        print("LinearSVM testing set accuracy:%.2f" % accuracy)
        dictScore = classification_report(test_y, y_pred, output_dict=True)
        print(classification_report(test_y,y_pred))
        LinearSVM_Accuracy.append(dictScore['weighted avg']['recall'])
        LinearSVM_Sensitivity.append(dictScore['1']['recall'])
        LinearSVM_Specificity.append(dictScore['0']['recall'])

        # 保存tpr与fpr
        y_predss = calibrated_svc.predict_proba(T_test)
        fpr, tpr, threshold = metrics.roc_curve(test_y, y_predss[:, 1])
        svm_fprs.append(fpr)
        svm_tprs.append(tpr)
        auc = metrics.auc(fpr, tpr)
        svm_aucs.append(auc)

        T_train = T_traincopy
        train_y = train_ycopy


        """
        LightGBM tuning parameters
        """
        def objective(trial):
            param = {
                "objective": "binary",
                "verbosity": -1,
                "train_metric": True,
                "boosting_type": trial.suggest_categorical("boosting_type", ['dart', 'rf', 'gbdt']),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 5, 100),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "n_estimators": trial.suggest_int("n_estimators", 10, 200),
                # "n_estimators": trial.suggest_categorical("n_estimators", [int(x) for x in np.linspace(start=1, stop=1000, num=100)]),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "min_child_samples": trial.suggest_int("min_child_samples", 1, 20),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 1),
                "max_bin": trial.suggest_int("max_bin", 10, 100),
                "cat_smooth": trial.suggest_int("cat_smooth", 0, 100),
            }

            kfold = StratifiedKFold(n_splits=5)

            scores = []
            X=T_train
            Y=train_y

            for train_index, valid_index in kfold.split(X, Y):
                x_train, x_valid = X[train_index], X[valid_index]
                y_train, y_valid = Y[train_index], Y[valid_index]

                smo = BorderlineSMOTE(kind='borderline-2', random_state=131)
                x_train, y_train = smo.fit_resample(x_train, y_train)

                gbm = lgb.LGBMClassifier(**param)
                gbm.fit(x_train, y_train)

                y_pred = gbm.predict(x_valid)
                pred_labels = np.rint(y_pred)
                accuracy = accuracy_score(y_valid, pred_labels)
                scores.append(accuracy)
            print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
            print(param)
            return np.mean(scores)

        #Optuna for LightGBM with 100 iterations and a time limit of 600 seconds
        studyLightgbm = optuna.create_study(direction="maximize")
        studyLightgbm.optimize(objective, n_trials=100,timeout=600)


        print("Number of finished trials: {}".format(len(studyLightgbm.trials)))

        print("Best trial:")
        print("  Value: {}".format(studyLightgbm.best_trial.value))

        print("  Params: ")
        for key, value in studyLightgbm.best_trial.params.items():
            print("{}: {}".format(key, value))

        gbm = lgb.LGBMClassifier(**studyLightgbm.best_params)
        T_train, train_y = smo.fit_resample(T_train, train_y)
        print(sorted(Counter(train_y).items()))
        gbm.fit(T_train, train_y)
        # joblib.dump(gbm, "./lightgbm"+str(cnt)+".pkl")
        joblib.dump(svm, "../visualizationAnalysis/LightGBM" + str(cnt) + ".pkl")

        y_pred = gbm.predict(T_test)
        pred_labels = np.rint(y_pred)
        accuracy = sklearn.metrics.accuracy_score(test_y, pred_labels)
        print("LightGBM testing set accuracy:%.2f" % accuracy)
        dictScore = classification_report(test_y, y_pred, output_dict=True)
        print(classification_report(test_y, y_pred))

        LightGBM_Accuracy.append(dictScore['weighted avg']['recall'])
        LightGBM_Sensitivity.append(dictScore['1']['recall'])
        LightGBM_Specificity.append(dictScore['0']['recall'])

        # 保存tpr与fpr
        y_predss = gbm.predict_proba(T_test)
        fpr, tpr, threshold = metrics.roc_curve(test_y, y_predss[:, 1])
        lgb_fprs.append(fpr)
        lgb_tprs.append(tpr)
        auc = metrics.auc(fpr, tpr)
        lgb_aucs.append(auc)

        cnt = cnt + 1


    print("Current running file path" + os.getcwd())
    print("LinearSVM five-fold cross-validation average Accuracy:%.4f"%(np.mean(LinearSVM_Accuracy)))
    print("LightGBM five-fold cross-validation average Accuracy:%.4f"%(np.mean(LightGBM_Accuracy)))

    print("LinearSVM five-fold cross-validation average Sensitivity:%.4f"%(np.mean(LinearSVM_Sensitivity)))
    print("LightGBM five-fold cross-validation average Sensitivity:%.4f"%(np.mean(LightGBM_Sensitivity)))

    print("LinearSVM five-fold cross-validation average Specificity:%.4f"%(np.mean(LinearSVM_Specificity)))
    print("LightGBM five-fold cross-validation average Specificity:%.4f"%(np.mean(LightGBM_Specificity)))

    fig, axs = plt.subplots(1, 2, figsize=(12, 6)) 


    for i in range(5):
        axs[0].plot(svm_fprs[i], svm_tprs[i], label='ROC fold %d (AUC = %0.2f)' % (i, svm_aucs[i]))

    axs[0].plot([0, 1], [0, 1], 'k--')
    axs[0].set_xlim([0.0, 1.0])
    axs[0].set_ylim([0.0, 1.05])
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].set_title('ROC Curves for 5-fold Cross Validation')
    axs[0].legend(loc="lower right")

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(svm_fprs, svm_tprs)], axis=0)
    mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
    mean_auc = np.mean(svm_aucs, axis=0)
    axs[1].plot(mean_fpr, mean_tpr, label='Mean ROC (AUC = %0.2f)' % mean_auc)
    axs[1].plot([0, 1], [0, 1], 'k--')
    axs[1].set_xlim([0.0, 1.0])
    axs[1].set_ylim([0.0, 1.05])
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].set_title('Mean ROC Curve')
    axs[1].legend(loc="lower right")


    plt.tight_layout()
    plt.savefig(f'../../data/lgbCompareWithsvm/svm_roc.tif', dpi=500, format='tif')
    plt.show()


    dict = {"fpr": mean_fpr, "tpr": mean_tpr, "auc": mean_auc}
    df = pd.DataFrame(dict)
    df.to_csv("../../data/lgbCompareWithsvm/svm_fpr_tpr.csv", index=False)

    print("Average AUC: ", np.mean(svm_aucs))

    
    for i in range(5):
        axs[0].plot(lgb_fprs[i], lgb_tprs[i], label='ROC fold %d (AUC = %0.2f)' % (i, lgb_aucs[i]))

    axs[0].plot([0, 1], [0, 1], 'k--')
    axs[0].set_xlim([0.0, 1.0])
    axs[0].set_ylim([0.0, 1.05])
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].set_title('ROC Curves for 5-fold Cross Validation')
    axs[0].legend(loc="lower right")

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(lgb_fprs, lgb_tprs)], axis=0)
    mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
    mean_auc = np.mean(lgb_aucs, axis=0)
    axs[1].plot(mean_fpr, mean_tpr, label='Mean ROC (AUC = %0.2f)' % mean_auc)
    axs[1].plot([0, 1], [0, 1], 'k--')
    axs[1].set_xlim([0.0, 1.0])
    axs[1].set_ylim([0.0, 1.05])
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].set_title('Mean ROC Curve')
    axs[1].legend(loc="lower right")


    plt.tight_layout()
    plt.savefig(f'../../data/lgbCompareWithsvm/lgb_roc.tif', dpi=500, format='tif')
    plt.show()

  
    dict = {"fpr": mean_fpr, "tpr": mean_tpr, "auc": mean_auc}
    df = pd.DataFrame(dict)
    df.to_csv("../../data/lgbCompareWithsvm/lgb_fpr_tpr.csv", index=False)

    print("Average AUC: ", np.mean(lgb_aucs))

if __name__ == "__main__":

    init()
    run()


