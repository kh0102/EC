from collections import Counter

import joblib
import numpy as np
import optuna
import os
import sklearn
import xgboost as xgboost
from imblearn.over_sampling import BorderlineSMOTE
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PowerTransformer

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

    RF_Accuracy = []
    RF_Specificity = []
    RF_Sensitivity = []

    xgb_Accuracy = []
    xgb_Specificity = []
    xgb_Sensitivity = []

    cnt = 1
    kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=131)

    for train_index, test_index in kfold.split(data, label):

        train_x, test_x = data[train_index], data[test_index]
        train_y, test_y = label[train_index], label[test_index]



        T_train=[]
        for i in importantLightgbmFeat:
            T_train.append(train_x[:,i])
        T_train=np.asarray(T_train)
        T_train=T_train.reshape(T_train.shape[1], T_train.shape[2])


        T_test=[]
        for i in importantLightgbmFeat:
            T_test.append(test_x[:,i])
        T_test = np.asarray(T_test)
        T_test = T_test.reshape(T_test.shape[1], T_test.shape[2])

        # svm = joblib.load(str("linearSvm"+str(cnt)+".m"))
        # gbm = joblib.load(str("lightgbm"+str(cnt)+".pkl"))

        """
        XGBoost tuning parameters
        """

        def objective(trial):
            param = {
                'tree_method': 'gpu_hist',
                # this parameter means using the GPU when training our model to speedup the training process
                'lambda': trial.suggest_float('lambda', 1e-3, 10.0),
                'alpha': trial.suggest_float('alpha', 1e-3, 10.0),
                'colsample_bytree': trial.suggest_categorical('colsample_bytree',
                                                              [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
                'learning_rate': trial.suggest_categorical('learning_rate',
                                                           [0.008, 0.009, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
                'n_estimators': trial.suggest_int("n_estimators", 10, 200),
                'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17, 20]),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
            }
            # Network attribute features use max_iter = 1e7, all others are 1e6
            xgb = xgboost.XGBClassifier(**param)

            kfold = StratifiedKFold(n_splits=5)
            X = T_train
            Y = train_y

            scores = []
            for train_index, valid_index in kfold.split(X, Y):
                x_train, x_valid = X[train_index], X[valid_index]
                y_train, y_valid = Y[train_index], Y[valid_index]

                smo = BorderlineSMOTE(kind='borderline-2', random_state=131)  # kind='borderline-2'
                x_train, y_train = smo.fit_resample(x_train, y_train)

                xgb.fit(x_train, y_train)
                y_pred = xgb.predict(x_valid)
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

        xgb = xgboost.XGBClassifier(**study.best_params)


        smo = BorderlineSMOTE(kind='borderline-2', random_state=131)

        T_traincopy = T_train
        train_ycopy = train_y

        T_train, train_y = smo.fit_resample(T_train, train_y)
        print(sorted(Counter(train_y).items()))
        xgb.fit(T_train, train_y)
        joblib.dump(xgb, "../visualizationAnalysis/xgb" + str(cnt) + ".pkl")

        y_pred = xgb.predict(T_test)
        pred_labels = np.rint(y_pred)
        accuracy = sklearn.metrics.accuracy_score(test_y, pred_labels)
        print("XGBoost testing set accuracy:%.2f" % accuracy)
        dictScore = classification_report(test_y, y_pred, output_dict=True)
        print(classification_report(test_y, y_pred))
        xgb_Accuracy.append(dictScore['weighted avg']['recall'])
        xgb_Sensitivity.append(dictScore['1']['recall'])
        xgb_Specificity.append(dictScore['0']['recall'])

        T_train = T_traincopy
        train_y = train_ycopy

        """
        RF tuning parameters
        """
        def objective(trial):
            param = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 200),
                "max_depth": trial.suggest_categorical("max_depth", [5,7,9,11,13,15,17,20]),
                "min_samples_split": trial.suggest_categorical('min_samples_split', [2, 5, 10]),
                "min_samples_leaf": trial.suggest_categorical('min_samples_leaf', [1, 2, 4]),
                "max_features": trial.suggest_categorical('max_features', ['auto', 'sqrt']),
            }
            # Network attribute features use max_iter = 1e7, all others are 1e6
            RF = RandomForestClassifier(n_estimators=param["n_estimators"],
                                        max_depth=param["max_depth"],
                                        min_samples_split=param["min_samples_split"],
                                        min_samples_leaf=param["min_samples_leaf"],
                                        max_features=param["max_features"])

            kfold = StratifiedKFold(n_splits=5)
            X=T_train
            Y=train_y

            scores = []
            for train_index, valid_index in kfold.split(X, Y):
                x_train, x_valid = X[train_index], X[valid_index]
                y_train, y_valid = Y[train_index], Y[valid_index]

                smo = BorderlineSMOTE(kind='borderline-2', random_state=131)  # kind='borderline-2'
                x_train, y_train = smo.fit_resample(x_train, y_train)

                RF.fit(x_train, y_train)
                y_pred = RF.predict(x_valid)
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

        RF = RandomForestClassifier(n_estimators=study.best_params["n_estimators"],
                                    max_depth=study.best_params["max_depth"],
                                    min_samples_split=study.best_params["min_samples_split"],
                                    min_samples_leaf=study.best_params["min_samples_leaf"],
                                    max_features=study.best_params["max_features"])


        smo = BorderlineSMOTE(kind='borderline-2', random_state=131)



        T_train, train_y = smo.fit_resample(T_train, train_y)
        print(sorted(Counter(train_y).items()))
        RF.fit(T_train, train_y)
        joblib.dump(RF, "../visualizationAnalysis/RF" + str(cnt) + ".pkl")

        y_pred = RF.predict(T_test)
        pred_labels = np.rint(y_pred)
        accuracy = sklearn.metrics.accuracy_score(test_y, pred_labels)
        print("RandomForest testing set accuracy:%.2f" % accuracy)
        dictScore = classification_report(test_y, y_pred, output_dict=True)
        print(classification_report(test_y,y_pred))
        RF_Accuracy.append(dictScore['weighted avg']['recall'])
        RF_Sensitivity.append(dictScore['1']['recall'])
        RF_Specificity.append(dictScore['0']['recall'])

        cnt += 1

    print("Current running file path" + os.getcwd())
    print("xgb five-fold cross-validation average Accuracy:%.4f" % (np.mean(xgb_Accuracy)))
    print("RF five-fold cross-validation average Accuracy:%.4f" % (np.mean(RF_Accuracy)))

    print("xgb five-fold cross-validation average Sensitivity:%.4f" % (np.mean(xgb_Sensitivity)))
    print("RF five-fold cross-validation average Sensitivity:%.4f" % (np.mean(RF_Sensitivity)))

    print("xgb five-fold cross-validation average Specificity:%.4f" % (np.mean(xgb_Specificity)))
    print("RF five-fold cross-validation average Specificity:%.4f" % (np.mean(RF_Specificity)))



if __name__ == "__main__":

    init()
    run()