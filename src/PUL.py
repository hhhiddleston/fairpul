import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from pulearn import BaggingPuClassifier,  WeightedElkanotoPuClassifier,ElkanotoPuClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from dataloader import *
from gen_syn import *
import random
random.seed(11223344)
from sklearn.neural_network import MLPClassifier

# load data
x_data, YL, x_control = load_german_data()
print(x_data.shape)

# scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
# x_data = scaler.fit_transform(x_data)
g = np.array(x_control == 1.)
y = np.array(YL == 1)
print("Positive labels: ")
print(len(np.where(YL == 1)[0]))
print("Sensitive ones:")
print(len(np.where(x_control == 1.)[0]))
print("Positive sensitive samples: ")
print(len(np.where(YL[g] == 1)[0]))

print_flag = False

ori_acc, ori_deo, ori_f1, elkan_acc, elkan_deo, elkan_f1, fair_acc, fair_deo, fair_f1 = [], [], [], [], [], [], [], [], []
for runs in range(10):
    idx_train = random.sample(range(len(x_data)), int(0.7 * len(x_data)))
    idx_test = list(set(range(len(x_data))) - set(idx_train))

    pos_ind = np.where(YL[idx_train] == 1)[0]
    #shuffle them
    np.random.shuffle(pos_ind)
    # leave 90% of the positives marked
    pos_sample_len = int(np.ceil(0.9 * len(pos_ind)))
    print(f'Using {pos_sample_len}/{len(pos_ind)} as positives and unlabeling the rest')
    pos_sample = pos_ind[:pos_sample_len]
    y_labeled = np.zeros(len(idx_train))
    y_labeled[pos_sample] = 1.

    g = np.array(x_control[idx_test] == 1.)
    y = np.array(YL[idx_test] == 1)

    # Choose different base models
    # pu_estimator = ElkanotoPuClassifier(svm.SVC(gamma='scale', C=1, kernel='rbf',probability=True))
    # pu_estimator = ElkanotoPuClassifier(svm.SVC(C=10, kernel='linear',probability=True), hold_out_ratio=0.2)
    # pu_estimator = ElkanotoPuClassifier(svm.SVC(C=0.1, kernel='poly', degree=2, gamma=2, probability=True), hold_out_ratio=0.2)
    pu_estimator = ElkanotoPuClassifier(LogisticRegression(max_iter=1000), hold_out_ratio=0.2)
    # pu_estimator = ElkanotoPuClassifier(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (8, 16, 2), random_state = 1), hold_out_ratio=0.2)

    pu_estimator.fit(x_data[idx_train, :], y_labeled)
    y_predict = pu_estimator.predict(x_data[idx_test, :])
    y_predict = np.where(y_predict<1., 0., 1.)
    e = (y_predict != YL[idx_test]).astype(float)
    f1 = f1_score(y_predict, YL[idx_test])
    print("");
    print(pu_estimator)

    # Step 2 in Alg.1
    eta = pu_estimator.predict_proba(x_data[idx_test, :])[:,1]
    EX1s1etaX1 = np.mean(eta[g])
    EX1s0etaX0 = np.mean(eta[~g])
    PY1S1 = np.mean(eta[g])*np.mean(g)
    PY1S0 = np.mean(eta[g])*(1-np.mean(g))
    obj = np.inf

    # Step 3 in Alg.1
    for theta in np.concatenate((-np.logspace(-2,2,10000), np.logspace(-2,2,10000))):
        tmp = abs(np.mean(eta[g] *(1<=eta[g] *(2.0-theta/PY1S1)).astype(float))/EX1s1etaX1 -
                np.mean(eta[~g]*(1<=eta[~g]*(2.0+theta/PY1S0)).astype(float))/EX1s0etaX0)
        if (obj > tmp):
            obj = tmp
            thetahat = theta

    # Step 4 in Alg.1
    yp = []
    for i in range(len(y)):
        if g[i]:
            yp.append(float(1<=eta[i]*(2.0-thetahat/PY1S1)))
        else:
            yp.append(float(1<=eta[i]*(2.0+thetahat/PY1S0)))

    yp = np.array(yp)


    e = (y != yp).astype(float)
    f1 = f1_score(y, yp)
    if print_flag:
        print(""); print("Fair")
        print("F1={}".format(f1))
        acc = 1 - np.mean(e);     print("ACC = {}".format(acc))
        err = np.mean(e[g]);      print("ERR Female = {}".format(err))
        err = np.mean(e[~g]);     print("ERR Male = {}".format(err))
        err = np.mean(e[g & y]);  print("ERR Female >50K = {}".format(err))
        err = np.mean(e[~g & y]); print("ERR Male >50K = {}".format(err))
        err = np.mean(e[g & ~y]);  print("ERR Female  <=50K = {}".format(err))
        err = np.mean(e[~g & ~y]); print("ERR Male  <=50K = {}".format(err))
        print("DEO = {}".format(abs(np.mean(e[g & y]) - np.mean(e[~g & y]))))
    fair_f1.append(f1)
    fair_acc.append(1 - np.mean(e))
    fair_deo.append(abs(np.mean(e[g & y]) - np.mean(e[~g & y])))


fair_f1 = np.array(fair_f1)
fair_acc = np.array(fair_acc)
fair_deo = np.array(fair_deo)

print("F1: {:.3f} {:.3f}".format(np.mean(fair_f1), np.std(fair_f1)))
print("Acc: {:.3f} {:.3f}".format(np.mean(fair_acc), np.std(fair_acc)))
print("DEO: {:.3f} {:.3f}".format(np.mean(fair_deo), np.std(fair_deo)))
