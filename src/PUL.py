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
# from gen_synthetic import *
import random
random.seed(11223344)
from fairlearn.preprocessing import CorrelationRemover
from sklearn.neural_network import MLPClassifier

# data_loader = generate_synthetic_data()
# # x_data, YL, x_control = generate_synthetic_data()
# x_data, YL, x_control = data_loader.gen()
# data_loader.plot(True)
# x_data, YL, x_control = generate_synthetic_data(2, plot_data=True)
# x_data, YL, x_control = generate_toy_data(1000, 200, 2)
x_data, YL, x_control = load_german_data()
print(x_data.shape)
# remover = CorrelationRemover(sensitive_feature_ids=[8],alpha=1.0)
# x_all = np.hstack((x_data,x_control.reshape(-1,1)))
# remover.fit(X=x_all)
# x_data = remover.transform(x_all)

scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
x_data = scaler.fit_transform(x_data)
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
    # leave just 25% of the positives marked
    pos_sample_len = int(np.ceil(0.9 * len(pos_ind)))
    # pos_sample_len = 1200
    print(f'Using {pos_sample_len}/{len(pos_ind)} as positives and unlabeling the rest')
    pos_sample = pos_ind[:pos_sample_len]
    y_labeled = np.zeros(len(idx_train))
    y_labeled[pos_sample] = 1.

    g = np.array(x_control[idx_test] == 1.)
    y = np.array(YL[idx_test] == 1)

    # clf = svm.SVC(C=10, kernel='linear',probability=True)
    # param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 'scale'],
    #                'C': [1, 10, 100]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    # clf = svm.SVC(gamma='scale', C=1, decision_function_shape='ovr', kernel='rbf', probability=True)
    # clf = svm.SVC(C=0.1, kernel='poly', degree=2, gamma=2, probability=True)
    # clf = GridSearchCV(estimator=svm.SVC(probability=True),
    #                    param_grid=param_grid,
    #                    scoring='%s_macro' % 'precision')
    # clf = RandomForestClassifier(max_depth=2)
    # clf.fit(x_data[idx_train, :], YL[idx_train])
    clf = LogisticRegression(max_iter=1000)
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (8, 16, 2), random_state = 1)
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (24, 48, 2), random_state = 1)

    clf.fit(x_data[idx_train, :], y_labeled)
    YF_R = clf.predict(x_data[idx_test, :])
    e = (YF_R != YL[idx_test]).astype(float)
    print(np.mean(e))
    YF_R = np.where(YF_R < 1., 0., 1.)
    # YF_P = clf.predict_proba(x_data[idx_test, :])
    e = (YF_R != YL[idx_test]).astype(float)
    print(np.mean(e))
    f1 = f1_score(YF_R, YL[idx_test])

    ori_f1.append(f1)
    ori_acc.append(1 - np.mean(e))
    ori_deo.append(abs(np.mean(e[g & y]) - np.mean(e[~g & y])))


    if print_flag:
        print(""); print(clf)
        print("F1 = {}".format(f1))
        acc = 1 - np.mean(e);     print("ACC = {}".format(acc))
        err = np.mean(e[g]);      print("ERR Female = {}".format(err))
        err = np.mean(e[~g]);     print("ERR Male = {}".format(err))
        err = np.mean(e[g & y]);  print("ERR Female >50K = {}".format(err))
        err = np.mean(e[~g & y]); print("ERR Male >50K = {}".format(err))
        err = np.mean(e[g & ~y]);  print("ERR Female  <=50K = {}".format(err))
        err = np.mean(e[~g & ~y]); print("ERR Male  <=50K = {}".format(err))
        print("DEO = {}".format(abs(np.mean(e[g & y]) - np.mean(e[~g & y]))))

    # pu_estimator = WeightedElkanotoPuClassifier(
    #            svm.SVC(C=1, kernel='linear',probability=True), pos_sample_len, len(x_data[idx_train]) - pos_sample_len)
    # pu_estimator = WeightedElkanotoPuClassifier(LogisticRegression(max_iter=1000), pos_sample_len, len(x_data[idx_train]) - pos_sample_len, hold_out_ratio=0.2)
    # pu_estimator = WeightedElkanotoPuClassifier(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (24, 48, 2), random_state = 1), pos_sample_len, len(x_data[idx_train]) - pos_sample_len, hold_out_ratio=0.2)
    # pu_estimator = WeightedElkanotoPuClassifier(svm.SVC(gamma='scale', C=1, kernel='rbf',probability=True), pos_sample_len, len(x_data[idx_train]) - pos_sample_len, hold_out_ratio=0.2)
    # pu_estimator = WeightedElkanotoPuClassifier(svm.SVC(C=0.1, kernel='poly', degree=2, gamma=2, probability=True), pos_sample_len, len(x_data[idx_train]) - pos_sample_len, hold_out_ratio=0.2)
    # pu_estimator = ElkanotoPuClassifier(svm.SVC(gamma='scale', C=1, kernel='rbf',probability=True))
    # pu_estimator = ElkanotoPuClassifier(svm.SVC(C=10, kernel='linear',probability=True), hold_out_ratio=0.2)
    # pu_estimator = ElkanotoPuClassifier(svm.SVC(C=0.1, kernel='poly', degree=2, gamma=2, probability=True), hold_out_ratio=0.2)
    # pu_estimator = ElkanotoPuClassifier(LogisticRegression(max_iter=1000), hold_out_ratio=0.2)
    # pu_estimator = ElkanotoPuClassifier(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (8, 16, 2), random_state = 1), hold_out_ratio=0.2)
    # pu_estimator = BaggingPuClassifier(base_estimator=svm.SVC(C=1, kernel='linear', probability=True), n_estimators=15)
    # pu_estimator = BaggingPuClassifier(base_estimator=svm.SVC(C=0.1, kernel='poly', degree=2, gamma=2, probability=True), n_estimators=15)
    pu_estimator = BaggingPuClassifier(base_estimator=LogisticRegression(max_iter=1000), n_estimators=15)
    pu_estimator.fit(x_data[idx_train, :], y_labeled)
    # y_predict = pu_estimator.predict(L[:, :-1])
    y_predict = pu_estimator.predict(x_data[idx_test, :])
    # y_predict = pu_estimator.predict(x_data[idx_train, :])
    y_predict = np.where(y_predict<1., 0., 1.)
    e = (y_predict != YL[idx_test]).astype(float)
    f1 = f1_score(y_predict, YL[idx_test])
    # e = (y_predict != YL[idx_train]).astype(float)
    # f1 = f1_score(y_predict, YL[idx_train])
    print("");
    print(pu_estimator)
    if print_flag:
        print("F1={}".format(f1))
        acc = 1 - np.mean(e);     print("ACC = {}".format(acc))
        err = np.mean(e[g]);      print("ERR Female = {}".format(err))
        err = np.mean(e[~g]);     print("ERR Male = {}".format(err))
        err = np.mean(e[g & y]);  print("ERR Female >50K = {}".format(err))
        err = np.mean(e[~g & y]); print("ERR Male >50K = {}".format(err))
        err = np.mean(e[g & ~y]);  print("ERR Female  <=50K = {}".format(err))
        err = np.mean(e[~g & ~y]); print("ERR Male  <=50K = {}".format(err))
        print("DEO = {}".format(abs(np.mean(e[g & y]) - np.mean(e[~g & y]))))

    elkan_f1.append(f1)
    elkan_acc.append(1 - np.mean(e))
    elkan_deo.append(abs(np.mean(e[g & y]) - np.mean(e[~g & y])))


    # Apply The proposed Method
    eta = pu_estimator.predict_proba(x_data[idx_test, :])[:,1]
    # eta = clf.predict_proba(x_data[idx_test, :])[:,1]
    # eta = pu_estimator.predict_proba(x_data[idx_train, :])
    EX1s1etaX1 = np.mean(eta[g])
    EX1s0etaX0 = np.mean(eta[~g])
    PY1S1 = np.mean(eta[g])*np.mean(g)
    PY1S0 = np.mean(eta[g])*(1-np.mean(g))
    obj = np.inf

    for theta in np.concatenate((-np.logspace(-2,2,10000), np.logspace(-2,2,10000))):
        tmp = abs(np.mean(eta[g] *(1<=eta[g] *(2.0-theta/PY1S1)).astype(float))/EX1s1etaX1 -
                np.mean(eta[~g]*(1<=eta[~g]*(2.0+theta/PY1S0)).astype(float))/EX1s0etaX0)
        if (obj > tmp):
            obj = tmp
            thetahat = theta
    yp = []
    for i in range(len(y)):
        if g[i]:
            yp.append(float(1<=eta[i]*(2.0-thetahat/PY1S1)))
        else:
            yp.append(float(1<=eta[i]*(2.0+thetahat/PY1S0)))

    yp = np.array(yp)

    # Print Results of Fair RF

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



ori_f1 = np.array(ori_f1)
ori_acc = np.array(ori_acc)
ori_deo = np.array(ori_deo)
elkan_f1 = np.array(elkan_f1)
elkan_acc = np.array(elkan_acc)
elkan_deo = np.array(elkan_deo)
fair_f1 = np.array(fair_f1)
fair_acc = np.array(fair_acc)
fair_deo = np.array(fair_deo)

print("F1: {:.3f} {:.3f}".format(np.mean(ori_f1), np.std(ori_f1)))
print("Acc: {:.3f} {:.3f}".format(np.mean(ori_acc), np.std(ori_acc)))
print("DEO: {:.3f} {:.3f}".format(np.mean(ori_deo), np.std(ori_deo)))
print("F1: {:.3f} {:.3f}".format(np.mean(elkan_f1), np.std(elkan_f1)))
print("Acc: {:.3f} {:.3f}".format(np.mean(elkan_acc), np.std(elkan_acc)))
print("DEO: {:.3f} {:.3f}".format(np.mean(elkan_deo), np.std(elkan_deo)))
print("F1: {:.3f} {:.3f}".format(np.mean(fair_f1), np.std(fair_f1)))
print("Acc: {:.3f} {:.3f}".format(np.mean(fair_acc), np.std(fair_acc)))
print("DEO: {:.3f} {:.3f}".format(np.mean(fair_deo), np.std(fair_deo)))
