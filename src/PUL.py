import csv
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from pulearn import ElkanotoPuClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from dataloader import *
from gen_syn import *
import random
random.seed(11223344)
from sklearn.neural_network import MLPClassifier

def main(args):
    # Load data
    if args.dataset == 'synthe':
        x_data, YL, x_control = generate_toy_data(1000, 200, 2)
    elif args.dataset == 'drug':
        x_data, YL, x_control = load_drug_data()
        scaler = MinMaxScaler()
        x_data = scaler.fit_transform(x_data)
    elif args.dataset == 'compas':
        x_data, YL, x_control = load_compas_data()
    elif args.dataset == 'german':
        x_data, YL, x_control = load_german_data()

    g = np.array(x_control == 1.)
    y = np.array(YL == 1)
    print("Positive labels: ")
    print(len(np.where(YL == 1)[0]))
    print("Sensitive ones:")
    print(len(np.where(x_control == 1.)[0]))
    print("Positive sensitive samples: ")
    print(len(np.where(YL[g] == 1)[0]))

    fair_acc, fair_deo, fair_aeo, fair_f1 = [], [], [], []

    for runs in range(args.epochs):
        # Split training and test set
        idx_train = random.sample(range(len(x_data)), int(args.trainrate * len(x_data)))
        idx_test = list(set(range(len(x_data))) - set(idx_train))

        # Upsample for unbalanced datasets
        if args.upsample:
            train_data, train_y = x_data[idx_train, :], YL[idx_train]
            i_class0 = np.where(train_y == 0)[0]
            i_class1 = np.where(train_y == 1)[0]
            s_class0 = len(i_class0)
            print("s_class0: ", s_class0)
            s_class1 = len(i_class1)
            print("s_class1: ", s_class1)
            i_class0_upsampled = np.random.choice(i_class0, size=s_class1, replace=True)
            new_y = np.hstack((train_y[i_class0_upsampled], train_y[i_class1]))
            new_data = np.vstack((train_data[i_class0_upsampled], train_data[i_class1]))
            shuf_idx = np.arange(len(new_data))
            np.random.shuffle(shuf_idx)
            new_data = new_data[shuf_idx, :]
            new_y = new_y[shuf_idx]
            print(new_data.shape, new_y.shape)
        else:
            new_data = x_data[idx_train, :]
            new_y = YL[idx_train]

        # Construct PUL datasets
        pos_ind = np.where(new_y == 1)[0]
        np.random.shuffle(pos_ind)
        pos_sample_len = int(np.ceil(args.labelr * len(pos_ind)))
        print(f'Using {pos_sample_len}/{len(pos_ind)} as positives and unlabeling the rest')
        pos_sample = pos_ind[:pos_sample_len]
        y_labeled = np.zeros(len(new_data))
        y_labeled[pos_sample] = 1.

        g = np.array(x_control[idx_test] == 1.)
        y = np.array(YL[idx_test] == 1)

        # Choose different base models. Please refer to the paper for more tested models and corresponding parameters
        if args.model == 'lr':
            pu_estimator = ElkanotoPuClassifier(LogisticRegression(max_iter=10000), hold_out_ratio=0.2)

        # Step 1 in Alg.1
        pu_estimator.fit(new_data, y_labeled)

        if args.method == 'fairpulaod':
            # Step 2 in Alg.1
            eta = pu_estimator.predict_proba(x_data[idx_test, :])
            EX1s1etaX1 = np.mean(eta[g])  # 1 - it gets for TNR
            EX1s0etaX0 = np.mean(eta[~g])

            PY1S1 = np.mean(eta[g])*np.mean(g)
            PY0S1 = np.mean(eta[~g]) * np.mean(g)
            PY1S0 = np.mean(eta[g])*(1-np.mean(g))
            PY0S0 = np.mean(eta[~g])*(1-np.mean(g))

            # Step 3 in Alg.1
            if not args.bruce:
                # Strategy 1: Simulated Annealing
                T = 1
                Tmin = 0.00000001
                alpha = 0.999
                theta_1 = np.random.uniform(-2, 2)
                theta_2 = np.random.uniform(-2, 2)
                while T >= Tmin:
                        GX1 = ((eta[g] * (1.0 - theta_1 / PY1S1) + (1 - eta[g]) * (1.0 - theta_2 / PY0S1)) >= 0).astype(
                            float)
                        GX0 = (eta[~g] * (1.0 + theta_1 / PY1S0) + (1 - eta[~g]) * (1.0 + theta_2 / PY0S0) >= 0).astype(
                            float)

                        tmp = abs(np.mean(eta[g] * GX1) / EX1s1etaX1 - np.mean(eta[~g] * GX0) / EX1s0etaX0) + abs(
                            np.mean((1 - eta[g]) * (1 - GX1)) / (1 - EX1s1etaX1) - np.mean((1 - eta[~g]) * (1 - GX0)) / (
                                        1 - EX1s0etaX0))

                        theta_1_n = theta_1 + np.random.uniform(-0.01, 0.01)
                        theta_2_n = theta_2 + np.random.uniform(-0.01, 0.01)

                        if -2 <= theta_1_n <= 2 and -2 <= theta_2_n <= 2:
                            GX1 = ((eta[g] * (1.0 - theta_1_n / PY1S1) + (1 - eta[g]) * (1.0 - theta_2_n / PY0S1)) >= 0).astype(
                                float)
                            GX0 = (eta[~g] * (1.0 + theta_1_n / PY1S0) + (1 - eta[~g]) * (1.0 + theta_2_n / PY0S0) >= 0).astype(
                                float)

                            tmp_n = abs(np.mean(eta[g] * GX1) / EX1s1etaX1 - np.mean(eta[~g] * GX0) / EX1s0etaX0) + abs(
                                np.mean((1 - eta[g]) * (1 - GX1)) / (1 - EX1s1etaX1) - np.mean(
                                    (1 - eta[~g]) * (1 - GX0)) / (
                                        1 - EX1s0etaX0))

                            if tmp_n < tmp:
                                theta_1 = theta_1_n
                                theta_2 = theta_2_n
                            else:
                                p = np.exp((tmp-tmp_n) / T)
                                r = random.uniform(0, 1)
                                if p > r:
                                    theta_1 = theta_1_n
                                    theta_2 = theta_2_n
                        T = T*alpha

            else:
                # Strategy 2: Do brute-force search for acceptable search space
                obj = np.inf
                for thetat_1 in np.concatenate((-np.logspace(-2, 2, 100), np.logspace(-2,2,100))):
                    for thetat_2 in np.concatenate((-np.logspace(-2, 2, 100), np.logspace(-2,2,100))):
                        GX1 = ((eta[g] * (1.0 - thetat_1 / PY1S1) + (1 - eta[g]) * (1.0 - thetat_2 / PY0S1)) >=0).astype(float)
                        GX0 = (eta[~g] * (1.0 + thetat_1 / PY1S0) + (1 - eta[~g]) * (1.0 + thetat_2 / PY0S0) >=0).astype(float)

                        tmp = abs(np.mean(eta[g] * GX1)/EX1s1etaX1 - np.mean(eta[~g]* GX0)/EX1s0etaX0) + abs(np.mean((1-eta[g]) * (1-GX1)) / (1-EX1s1etaX1) - np.mean((1-eta[~g]) * (1-GX0)) / (1-EX1s0etaX0))
                        if (obj > tmp):
                            obj = tmp
                            theta_1 = thetat_1
                            theta_2 = thetat_2

            # Step 4 in Alg.1
            yp = []
            for i in range(len(y)):
                if g[i]:
                    yp.append(float((eta[i] * (1.0 - theta_1 / PY1S1) + (1 - eta[i]) * (1.0 - theta_2 / PY0S1)) >= 0))
                else:
                    yp.append(float(eta[i] * (1.0 + theta_1 / PY1S0) + (1 - eta[i]) * (1.0 + theta_2 / PY0S0) >= 0))

            yp = np.array(yp)

            e = (y != yp).astype(float)
            f1 = f1_score(y, yp)
            fair_f1.append(f1)
            fair_acc.append(1 - np.mean(e))
            fair_aeo.append(abs(np.mean(e[g & y]) - np.mean(e[~g & y])))
            fair_deo.append((abs(np.mean(e[g & y]) - np.mean(e[~g & y])) + abs(np.mean(e[g & (~y)]) - np.mean(e[~g & (~y)]))) / 2 )

        elif args.method == 'fairpuleod':
            # Step 2 in Alg.1
            eta = pu_estimator.predict_proba(x_data[idx_test, :])
            EX1s1etaX1 = np.mean(eta[g])
            EX1s0etaX0 = np.mean(eta[~g])
            PY1S1 = np.mean(eta[g]) * np.mean(g)
            PY1S0 = np.mean(eta[g]) * (1 - np.mean(g))

            # Step 3 in Alg.1
            # For FairPUL-EOD, brute-force search is acceptable since search space is not large
            obj = np.inf
            for theta in np.concatenate((-np.logspace(-2, 2, 10000), np.logspace(-2, 2, 10000))):
                tmp = abs(np.mean(eta[g] * (1 <= eta[g] * (2.0 - theta / PY1S1)).astype(float)) / EX1s1etaX1 -
                          np.mean(eta[~g] * (1 <= eta[~g] * (2.0 + theta / PY1S0)).astype(float)) / EX1s0etaX0)
                if (obj > tmp):
                    obj = tmp
                    thetahat = theta

            # Step 4 in Alg.1
            yp = []
            for i in range(len(y)):
                if g[i]:
                    yp.append(float(1 <= eta[i] * (2.0 - thetahat / PY1S1)))
                else:
                    yp.append(float(1 <= eta[i] * (2.0 + thetahat / PY1S0)))

            yp = np.array(yp)

            e = (y != yp).astype(float)
            f1 = f1_score(y, yp)
            fair_f1.append(f1)
            fair_acc.append(1 - np.mean(e))
            fair_aeo.append(abs(np.mean(e[g & y]) - np.mean(e[~g & y])))
            fair_deo.append((abs(np.mean(e[g & y]) - np.mean(e[~g & y])) + abs(np.mean(e[g & (~y)]) - np.mean(e[~g & (~y)]))) / 2 )


    fair_f1 = np.array(fair_f1)
    fair_acc = np.array(fair_acc)
    fair_aeo = np.array(fair_aeo)
    fair_deo = np.array(fair_deo)

    print("F1: {:.3f} {:.3f}".format(np.mean(fair_f1), np.std(fair_f1)))
    print("Acc: {:.3f} {:.3f}".format(np.mean(fair_acc), np.std(fair_acc)))
    print("DEO: {:.3f} {:.3f}".format(np.mean(fair_aeo), np.std(fair_aeo)))
    print("AEO: {:.3f} {:.3f}".format(np.mean(fair_deo), np.std(fair_deo)))

def parsers_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='compas',
                        help='compas, germna, adults, synthe')
    parser.add_argument("--trainrate", help="train test rates", type=float, default=0.7)
    parser.add_argument("--labelr", help="label rates", type=float, default=0.9)
    parser.add_argument("--epochs", help="Number of training epochs", type=int, default=10)
    parser.add_argument('--model', type=str, default='lsvm', help='lsvm,ssvm,psvm')
    parser.add_argument('--method', type=str, default='fairpuleod', help='fairpuleod, fairpulaod, uPU, wPU, bagging')
    parser.add_argument('--c', type=float, default=0.1)
    parser.add_argument('--upsample', type=bool, default=True)
    parser.add_argument('--bruce', type=bool, default=True)
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parsers_parser()
    main(args)
