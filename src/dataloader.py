from __future__ import division
import os, sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import feature_extraction
from sklearn import preprocessing
from random import seed, shuffle
import csv
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from collections import namedtuple


SEED = 1234
seed(SEED)
np.random.seed(SEED)


def add_intercept(x):

    """ Add intercept to the data before linear classification """
    m,n = x.shape
    intercept = np.ones(m).reshape(m, 1) # the constant b
    return np.concatenate((intercept, x), axis = 1)


def get_one_hot_encoding(in_arr):
    """
        input: 1-D arr with int vals -- if not int vals, will raise an error
        output: m (ndarray): one-hot encoded matrix
                d (dict): also returns a dictionary original_val -> column in encoded matrix
    """

    for k in in_arr:
        if str(type(k)) != "<type 'numpy.float64'>" and type(k) != int and type(k) != np.int64:
            print(str(type(k)))
            print("************* ERROR: Input arr does not have integer types")
            return None

    in_arr = np.array(in_arr, dtype=int)
    assert (len(in_arr.shape) == 1)  # no column, means it was a 1-D arr
    attr_vals_uniq_sorted = sorted(list(set(in_arr)))
    num_uniq_vals = len(attr_vals_uniq_sorted)
    if (num_uniq_vals == 2) and (attr_vals_uniq_sorted[0] == 0 and attr_vals_uniq_sorted[1] == 1):
        return in_arr, None

    index_dict = {}  # value to the column number
    for i in range(0, len(attr_vals_uniq_sorted)):
        val = attr_vals_uniq_sorted[i]
        index_dict[val] = i

    out_arr = []
    for i in range(0, len(in_arr)):
        tup = np.zeros(num_uniq_vals)
        val = in_arr[i]
        ind = index_dict[val]
        tup[ind] = 1  # set that value of tuple to 1
        out_arr.append(tup)

    return np.array(out_arr), index_dict

def check_data_file(fname):
    files = os.listdir(".")  # get the current directory listing
    print("Looking for file {} in the current directory...".format(fname))

    if fname not in files:
        print("File not found!")
        exit(0)
    else:
        print("File found in current directory..")

def load_german_data():
    data = []
    with open('german.data-numeric', 'r') as file:
        for row in file:
            data.append([int(x) for x in row.split()])
    data = np.array(data)
    x = data[:, :-1]
    y = data[:, -1] - 1

    z = []
    with open('german.data', 'r') as file:
        for row in file:
            line = [x for x in row.split()]
            if line[8] == 'A92' or line[8] == 'A95':
                z.append(1)
            elif line[8] == 'A91' or line[8] == 'A93' or line[8] == 'A94':
                z.append(0.)
            else:
                print("Wrong gender key!")
                exit(0)
    return x,y, np.array(z)

def load_compas_data():
    FEATURES_CLASSIFICATION = ["age_cat", "race", "sex", "priors_count",
                               "c_charge_degree"]  # features to be used for classification
    CONT_VARIABLES = [
        "priors_count"]  # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = "two_year_recid"  # the decision variable
    SENSITIVE_ATTRS = ["race"]

    COMPAS_INPUT_FILE = "compas-scores-two-years.csv"
    check_data_file(COMPAS_INPUT_FILE)

    # load the data and get some stats
    df = pd.read_csv(COMPAS_INPUT_FILE)
    df = df.dropna(subset=["days_b_screening_arrest"])  # dropping missing vals

    # convert to np array
    data = df.to_dict('list')
    for k in data.keys():
        data[k] = np.array(data[k])

    """ Filtering the data """

    # These filters are the same as propublica (refer to https://github.com/propublica/compas-analysis)
    # If the charge date of a defendants Compas scored crime was not within 30 days from when the person was arrested, we assume that because of data quality reasons, that we do not have the right offense.
    idx = np.logical_and(data["days_b_screening_arrest"] <= 30, data["days_b_screening_arrest"] >= -30)

    # We coded the recidivist flag -- is_recid -- to be -1 if we could not find a compas case at all.
    idx = np.logical_and(idx, data["is_recid"] != -1)

    # In a similar vein, ordinary traffic offenses -- those with a c_charge_degree of 'O' -- will not result in Jail time are removed (only two of them).
    idx = np.logical_and(idx, data["c_charge_degree"] != "O")  # F: felony, M: misconduct

    # We filtered the underlying data from Broward county to include only those rows representing people who had either recidivated in two years, or had at least two years outside of a correctional facility.
    idx = np.logical_and(idx, data["score_text"] != "NA")

    # we will only consider blacks and whites for this analysis
    idx = np.logical_and(idx, np.logical_or(data["race"] == "African-American", data["race"] == "Caucasian"))

    # select the examples that satisfy this criteria
    for k in data.keys():
        data[k] = data[k][idx]

    """ Feature normalization and one hot encoding """

    # convert class label 0 to -1
    y = data[CLASS_FEATURE]
    # y[y == 0] = -1

    print("\nNumber of people recidivating within two years")
    print(pd.Series(y).value_counts())
    print()

    X = np.array([]).reshape(len(y),
                             0)  # empty array with num rows same as num examples, will hstack the features to it
    x_control = defaultdict(list)

    feature_names = []
    for attr in FEATURES_CLASSIFICATION:
        vals = data[attr]
        if attr in CONT_VARIABLES:
            vals = [float(v) for v in vals]
            vals = preprocessing.scale(vals)  # 0 mean and 1 variance
            vals = np.reshape(vals, (len(y), -1))  # convert from 1-d arr to a 2-d arr with one col
        elif attr in SENSITIVE_ATTRS:
            new_val = np.zeros(len(vals))
            for _ in range(len(vals)):
                if vals[_] == 'African-American':
                    new_val[_] = 1.
                elif vals[_] == 'Caucasian':
                    new_val[_] = 0.
                else:
                    print("Wrong race!")
                    exit(0)

            vals = np.reshape(new_val, (len(y), -1))

        else:  # for binary categorical variables, the label binarizer uses just one var instead of two
            lb = preprocessing.LabelBinarizer()
            lb.fit(vals)
            vals = lb.transform(vals)

        # add to sensitive features dict
        if attr in SENSITIVE_ATTRS:
            x_control[attr] = vals
            # continue

        # add to learnable features
        X = np.hstack((X, vals))

        if attr in CONT_VARIABLES:  # continuous feature, just append the name
            feature_names.append(attr)
        else:  # categorical features
            if vals.shape[1] == 1:  # binary features that passed through lib binarizer
                feature_names.append(attr)
            else:
                for k in lb.classes_:  # non-binary categorical features, need to add the names for each cat
                    feature_names.append(attr + "_" + str(k))

    # convert the sensitive feature to 1-d array
    x_control = dict(x_control)
    for k in x_control.keys():
        assert (x_control[k].shape[1] == 1)  # make sure that the sensitive feature is binary after one hot encoding
        x_control[k] = np.array(x_control[k]).flatten()


    """permute the date randomly"""
    perm = list(range(0, X.shape[0]))
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in x_control.keys():
        x_control[k] = x_control[k][perm]

    X = add_intercept(X)

    feature_names = ["intercept"] + feature_names
    assert (len(feature_names) == X.shape[1])
    print("Features we will be using for classification are:", feature_names, "\n")

    print(X.shape, y.shape, len(x_control['race']))
    return X, y, x_control['race']