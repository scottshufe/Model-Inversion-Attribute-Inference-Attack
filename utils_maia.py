import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn import preprocessing
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score
from tabulate import tabulate


def load_data_adult():
    df = pd.read_csv('data/Adult_35222.csv')
    df_copy = df.copy()

    target_col = 'income'
    numerical_cols = ['fnlwgt', 'capitalgain', 'capitalloss', 'hoursperweek']
    categorical_cols = ['work', 'education', 'marital', 'occupation', 'sex', 'race']

    categorical_label_encoders = {
        'work': preprocessing.LabelEncoder(),
        'education': preprocessing.LabelEncoder(),
        'marital': preprocessing.LabelEncoder(),
        'occupation': preprocessing.LabelEncoder(),
        'sex': preprocessing.LabelEncoder(),
        'race': preprocessing.LabelEncoder()
    }

    categorical_onehot_encoders = {
        'work': preprocessing.OneHotEncoder(drop='first'),
        'education': preprocessing.OneHotEncoder(drop='first'),
        'marital': preprocessing.OneHotEncoder(drop='first'),
        'occupation': preprocessing.OneHotEncoder(drop='first'),
        'sex': preprocessing.OneHotEncoder(drop='first'),
        'race': preprocessing.OneHotEncoder(drop='first')
    }

    nss1 = preprocessing.StandardScaler()
    nss2 = preprocessing.StandardScaler()
    nss3 = preprocessing.StandardScaler()
    nss4 = preprocessing.StandardScaler()

    le_y = preprocessing.LabelEncoder()

    df_copy.loc[:, 'fnlwgt'] = nss1.fit_transform(df[['fnlwgt']])
    df_copy.loc[:, 'capitalgain'] = nss2.fit_transform(df[['capitalgain']])
    df_copy.loc[:, 'capitalloss'] = nss3.fit_transform(df[['capitalloss']])
    df_copy.loc[:, 'hoursperweek'] = nss4.fit_transform(df[['hoursperweek']])

    encoded_data_dict = {}
    for cat_col in categorical_cols:
        df_copy[cat_col] = categorical_label_encoders[cat_col].fit_transform(df[cat_col])
        encoded_col = categorical_onehot_encoders[cat_col].fit_transform(df_copy[[cat_col]]).toarray()
        _arr = pd.DataFrame(encoded_col, columns=categorical_onehot_encoders[cat_col].get_feature_names_out())
        encoded_data_dict[cat_col] = _arr

    df_encoded = df_copy.drop(categorical_cols, axis=1)

    for cat_col in categorical_cols:
        df_encoded = pd.concat([df_encoded, encoded_data_dict[cat_col]], axis=1)

    df_encoded = df_encoded.drop(['income'], axis=1)
    df_encoded['income'] = le_y.fit_transform(df_copy[['income']])

    return df, df_encoded, categorical_label_encoders, categorical_onehot_encoders


def construct_querying_data(df, df_encoded, sensitive_attr, sensitive_val, categorical_label_encoders,
                            categorical_onehot_encoders):
    df_copy = df.copy()
    n = len(df_copy)
    df_copy[sensitive_attr] = np.concatenate([np.repeat(sensitive_val, n)])
    df_copy[sensitive_attr] = categorical_label_encoders[sensitive_attr].transform(df_copy[sensitive_attr])
    encoded_col = categorical_onehot_encoders[sensitive_attr].transform(df_copy[[sensitive_attr]]).toarray()
    encoded_df = pd.DataFrame(encoded_col, columns=categorical_onehot_encoders[sensitive_attr].get_feature_names_out())

    df_query = df_encoded.copy()

    for col in encoded_df.columns:
        df_query[col] = encoded_df[col]

    return df_query


def cal_score(df, df_encoded, sensitive_attr, pred_sensitive_vals):
    unique_sensitive_vals = df[sensitive_attr].unique()
    gt_sensitive_vals = df[sensitive_attr].to_numpy()
    print(
        f'\n\n\nOverall report for inferring sensitive attribute {sensitive_attr} of dataset Adult and target model '
        f'type DNN when performing CSMIA\n')
    print(get_all_scores(gt_sensitive_vals, pred_sensitive_vals, labels=unique_sensitive_vals))


def get_all_scores(actual, pred, labels):
    ((tp,fn),(fp,tn))= confusion_matrix(actual, pred, labels=labels)
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    fpr = fp/(fp+tn)

    acc = (tp+tn)/(tp+fp+fn+tn)
    gmean = geometric_mean_score(actual, pred, labels=labels)
    mcc = matthews_corrcoef(actual, pred)
    # f1 = f1_score(actual, pred, labels=labels)
    f1 = 2* precision * recall /(precision + recall)

    all_scores = [tp, tn, fp, fn, precision, recall, acc, f1, gmean, mcc, fpr]
    # print(all_scores)
    all_scores = [all_scores]
    all_scores_header = ['TP', 'TN', 'FP', 'FN', 'Precision', 'Recall', 'Accuracy', 'F1-score', 'G-mean', 'MCC', 'FPR']
    return tabulate(all_scores, headers=all_scores_header)
