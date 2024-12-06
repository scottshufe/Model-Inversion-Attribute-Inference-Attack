import random

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

from utils_maia import construct_querying_data


def run_csmia(df, df_encoded, sensitive_attr, categorical_label_encoders, categorical_onehot_encoders, y_attr, model):
    model.eval()

    adv_queries_dict_by_sensitive_val = {}

    unique_sensitive_vals = df[sensitive_attr].unique()
    unique_y_vals = df_encoded[y_attr].unique()

    # Construct querying data with each unique sensitive value
    for sensitive_val in unique_sensitive_vals:
        adv_query = construct_querying_data(df, df_encoded, sensitive_attr, sensitive_val, categorical_label_encoders,
                                            categorical_onehot_encoders)
        adv_queries_dict_by_sensitive_val[sensitive_val] = adv_query

    list_of_prediction_count_for_sensitive_val = {}
    list_of_confidence_sum_for_sensitive_val = {}
    # Query with each constructed querying data
    for sensitive_val in unique_sensitive_vals:
        list_of_prediction_count_for_sensitive_val[sensitive_val] = []
        list_of_confidence_sum_for_sensitive_val[sensitive_val] = []
        adv_query = adv_queries_dict_by_sensitive_val[sensitive_val]
        X_query = adv_query.copy().drop([y_attr], axis=1)
        y_query = adv_query[[y_attr]]

        print(f"Querying with sensitive value setting to {sensitive_val}.")
        for i in tqdm(range(X_query.shape[0])):
            prediction_count_for_yval = {}
            confidence_sum_yval = {}
            for yval in unique_y_vals:
                prediction_count_for_yval[yval] = 0
                confidence_sum_yval[yval] = 0.

            input_data = X_query.iloc[i].copy()
            input_data = torch.Tensor(input_data.values)

            # if there are missing nsa, then it is partial knowledge CSMIA
            # if len(self.dataset.missing_nonsensitive_attributes) != 0:
            #     for nsa in self.dataset.missing_nonsensitive_attributes:
            #         nsa_vals = self.dataset.missing_nsa_vals[nsa]
            #
            #         for nsa_val in nsa_vals:
            #             input_data[nsa] = nsa_val
            #             prediction = self.target_model.model.predict(input_data, full=True)
            #             predicted_yval = prediction['prediction']
            #
            #             prediction_count_for_yval[predicted_yval] += 1
            #             if self.target_model.model_type == 'DNN':
            #                 confidence_sum_yval[predicted_yval] += prediction['probability']
            #             else:
            #                 confidence_sum_yval[predicted_yval] += prediction['confidence']

            output = model(input_data).detach().cpu()
            predicted_yval = torch.argmax(output.view(-1, 2), dim=1).numpy()[0]
            prediction_count_for_yval[predicted_yval] += 1
            predicted_conf = F.softmax(output.view(-1, 2), dim=1).numpy()[0][predicted_yval]
            confidence_sum_yval[predicted_yval] += predicted_conf
            # 注意一下这里的confidence是所有的还是单个

            list_of_prediction_count_for_sensitive_val[sensitive_val].append(prediction_count_for_yval)
            list_of_confidence_sum_for_sensitive_val[sensitive_val].append(confidence_sum_yval)

    # Match with ground truth y
    gt_yvals = df_encoded[y_attr]
    predicted_sensitive_vals = []
    case_of_pred = []

    for i in range(df_encoded.shape[0]):
        prediction_match = []
        confidence_score_for_correct_y = []
        total_confidence_scores = []

        for sensitive_val in unique_sensitive_vals:
            prediction_match.append(list_of_prediction_count_for_sensitive_val[sensitive_val][i][gt_yvals[i]])
            confidence_score_for_correct_y.append(list_of_confidence_sum_for_sensitive_val[sensitive_val][i]
                                                  [gt_yvals[i]])
            total_confidence_scores.append(sum([list_of_confidence_sum_for_sensitive_val[sensitive_val][i][y_val]
                                                for y_val in unique_y_vals]))

        prediction_match = np.array(prediction_match)
        num_of_pred_match = len(np.argwhere(prediction_match > 0))

        if num_of_pred_match == 0:
            case = 3
            mins = np.argwhere(total_confidence_scores == np.min(total_confidence_scores))
            index = random.choice(mins)[0]
        elif num_of_pred_match == 1:
            case = 1
            index = np.argwhere(prediction_match > 0)[0][0]
        else:
            case = 2
            confidence_score_for_correct_y = [confidence_score_for_correct_y[i]*(prediction_match[i] > 0) for i
                                              in range(len(confidence_score_for_correct_y))]
            maxes = np.argwhere(confidence_score_for_correct_y == np.max(confidence_score_for_correct_y))
            index = random.choice(maxes)[0]

        # final predicted sensitive value
        predicted_sensitive_vals.append(unique_sensitive_vals[index])
        # case num. of each attacked record
        case_of_pred.append(case)

    return predicted_sensitive_vals, case_of_pred


def run_csmia_coreg(df, df_encoded, sensitive_attr, categorical_label_encoders, categorical_onehot_encoders, y_attr,
                    models):
    for i in range(len(models)):
        models[i].eval()

    adv_queries_dict_by_sensitive_val = {}

    unique_sensitive_vals = df[sensitive_attr].unique()
    unique_y_vals = df_encoded[y_attr].unique()

    # Construct querying data with each unique sensitive value
    for sensitive_val in unique_sensitive_vals:
        adv_query = construct_querying_data(df, df_encoded, sensitive_attr, sensitive_val, categorical_label_encoders,
                                            categorical_onehot_encoders)
        adv_queries_dict_by_sensitive_val[sensitive_val] = adv_query

    list_of_prediction_count_for_sensitive_val = {}
    list_of_confidence_sum_for_sensitive_val = {}
    # Query with each constructed querying data
    for sensitive_val in unique_sensitive_vals:
        list_of_prediction_count_for_sensitive_val[sensitive_val] = []
        list_of_confidence_sum_for_sensitive_val[sensitive_val] = []
        adv_query = adv_queries_dict_by_sensitive_val[sensitive_val]
        X_query = adv_query.copy().drop([y_attr], axis=1)
        y_query = adv_query[[y_attr]]

        print(f"Querying with sensitive value setting to {sensitive_val}.")
        for i in tqdm(range(X_query.shape[0])):
            prediction_count_for_yval = {}
            confidence_sum_yval = {}
            for yval in unique_y_vals:
                prediction_count_for_yval[yval] = 0
                confidence_sum_yval[yval] = 0.

            input_data = X_query.iloc[i].copy()
            input_data = torch.Tensor(input_data.values)

            # if there are missing nsa, then it is partial knowledge CSMIA
            # if len(self.dataset.missing_nonsensitive_attributes) != 0:
            #     for nsa in self.dataset.missing_nonsensitive_attributes:
            #         nsa_vals = self.dataset.missing_nsa_vals[nsa]
            #
            #         for nsa_val in nsa_vals:
            #             input_data[nsa] = nsa_val
            #             prediction = self.target_model.model.predict(input_data, full=True)
            #             predicted_yval = prediction['prediction']
            #
            #             prediction_count_for_yval[predicted_yval] += 1
            #             if self.target_model.model_type == 'DNN':
            #                 confidence_sum_yval[predicted_yval] += prediction['probability']
            #             else:
            #                 confidence_sum_yval[predicted_yval] += prediction['confidence']
            output = None
            for i in range(len(models)):
                if output is None:
                    output = models[i](input_data).detach().cpu()
                else:
                    output = output + models[i](input_data).detach().cpu()
            output = output / len(models)

            # output = model(input_data).detach().cpu()
            predicted_yval = torch.argmax(output.view(-1, 2), dim=1).numpy()[0]
            prediction_count_for_yval[predicted_yval] += 1
            predicted_conf = F.softmax(output.view(-1, 2), dim=1).numpy()[0][predicted_yval]
            confidence_sum_yval[predicted_yval] += predicted_conf
            # 注意一下这里的confidence是所有的还是单个

            list_of_prediction_count_for_sensitive_val[sensitive_val].append(prediction_count_for_yval)
            list_of_confidence_sum_for_sensitive_val[sensitive_val].append(confidence_sum_yval)

    # Match with ground truth y
    gt_yvals = df_encoded[y_attr]
    predicted_sensitive_vals = []
    case_of_pred = []

    for i in range(df_encoded.shape[0]):
        prediction_match = []
        confidence_score_for_correct_y = []
        total_confidence_scores = []

        for sensitive_val in unique_sensitive_vals:
            prediction_match.append(list_of_prediction_count_for_sensitive_val[sensitive_val][i][gt_yvals[i]])
            confidence_score_for_correct_y.append(list_of_confidence_sum_for_sensitive_val[sensitive_val][i]
                                                  [gt_yvals[i]])
            total_confidence_scores.append(sum([list_of_confidence_sum_for_sensitive_val[sensitive_val][i][y_val]
                                                for y_val in unique_y_vals]))

        prediction_match = np.array(prediction_match)
        num_of_pred_match = len(np.argwhere(prediction_match > 0))

        if num_of_pred_match == 0:
            case = 3
            mins = np.argwhere(total_confidence_scores == np.min(total_confidence_scores))
            index = random.choice(mins)[0]
        elif num_of_pred_match == 1:
            case = 1
            index = np.argwhere(prediction_match > 0)[0][0]
        else:
            case = 2
            confidence_score_for_correct_y = [confidence_score_for_correct_y[i]*(prediction_match[i] > 0) for i
                                              in range(len(confidence_score_for_correct_y))]
            maxes = np.argwhere(confidence_score_for_correct_y == np.max(confidence_score_for_correct_y))
            index = random.choice(maxes)[0]

        # final predicted sensitive value
        predicted_sensitive_vals.append(unique_sensitive_vals[index])
        # case num. of each attacked record
        case_of_pred.append(case)

    return predicted_sensitive_vals, case_of_pred
