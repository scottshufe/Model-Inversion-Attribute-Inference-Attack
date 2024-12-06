from csmia import run_csmia
from utils_maia import load_data_adult, cal_score
from train_target_model import Model

import torch
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    df, df_encoded, categorical_label_encoders, categorical_onehot_encoders = load_data_adult()
    model = Model(n_inputs=df_encoded.shape[1]-1)
    PATH = "tmp_model/target_model_adult_dnn.pt"
    model.load_state_dict(torch.load(PATH, weights_only=True))
    predicted_sensitive_vals, case_of_pred = run_csmia(df, df_encoded, 'marital', categorical_label_encoders,
                                                      categorical_onehot_encoders, 'income', model)
    cal_score(df, df_encoded, 'marital', predicted_sensitive_vals)
