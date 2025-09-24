# %%
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from tqdm import tqdm  # for progress tracking
# Import necessary libraries
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.utils import shuffle
from sklearn.metrics import (
    roc_auc_score, 
    roc_curve, 
    confusion_matrix, 
    classification_report, 
    accuracy_score, 
    precision_score, 
    recall_score,
    ConfusionMatrixDisplay,
    auc,
    f1_score
)
from xgboost import XGBClassifier
import joblib  # For saving the model
import optuna


pd.set_option('display.max_columns', None)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc

BG_FRAC = 0.8

# %%


import xgboost as xgb
# Check if GPU is available
try:
    # Test GPU availability
    test_data = xgb.DMatrix([[1, 2], [3, 4]], label=[0, 1])
    gpu_available = True
    print("GPU is available for XGBoost")
except:
    gpu_available = False
    print("GPU is not available, using CPU")

# %%
THE_CHW_TO_STUDY = 0.1

import src
cfg = src.configs.get_configs()
columns_to_keep = cfg.columns_to_keep
num_of_features = len(columns_to_keep)
print("Num of columns_to_keep", num_of_features)
df = pd.read_parquet(SOME_PATH, 
                     engine='pyarrow', 
                     columns=columns_to_keep)
df_background = pd.read_parquet(SOME_OTHER_PATH,
                                engine='pyarrow',
                                columns=columns_to_keep)
# %%
num_total_sig = 1_000_000
num_test_sig = int(num_total_sig * 0.2)
num_train_sig = num_total_sig - num_test_sig
bg_all = df_background.drop(columns=['cHW'], errors='ignore').to_numpy()
bg_train = bg_all[:2_000_000]
bg_val = bg_all[2_000_000:2_500_000]
bg_test = bg_all[2_500_000:3_000_000]
del bg_all, df_background
gc.collect()
sm_all = df[df['cHW'] == 0].drop(columns=['cHW'], errors='ignore').to_numpy()
bsm_all = df[df['cHW'] == THE_CHW_TO_STUDY].drop(columns=['cHW'], errors='ignore').to_numpy()
sm_train_src, sm_test_src = sm_all[num_test_sig:], sm_all[:num_test_sig]
bsm_train_src, bsm_test_src = bsm_all[num_test_sig:], bsm_all[:num_test_sig]

# %%

if BG_FRAC > 0:
    num_train_bg, num_test_bg = int(num_train_sig * BG_FRAC), int(num_test_sig * BG_FRAC)
    bg_train_src, bg_test_src = bg_train[:2*num_train_bg], bg_test[:2*num_test_bg]
    print(f'bg_train_src shape: {bg_train_src.shape}, bg_test_src shape: {bg_test_src.shape}')
    bsm_train = np.concatenate((bsm_train_src, bg_train_src[len(bg_train_src)//2:]), axis=0)
    bsm_test = np.concatenate((bsm_test_src, bg_test_src[len(bg_test_src)//2:]), axis=0)
    sm_train = np.concatenate((sm_train_src, bg_train_src[:len(bg_train_src)//2]), axis=0)
    sm_test = np.concatenate((sm_test_src, bg_test_src[:len(bg_test_src)//2]), axis=0)
    print(f'bsm_train shape: {bsm_train.shape}, bsm_test shape: {bsm_test.shape}, sm_train shape: {sm_train.shape}, sm_test shape: {sm_test.shape}')
    bsm_train = shuffle(bsm_train, random_state=42)
    bsm_test = shuffle(bsm_test, random_state=42)
    sm_train = shuffle(sm_train, random_state=42)
    sm_test = shuffle(sm_test, random_state=42)
else:
    bsm_train, bsm_test = bsm_train_src, bsm_test_src
    sm_train, sm_test = sm_train_src, sm_test_src
    print(f'bsm_train shape: {bsm_train.shape}, bsm_test shape: {bsm_test.shape}, sm_train shape: {sm_train.shape}, sm_test shape: {sm_test.shape}')


# %%
all_train = np.concatenate((bsm_train, sm_train), axis=0)
all_test = np.concatenate((bsm_test, sm_test), axis=0)
# %%
train_labels = np.concatenate((np.ones((bsm_train.shape[0], 1)), np.zeros((sm_train.shape[0], 1))), axis=0)
test_labels = np.concatenate((np.ones((bsm_test.shape[0], 1)), np.zeros((sm_test.shape[0], 1))), axis=0)
print(f'all_train shape: {all_train.shape}, all_test shape: {all_test.shape}, train_labels shape: {train_labels.shape}, test_labels shape: {test_labels.shape}')
# %%
X_train, y_train = shuffle(all_train, train_labels, random_state=42)
X_test, y_test = shuffle(all_test, test_labels, random_state=42)
del all_train, all_test, train_labels, test_labels, bsm_train, bsm_test, sm_train, sm_test
print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')
del df

# %%

RANDOM_SEED = 42
N_TRIALS = 50
CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 400]),
        'max_depth': trial.suggest_categorical('max_depth', [3, 6, 9, 11, 13]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.001, 0.003, 0.01, 0.03, 0.1]),
        'min_child_weight': trial.suggest_categorical('min_child_weight', [1, 10, 100]),
        'subsample': trial.suggest_categorical('subsample', [0.5, 0.7, 1]),
        'reg_lambda': trial.suggest_categorical('reg_lambda', [0.1, 1, 10]),
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',  
        'device': 'cuda',      
        'random_state': 42,
        'n_jobs': 1
    }

    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train,
                             cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                             scoring='roc_auc',
                             n_jobs=1)
    return scores.mean()

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("Best AUC:", study.best_value)
print("Best params:", study.best_params)


# %% [markdown]
# # optuna
# Best AUC: 0.5084543794366093
# Best params: {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.003, 'min_child_weight': 10, 'subsample': 0.5, 'reg_lambda': 1}

# %%

# Refit final model on full train with best params
# best_params = study.best_params
best_params = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.003, 'min_child_weight': 10, 'subsample': 0.5, 'reg_lambda': 1}
best_params.update({
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'n_jobs': -1,
    'random_state': RANDOM_SEED})

# %%

for i in range(5):
    final_model = XGBClassifier(**best_params)
    final_model.fit(
        X_train, 
        y_train, 
        # eval_set=[(X_test, y_test)],
        verbose=100  # Adjust verbosity as needed
    )

    # 7) Test AUC
    from sklearn.metrics import roc_auc_score
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    print("Test  AUC:", roc_auc_score(y_test, y_pred_proba))


    # final_model.save_model(f'xgb_chw{THE_CHW_TO_STUDY}_bg08_seed{i}.json')

    del final_model, y_pred_proba
    