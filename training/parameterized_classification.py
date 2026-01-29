# %%

# Our previous experiments showed that dynamic bagging doesn't affect the performance of PNNs
# and it takes a lot of time to train. Final training was done according to the following script.

from logging import config
import os
import sys
import io
from unicodedata import name
import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbCallback, WandbModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle
import imageio
from PIL import Image
from omegaconf import OmegaConf
import src
import gc
import datetime
import matplotlib
matplotlib.use('Agg')

pd.set_option('display.max_columns', None)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "True"

#%%

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("GPU:", gpu)
    tf.config.experimental.set_memory_growth(gpu, True)


try:
    EXPERIMENT_NAME = os.path.splitext(os.path.basename(__file__))[0]
except NameError:
    EXPERIMENT_NAME = "placeholder_name"

timestamp = datetime.datetime.now().strftime("%d-%m_%H-%M")
SWEEP_NAME = f"sweep_name_{timestamp}"


def train():
    # %%
    tf.keras.backend.clear_session()
    gc.collect()
    cfg = src.configs.get_configs()
    cfg.experiment.save = True
    cfg.experiment.wandb = True
    cfg.experiment.sweep = True
    cfg.experiment.name = EXPERIMENT_NAME
    MONITOR = "val_loss" if cfg.validation_split > 0.0 else "loss"
    # %%
    if cfg.experiment.sweep:
        run = wandb.init()
        run.name = f"Bag{wandb.config['bag_size']}_Seed{wandb.config['seed']}"
    print(f' Changes made to the default config are: '.center(80, "-"))
    if not cfg.experiment.wandb:
        cfg.epochs = 2
    cfg.patience = 10
    cfg.validation_split = 0.2
    cfg.lr = 1e-3
    if cfg.experiment.wandb:
        cfg.bag_size = wandb.config["bag_size"]
        if cfg.experiment.db:
            cfg.batch_size = 8*(10_000//cfg.bag_size)
    else:
        cfg.bag_size = 1
        if cfg.experiment.db:
            cfg.batch_size = 8*(10_000//cfg.bag_size)
    print(f' - bag_size: {cfg.bag_size}')
    print(f' - batch_size: {cfg.batch_size}')
    print(f' - epochs: {cfg.epochs}')
    print(f' - test_split: {cfg.test_split}')
    print(f' - validation_split: {cfg.validation_split}')
    print(f' - patience: {cfg.patience}')
    print(f' - lr: {cfg.lr}')
    print(f' - save: {cfg.experiment.save}')
    # %%
    ANALYSIS_RANGE = 1
    start = -ANALYSIS_RANGE
    stop = ANALYSIS_RANGE
    step = 0.1
    chw_grid = np.arange(start, stop + step, step)
    chw_grid = np.round(chw_grid, 10) 
    # %%
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["monitor"] = MONITOR
    if cfg.experiment.wandb:
        wandb.config.update(cfg_dict, allow_val_change=True)
    if cfg.experiment.save:
        if cfg.experiment.sweep:
            for path in [cfg.save_pth.sweep_model_save, cfg.save_pth.sweep_CI_plt, 
                         cfg.save_pth.sweep_roc_auc_plt, cfg.save_pth.wandb_artifact_dir]:
                os.makedirs(path, exist_ok=True)
        else:
            for path in [cfg.save_pth.model_save, cfg.save_pth.plots, cfg.save_pth.cm, cfg.save_pth.checkpoint]:
                os.makedirs(path, exist_ok=True)
    # %%
    columns_to_keep = cfg.columns_to_keep
    num_of_features = len(columns_to_keep)
    print("Num of columns_to_keep", num_of_features)
    df = pd.read_parquet(SOME_PATH, 
                         engine='pyarrow', 
                         columns=columns_to_keep)
    sm_data = df[df['cHW'] == 0].copy()
    all_chw_vals = np.sort(df['cHW'].unique())
    num_of_train_samples = int(1_000_000*(1-cfg.test_split))
    num_of_test_samples = int(1_000_000*cfg.test_split)
    train_bag_num = num_of_train_samples//cfg.bag_size
    test_bag_num = num_of_test_samples//cfg.bag_size
    # %%
    sm_test_all_features = sm_data.copy().to_numpy()[num_of_train_samples:].reshape(test_bag_num, cfg.bag_size, num_of_features).copy()
    #
    true_match_train = np.zeros((len(all_chw_vals), train_bag_num, cfg.bag_size, num_of_features))
    true_match_test = np.zeros((len(all_chw_vals), test_bag_num, cfg.bag_size, num_of_features))
    sm_duplicate_train = np.zeros((len(all_chw_vals), train_bag_num, cfg.bag_size, num_of_features))
    sm_duplicate_test = np.zeros((len(all_chw_vals), test_bag_num, cfg.bag_size, num_of_features))
    # 
    true_match_train_labels = np.ones((len(all_chw_vals), train_bag_num, 1))
    true_match_test_labels = np.ones((len(all_chw_vals), test_bag_num, 1))
    sm_duplicate_train_labels = np.zeros((len(all_chw_vals), train_bag_num, 1))
    sm_duplicate_test_labels = np.zeros((len(all_chw_vals), test_bag_num, 1))
    #
    # %%
    for i, chw_val in enumerate(all_chw_vals):    
        temp_true_match = df[df['cHW'] == chw_val].copy().to_numpy()
        temp_sm_train = sm_data.copy().to_numpy()[:num_of_train_samples]
        temp_sm_train[:, -1] = chw_val
        temp_sm_test = sm_data.copy().to_numpy()[num_of_train_samples:]
        temp_sm_test[:, -1] = chw_val
        if chw_val == 0:
            chw_0_index = i
        if cfg.experiment.sweep:
            temp_true_match = shuffle(temp_true_match, random_state=wandb.config["seed"]+i)
            temp_sm_train = shuffle(temp_sm_train, random_state=wandb.config["seed"]+i)
        else:
            temp_true_match = shuffle(temp_true_match, random_state=42+i)
            temp_sm_train = shuffle(temp_sm_train, random_state=42+i)
        true_match_train[i] = temp_true_match[0:num_of_train_samples].reshape(train_bag_num, cfg.bag_size, num_of_features)
        true_match_test[i] = temp_true_match[num_of_train_samples:].reshape(test_bag_num, cfg.bag_size, num_of_features)
        sm_duplicate_train[i] = temp_sm_train.reshape(train_bag_num, cfg.bag_size, num_of_features)
        sm_duplicate_test[i] = temp_sm_test.reshape(test_bag_num, cfg.bag_size, num_of_features)
    print(f'true_match_train shape: {true_match_train.shape}, true_match_test shape: {true_match_test.shape}')
    del df, temp_true_match, temp_sm_train, temp_sm_test
    gc.collect()
    # %%
    temp_chw_n2_data = shuffle(true_match_train[chw_0_index-2].copy().reshape(-1, num_of_features), random_state=84)
    temp_chw_n2_data = temp_chw_n2_data.reshape(train_bag_num, cfg.bag_size, num_of_features)[:int(train_bag_num*0.2),:,:]
    temp_chw_n1_data = shuffle(true_match_train[chw_0_index-1].copy().reshape(-1, num_of_features), random_state=84)
    temp_chw_n1_data = temp_chw_n1_data.reshape(train_bag_num, cfg.bag_size, num_of_features)[:int(train_bag_num*0.3),:,:]
    temp_chw_p1_data = shuffle(true_match_train[chw_0_index+1].copy().reshape(-1, num_of_features), random_state=84)
    temp_chw_p1_data = temp_chw_p1_data.reshape(train_bag_num, cfg.bag_size, num_of_features)[:int(train_bag_num*0.3),:,:]
    temp_chw_p2_data = shuffle(true_match_train[chw_0_index+2].copy().reshape(-1, num_of_features), random_state=84)
    temp_chw_p2_data = temp_chw_p2_data.reshape(train_bag_num, cfg.bag_size, num_of_features)[:int(train_bag_num*0.2),:,:]
    sm_duplicate_train[chw_0_index] = np.concatenate((temp_chw_n2_data, temp_chw_n1_data, temp_chw_p1_data, temp_chw_p2_data), axis=0)
    sm_duplicate_train[chw_0_index][:,:, -1] = 0  # Set cHW to 0 for SM bags
    del temp_chw_p1_data, temp_chw_p2_data
    gc.collect()
    test_temp_chw_n2_data = shuffle(true_match_test[chw_0_index-2].copy().reshape(-1, num_of_features), random_state=84)
    test_temp_chw_n2_data = test_temp_chw_n2_data.reshape(test_bag_num, cfg.bag_size, num_of_features)[:int(test_bag_num*0.2),:,:]
    test_temp_chw_n1_data = shuffle(true_match_test[chw_0_index-1].copy().reshape(-1, num_of_features), random_state=84)
    test_temp_chw_n1_data = test_temp_chw_n1_data.reshape(test_bag_num, cfg.bag_size, num_of_features)[:int(test_bag_num*0.3),:,:]
    test_temp_chw_p1_data = shuffle(true_match_test[chw_0_index+1].copy().reshape(-1, num_of_features), random_state=84)
    test_temp_chw_p1_data = test_temp_chw_p1_data.reshape(test_bag_num, cfg.bag_size, num_of_features)[:int(test_bag_num*0.3),:,:]
    test_temp_chw_p2_data = shuffle(true_match_test[chw_0_index+2].copy().reshape(-1, num_of_features), random_state=84)
    test_temp_chw_p2_data = test_temp_chw_p2_data.reshape(test_bag_num, cfg.bag_size, num_of_features)[:int(test_bag_num*0.2),:,:]
    sm_duplicate_test[chw_0_index] = np.concatenate((test_temp_chw_n2_data, test_temp_chw_n1_data, test_temp_chw_p1_data, test_temp_chw_p2_data), axis=0)
    sm_duplicate_test[chw_0_index][:,:, -1] = 0  # Set cHW to 0 for SM bags
    del test_temp_chw_n2_data, test_temp_chw_n1_data, test_temp_chw_p1_data, test_temp_chw_p2_data
    gc.collect()
    # %%
    true_match_train = true_match_train.reshape(-1, cfg.bag_size, num_of_features)
    true_match_test = true_match_test.reshape(-1, cfg.bag_size, num_of_features)
    sm_duplicate_train = sm_duplicate_train.reshape(-1, cfg.bag_size, num_of_features)
    sm_duplicate_test = sm_duplicate_test.reshape(-1, cfg.bag_size, num_of_features)
    #
    true_match_train_labels = true_match_train_labels.reshape(-1, 1)
    true_match_test_labels = true_match_test_labels.reshape(-1, 1)
    sm_duplicate_train_labels = sm_duplicate_train_labels.reshape(-1, 1)
    sm_duplicate_test_labels = sm_duplicate_test_labels.reshape(-1, 1)
    #
    X_train = np.concatenate((true_match_train, sm_duplicate_train), axis=0)
    X_test = np.concatenate((true_match_test, sm_duplicate_test), axis=0)
    y_train = np.concatenate((true_match_train_labels, sm_duplicate_train_labels), axis=0)
    y_test = np.concatenate((true_match_test_labels, sm_duplicate_test_labels), axis=0)
    print(f'X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}')
    del sm_duplicate_train, sm_duplicate_test, true_match_train, true_match_test, true_match_train_labels, true_match_test_labels 
    del sm_duplicate_train_labels, sm_duplicate_test_labels
    gc.collect()
    # %%
    if cfg.experiment.sweep:
        X_train, y_train = shuffle(X_train, y_train, random_state=wandb.config["seed"])
        X_test, y_test = shuffle(X_test, y_test, random_state=wandb.config["seed"])
    else:
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)
    print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')
    # %%
    if cfg.experiment.sweep:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=cfg.validation_split, random_state=wandb.config["seed"])
    else:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=cfg.validation_split, random_state=42)
    print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, X_val shape: {X_val.shape}, y_val shape: {y_val.shape}')
    # With data generator
    steps_per_epoch = len(X_train) // cfg.batch_size
    validation_steps = len(X_val) // cfg.batch_size
    # %%
    batch_size = cfg.batch_size
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    del X_val, y_train, y_val
    gc.collect()
    # %%
    def create_binary_clf_model(data, cfg, dense_units=256, num_of_layers=6, activation='elu', dropout=0.1):
        normalization = k.layers.Normalization()
        normalization.adapt(data)
        inputs = k.Input(shape=data.shape[1:])
        x = normalization(inputs)
        for _ in range(num_of_layers):
            x = k.layers.Dense(dense_units)(x)
            x = k.layers.BatchNormalization()(x)
            x = k.layers.Activation(activation)(x)
            if dropout > 0:
                x = k.layers.Dropout(dropout)(x)
        if cfg.experiment.db:
            x = k.layers.GlobalAveragePooling1D()(x)
        outputs = k.layers.Dense(1, activation='sigmoid')(x)
        model = k.Model(inputs=inputs, outputs=outputs)
        return model
    if cfg.experiment.wandb:
        model = create_binary_clf_model(X_train, cfg, dense_units=wandb.config["dense_units"], num_of_layers=wandb.config["num_layers"])
    else:
        model = create_binary_clf_model(X_train, cfg, dense_units=64, num_of_layers=3)
    print(model.summary())
    # %%
    optimizer = k.optimizers.AdamW(learning_rate=cfg.lr)
    model.compile(optimizer=optimizer,
                  loss=k.losses.BinaryCrossentropy(),
                  metrics=[k.metrics.AUC(name='ROC-AUC', curve='ROC'),
                           k.metrics.Precision(name='precision'),
                           k.metrics.Recall(name='recall'),
                           k.metrics.F1Score(name='F1'),
                           k.metrics.Accuracy(name='accuracy')])
    # %%
    callbacks = []
    if cfg.early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(MONITOR, patience=cfg.patience, restore_best_weights=True))
    if cfg.reduce_lr_on_plateau:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor=MONITOR, patience=int(cfg.patience/2), min_delta=0.0, min_lr=cfg.min_lr))
    if cfg.experiment.wandb:
        callbacks.append(WandbMetricsLogger(log_freq= "epoch"))
    print(callbacks)
    del X_train
    gc.collect()
    #
    # %%
    history = model.fit(train_dataset, 
                        epochs=cfg.epochs, 
                        verbose=2, 
                        validation_data=val_dataset, 
                        callbacks=callbacks, 
                        steps_per_epoch=steps_per_epoch, 
                        validation_steps=validation_steps)
    del train_dataset, val_dataset
    gc.collect()
    # %%
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    loss_fig = plt.gcf()
    if cfg.experiment.wandb:
        wandb.log({"training_validation_loss": wandb.Image(loss_fig)})
    plt.show()
    plt.clf()
    plt.close()
    # %%
    if cfg.experiment.save:
        if cfg.experiment.sweep:
            save_path = f"{cfg.save_pth.sweep_model_save}/{SWEEP_NAME}"
            os.makedirs(save_path, exist_ok=True)
        else:
            save_path = cfg.save_pth.model_save
        model.save(f"{save_path}/{run.name}.keras")
        print(f"Model saved at {save_path}")
        if cfg.experiment.wandb:
            wandb.save(save_path)
    if cfg.experiment.wandb:
        wandb.finish()

    k.backend.clear_session()
    gc.collect()


# %%

cfg = src.configs.get_configs()
cfg.experiment.name = EXPERIMENT_NAME
print(f' Experiment name: {cfg.experiment.name} '.center(80, "="))
MONITOR = "val_loss" if cfg.validation_split > 0.0 else "loss"


cfg_dict = OmegaConf.to_container(cfg, resolve=True)
cfg_dict["monitor"] = MONITOR


sweep_config = {
    "method": "grid",  # Using grid search to get all combinations sequentially
    "name": SWEEP_NAME, 
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "num_layers": {"values": [3]},
        "dense_units": {"values": [64]},
        "events_tested": {"values": [1000]},
        # "bag_size": {"values": [250, 125, 25, 20, 10]},
        "bag_size": {"values": [1, 100, 50, 200]},
        "seed": {"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
    }
}

sweep_id = wandb.sweep(sweep_config, project=cfg.experiment.name)

wandb.agent(sweep_id, function=train)

