# %%

# You can't use dynamic bagging when your bag size is 1.
# That's why this script exists.

from logging import config
import os
import sys
import io
from unicodedata import name
import tensorflow as tf
import tensorflow.keras as k
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbCallback, WandbModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import imageio
from PIL import Image
from omegaconf import OmegaConf
import src
import datetime
import gc
from sklearn.utils import shuffle


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
SWEEP_NAME = f"sweepname_{timestamp}"
THE_CHW_TO_STUDY = 0.1


def train():
    # %%
    cfg = src.configs.get_configs()
    cfg.experiment.save = True
    cfg.experiment.wandb = True
    cfg.experiment.sweep = True
    cfg.experiment.name = EXPERIMENT_NAME
    print(f' Experiment name: {cfg.experiment.name} '.center(80, "="))
    # print(f' Run name: {run.name} '.center(80, "="))
    MONITOR = "val_loss" if cfg.validation_split > 0.0 else "loss"
    # %%
    if cfg.experiment.sweep:
        run = wandb.init()  
        run.name = f"Bag{wandb.config['bag_size']}_Bckgrd{wandb.config['bckgrnd_percent']}_Seed{wandb.config['seed']}"
    cfg.patience = 25
    cfg.validation_split = 0.2
    cfg.lr = 1e-3
    if cfg.experiment.wandb:
        cfg.bag_size = wandb.config["bag_size"]
        if cfg.experiment.db:
            cfg.batch_size = 8*(10_000//cfg.bag_size)
    else:
        if cfg.experiment.db:
            cfg.batch_size = 8*(10_000//cfg.bag_size)
        else:
            cfg.batch_size = 2**13
        cfg.bag_size = 1
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
    chw_grid = np.round(chw_grid, 10) # round to avoid floating point errors
    NUM_CHW_VALS_TO_SCAN = len(chw_grid)
    # %%
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["monitor"] = MONITOR
    if cfg.experiment.wandb:
        wandb.init(project=cfg.experiment.name, config=cfg_dict)
        wandb.init(config=cfg_dict)
    if cfg.experiment.save:
        for path in [cfg.save_pth.model_save, cfg.save_pth.plots, cfg.save_pth.cm, cfg.save_pth.checkpoint]:
            os.makedirs(path, exist_ok=True)
    # %%
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
    num_test_sig = int(num_total_sig * cfg.test_split)
    num_train_sig = num_total_sig - num_test_sig
    bg_all = df_background.drop(columns=['cHW'], errors='ignore').to_numpy()
    bg_train = bg_all[:2_000_000] # keeping it safe
    bg_val = bg_all[2_000_000:2_500_000]
    bg_test = bg_all[2_500_000:3_000_000]
    sm_all = df[df['cHW'] == 0].drop(columns=['cHW'], errors='ignore').to_numpy()
    bsm_all = df[df['cHW'] == THE_CHW_TO_STUDY].drop(columns=['cHW'], errors='ignore').to_numpy() # Adjust BSM cHW if needed
    sm_train_src, sm_test_src = sm_all[num_test_sig:], sm_all[:num_test_sig]
    bsm_train_src, bsm_test_src = bsm_all[num_test_sig:], bsm_all[:num_test_sig]
    del bg_all, df_background, sm_all, bsm_all
    gc.collect()
    # %%
    if not cfg.experiment.wandb:
        BG_FRAC = 0.2
    else: 
        BG_FRAC = wandb.config.get("bckgrnd_percent")
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
    #%%
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
    # %%
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=cfg.validation_split, random_state=42)
    print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, X_val shape: {X_val.shape}, y_val shape: {y_val.shape}')
    del df
    # %%
    batch_size = cfg.batch_size
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    if cfg.experiment.db:
        train_dataset = (
            train_dataset
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))
        val_dataset = (
            val_dataset
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))
    else:
        train_dataset = (
            train_dataset
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))
        val_dataset = (
            val_dataset
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))
    # %%
    def create_binary_clf(data, cfg, dense_units=256, num_of_layers=6, activation='elu', dropout=0.1):
        normalization = k.layers.Normalization()
        normalization.adapt(data)
        if cfg.bag_size == 1: 
            inputs = k.Input(shape=(data.shape[1],))
        else:
            inputs = k.Input(shape=(None, data.shape[2]))
        x = normalization(inputs)
        for _ in range(num_of_layers):
            x = k.layers.Dense(dense_units)(x)
            x = k.layers.BatchNormalization()(x)
            x = k.layers.Activation(activation)(x)
            x = k.layers.Dropout(dropout)(x)
        if cfg.bag_size != 1:
            x = k.layers.GlobalAveragePooling1D()(x)
        outputs = k.layers.Dense(1, activation='sigmoid')(x)
        model = k.Model(inputs=inputs, outputs=outputs)
        return model
    if cfg.experiment.wandb:
        model = create_binary_clf(X_train, cfg, dense_units=wandb.config["dense_units"], num_of_layers=wandb.config["num_layers"])
    else:
        model = create_binary_clf(X_train, cfg)
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
    class LossLoggerCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is not None:
                wandb.log({
                    "ROC_AUC": logs.get("ROC-AUC"),
                    "val_loss": logs.get("val_loss"),
                    "loss": logs.get("loss"),
                    "accuracy": logs.get("accuracy")
                }, step=epoch)
    # %%
    callbacks = []
    if cfg.early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(MONITOR, patience=cfg.patience, restore_best_weights=True))
    if cfg.reduce_lr_on_plateau:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor=MONITOR, patience=int(cfg.patience/2), min_delta=0.0, min_lr=cfg.min_lr))
    if cfg.experiment.wandb:
        callbacks.append(WandbMetricsLogger(log_freq= "epoch"))
        callbacks.append(LossLoggerCallback())
    print(callbacks)
    # %%
    history = model.fit(train_dataset, epochs=cfg.epochs, verbose=2, validation_data=val_dataset, callbacks=callbacks)
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
        "bckgrnd_percent": {"values": [0, 0.2, 0.4, 0.8]},
        "bag_size": {"values": [1]},
        "seed": {"values": [1, 2, 3, 4, 5]},
    }
}

# Create the sweep and obtain the sweep ID
sweep_id = wandb.sweep(sweep_config, project=cfg.experiment.name)

wandb.agent(sweep_id, function=train)

