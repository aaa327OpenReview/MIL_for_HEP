# %%

# Dynamic bagging also didn't help much for binary classification, but, this was the script I used

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
        cfg.bag_size = 1000
    print(f' - bag_size: {cfg.bag_size}')
    print(f' - batch_size: {cfg.batch_size}')
    print(f' - epochs: {cfg.epochs}')
    print(f' - test_split: {cfg.test_split}')
    print(f' - validation_split: {cfg.validation_split}')
    print(f' - patience: {cfg.patience}')
    print(f' - lr: {cfg.lr}')
    print(f' - save: {cfg.experiment.save}')
    # =============================================================================
    # =============================================================================
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
    DYNAMIC_BAGGING = bool(getattr(cfg.experiment, "dynamic_bagging", True))
    print(f"Dynamic bagging: {DYNAMIC_BAGGING}")
    df = pd.read_parquet(SOME_PATH, 
                         engine='pyarrow', 
                         columns=columns_to_keep)
    df_background = pd.read_parquet(SOME_OTHER_PATH,
                                    engine='pyarrow',
                                    columns=columns_to_keep)
    # %%
    if not cfg.experiment.wandb:
        BG_FRAC = 0.0
    else:
        BG_FRAC = wandb.config.get("bckgrnd_percent")
    sm_all = df[df['cHW'] == 0].drop(columns=['cHW'], errors='ignore').to_numpy(dtype=np.float32)
    bsm_all = df[df['cHW'] == THE_CHW_TO_STUDY].drop(columns=['cHW'], errors='ignore').to_numpy(dtype=np.float32)
    bg_all = df_background.drop(columns=['cHW'], errors='ignore').to_numpy(dtype=np.float32)
    bg_train = bg_all[:2_000_000]
    bg_val = bg_all[2_000_000:2_500_000]
    bg_test = bg_all[2_500_000:3_000_000]
    del bg_all, df_background

    print(f"Instances: SM={len(sm_all):,}, BSM(cHW={THE_CHW_TO_STUDY})={len(bsm_all):,}, BG={len(bg_train):,}")
    num_total_sig = 1_000_000
    num_test_sig = int(num_total_sig * cfg.test_split)
    num_train_sig = num_total_sig - num_test_sig

    sm_train_val, sm_test = sm_all[num_test_sig:], sm_all[:num_test_sig]
    bsm_train_val, bsm_test = bsm_all[num_test_sig:], bsm_all[:num_test_sig]
    sm_train, sm_val = train_test_split(sm_train_val, test_size=cfg.validation_split, random_state=42, shuffle=True)
    bsm_train, bsm_val = train_test_split(bsm_train_val, test_size=cfg.validation_split, random_state=42, shuffle=True)
    del df, sm_train_val, bsm_train_val
    gc.collect()
    # %%
    def _bags_for_class(instances, label_value, bag_sig, bg_frac, bg_pool,
                        shuffle_data, drop_remainder=True, buffer_cap=20000):
        num_instances = instances.shape[0]
        # Signal instance dataset
        sig_ds = tf.data.Dataset.from_tensor_slices(instances)
        if shuffle_data:
            sig_ds = sig_ds.shuffle(buffer_size=min(num_instances, buffer_cap), reshuffle_each_iteration=True)
        sig_bags = sig_ds.batch(bag_sig, drop_remainder=drop_remainder)
        #
        bg_cnt = int(round(bag_sig * bg_frac))
        if bg_cnt > 0:
            bg_ds = tf.data.Dataset.from_tensor_slices(bg_pool)
            if shuffle_data:
                bg_ds = bg_ds.shuffle(buffer_size=min(len(bg_pool), buffer_cap), reshuffle_each_iteration=True)
            n_sig_bags = instances.shape[0] // bag_sig
            bg_bags = bg_ds.batch(bg_cnt, drop_remainder=True)
            bag_ds = tf.data.Dataset.zip((sig_bags, bg_bags)).map(
                lambda s, b: tf.concat([s, b], axis=0),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            bag_ds = sig_bags
        label = tf.constant([float(label_value)], dtype=tf.float32)
        bag_ds = bag_ds.map(lambda bag: (bag, label), num_parallel_calls=tf.data.AUTOTUNE)
        num_bags = instances.shape[0] // bag_sig
        return bag_ds, num_bags
    #
    def create_dynamic_binary_dataset(sm_split, bsm_split, bg_pool, bag_size, bg_frac, batch_size,
                                      shuffle_data=True, drop_remainder=True, cache=False):
        bg_sm, bg_bsm = bg_pool[:len(bg_pool)//2], bg_pool[len(bg_pool)//2:]
        sm_ds, sm_n = _bags_for_class(sm_split, 0.0, bag_size, bg_frac, bg_sm, shuffle_data, drop_remainder)
        bsm_ds, bsm_n = _bags_for_class(bsm_split, 1.0, bag_size, bg_frac, bg_bsm, shuffle_data, drop_remainder)
        #
        full_bag_ds = sm_ds.concatenate(bsm_ds)
        total_bags = sm_n + bsm_n
        #
        if shuffle_data:
            full_bag_ds = full_bag_ds.shuffle(buffer_size=total_bags, reshuffle_each_iteration=True)
        if cache:
            full_bag_ds = full_bag_ds.cache()
        dataset = full_bag_ds.batch(batch_size, drop_remainder=drop_remainder).prefetch(tf.data.AUTOTUNE)
        steps = (total_bags // batch_size) if drop_remainder else (total_bags + batch_size - 1) // batch_size
        return dataset, steps, total_bags
    # %%
    bag_sig = cfg.bag_size
    bg_cnt = int(round(bag_sig * BG_FRAC))
    bag_len = bag_sig + bg_cnt
    train_dataset, train_steps, train_bags = create_dynamic_binary_dataset(
        sm_train, bsm_train, bg_train, bag_sig, BG_FRAC, cfg.batch_size, shuffle_data=DYNAMIC_BAGGING, drop_remainder=False, cache=False
    )
    val_dataset, val_steps, val_bags = create_dynamic_binary_dataset(
        sm_val, bsm_val, bg_val, bag_sig, BG_FRAC, cfg.batch_size, shuffle_data=False, drop_remainder=False, cache=True
    )
    test_dataset, test_steps, test_bags = create_dynamic_binary_dataset(
        sm_test, bsm_test, bg_test, bag_sig, BG_FRAC, cfg.batch_size, shuffle_data=False, drop_remainder=False, cache=True
    )
    print(f"[Bags] train={train_bags} ({train_steps} steps), val={val_bags} ({val_steps} steps), test={test_bags} ({test_steps} steps)")
    print(f"Per-bag sizes: signal={bag_sig}, background={bg_cnt}, total={bag_len}")

    # %%
    def make_norm_sample(sm_src, bsm_src, bg_src, bag_sig, bg_frac, max_bags=64, seed=42):
        rng = np.random.default_rng(seed)
        bg_cnt = int(round(bag_sig * bg_frac))
        bag_len = bag_sig + bg_cnt
        n_sm_bags = max_bags // 2
        n_bsm_bags = max_bags - n_sm_bags
        feat = sm_src.shape[1]
        def build_bags(src, n_bags):
            # Sample without replacement as much as possible; fall back to wrap if needed
            total_needed = n_bags * bag_sig
            if len(src) >= total_needed:
                idx = rng.choice(len(src), size=total_needed, replace=False)
            else:
                idx = rng.choice(len(src), size=total_needed, replace=True)
            sig = src[idx].reshape(n_bags, bag_sig, feat)
            if bg_cnt > 0:
                total_bg_needed = n_bags * bg_cnt
                if len(bg_src) >= total_bg_needed:
                    bidx = rng.choice(len(bg_src), size=total_bg_needed, replace=False)
                else:
                    bidx = rng.choice(len(bg_src), size=total_bg_needed, replace=True)
                bg = bg_src[bidx].reshape(n_bags, bg_cnt, feat)
                bags = np.concatenate([sig, bg], axis=1)
            else:
                bags = sig
            return bags
        sm_bags = build_bags(sm_src, n_sm_bags)
        bsm_bags = build_bags(bsm_src, n_bsm_bags)
        sample = np.concatenate([sm_bags, bsm_bags], axis=0).astype(np.float32)
        return sample
    X_norm_sample = make_norm_sample(sm_train, bsm_train, bg_train, bag_sig, BG_FRAC, max_bags=64)
    print(f"X_norm_sample shape: {X_norm_sample.shape}")
    # %%
    def create_binary_clf(data, cfg, dense_units=64, num_of_layers=3, activation='elu', dropout=0.1):
        normalization = k.layers.Normalization()
        normalization.adapt(data)
        inputs = k.Input(shape=(None, data.shape[2]))
        x = normalization(inputs)
        for _ in range(num_of_layers):
            x = k.layers.Dense(dense_units, kernel_regularizer=k.regularizers.l2(1e-3))(x)
            x = k.layers.BatchNormalization()(x)
            x = k.layers.Activation(activation)(x)
            x = k.layers.Dropout(dropout)(x)
        if cfg.experiment.db:
            x = k.layers.GlobalAveragePooling1D()(x)
        outputs = k.layers.Dense(1, activation='sigmoid')(x)
        model = k.Model(inputs=inputs, outputs=outputs)
        return model
    if cfg.experiment.wandb:
        model = create_binary_clf(X_norm_sample, cfg, dense_units=wandb.config["dense_units"], num_of_layers=wandb.config["num_layers"])
    else:
        model = create_binary_clf(X_norm_sample, cfg)
    print(model.summary())
    del X_norm_sample, sm_all, bsm_all, sm_train, sm_val, sm_test, bsm_train, bsm_val, bsm_test
    gc.collect()
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
        "bag_size": {"values": [1000, 500, 250, 100, 10]},
        "seed": {"values": [1, 2, 3, 4, 5]},
    }
}

# Create the sweep and obtain the sweep ID
sweep_id = wandb.sweep(sweep_config, project=cfg.experiment.name)

wandb.agent(sweep_id, function=train)

