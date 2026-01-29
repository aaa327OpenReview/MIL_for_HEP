# %%

# You can't use dynamic bagging when your bag size is 1.
# That's why this script exists.

import os
import sys
import io
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

# Generate unique timestamp-based name
timestamp = datetime.datetime.now().strftime("%d-%m_%H-%M")
SWEEP_NAME = f"sweep_name_{timestamp}"


def train():
    # %%
    tf.keras.backend.clear_session()
    gc.collect()
    cfg = src.configs.get_configs()
    cfg.experiment.save = False
    cfg.experiment.wandb = False
    cfg.experiment.sweep = False
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
    if cfg.experiment.wandb:
        cfg.bag_size = wandb.config["bag_size"]
        cfg.bckgrnd_percnt = wandb.config["bckgrnd_percnt"]
        cfg.chw_classes = wandb.config["chw_classes"] 
        if cfg.experiment.db:
            cfg.batch_size = 8*(10_000//cfg.bag_size)
    else:
        cfg.bag_size = 1
        cfg.chw_classes = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        cfg.bckgrnd_percnt = 0
        if cfg.experiment.db:
            cfg.batch_size = 8*(10_000//cfg.bag_size)
    #
    cfg.num_classes = len(cfg.chw_classes)
    chw_grid = np.array(cfg.chw_classes)
    #
    print(f' - chw_classes: {cfg.chw_classes}')
    print(f' - bag_size: {cfg.bag_size}')
    print(f' - batch_size: {cfg.batch_size}')
    print(f' - epochs: {cfg.epochs}')
    print(f' - test_split: {cfg.test_split}')
    print(f' - validation_split: {cfg.validation_split}')
    print(f' - patience: {cfg.patience}')
    print(f' - lr: {cfg.lr}')
    print(f' - save: {cfg.experiment.save}')
    # %%
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["monitor"] = MONITOR
    if cfg.experiment.wandb:
        wandb.init(project=cfg.experiment.name, config=cfg_dict)
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
    class_map = {val: i for i, val in enumerate(cfg.chw_classes)}
    print(f"Class map set, the 'chw_val: class_id' mapping is:\n{class_map}")
    # %%
    df = pd.read_parquet(SOME_PATH, 
                         engine='pyarrow', 
                         columns=columns_to_keep)
    df = df[df['cHW'].isin(cfg.chw_classes)].copy()
    df['class_id'] = df['cHW'].map(class_map)
    if cfg.bckgrnd_percnt != 0:
        bg_source = pd.read_parquet(SOME_PATH,
                                        engine='pyarrow',
                                        columns=columns_to_keep).drop(columns=['cHW'], errors='ignore').to_numpy()
    features = df.drop(columns=['cHW', 'class_id']).to_numpy()
    labels = df['class_id'].to_numpy()
    num_of_features = features.shape[1]
    print(f"Loaded data with {num_of_features} features for {cfg.num_classes} classes.")
    print("Class distribution:\n", df['cHW'].value_counts())
    del df
    gc.collect()
    # %%
    # Standard instance-level data splitting
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features, labels, test_size=cfg.test_split, random_state=42, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=cfg.validation_split, random_state=42, stratify=y_train_val)
    del features, labels, X_train_val, y_train_val
    gc.collect()
    # Simplified data pipeline for bag_size=1 (instance-level training)
    # Reshape features to have a "bag" dimension of 1 for model compatibility
    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    train_steps = (len(X_train) // cfg.batch_size)  # Ceiling division
    val_steps = (len(X_val) // cfg.batch_size)
    test_steps = (len(X_test) // cfg.batch_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, tf.one_hot(y_train, cfg.num_classes)))
    train_dataset = train_dataset.shuffle(buffer_size=3*cfg.batch_size).batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    del X_train, y_train
    gc.collect()
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, tf.one_hot(y_val, cfg.num_classes)))
    val_dataset = val_dataset.batch(cfg.batch_size).cache().prefetch(tf.data.AUTOTUNE)
    del X_val, y_val
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, tf.one_hot(y_test, cfg.num_classes)))
    test_dataset = test_dataset.batch(cfg.batch_size).cache().prefetch(tf.data.AUTOTUNE)
    del X_test, y_test
    gc.collect()
    # %%
    normalization_layer = tf.keras.layers.Normalization()
    # We use .take(100) to grab only 100 batches, which is more than enough and very fast.
    print("Adapting normalization layer...")
    adaptation_ds = train_dataset.map(lambda features, label: features).take(int(100*(1000//cfg.bag_size))) 
    normalization_layer.adapt(adaptation_ds)
    print("Adaptation complete.")
    del adaptation_ds
    gc.collect()
    # %%
    def create_multi_class_model(input_shape, adapted_normalization_layer, cfg, dense_units=256, num_of_layers=6, activation='elu', dropout=0.1):
        inputs = k.Input(shape=input_shape)
        x = adapted_normalization_layer(inputs)
        for _ in range(num_of_layers):
            x = k.layers.Dense(dense_units, kernel_regularizer=k.regularizers.l2(1e-3))(x)
            x = k.layers.BatchNormalization()(x)
            x = k.layers.Activation(activation)(x)
            if dropout > 0:
                x = k.layers.Dropout(dropout)(x)
        if len(input_shape) > 1: 
            x = k.layers.GlobalAveragePooling1D()(x)
        outputs = k.layers.Dense(cfg.num_classes, activation='softmax')(x)
        model = k.Model(inputs=inputs, outputs=outputs)
        return model
    input_shape = train_dataset.element_spec[0].shape[1:] 
    if cfg.experiment.wandb:
        model = create_multi_class_model(
            input_shape, normalization_layer, cfg, 
            dense_units=wandb.config["dense_units"], 
            num_of_layers=wandb.config["num_layers"],
            dropout=cfg.dropout if hasattr(cfg, 'dropout') else 0.1
        )
    else:
        model = create_multi_class_model(
            input_shape, normalization_layer, cfg, 
            dense_units=64, 
            num_of_layers=3,
            dropout=cfg.dropout if hasattr(cfg, 'dropout') else 0.1
        )
    print(model.summary())
    # %%
    optimizer = k.optimizers.AdamW(learning_rate=cfg.lr)
    model.compile(optimizer=optimizer,
                  loss=k.losses.CategoricalCrossentropy(), 
                  metrics=[k.metrics.CategoricalAccuracy(name='accuracy'), 
                           k.metrics.AUC(name='auc', curve='ROC', multi_label=True, num_labels=cfg.num_classes)])
    # %%
    callbacks = []
    if cfg.early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(MONITOR, patience=cfg.patience, restore_best_weights=True))
    if cfg.reduce_lr_on_plateau:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor=MONITOR, patience=int(cfg.patience/2), min_delta=0.0, min_lr=cfg.min_lr))
    if cfg.experiment.wandb:
        callbacks.append(WandbMetricsLogger(log_freq= "epoch")) # for batch level logging, set some number I guess
    print(callbacks)
    # %%
    model_size = model.count_params()
    if cfg.experiment.sweep:
        run.config.update({"model_params": model_size})
    # %%
    history = model.fit(train_dataset, 
                        epochs=cfg.epochs, 
                        verbose=2, 
                        validation_data=val_dataset, 
                        callbacks=callbacks, 
                        steps_per_epoch=train_steps, 
                        validation_steps=val_steps)
    # %%
    try:
        plt.clf()
        plt.close()
    except:
        pass
    # Plot and log training/validation loss
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
        "bckgrnd_percnt": {"values": [0]},
        "chw_classes": {"values": [[-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]]},
        "bag_size": {"values": [1]},
        "seed": {"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
    }
}

sweep_id = wandb.sweep(sweep_config, project=cfg.experiment.name)

wandb.agent(sweep_id, function=train)

