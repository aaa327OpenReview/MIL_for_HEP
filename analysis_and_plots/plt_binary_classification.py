# %% 
from math import e
import os, re, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
import tensorflow.keras as k
import tensorflow as tf
from xgboost import XGBClassifier
import src


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


# %%
MODEL_SWEEP_DIR = 
SIGNAL_DATA_PATH = 
BACKGROUND_DATA_PATH = 
XGB_PRE_PATH = 
XGB_FILENAME_PATTERN = 
THE_CHW_TO_STUDY = 0.1
TEST_SPLIT = 0.2
NUM_TOTAL_SIG = 1_000_000
PRED_BATCH = 8192
COMMON_FPR = np.linspace(0,1,201)

# %%
def safe_logit(p, eps=1e-12):
    p = np.clip(p, eps, 1-eps)
    return np.log(p/(1-p))
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
def interp_tpr(fpr, tpr, grid):
    u, idx = np.unique(fpr, return_index=True)
    return np.interp(grid, u, tpr[idx])
def roc_dict(y, probs):
    fpr, tpr, _ = roc_curve(y, probs)
    return {"fpr": fpr, "tpr": tpr, "auc": float(auc(fpr,tpr))}
def aggregate_models_rocs(keys, container, fpr_grid=COMMON_FPR):
    tprs = []
    aucs = []
    for k in keys:
        entry = container[k]
        r = roc_dict(entry['y'], entry['probs'])
        tprs.append(interp_tpr(r['fpr'], r['tpr'], fpr_grid))
        aucs.append(r['auc'])
    if len(aucs) == 0:
        return None
    tprs = np.vstack(tprs)
    return {"mean_tpr": tprs.mean(axis=0), "std_tpr": tprs.std(axis=0),
            "mean_auc": float(np.mean(aucs)), "std_auc": float(np.std(aucs)),
            "n": len(aucs), "fpr_grid": fpr_grid}
# %%
cfg = src.configs.get_configs()
cols = cfg.columns_to_keep

df_sig = pd.read_parquet(SIGNAL_DATA_PATH, engine='pyarrow', columns=cols)
df_bg  = pd.read_parquet(BACKGROUND_DATA_PATH, engine='pyarrow', columns=cols)

num_test_sig = int(NUM_TOTAL_SIG * TEST_SPLIT)
num_train_sig = NUM_TOTAL_SIG - num_test_sig

bg_all = df_bg.drop(columns=['cHW'], errors='ignore').to_numpy()
bg_test = bg_all[2_500_000:3_000_000] # like in the training script
del bg_all
sm_all = df_sig[df_sig['cHW'] == 0].drop(columns=['cHW'], errors='ignore').to_numpy()
bsm_all = df_sig[df_sig['cHW'] == THE_CHW_TO_STUDY].drop(columns=['cHW'], errors='ignore').to_numpy()

sm_train_src, sm_test_src = sm_all[num_test_sig:], sm_all[:num_test_sig] # like in the training script
bsm_train_src, bsm_test_src = bsm_all[num_test_sig:], bsm_all[:num_test_sig] # like in the training script

del df_sig, df_bg, sm_all, bsm_all
gc.collect()

print("Loaded data. shapes:", sm_test_src.shape, bsm_test_src.shape, bg_test.shape)

# %%

keras_files = [f for f in os.listdir(MODEL_SWEEP_DIR) if f.endswith('.keras')]
rg = re.compile(r"Bag(\d+)_Bckgrd([\d.]+)_Seed(\d+)\.keras")
models_by_bgrnd = {}
for fn in keras_files:
    m = rg.match(fn)
    if not m: continue
    bag_size = int(m.group(1))
    bgrnd_frac = float(m.group(2))
    bgrnd_frac_str = f"{bgrnd_frac:.6f}"
    models_by_bgrnd.setdefault(bgrnd_frac_str, {}).setdefault(bag_size, []).append(os.path.join(MODEL_SWEEP_DIR, fn))

print("Found bg fracs:", sorted(models_by_bgrnd.keys()))

# %%
all_predictions_all_models = {}

for bgrnd_frac_str in sorted(models_by_bgrnd.keys(), key=float):
    bgrnd_frac = float(bgrnd_frac_str)
    print(f"\n\n>>> bgrnd_frac = {bgrnd_frac_str}")

    bsm_test_events = bsm_test_src.copy()
    sm_test_events  = sm_test_src.copy()
    bg_test_src = bg_test.copy()    

    bag_testsets = {}
    orig_bsm = bsm_test_events  # 2D (N_bsm, F)
    orig_sm  = sm_test_events   # 2D (N_sm, F)
    for bag_size in sorted(models_by_bgrnd[bgrnd_frac_str].keys()):
        if bag_size == 1:
            Xb = np.concatenate((orig_bsm, orig_sm), axis=0)
            yb = np.concatenate((np.ones(len(orig_bsm)), np.zeros(len(orig_sm))), axis=0)
            Xb, yb = shuffle(Xb, yb, random_state=42)
            bag_testsets[bag_size] = (Xb, yb)
        else:
            feat = orig_bsm.shape[1]
            nbsm = (orig_bsm.shape[0] // bag_size) * bag_size
            nsm  = (orig_sm.shape[0]  // bag_size) * bag_size
            bsm_bags = orig_bsm[:nbsm].copy().reshape(-1, bag_size, feat)
            sm_bags  = orig_sm[:nsm].copy().reshape(-1, bag_size, feat)
            bg_cnt = int(round(bag_size * bgrnd_frac))
            if bg_cnt > 0:
                total_bags = len(bsm_bags) + len(sm_bags)
                bg_events_needed = total_bags * bg_cnt
                if bg_test_src.shape[0] < bg_events_needed:
                    raise ValueError(f"Not enough bg events ({bg_test_src.shape[0]}) to fill bags (need {bg_events_needed}). Reduce bgrnd_frac or bag_size.")
                else:
                    bg_pool = bg_test_src[:bg_events_needed]
                bsm_bg = bg_pool[:len(bsm_bags)*bg_cnt].reshape(len(bsm_bags), bg_cnt, feat)
                sm_bg  = bg_pool[len(bsm_bags)*bg_cnt:].reshape(len(sm_bags),  bg_cnt, feat)
                bsm_bags = np.concatenate([bsm_bags, bsm_bg], axis=1)
                sm_bags  = np.concatenate([sm_bags,  sm_bg],  axis=1)
            Xb = np.concatenate((bsm_bags, sm_bags), axis=0)
            yb = np.concatenate((np.ones(len(bsm_bags)), np.zeros(len(sm_bags))), axis=0)
            Xb, yb = shuffle(Xb, yb, random_state=42)
            bag_testsets[bag_size] = (Xb, yb)
    
    print("  Prepared bag testsets for sizes:", {k: v[0].shape for k,v in bag_testsets.items()})
    print(f'{bag_testsets[1][0].shape}', 'events in bag=1 test set')
    print(f'{bag_testsets[1][1]}', 'labels in bag=1 test set')

    predictions_container = {"event_models": {}, "bag_models": {}, "ensemble_from_event": {}}

    # ---- MLP event models (bag_size==1) ----
    for model_pth in models_by_bgrnd[bgrnd_frac_str].get(1, []):
        print("  load MLP event model:", os.path.basename(model_pth))
        model = k.models.load_model(model_pth)
        X_test = bag_testsets[1][0]
        y_test = bag_testsets[1][1]
        probs = model.predict(X_test, batch_size=PRED_BATCH).ravel()
        predictions_container["event_models"][model_pth] = {"X": X_test, "y": y_test, "probs": probs}

        # build ensembles (1->N) for every bag size present
        for bag_size, (Xb, yb) in bag_testsets.items():
            if bag_size == 1:
                continue
            flat = Xb.reshape((-1, Xb.shape[2]))
            pflat = model.predict(flat, batch_size=PRED_BATCH).ravel()
            n_bags = Xb.shape[0]
            total_bag = Xb.shape[1]
            assert pflat.size == n_bags * total_bag, "MLP ensemble: unexpected prediction length"
            pperbag = pflat.reshape((n_bags, total_bag))
            logit_sum = np.sum(safe_logit(pperbag), axis=1)
            bag_probs = sigmoid(logit_sum)
            predictions_container["ensemble_from_event"][(model_pth, bag_size)] = {"X": Xb, "y": yb, "probs": bag_probs}

        k.backend.clear_session(); del model; gc.collect()

    # ---- XGBoost event model (bag_size==1) ----
    bgrnd_code = f"{int(round(bgrnd_frac*10)):02d}"
    xgb_model_path = os.path.join(XGB_PRE_PATH, XGB_FILENAME_PATTERN.format(bgrnd_code=bgrnd_code))
    print("  load XGB event model:", os.path.basename(xgb_model_path))
    xgb_model = XGBClassifier()
    xgb_model.load_model(xgb_model_path)

    X_test, y_test = bag_testsets[1]
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    predictions_container["event_models"][f"xgb::{xgb_model_path}"] = {"X": X_test, "y": y_test, "probs": xgb_probs}

    # artificial bag ensembles from the event-level XGB model
    for bag_size, (Xb, yb) in bag_testsets.items():
        if bag_size == 1:
            continue
        flat = Xb.reshape((-1, Xb.shape[2]))
        pflat = xgb_model.predict_proba(flat)[:, 1]
        n_bags = Xb.shape[0]
        total_bag = Xb.shape[1]
        assert pflat.size == n_bags * total_bag, "XGB ensemble: unexpected prediction length"
        pperbag = pflat.reshape((n_bags, total_bag))
        logit_sum = np.sum(safe_logit(pperbag), axis=1)
        bag_probs = sigmoid(logit_sum)
        predictions_container["ensemble_from_event"][(f"xgb::{xgb_model_path}", bag_size)] = {
            "X": Xb, "y": yb, "probs": bag_probs
        }
    del xgb_model

    # ---- Evaluate bag-level MLP models (bag_size > 1) ----
    for bag_size, mpaths in models_by_bgrnd[bgrnd_frac_str].items():
        if bag_size == 1: continue
        Xb, yb = bag_testsets[bag_size]
        for model_pth in mpaths:
            print("  load bag model:", os.path.basename(model_pth))
            model = k.models.load_model(model_pth)
            probs = model.predict(Xb, batch_size=min(4096, Xb.shape[0])).ravel()
            predictions_container["bag_models"][model_pth] = {
                "bag_size": bag_size, "src_bag_size": bag_size,
                "X": Xb, "y": yb, "probs": probs
            }

            k.backend.clear_session(); del model; gc.collect()
    
    all_predictions_all_models[bgrnd_frac_str] = predictions_container


# %%

summary_rows = []

for bgrnd_frac_str, predictions_container in all_predictions_all_models.items():
    bgrnd_frac = float(bgrnd_frac_str)
    
    # ---- AGGREGATE & PLOT for this bg_frac ----
    event_container = predictions_container["event_models"]
    mlp_event_keys = [k for k in event_container if not str(k).startswith("xgb::")]
    xgb_event_keys = [k for k in event_container if str(k).startswith("xgb::")]

    agg_mlp_event = aggregate_models_rocs(mlp_event_keys, event_container) if mlp_event_keys else None
    agg_xgb_event = aggregate_models_rocs(xgb_event_keys, event_container) if xgb_event_keys else None
    bag_container   = predictions_container["bag_models"]
    
    ens_container   = predictions_container["ensemble_from_event"]

    available_bag_sizes = sorted(set(
        [v["bag_size"] for v in bag_container.values()] +
        [k[1] for k in ens_container.keys()]))
    
    plt.figure(figsize=(12,9))
    plt.style.use('seaborn-v0_8-whitegrid')

    # plot event baselines
    if agg_mlp_event is not None:
        plt.plot(agg_mlp_event['fpr_grid'], agg_mlp_event['mean_tpr'], color='black', lw=2.2,
                 label=f"MLP event (bag=1) AUC={agg_mlp_event['mean_auc']:.3f} (n={agg_mlp_event['n']})")
        plt.fill_between(agg_mlp_event['fpr_grid'], agg_mlp_event['mean_tpr']-agg_mlp_event['std_tpr'],
                         agg_mlp_event['mean_tpr']+agg_mlp_event['std_tpr'], color='black', alpha=0.12)
    if agg_xgb_event is not None:
        plt.plot(agg_xgb_event['fpr_grid'], agg_xgb_event['mean_tpr'], color='purple', lw=2.2, linestyle='-',
                 label=f"XGB event AUC={agg_xgb_event['mean_auc']:.3f}")
        plt.fill_between(agg_xgb_event['fpr_grid'], agg_xgb_event['mean_tpr']-agg_xgb_event['std_tpr'],
                         agg_xgb_event['mean_tpr']+agg_xgb_event['std_tpr'], color='purple', alpha=0.12)

    cmap = plt.cm.get_cmap("tab10", max(3, len(available_bag_sizes)))
    for i, bag_size in enumerate(available_bag_sizes):
        color = cmap(i % cmap.N)
        bag_keys = [p for p,v in bag_container.items() if v["bag_size"]==bag_size]
        agg_bag = aggregate_models_rocs(bag_keys, bag_container) if len(bag_keys)>0 else None

        mlp_ens_keys = [k for k in ens_container.keys() if k[1]==bag_size and not str(k[0]).startswith("xgb")]
        mlp_ens_temp = {k: {"y": ens_container[k]["y"], "probs": ens_container[k]["probs"]} for k in mlp_ens_keys}
        agg_mlpens = aggregate_models_rocs(list(mlp_ens_temp.keys()), mlp_ens_temp) if len(mlp_ens_temp)>0 else None

        xgb_ens_keys = [k for k in ens_container.keys() if isinstance(k, tuple) and str(k[0]).startswith("xgb") and k[1]==bag_size]
        xgb_ens_temp = {k: {"y": ens_container[k]["y"], "probs": ens_container[k]["probs"]} for k in xgb_ens_keys}
        agg_xgbens = aggregate_models_rocs(list(xgb_ens_temp.keys()), xgb_ens_temp) if len(xgb_ens_temp)>0 else None

        if agg_bag is not None:
            plt.plot(agg_bag['fpr_grid'], agg_bag['mean_tpr'], color=color, lw=2.0,
                     label=f"Bag N={bag_size} (MLP) {agg_bag['mean_auc']:.3f}±{agg_bag['std_auc']:.3f}")
            plt.fill_between(agg_bag['fpr_grid'], agg_bag['mean_tpr']-agg_bag['std_tpr'],
                            #  agg_bag['mean_tpr']+agg_bag['std_tpr'], color=color, alpha=0.14)
                             agg_bag['mean_tpr']+agg_bag['std_tpr'], color=color, alpha=0.16)
        if agg_mlpens is not None:
            plt.plot(agg_mlpens['fpr_grid'], agg_mlpens['mean_tpr'], color='black', lw=1.8, linestyle='--',
                     label=f"Ensemble 1→{bag_size} (MLP) {agg_mlpens['mean_auc']:.3f}±{agg_mlpens['std_auc']:.3f}")
            plt.fill_between(agg_mlpens['fpr_grid'], agg_mlpens['mean_tpr']-agg_mlpens['std_tpr'],
                            #  agg_mlpens['mean_tpr']+agg_mlpens['std_tpr'], color='grey', alpha=0.08)
                             agg_mlpens['mean_tpr']+agg_mlpens['std_tpr'], color='grey', alpha=0.12)
        # if agg_xgbens is not None:
        #     plt.plot(agg_xgbens['fpr_grid'], agg_xgbens['mean_tpr'], color='black', lw=2.0, linestyle=':',
        #              label=f"Ensemble 1→{bag_size} (XGB) {agg_xgbens['mean_auc']:.3f}±{agg_xgbens['std_auc']:.3f}")
        #     plt.fill_between(agg_xgbens['fpr_grid'], agg_xgbens['mean_tpr']-agg_xgbens['std_tpr'],
        #                      agg_xgbens['mean_tpr']+agg_xgbens['std_tpr'], color='purple', alpha=0.08)

        # summary row
        summary_rows.append({
            "bg_frac": float(bgrnd_frac_str),
            "bag_size": bag_size,
            "mlp_bag_mean_auc": agg_bag['mean_auc'] if agg_bag is not None else np.nan,
            "mlp_bag_std_auc": agg_bag['std_auc'] if agg_bag is not None else np.nan,
            "mlp_ens_mean_auc": agg_mlpens['mean_auc'] if agg_mlpens is not None else np.nan,
            "mlp_ens_std_auc": agg_mlpens['std_auc'] if agg_mlpens is not None else np.nan,
            "xgb_ens_mean_auc": agg_xgbens['mean_auc'] if agg_xgbens is not None else np.nan,
            "xgb_ens_std_auc": agg_xgbens['std_auc'] if agg_xgbens is not None else np.nan,
            "mlp_event_mean_auc": agg_mlp_event['mean_auc'] if agg_mlp_event is not None else np.nan,
            "mlp_event_std_auc": agg_mlp_event['std_auc'] if agg_mlp_event is not None else np.nan,
            "xgb_event_mean_auc": agg_xgb_event['mean_auc'] if agg_xgb_event is not None else np.nan,
            "xgb_event_std_auc": agg_xgb_event['std_auc'] if agg_xgb_event is not None else np.nan
        })

    plt.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--', label='Chance (AUC = 0.500)')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=18, labelpad=10)
    plt.ylabel('True Positive Rate', fontsize=18, labelpad=10)
    plt.title(f'ROC Curves: $c_{{HW}}={THE_CHW_TO_STUDY}$, Background Contamination={int(float(bgrnd_frac)*100)}%', fontsize=20, pad=15)
    handles, labels = plt.gca().get_legend_handles_labels()
    seen = set(); uniq = []
    for h,l in zip(handles, labels):
        if l not in seen:
            uniq.append((h,l)); seen.add(l)
    if uniq:
        hs, ls = zip(*uniq)
        plt.legend(hs, ls, loc='lower right', fontsize=13)
    plt.grid(True, which='both', linestyle=':', linewidth=0.7)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    # plt.savefig(f"MILvsXGB_chw01_bg{int(float(bgrnd_frac)*100)}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()



# %%
# --- print combined numeric summary ---
df_summary = pd.DataFrame(summary_rows).sort_values(["bg_frac","bag_size"]).reset_index(drop=True)
def fmt(x): return f"{x:.4f}" if (pd.notnull(x) and not np.isnan(x)) else "nan"
if df_summary.shape[0] == 0:
    print("No results collected.")
else:
    for c in df_summary.columns:
        if 'auc' in c:
            df_summary[c] = df_summary[c].apply(lambda v: fmt(v))
    print("\nCombined summary (one row per bg_frac, bag_size):")
    print(df_summary.to_string(index=False))
