# %%
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
import pickle

import glob
import re
from matplotlib.lines import Line2D

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


# %%

def calculate_ci_from_log_likelihood(chw_grid_local, sum_log_prob_over_chw_grid_local, fit_window_half_size=4):
    """Calculates CI by fitting a parabola to the log-likelihood."""
    max_ll_idx = np.argmax(sum_log_prob_over_chw_grid_local)
    start_idx = max(0, max_ll_idx - fit_window_half_size)
    end_idx = min(len(chw_grid_local) - 1, max_ll_idx + fit_window_half_size)
    chw_fit_data = chw_grid_local[start_idx:end_idx + 1]
    sum_log_prob_fit_data = sum_log_prob_over_chw_grid_local[start_idx:end_idx + 1]
    if len(chw_fit_data) < 3:
        chw_fit_data = chw_grid_local
        sum_log_prob_fit_data = sum_log_prob_over_chw_grid_local
    coeffs = np.polyfit(chw_fit_data, sum_log_prob_fit_data, 2)
    a_fit, p1, _ = coeffs
    fitted_y_values = np.polyval(coeffs, chw_fit_data)
    mse = np.mean((sum_log_prob_fit_data - fitted_y_values)**2)
    mu_fit = -p1 / (2 * a_fit)
    w1_sigma = np.sqrt(-0.5 / a_fit)

    return mu_fit, mu_fit - w1_sigma, mu_fit + w1_sigma, a_fit, mse

def get_coverage_and_fisher(predictions, bag_size, chw_grid, ci_constant):
    """
    For a single set of predictions (one model or ensemble), calculates the
    bias-corrected CI coverage and mean Fisher information.
    """
    num_bags_per_chunk = 1000 // bag_size
    total_bags_available = predictions.shape[1]
    num_chunks = total_bags_available // num_bags_per_chunk
    true_value = 0.0
    original_cis = []
    fisher_info_values = []
    mse_values = [] 
    for i in range(num_chunks):
        start_bag = i * num_bags_per_chunk
        end_bag = start_bag + num_bags_per_chunk
        chunk_preds = predictions[:, start_bag:end_bag]
        log_prob = np.log(chunk_preds / (1 - chunk_preds + EPSILON))
        # log_prob = np.log(chunk_preds / (chunk_preds[10] + EPSILON))
        sum_log_prob = np.sum(log_prob, axis=1) * ci_constant
        if bag_size == 1:
            mu, ci_low, ci_high, a_fit, mse= calculate_ci_from_log_likelihood(chw_grid, sum_log_prob, fit_window_half_size=7)
        else:
            mu, ci_low, ci_high, a_fit, mse= calculate_ci_from_log_likelihood(chw_grid, sum_log_prob, fit_window_half_size=4)
        original_cis.append((ci_low, ci_high))
        fisher_info_values.append(-2 * a_fit)
        mse_values.append(mse) 
    # Perform bias correction
    midpoints = [((low + high) / 2.0) for low, high in original_cis]
    average_bias = np.nanmean(midpoints)
    corrected_cis = [(low - average_bias, high - average_bias) for low, high in original_cis]
    # Calculate coverage of the bias-corrected CIs
    coverage_count = sum(1 for low, high in corrected_cis if low <= true_value <= high)
    coverage_percentage = (coverage_count / num_chunks) * 100
    return coverage_percentage, fisher_info_values, average_bias, np.nanmean(mse_values)

def find_optimal_constant(predictions, bag_size, chw_grid, is_ensemble=False, tolerance=0.1, max_iterations=150, learning_rate=0.3):
    """
    Performs a numerical search to find the CI constant that yields coverage
    closest to 68.3%.
    """
    TARGET_COVERAGE = 68.3
    TOLERANCE = tolerance # Target: 68.3 +/- tolerance
    MAX_ITERATIONS = max_iterations
    LEARNING_RATE = learning_rate # A factor to control the update speed
    ci_constant = 1.0 # Initial guess
    if is_ensemble:
        initial_cov, _, _, _ = get_coverage_and_fisher(predictions, bag_size, chw_grid, 1.0)
        print(f"DEBUG (BagSize={bag_size}, Ensemble): Initial uncalibrated coverage is {initial_cov:.2f}%")
    else:
        initial_coverages = [get_coverage_and_fisher(p, bag_size, chw_grid, 1.0)[0] for p in predictions]
        mean_initial_cov = np.nanmean(initial_coverages)
        print(f"DEBUG (BagSize={bag_size}, Individual): Initial uncalibrated coverage is {mean_initial_cov:.2f}%")
    for i in range(MAX_ITERATIONS):
        if is_ensemble:
            coverage, _, _, _ = get_coverage_and_fisher(predictions, bag_size, chw_grid, ci_constant)
        else:
            coverages = [get_coverage_and_fisher(p, bag_size, chw_grid, ci_constant)[0] for p in predictions]
            coverage = np.nanmean(coverages)

        error = coverage - TARGET_COVERAGE
        if abs(error) < TOLERANCE:
            break
        if i == MAX_ITERATIONS - 1:
            print(f"Max iterations reached for BagSize {bag_size}. Final constant: {ci_constant:.4f} with coverage {coverage:.2f}%")
        ci_constant *= (1 + LEARNING_RATE * (error / 100.0))

    if is_ensemble:
        final_coverage, final_fisher_list, final_bias, final_mse = get_coverage_and_fisher(predictions, bag_size, chw_grid, ci_constant)
    else:
        results = [get_coverage_and_fisher(p, bag_size, chw_grid, ci_constant) for p in predictions]
        final_coverage = np.nanmean([r[0] for r in results])
        final_fisher_list = [r[1] for r in results]
        final_bias = np.nanmean([r[2] for r in results])
        final_mse = np.nanmean([r[3] for r in results]) 

    return ci_constant, final_coverage, final_fisher_list, final_bias, final_mse




# %%

def plot_combined_confidence_intervals(original_cis, corrected_cis, y_labels, title, true_value, bias, x_lim=None):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    y_coords = np.arange(len(y_labels))
    datasets = {
        "original": original_cis,
        "corrected": corrected_cis
    }
    plot_params = {
        "original": {'fmt': 'o', 'label_suffix': 'Original'},
        "corrected": {'fmt': 'D', 'label_suffix': 'Bias-Corrected'} 
    }
    for name, data in datasets.items():
        params = plot_params[name]
        covers_data = {'x': [], 'y': [], 'xerr': []}
        misses_data = {'x': [], 'y': [], 'xerr': []}
        
        for i, (_, lower, upper) in enumerate(data):
            if np.isnan(lower) or np.isnan(upper):
                continue
            mid = (lower + upper) / 2
            err = (upper - lower) / 2
            if lower <= true_value <= upper:
                covers_data['x'].append(mid)
                covers_data['y'].append(y_coords[i])
                covers_data['xerr'].append(err)
            else:
                misses_data['x'].append(mid)
                misses_data['y'].append(y_coords[i])
                misses_data['xerr'].append(err)
        ax.errorbar(x=covers_data['x'], y=covers_data['y'], xerr=covers_data['xerr'],
                    fmt=params['fmt'], color='#2976bb', ecolor='#2976bb', capsize=4, elinewidth=1.5,
                    markeredgecolor='black', markersize=6, label=f"{params['label_suffix']} (Covers True Value)")
        ax.errorbar(x=misses_data['x'], y=misses_data['y'], xerr=misses_data['xerr'],
                    fmt=params['fmt'], color='#c74a45', ecolor='#c74a45', capsize=4, elinewidth=1.5,
                    markeredgecolor='black', markersize=6, label=f"{params['label_suffix']} (Misses True Value)")

    ax.axvline(true_value, color='green', linestyle='--', label=f'True $c_{{HW}}$ ({true_value})')
    ax.set_yticks(y_coords)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    ax.set_xlabel('$c_{HW}$ Value', fontsize=14)
    ax.set_ylabel('Model Identifier (Seed)', fontsize=14)
    ax.set_title(title, fontsize=16, pad=20)
    if x_lim:
        ax.set_xlim(x_lim)
    handles, labels = ax.get_legend_handles_labels()
    handles = sorted(handles, key=lambda h: 'Original' in h.get_label(), reverse=True)
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, fontsize=12)
    ax.grid(True, which='major', axis='x', linestyle=':', alpha=0.7)
    ax.grid(False, which='major', axis='y')
    fig.tight_layout()
    plt.show()


def plot_ci_subplots(original_cis, corrected_cis, y_labels, bag_size, n_events, true_value, bias, x_lim=None):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
    y_coords = np.arange(len(y_labels))
    COLOR_COVERS = '#2976bb'
    COLOR_MISSES = '#c74a45'
    COLOR_TRUE_VAL = '#009E73'
    fig.suptitle('Confidence Interval Coverage from Expected Log-Likelihood Ratios', fontsize=25, fontweight='bold')
    plot_data = {
        'Original Confidence Intervals': (axes[0], original_cis),
        'Bias-Corrected Confidence Intervals': (axes[1], corrected_cis)
    }
    for title, (ax, data) in plot_data.items():
        covers_data = {'x': [], 'y': [], 'xerr': []}
        misses_data = {'x': [], 'y': [], 'xerr': []}
        for i, (_, lower, upper) in enumerate(data):
            if np.isnan(lower) or np.isnan(upper): continue
            mid, err = (lower + upper) / 2, (upper - lower) / 2
            if lower <= true_value <= upper:
                covers_data['y'].append(y_coords[i]); covers_data['x'].append(mid); covers_data['xerr'].append(err)
            else:
                misses_data['y'].append(y_coords[i]); misses_data['x'].append(mid); misses_data['xerr'].append(err)
        
        ax.errorbar(x=covers_data['x'], y=covers_data['y'], xerr=covers_data['xerr'],
                    fmt=' ', ecolor='blue', elinewidth=2.5, capsize=8, capthick=2.5)
        ax.errorbar(x=misses_data['x'], y=misses_data['y'], xerr=misses_data['xerr'],
                    fmt=' ', ecolor='red', elinewidth=2.5, capsize=8, capthick=2.5)
        ax.set_title(title, fontsize=16, pad=10)
        ax.axvline(true_value, color=COLOR_TRUE_VAL, linestyle='--', lw=2)
        ax.set_xlabel('$c_{HW}$ Value', fontsize=14)
        if x_lim: ax.set_xlim(x_lim)
        ax.grid(True, axis='x', linestyle=':', color='gray', alpha=0.6)
        ax.set_yticks(y_coords)
        ax.set_yticklabels(y_labels, fontsize=12)
    legend_elements = [
        Line2D([0], [0], color=COLOR_TRUE_VAL, lw=2, linestyle='--', label=f'True $c_{{HW}}$ ({true_value})'),
        Line2D([0], [0], color=COLOR_COVERS, lw=3, label='Covers True Value'),
        Line2D([0], [0], color=COLOR_MISSES, lw=3, label='Misses True Value')
    ]
    axes[0].legend(handles=legend_elements, loc='lower left', fontsize=12, fancybox=True, framealpha=0.9)
    orig_total = sum(1 for _, low, high in original_cis if not np.isnan(low))
    corr_total = sum(1 for _, low, high in corrected_cis if not np.isnan(low))
    orig_covers_count = sum(1 for _, low, high in original_cis if not np.isnan(low) and low <= true_value <= high)
    corr_covers_count = sum(1 for _, low, high in corrected_cis if not np.isnan(low) and low <= true_value <= high)
    orig_coverage_perc = (orig_covers_count / orig_total) * 100 if orig_total > 0 else 0
    corr_coverage_perc = (corr_covers_count / corr_total) * 100 if corr_total > 0 else 0
    text_str = (
        r'$\bf{Analysis\ Summary}$' + '\n'
        '----------------------------------------\n'
        f'BagSize: {bag_size}, N_events: {n_events}\n'
        f'Bias Shift: {bias:.4f}\n\n'
        r'$\bf{Coverage\ Statistics}$' + '\n'
        f'Original:   {orig_covers_count}/{orig_total} ({orig_coverage_perc:.1f}%)\n'
        f'Corrected:  {corr_covers_count}/{corr_total} ({corr_coverage_perc:.1f}%)\n'
        f'Expected 1-Sigma:     ~68.3%\n'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[1].text(0.97, 0.03, text_str, transform=axes[1].transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    axes[0].set_ylabel('Model Identifier (Seed)', fontsize=14)
    axes[0].invert_yaxis()
    fig.tight_layout()
    plt.show()


# %%


def plot_confidence_intervals_horizontal(ci_data_list, y_axis_labels, title, true_value=0.0, x_lim=None, info_text=None, bag_size=None):
    plt.figure(figsize=(12, max(6, len(ci_data_list) * 0.3))) 
    y_coords = [d[0] for d in ci_data_list]
    for y_coord, lower, upper in ci_data_list:
        if np.isnan(lower) or np.isnan(upper):
            plt.plot([np.nan], [y_coord], marker='x', color='gray', markersize=10, label='Fit Failed' if not plt.gca().get_legend() else "")
            continue
        color = 'blue' if lower <= true_value <= upper else 'red'
        plt.plot([lower, upper], [y_coord, y_coord], marker='|', linestyle='-', color=color, markersize=10, mew=2)
    plt.axvline(true_value, color='green', linestyle='--', label=f'True $c_{{HW}}$ ({true_value})')
    if y_coords:
        plt.yticks(ticks=y_coords, labels=y_axis_labels)
    plt.xlabel('$c_{HW}$ Value', fontsize=15)
    plt.ylabel('Identifier (Data Chunk)', fontsize=15)
    plt.title(title, fontsize=18)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize=15)
    plt.grid(True, axis='x', linestyle=':', alpha=0.7)
    if info_text:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.gca().text(0.05, 0.95, info_text, transform=plt.gca().transAxes, fontsize=15,
                     verticalalignment='top', bbox=props)
    if y_coords:
        y_min, y_max = min(y_coords), max(y_coords)
        y_padding = (y_max - y_min) * 0.02  # 2% padding
        plt.ylim(y_min - y_padding, y_max + y_padding)
    if x_lim:
        plt.xlim(x_lim)
    else: 
        all_ci_values = [val for _, low, high in ci_data_list for val in (low, high) if not np.isnan(val)]
        if all_ci_values:
            min_val, max_val = min(all_ci_values), max(all_ci_values)
            padding = (max_val - min_val) * 0.1 if (max_val - min_val) > 0 else 0.5
            plt.xlim(min_val - padding, max_val + padding)
        else: 
             plt.xlim(chw_grid.min() - 0.1, chw_grid.max() + 0.1)
    plt.tight_layout()
    # plt.savefig(f'0_param_ci_{bag_size}.png', bbox_inches='tight')  # Tight bounding box
    plt.show()
    plt.clf()
    plt.close()
# %%

MODEL_DIR = 
DEFAULT_TEST_SPLIT = 0.2
EPSILON = 1e-15
BAG_SIZE_EXPECTED = [1, 10, 20, 25, 50, 100, 125, 200, 250]
cfg = src.configs.get_configs()
cfg.test_split = DEFAULT_TEST_SPLIT
cfg.chw_classes = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
cfg.bckgrnd_percnt = 0
cfg.num_classes = len(cfg.chw_classes)
chw_grid = np.array(cfg.chw_classes)

# %%

columns_to_keep = cfg.columns_to_keep
num_of_features = len(columns_to_keep)
print("Num of columns_to_keep", num_of_features)
df = pd.read_parquet(SOME_PATH, 
                     engine='pyarrow', 
                     columns=columns_to_keep)
sm_data = df[df['cHW'] == 0].copy()
del df
gc.collect()

# %%

# %%


BAG_SIZE_EXPECTED = [1, 10, 20, 25, 50, 100, 125, 200, 250]
all_bag_sizes_all_seed_predictions = {}

for bag_size in BAG_SIZE_EXPECTED:
    print(f"Processing bag size: {bag_size}")
    model_files = sorted(glob.glob(os.path.join(MODEL_DIR, f'Bag{bag_size}_Seed*.keras')))
    model_files = sorted(model_files, key=lambda x: int(re.search(r'Seed(\d+)', x).group(1)))

    cfg.bag_size = bag_size
    cfg.batch_size = 8 * (10_000 // cfg.bag_size)

    num_of_train_samples = int(1_000_000*(1-cfg.test_split))
    num_of_test_samples = int(1_000_000*cfg.test_split)
    train_bag_num = num_of_train_samples//cfg.bag_size
    test_bag_num = num_of_test_samples//cfg.bag_size
    sm_test_all_features = sm_data.copy().to_numpy()[num_of_train_samples:].reshape(test_bag_num, cfg.bag_size, num_of_features).copy()
    sm_test_without_chw_val = sm_test_all_features[:,:,:-1]  # Exclude cHW column

    all_models_predictions = []
    all_models_pred_over_chw_grid = []
    y_labels_part1 = []
    for model_idx, model_path in enumerate(model_files):
        seed = int(re.search(r'Seed(\d+)', os.path.basename(model_path)).group(1))
        y_labels_part1.append(f"Seed {seed}")
        
        pred_over_chw_grid_np = np.zeros((chw_grid.shape[-1], sm_test_without_chw_val.shape[0]))
        
        model = k.models.load_model(model_path, compile=False)
        print(f"Loaded model {model_idx+1}/{len(model_files)}: {model_path} with seed {seed}")

        for i, chw in enumerate(chw_grid):
            temp_chw_column = np.full((sm_test_without_chw_val.shape[0], cfg.bag_size, 1), chw)
            input_to_model = np.concatenate((sm_test_without_chw_val, temp_chw_column), axis=2)
            model_predicton = model.predict(input_to_model, batch_size=3*cfg.batch_size)
            pred_over_chw_grid_np[i] = model_predicton.squeeze()
            del temp_chw_column, input_to_model, model_predicton

        all_models_pred_over_chw_grid.append(pred_over_chw_grid_np.copy())
        del model, pred_over_chw_grid_np
        k.backend.clear_session()
        gc.collect()

    all_models_predictions = np.array(all_models_predictions)
    all_models_pred_over_chw_grid = np.array(all_models_pred_over_chw_grid)

    all_predictions = np.copy(all_models_pred_over_chw_grid) # Shape: (num_models, num_chw_values, num_bags)
    all_bag_sizes_all_seed_predictions[bag_size] = all_predictions
    
    del all_models_predictions, all_models_pred_over_chw_grid
    if bag_size != BAG_SIZE_EXPECTED[-1]:
        del y_labels_part1
    print(f"Processed bag size {bag_size} with {len(model_files)} models.")
    gc.collect()



# %%

# %%


# predictions_to_save = {str(key): value for key, value in all_bag_sizes_all_seed_predictions.items()}

# save_path = 
# np.savez_compressed(save_path, **predictions_to_save)

# print(f"Successfully saved all prediction data to: {save_path}")



# %%
BAG_SIZE_EXPECTED = [1, 10, 20, 25, 50, 100, 125, 200, 250]

save_path = 
loaded_data = np.load(save_path)
all_bag_sizes_all_seed_predictions = {int(key): value for key, value in loaded_data.items()}
print("Restored dictionary keys:", list(all_bag_sizes_all_seed_predictions.keys()))
print("Shape of data for bag size 50:", all_bag_sizes_all_seed_predictions[50].shape)

# %%

for bag_size, all_predictions in all_bag_sizes_all_seed_predictions.items():
    print(f"Bag Size: {bag_size}, Predictions Shape: {all_predictions.shape}")


# %%


















# %%
# =====================================================================================
# MAIN EXECUTION LOOP
# =====================================================================================

results_list = []
for bag_size, all_predictions in all_bag_sizes_all_seed_predictions.items():
    print(f"--- Processing Bag Size: {bag_size} ---")
    print("Searching for optimal constant for Individual Models...")
    ind_const, ind_cov, ind_fisher, ind_bias, ind_mse = find_optimal_constant(
        all_predictions, bag_size, chw_grid, is_ensemble=False, tolerance=0.25, max_iterations=150, learning_rate=0.3
    )
    results_list.append({
        'Bag Size': bag_size,
        'Model Type': 'Individual (Avg)',
        'CI Constant': ind_const,
        'Bias Shift': ind_bias,
        'Coverage (%)': ind_cov,
        'Mean Fisher Info': ind_fisher,
        'Fit MSE': ind_mse
    })
    print("Searching for optimal constant for Ensemble Model...")
    ensemble_predictions = np.mean(all_predictions, axis=0)
    ens_const, ens_cov, ens_fisher, ens_bias, ind_mse = find_optimal_constant(
        ensemble_predictions, bag_size, chw_grid, is_ensemble=True, tolerance=0.25, max_iterations=150, learning_rate=0.3
    )
    results_list.append({
        'Bag Size': bag_size,
        'Model Type': 'Ensemble',
        'CI Constant': ens_const,
        'Bias Shift': ens_bias,
        'Coverage (%)': ens_cov,
        'Mean Fisher Info': ens_fisher,
        'Fit MSE': ind_mse
    })
results_df = pd.DataFrame(results_list)
print("="*80)
print("AUTOMATED CALIBRATION SUMMARY")
print("="*80)
print(results_df.to_string(index=False, float_format="%.4f"))
print("="*80)

# %%

print("\n" + "="*80)
print("GENERATING DETAILED DATA FOR RESEARCH PAPER")
print("="*80)

paper_results_list = []
for bag_size, all_predictions_for_bag in all_bag_sizes_all_seed_predictions.items():
    print(f"--- Analyzing Bag Size: {bag_size} for Paper ---")
    ind_const = results_df[(results_df['Bag Size'] == bag_size) & (results_df['Model Type'] == 'Individual (Avg)')]['CI Constant'].values[0]
    ens_const = results_df[(results_df['Bag Size'] == bag_size) & (results_df['Model Type'] == 'Ensemble')]['CI Constant'].values[0]
    
    for model_idx, model_preds in enumerate(all_predictions_for_bag):
        # A. UNCALIBRATED (constant=1.0)
        uncal_cov, uncal_fisher, uncal_bias, uncal_mse = get_coverage_and_fisher(model_preds, bag_size, chw_grid, ci_constant=1.0)
        paper_results_list.append({
            'Bag Size': bag_size, 'Model Type': 'Individual', 'Model ID': model_idx + 1, 'Calibrated': 'No',
            'Coverage (%)': uncal_cov, 'Mean Fisher Info': np.mean(uncal_fisher), 'Bias Shift': uncal_bias, 'Fit MSE': uncal_mse
        })
        # B. CALIBRATED (using the constant found for the average of individuals)
        cal_cov, cal_fisher, cal_bias, cal_mse = get_coverage_and_fisher(model_preds, bag_size, chw_grid, ci_constant=ind_const)
        paper_results_list.append({
            'Bag Size': bag_size, 'Model Type': 'Individual', 'Model ID': model_idx + 1, 'Calibrated': 'Yes',
            'Coverage (%)': cal_cov, 'Mean Fisher Info': np.mean(cal_fisher), 'Bias Shift': cal_bias, 'Fit MSE': cal_mse
        })
    # Ensemble Models
    ensemble_predictions = np.mean(all_predictions_for_bag, axis=0)
    # A. UNCALIBRATED
    ens_uncal_cov, ens_uncal_fisher, ens_uncal_bias, ens_uncal_mse = get_coverage_and_fisher(ensemble_predictions, bag_size, chw_grid, ci_constant=1.0)
    paper_results_list.append({
        'Bag Size': bag_size, 'Model Type': 'Ensemble', 'Model ID': 'N/A', 'Calibrated': 'No',
        'Coverage (%)': ens_uncal_cov, 'Mean Fisher Info': np.mean(ens_uncal_fisher), 'Bias Shift': ens_uncal_bias, 'Fit MSE': ens_uncal_mse
    })
    # B. CALIBRATED
    ens_cal_cov, ens_cal_fisher, ens_cal_bias, ens_cal_mse = get_coverage_and_fisher(ensemble_predictions, bag_size, chw_grid, ci_constant=ens_const)
    paper_results_list.append({
        'Bag Size': bag_size, 'Model Type': 'Ensemble', 'Model ID': 'N/A', 'Calibrated': 'Yes',
        'Coverage (%)': ens_cal_cov, 'Mean Fisher Info': np.mean(ens_cal_fisher), 'Bias Shift': ens_cal_bias, 'Fit MSE': ens_cal_mse
    })

paper_df = pd.DataFrame(paper_results_list)
summary = paper_df.groupby(['Bag Size', 'Model Type', 'Calibrated']).agg(
    Mean_Coverage=('Coverage (%)', 'mean'),
    Std_Coverage=('Coverage (%)', 'std'),
    Mean_Fisher=('Mean Fisher Info', 'mean'),
    Std_Fisher=('Mean Fisher Info', 'std'),
    Mean_Bias=('Bias Shift', 'mean'),
    Mean_MSE=('Fit MSE', 'mean')
).reset_index()
summary = summary.fillna(0) # Std Dev is NaN for Ensemble, so fill with 0
summary['Coverage (mean±std)'] = summary.apply(lambda row: f"{row['Mean_Coverage']:.1f} ± {row['Std_Coverage']:.1f}", axis=1)
summary['Fisher (mean±std)'] = summary.apply(lambda row: f"{row['Mean_Fisher']:.1f} ± {row['Std_Fisher']:.1f}", axis=1)

print("\n\n" + "="*80)
print("SUMMARY TABLE FOR PAPER")
print("="*80)
print(summary[['Bag Size', 'Model Type', 'Calibrated', 'Coverage (mean±std)', 'Fisher (mean±std)', 'Mean_Bias', 'Mean_MSE']].to_string(index=False, float_format="%.3f"))


# %%

# print("\n\n" + "="*80)
# print("FULL DETAILED DATAFRAME")
# print("="*80)
# with pd.option_context('display.max_rows', None):
#     print(paper_df[paper_df['Bag Size'] == 1]) # Example: view full data for bag size 1

# %%

# paper_df.to_csv('param_detailed_calibration_results.csv', index=False)



# %%



# =========================================
# Plotting Fisher Information Reach
# =========================================
from scipy.optimize import curve_fit

results_list = []
individual_fisher_data = {}
ensemble_fisher_data = {}

for bag_size, all_predictions in all_bag_sizes_all_seed_predictions.items():
    print(f"--- Processing Bag Size: {bag_size} ---")
    # --- 1. Calibrate for Individual Models (Averaged) ---
    print("Searching for optimal constant for Individual Models...")
    ind_const, ind_cov, ind_fishers_list_of_lists, ind_bias, ind_mse = find_optimal_constant(
        all_predictions, bag_size, chw_grid, is_ensemble=False, tolerance=0.25
    )
    flat_ind_fishers = [item for sublist in ind_fishers_list_of_lists for item in sublist]
    individual_fisher_data[bag_size] = flat_ind_fishers
    results_list.append({
        'Bag Size': bag_size, 'Model Type': 'Individual (Avg)', 'CI Constant': ind_const,
        'Bias Shift': ind_bias, 'Coverage (%)': ind_cov, 'Mean Fisher Info': np.mean(flat_ind_fishers),
        'Fit MSE': ind_mse 
    })
    # --- 2. Calibrate for the Ensemble Model ---
    print("Searching for optimal constant for Ensemble Model...")
    ensemble_predictions = np.mean(all_predictions, axis=0)
    ens_const, ens_cov, ens_fishers_list, ens_bias, ens_mse = find_optimal_constant(
        ensemble_predictions, bag_size, chw_grid, is_ensemble=True, tolerance=0.25
    )
    ensemble_fisher_data[bag_size] = ens_fishers_list
    results_list.append({
        'Bag Size': bag_size, 'Model Type': 'Ensemble', 'CI Constant': ens_const,
        'Bias Shift': ens_bias, 'Coverage (%)': ens_cov, 'Mean Fisher Info': np.mean(ens_fishers_list),
        'Fit MSE': ens_mse
    })
    print("\n")



# %%
# =====================================================================================
# FISHER INFORMATION PLOTTING AND CURVE FITTING
# =====================================================================================

def asymptotic_fisher_model(N, I_true, C):
    return I_true / (1 + C / np.sqrt(N))


def plot_fisher_information(ax, data, color, marker, label):
    bag_sizes = np.array(sorted(data.keys()))
    mean_info = np.array([np.mean(data[bs]) for bs in bag_sizes])
    std_info = np.array([np.std(data[bs]) for bs in bag_sizes])
    try:
        initial_guess = [max(mean_info) * 1.1, 1.0]
        params, covariance = curve_fit(
            asymptotic_fisher_model, bag_sizes, mean_info,
            p0=initial_guess, sigma=std_info, absolute_sigma=True
        )
        I_true_fit, C_fit = params
        I_true_err, C_err = np.sqrt(np.diag(covariance))
        fit_successful = True
    except RuntimeError:
        print(f"Warning: Curve fit failed for '{label}'.")
        fit_successful = False
    ax.errorbar(
        bag_sizes, mean_info, yerr=std_info, fmt=marker, color=color,
        ecolor=color, alpha=0.6, elinewidth=3, capsize=5, markersize=8,
        markeredgecolor='black', label=f'{label} (±1σ)'
    )
    
    return I_true_fit if fit_successful else None

# %%
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8, 6))

I_true_ind = plot_fisher_information(ax, individual_fisher_data, 'royalblue', 'o', 'Individual Models')
I_true_ens = plot_fisher_information(ax, ensemble_fisher_data, 'firebrick', 's', 'Ensemble Model')

ax.set_xscale('log')
ax.set_xlabel('Bag Size ($N_B$)', fontsize=14)
ax.set_ylabel('Calibrated Fisher Information', fontsize=14)
ax.set_title('Fisher Information vs. Bag Size', fontsize=18, fontweight='bold')
ax.legend(fontsize=12, loc='best')
ax.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()


# %%

paper_table_values = pd.DataFrame(results_list)

# %%

# paper_table_values.to_csv('param_table_values.csv', index=False)


# %%



















# %%


ensamble_pred_list = []
for bag_size, all_predictions in all_bag_sizes_all_seed_predictions.items():
    print(f"--- Processing Bag Size: {bag_size} ---")
    ensemble_predictions = np.mean(all_predictions, axis=0)
    ensamble_pred_list.append(ensemble_predictions)


#%%


# ========================================
# Individual bag prediction
# ========================================

for pred_over_chw_grid_np in ensamble_pred_list:
    print(f'Working on data shape: {pred_over_chw_grid_np.shape}\n\n\n')
    plot_images = []
    for j in range(3):
        num_lines_to_plot = 20 
        collected_bag_predictions = []
        next_index_for_showing = 20*j
        for i in range(num_lines_to_plot):
            current_plot_index = i + next_index_for_showing 
            if current_plot_index >= pred_over_chw_grid_np.shape[1]: # pred_over_chw_grid_np has shape (models, chw, bags)
                print(f"Warning: Requested plot index {current_plot_index} is out of bounds for available bags ({pred_over_chw_grid_np.shape[1]}). Stopping.")
                break
            single_bag_prediction_data = pred_over_chw_grid_np[:, current_plot_index]
            collected_bag_predictions.append(single_bag_prediction_data)
        
        plt.figure(figsize=(8, 6)) 
        x_interp = np.linspace(chw_grid.min(), chw_grid.max(), 1000)
        colors = plt.cm.viridis(np.linspace(0, 1, len(collected_bag_predictions)))
        for idx, bag_pred_squeezed in enumerate(collected_bag_predictions):
            if bag_pred_squeezed.ndim == 0 or bag_pred_squeezed.size != len(chw_grid):
                print(f"Skipping bag {idx+1} due to unexpected shape: {bag_pred_squeezed.shape}")
                continue
            y_interp_bag = np.interp(x_interp, chw_grid, bag_pred_squeezed)
            plt.plot(x_interp, y_interp_bag, linestyle='-', color=colors[idx], alpha=0.6, label=f'Bag {idx + 1 + next_index_for_showing}')
        if collected_bag_predictions:
            avg_of_collected_bags = np.mean(np.array(collected_bag_predictions), axis=0)
        
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Threshold (0.5)')
        plt.axvline(x=0, color='darkgreen', linestyle='--', alpha=0.7, label='SM ($c_{HW}=0$)')
        plt.xlabel('$c_{HW}$ Value', fontsize=14)
        plt.ylabel('Probability Prediction', fontsize=14)
        plt.title(f'{len(collected_bag_predictions)} Individual Bag Predictions vs $c_{{HW}}$', fontsize=16)
        plt.grid(True, alpha=0.3)
        padding = (chw_grid.max() - chw_grid.min()) * 0.05 if len(chw_grid) > 1 else 0.1
        plt.xlim(chw_grid.min() - padding, chw_grid.max() + padding)
        plt.ylim(-0.05, 1.05)
        plt.tight_layout(rect=[0, 0.05, 1, 1] if len(collected_bag_predictions) > 10 else None)
        plt.show()
        buf = io.BytesIO()
        buf.seek(0)
        plot_images.append(Image.open(buf))    
        plt.clf() 
        plt.close()




# %%

# ========================================
# Average prediction boxplot
# ========================================

for i, pred_over_chw_grid in enumerate(ensamble_pred_list):
    print(f'Working on data shape: {pred_over_chw_grid.shape} for bag size {BAG_SIZE_EXPECTED[i]}\n\n\n')
    pred_over_chw_grid_np = np.array(pred_over_chw_grid)
    avg_predictions = np.mean(pred_over_chw_grid_np, axis=1)
    std_predictions = np.std(pred_over_chw_grid_np, axis=1)
    
    plt.figure(figsize=(12, 8))
    plt.boxplot(pred_over_chw_grid_np.T,
                positions=chw_grid,
                widths=(chw_grid[1] - chw_grid[0]) * 0.6,
                showfliers=False,
                patch_artist=True,
                boxprops=dict(facecolor='lightgray', alpha=0.5))
    x_interp = np.linspace(chw_grid.min(), chw_grid.max(), 1000)
    y_interp = np.interp(x_interp, chw_grid, avg_predictions)
    
    plt.plot(x_interp, y_interp, 'r-', label='Interpolated Line', linewidth=2)
    plt.scatter(chw_grid, avg_predictions, color='blue', s=30, label='Data Points')
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Threshold (0.5)')
    plt.axvline(x=0, color='green', linestyle='--', alpha=0.7, label='SM ($c_{HW}=0$)')
    plt.xlabel('$c_{HW}$ Value', fontsize=15)
    plt.ylabel('Probability Prediction', fontsize=15)
    plt.title('Multi-class Classifier Probabilities vs $c_{HW}$', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend()

    padding = (chw_grid.max() - chw_grid.min()) * 0.05  # 5% padding
    plt.xlim(chw_grid.min() - padding, chw_grid.max() + padding)
    plt.ylim(max(0, avg_predictions.min() - 0.1), min(1, avg_predictions.max() + 0.1))
    plt.tight_layout()
    plt.show()
    plt.close()
    



# %%

# 50 confidence intervals plot

for i, pred_over_chw_grid_np in enumerate(ensamble_pred_list):
    cfg.bag_size = BAG_SIZE_EXPECTED[i]
    cfg.batch_size = 8 * (10_000 // cfg.bag_size)
    print(f'Working on data shape: {pred_over_chw_grid_np.shape} for bag size {cfg.bag_size}\n\n\n')

    CI_CONSTANT = results_df[(results_df['Bag Size'] == cfg.bag_size) & (results_df['Model Type'] == 'Ensemble')]['CI Constant'].values[0]
    BIAS_SHIFT = results_df[(results_df['Bag Size'] == cfg.bag_size) & (results_df['Model Type'] == 'Ensemble')]['Bias Shift'].values[0]
    if cfg.bag_size == 1:
        FIT_WINDOW_HALF_SIZE = 7
    else:
        FIT_WINDOW_HALF_SIZE = 4
    test_bag_num = 200_000// cfg.bag_size 
    EPSILON = 1e-15
    num_bags_per_chunk = 1000 // cfg.bag_size  # Number of bags for 1000 events
    total_bags_available = test_bag_num
    num_chunks = total_bags_available // num_bags_per_chunk
    print(f"Available bags: {total_bags_available}, Bags per chunk: {num_bags_per_chunk}, Total chunks: {num_chunks}")
    
    individual_model_cis_all_chunks = []
    y_labels_part1 = []
    true_value = 0.0  # SM hypothesis
    coverage_results = []
    model_chunk_cis = []
    coverage_count = 0
    fisher_info_values = []
    
    for chunk_idx in range(num_chunks):
        start_bag = chunk_idx * num_bags_per_chunk
        end_bag = start_bag + num_bags_per_chunk
        predictions_for_log_calc = pred_over_chw_grid_np[:, start_bag:end_bag]
        log_prob_over_chw_grid = np.log(predictions_for_log_calc / (1 - predictions_for_log_calc + EPSILON))
        # log_prob_over_chw_grid = np.log(predictions_for_log_calc / (predictions_for_log_calc[10] + EPSILON))
        sum_log_prob = np.sum(log_prob_over_chw_grid, axis=1)*CI_CONSTANT
        mu, ci_low, ci_high, a_fit, mse = calculate_ci_from_log_likelihood(chw_grid, sum_log_prob, fit_window_half_size=FIT_WINDOW_HALF_SIZE)
        ci_low, ci_high = ci_low - BIAS_SHIFT, ci_high - BIAS_SHIFT
        if a_fit < 0:
            fisher_info = -2 * a_fit
            fisher_info_values.append(fisher_info)

        contains_true = not (np.isnan(ci_low) or np.isnan(ci_high)) and ci_low <= true_value <= ci_high
        if contains_true:
            coverage_count += 1
        model_chunk_cis.append((chunk_idx, ci_low, ci_high, contains_true))
        y_coord = 1 * num_chunks + chunk_idx  
        individual_model_cis_all_chunks.append((y_coord, ci_low, ci_high))
        y_labels_part1.append(f"Chunk{chunk_idx+1}")
    coverage_percentage = (coverage_count / num_chunks) * 100 if num_chunks > 0 else 0
    coverage_results.append((1, coverage_count, num_chunks, coverage_percentage))
    print(f"  Model {1} (Seed {1}): {coverage_count}/{num_chunks} CIs contain true value ({coverage_percentage:.1f}%)")
    for chunk_idx, ci_low, ci_high, contains_true in model_chunk_cis[:3]:  # Show first 3 chunks
        status = "✓" if contains_true else "✗"
        print(f"    Chunk {chunk_idx+1}: CI=[{ci_low:.3f}, {ci_high:.3f}] {status}")
    if len(model_chunk_cis) > 3:
        print(f"    ... and {len(model_chunk_cis)-3} more chunks")
    gc.collect()

    mean_fisher_info_text = ""
    if fisher_info_values:
        mean_fisher_info = np.mean(fisher_info_values)
        mean_fisher_info_text = f"Mean Fisher Information: {mean_fisher_info:.2f}"

    if individual_model_cis_all_chunks:
        plot_confidence_intervals_horizontal(individual_model_cis_all_chunks[:50], y_labels_part1[:50], 
                                             f'$1\sigma$ Confidence Invervals for 1000 event data chunks',
                                             x_lim=(-0.3, 0.3),
                                             info_text=mean_fisher_info_text,
                                             bag_size=cfg.bag_size)
    else:
        print("No CI data to plot for Part 1.")

    print("\n" + "="*80)
    print("COVERAGE ANALYSIS SUMMARY")
    print("="*80)
    print(f"Expected coverage for 1-sigma CIs: ~68.3%")
    print(f"Number of chunks per model: {num_chunks}")
    print(f"Events per chunk: 1000")
    print("-"*80)
    total_cis = 0
    total_coverage = 0
    for seed, coverage_count, total_chunks, coverage_percentage in coverage_results:
        print(f"Seed {seed:2d}: {coverage_count:2d}/{total_chunks} CIs contain true value ({coverage_percentage:5.1f}%)")
        total_cis += total_chunks
        total_coverage += coverage_count
    overall_coverage = (total_coverage / total_cis) * 100 if total_cis > 0 else 0
    coverage_difference = overall_coverage - 68.3
    print("-"*80)
    print(f"OVERALL: {total_coverage}/{total_cis} CIs contain true value ({overall_coverage:.1f}%)")
    print(f"Difference from expected 68.3%: {overall_coverage - 68.3:+.1f} percentage points")
    plt.figure(figsize=(6, 6))
    seeds = [result[0] for result in coverage_results]
    coverages = [result[3] for result in coverage_results]

    print("="*80)




# %%


# --- VIOLIN PLOT---
ensemble_peak_distributions = []
bag_size_labels = []
sorted_bag_sizes = sorted(all_bag_sizes_all_seed_predictions.keys())

for bag_size in sorted_bag_sizes:
    print(f"Processing Bag Size: {bag_size} for violin plot...")
    all_predictions = all_bag_sizes_all_seed_predictions[bag_size]
    ensemble_predictions = np.mean(all_predictions, axis=0)
    max_chw_indices = np.argmax(ensemble_predictions, axis=0)
    peak_chw_values = chw_grid[max_chw_indices]
    ensemble_peak_distributions.append(peak_chw_values)
    bag_size_labels.append(bag_size)

print("Generating the violin plot...")
fig, ax = plt.subplots(figsize=(12, 9))
parts = ax.violinplot(
    dataset=ensemble_peak_distributions,
    showmeans=True,
    showmedians=True,
    showextrema=True
)
for pc in parts['bodies']:
    pc.set_facecolor('lightgreen')
    pc.set_edgecolor('black')
    pc.set_alpha(0.8)
parts['cmeans'].set_color('red')
parts['cmedians'].set_color('black')
parts['cmins'].set_color('gray')
parts['cmaxes'].set_color('gray')
parts['cbars'].set_color('gray')

ax.set_title('Distribution of $c_{HW}$ Values Where The Probability Is Peaked', fontsize=18)
ax.set_xlabel('Bag Size', fontsize=17)
ax.set_ylabel('$c_{HW}$ Values', fontsize=17)
ax.set_xticks(np.arange(1, len(bag_size_labels) + 1))
ax.set_xticklabels(bag_size_labels)
ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
ax.grid(True, linestyle='--', alpha=0.6, axis='y')
plt.tight_layout()
plt.show()

# %%




# %%

for k, ensamble_predictions in enumerate(ensamble_pred_list):
    print(f'Working on data shape: {ensamble_predictions.shape} for bag size {BAG_SIZE_EXPECTED[k]}\n\n\n')
    cfg.bag_size = BAG_SIZE_EXPECTED[k]
    cfg.batch_size = 8 * (10_000 // cfg.bag_size)
    
    if cfg.bag_size == 1:
        CI_fit_window_half_size = 4
    else:
        CI_fit_window_half_size = 4
    
    THE_CI_CONSTANT = results_df[(results_df['Bag Size'] == cfg.bag_size) & (results_df['Model Type'] == 'Ensemble')]['CI Constant'].values[0]
    BIAS_SHIFT = 0.0 

# for j in range(1):
    pred_over_chw_grid_np = ensamble_predictions
    for i in range(5):
        index_for_next_plot = i
        epsilon = 1e-15
        num_of_bags_to_calculate = 1000//cfg.bag_size
        predictions_for_log_calculations = pred_over_chw_grid_np[:, index_for_next_plot*num_of_bags_to_calculate:(index_for_next_plot+1)*num_of_bags_to_calculate]
        
        # log_prob_over_chw_grid = np.log(predictions_for_log_calculations/(predictions_for_log_calculations[10] + epsilon))
        log_prob_over_chw_grid = np.log(predictions_for_log_calculations/(1- predictions_for_log_calculations + epsilon))
        
        sum_log_prob_over_chw_grid = np.sum(log_prob_over_chw_grid, axis=1)*THE_CI_CONSTANT
        sum_log_prob_over_chw_grid.shape
        max_ll_idx = np.argmax(sum_log_prob_over_chw_grid)
        fit_window_half_size = CI_fit_window_half_size
        
        start_idx = max(0, max_ll_idx - fit_window_half_size)
        end_idx = min(len(chw_grid) - 1, max_ll_idx + fit_window_half_size)
        num_points_in_window = end_idx - start_idx + 1
        desired_num_points = (2 * fit_window_half_size) + 1
        if num_points_in_window < desired_num_points:
            if start_idx == 0 and end_idx < len(chw_grid) -1 : # Max is near the beginning
                end_idx = min(len(chw_grid) - 1, start_idx + desired_num_points - 1)
            elif end_idx == len(chw_grid) - 1 and start_idx > 0: # Max is near the end
                start_idx = max(0, end_idx - desired_num_points + 1)
        
        # Final selection of data for fitting
        chw_fit_data = chw_grid[start_idx : end_idx + 1]
        sum_log_prob_fit_data = sum_log_prob_over_chw_grid[start_idx : end_idx + 1]
        print(f"Fitting parabola to cHW values: {chw_fit_data}")
        if len(chw_fit_data) < 3:
            print("Warning: Not enough points around the maximum to perform a robust parabolic fit. Fitting over the entire range instead.")
            chw_fit_data = chw_grid
            sum_log_prob_fit_data = sum_log_prob_over_chw_grid
        elif len(chw_fit_data) < desired_num_points:
            print(f"Warning: Fit window truncated to {len(chw_fit_data)} points due to proximity of maximum to data boundary.")
        coeffs = np.polyfit(chw_fit_data, sum_log_prob_fit_data, 2)
        p2, p1, p0 = coeffs
        a_fit = p2  # This is the 'a' in a(x-mu)^2+K, determining curvature
        mu_fit = -p1 / (2 * p2) # MLE for c_HW
        K_fit = np.polyval(coeffs, mu_fit) # Max value of the fitted log-likelihood proxy
        
        fitted_y_values = np.polyval(coeffs, chw_fit_data)
        mse_fit = np.mean((sum_log_prob_fit_data - fitted_y_values)**2)
        
        print(f"Parabolic fit coefficients: p2(a)={a_fit:.3f}, p1={p1:.3f}, p0={p0:.3f}")
        print(f"Derived fit parameters: mu_hat={mu_fit:.3f}, K_max={K_fit:.3f}")
        print(f"MSE of fit (on fit range): {mse_fit:.3f}")
        # fit_margin = (chw_fit_data[-1] - chw_fit_data[0]) * 0.1
        fit_margin = (chw_fit_data[-1] - chw_fit_data[0]) * 0
        fit_plot_min = chw_fit_data[0] - fit_margin
        fit_plot_max = chw_fit_data[-1] + fit_margin
        fit_plot_grid = np.linspace(fit_plot_min, fit_plot_max, 50)
        fit_plot_curve = np.polyval(coeffs, fit_plot_grid)
        w1_sigma, w2_sigma = np.nan, np.nan
        ci_1sigma_lower, ci_1sigma_upper = np.nan, np.nan
        ci_2sigma_lower, ci_2sigma_upper = np.nan, np.nan
        observed_fisher_information = np.nan
        xlim_min, xlim_max = chw_grid.min(), chw_grid.max()
        if a_fit < 0:
            observed_fisher_information = -2 * a_fit
            print(f"Observed Fisher Information J(mu_fit) = {observed_fisher_information:.3f}")
            delta_LL_1sigma = 0.5  # For 1-sigma (2*DeltaLL = 1)
            delta_LL_2sigma = 2.0  # For 2-sigma (2*DeltaLL = 4)
            
            w1_sigma = np.sqrt(-delta_LL_1sigma / a_fit)
            w2_sigma = np.sqrt(-delta_LL_2sigma / a_fit)
            ci_1sigma_lower, ci_1sigma_upper = mu_fit - w1_sigma, mu_fit + w1_sigma
            ci_2sigma_lower, ci_2sigma_upper = mu_fit - w2_sigma, mu_fit + w2_sigma

            ci_1sigma_lower -= BIAS_SHIFT
            ci_1sigma_upper -= BIAS_SHIFT
            ci_2sigma_lower -= BIAS_SHIFT
            ci_2sigma_upper -= BIAS_SHIFT
            
            print(f"1-sigma CI (Corrected): [{ci_1sigma_lower:.3f}, {ci_1sigma_upper:.3f}] (width_half={w1_sigma:.3f})")
            print(f"2-sigma CI (Corrected): [{ci_2sigma_lower:.3f}, {ci_2sigma_upper:.3f}] (width_half={w2_sigma:.3f})")
            
            # Adjust plot limits dynamically based on 2-sigma CI
            plot_margin_factor = 0.15 * (w2_sigma if not np.isnan(w2_sigma) and w2_sigma > 1e-6 else abs(mu_fit)*0.2+0.2)
            xlim_min_data = chw_grid.min()
            xlim_max_data = chw_grid.max()
            xlim_min = min(xlim_min_data, ci_2sigma_lower - plot_margin_factor if not np.isnan(ci_2sigma_lower) else xlim_min_data)
            xlim_max = max(xlim_max_data, ci_2sigma_upper + plot_margin_factor if not np.isnan(ci_2sigma_upper) else xlim_max_data)
        else:
            print("Warning: Coefficient a_fit (p2) is not negative. Parabola is not concave down. Cannot calculate CIs or Fisher Info as expected for a maximum.")
        #
        # --- Plotting: Log-Likelihood Fit ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig_ll_fit, ax_ll_fit = plt.subplots(figsize=(12, 8))

        y_data_plot = -2 * (sum_log_prob_over_chw_grid - K_fit)
        y_fit_points_plot = -2 * (sum_log_prob_fit_data - K_fit)
        y_fit_curve_plot = -2 * (fit_plot_curve - K_fit)
        ax_ll_fit.plot(chw_grid, y_data_plot, 'o', color='royalblue', ms=6, label='Data ($-2\\Delta\\ln L$)')
        ax_ll_fit.plot(chw_fit_data, y_fit_points_plot, 'o', color='green', ms=8,
                      label='Fit Points', alpha=0.7)
        ax_ll_fit.plot(fit_plot_grid, y_fit_curve_plot, '-', color='crimson', lw=2.5,
                      label=f'Parabolic Fit (MSE: {mse_fit:.3f})')
        ax_ll_fit.axvline(mu_fit, color='black', linestyle=':', lw=1.5,
                         label=f'Est. $c_{{HW}} (\\hat{{\\mu}}) = {mu_fit:.3f}$')
        if a_fit < 0:  # a_fit is from the original LL fit, so it should be negative for a maximum
            ax_ll_fit.axvspan(ci_1sigma_lower, ci_1sigma_upper, alpha=0.2, color='orange',
                          label=f'1σ CI ($\\pm {w1_sigma:.3f}$)')
            ax_ll_fit.axvspan(ci_2sigma_lower, ci_2sigma_upper, alpha=0.2, color='skyblue',
                          label=f'2σ CI ($\\pm {w2_sigma:.3f}$)')
            textstr = '\n'.join([
                f'Fisher Information: {observed_fisher_information:.3f}',
                f'1σ CI: [{ci_1sigma_lower:.3f}, {ci_1sigma_upper:.3f}]',
                f'2σ CI: [{ci_2sigma_lower:.3f}, {ci_2sigma_upper:.3f}]'
            ])
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax_ll_fit.text(0.05, 0.95, textstr, transform=ax_ll_fit.transAxes, fontsize=15,
                    verticalalignment='top', bbox=props)
        plot_bag_size = cfg.bag_size
        ax_ll_fit.set_xlabel('$c_{HW}$ Value', fontsize=18)
        ax_ll_fit.set_ylabel('$-2\\Delta\\ln L$', fontsize=18) # Updated Y-axis label
        ax_ll_fit.set_title(f'Profile Likelihood Scan: Bag Size = {plot_bag_size}', fontsize=20)
        ax_ll_fit.legend(loc='upper right', fontsize=16)
        ax_ll_fit.grid(True, linestyle='--', alpha=0.6)
        ax_ll_fit.set_xlim(xlim_min, xlim_max) # Apply dynamic xlim
        #
        plt.tight_layout()
        # plt.savefig(f'0_param_calib_LLR_bag_{cfg.bag_size}_{index_for_next_plot}.png', bbox_inches='tight')  # Tight bounding box
        plt.show()
  


# %%





num_of_features = len(columns_to_keep)
print("Num of columns_to_keep", num_of_features)
df = pd.read_parquet(SOME_PATH 
                     engine='pyarrow', 
                     columns=columns_to_keep)
sm_data = df[df['cHW'] == 0].copy()
# %%
# 
all_chw_vals = np.sort(df['cHW'].unique())
# %%
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
#
temp_chw_p1_data = shuffle(true_match_train[chw_0_index+1].copy().reshape(-1, num_of_features), random_state=84)
temp_chw_p1_data = temp_chw_p1_data.reshape(train_bag_num, cfg.bag_size, num_of_features)[:int(train_bag_num*0.3),:,:]
temp_chw_p2_data = shuffle(true_match_train[chw_0_index+2].copy().reshape(-1, num_of_features), random_state=84)
temp_chw_p2_data = temp_chw_p2_data.reshape(train_bag_num, cfg.bag_size, num_of_features)[:int(train_bag_num*0.2),:,:]
#
sm_duplicate_train[chw_0_index] = np.concatenate((temp_chw_n2_data, temp_chw_n1_data, temp_chw_p1_data, temp_chw_p2_data), axis=0)
sm_duplicate_train[chw_0_index][:,:, -1] = 0  # Set cHW to 0 for SM bags
#
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
#
# %%
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

chw_to_plot = [-1.0, -0.5, 0.0, 0.5, 1.0]
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(chw_to_plot)))


roc_prediction_results = {}
full_df = pd.read_parquet(SOME_PATH, 
                          engine='pyarrow', 
                          columns=columns_to_keep)
all_chw_vals = np.sort(full_df['cHW'].unique())
chw_0_index = np.where(all_chw_vals == 0)[0][0]
chw_to_evaluate = [-1.0, -0.5, 0.0, 0.5, 1.0]

for bag_size in BAG_SIZE_EXPECTED:
    print(f"\n--- Generating predictions for Bag Size: {bag_size} ---")
    cfg.bag_size = bag_size
    num_of_features = len(columns_to_keep)
    model_files = sorted(glob.glob(os.path.join(MODEL_DIR, f'Bag{bag_size}_Seed*.keras')))
    if not model_files:
        print(f"Warning: No models found for bag size {bag_size}. Skipping.")
        continue
    models = [k.models.load_model(p, compile=False) for p in model_files]
    print(f"Loaded {len(models)} models for ensemble.")
    temp_results_for_bag_size = {}
    
    for chw_val in chw_to_evaluate:
        print(f"  Generating test data for cHW hypothesis = {chw_val:.1f}")
        true_match_data = full_df[full_df['cHW'] == chw_val].copy()
        num_test_samples = int(len(true_match_data) * cfg.test_split)
        num_of_test_bags = num_test_samples // cfg.bag_size
        if num_of_test_bags == 0:
            print(f"    Skipping cHW={chw_val}, not enough test data for bag size {bag_size}.")
            continue
        X_true_match = true_match_data.tail(num_of_test_bags * cfg.bag_size).to_numpy().reshape(num_of_test_bags, cfg.bag_size, num_of_features)
        y_true_match = np.ones(num_of_test_bags)

        if chw_val == 0.0:
            test_temp_chw_n2 = full_df[full_df['cHW'] == all_chw_vals[chw_0_index-2]].tail(int(num_of_test_bags * 0.2) * cfg.bag_size).to_numpy()
            test_temp_chw_n1 = full_df[full_df['cHW'] == all_chw_vals[chw_0_index-1]].tail(int(num_of_test_bags * 0.3) * cfg.bag_size).to_numpy()
            test_temp_chw_p1 = full_df[full_df['cHW'] == all_chw_vals[chw_0_index+1]].tail(int(num_of_test_bags * 0.3) * cfg.bag_size).to_numpy()
            test_temp_chw_p2 = full_df[full_df['cHW'] == all_chw_vals[chw_0_index+2]].tail(int(num_of_test_bags * 0.2) * cfg.bag_size).to_numpy()
            
            X_fake_kin_combined = np.concatenate((test_temp_chw_n2, test_temp_chw_n1, test_temp_chw_p1, test_temp_chw_p2), axis=0)
            X_fake_kin_combined = shuffle(X_fake_kin_combined, random_state=42) # Shuffle to mix them up
            
            num_fake_bags = len(X_fake_kin_combined) // cfg.bag_size
            X_fake_kin = X_fake_kin_combined[:num_fake_bags * cfg.bag_size].reshape(num_fake_bags, cfg.bag_size, num_of_features)
            X_fake_kin[:, :, -1] = 0.0 # Set hypothesis to cHW=0
            y_fake_kin = np.zeros(num_fake_bags)

        else:
            sm_data = full_df[full_df['cHW'] == 0].copy()
            X_fake_kin = sm_data.tail(num_of_test_bags * cfg.bag_size).to_numpy().reshape(num_of_test_bags, cfg.bag_size, num_of_features)
            X_fake_kin[:, :, -1] = chw_val # Set the cHW hypothesis
            y_fake_kin = np.zeros(num_of_test_bags)

        X_test_roc = np.concatenate([X_true_match, X_fake_kin], axis=0)
        y_test_roc = np.concatenate([y_true_match, y_fake_kin], axis=0)

        all_model_preds = [model.predict(X_test_roc, batch_size=8 * (10_000 // cfg.bag_size)) for model in models]
        ensemble_predictions = np.mean(all_model_preds, axis=0)
        temp_results_for_bag_size[chw_val] = {'y_true': y_test_roc, 'y_pred': ensemble_predictions.squeeze()}
    roc_prediction_results[bag_size] = temp_results_for_bag_size
    
    del models
    k.backend.clear_session()
    gc.collect()

#%%

# import pickle
# save_path_roc = 'parameterized_roc_predictions.pkl'
# with open(save_path_roc, 'wb') as f:
#     pickle.dump(roc_prediction_results, f)

# del full_df
# gc.collect()

# %%


save_path_roc = 
with open(save_path_roc, 'rb') as f:
    roc_prediction_results = pickle.load(f)


# %%

i = 0

for bag_size, results_for_bag in roc_prediction_results.items():
    print(f"\n--- Plotting ROC Analysis for Bag Size: {bag_size} ---")
    plt.figure(figsize=(12, 9))
    plt.style.use('seaborn-v0_8-whitegrid')
    chw_to_plot = [chw_grid[len(chw_grid)//2], chw_grid[-len(chw_grid)//4], chw_grid[len(chw_grid)//4], chw_grid[0], chw_grid[-1]]
    colors = plt.cm.plasma(np.linspace(0, 1, len(chw_to_plot)))

    for idx, chw_val in enumerate(chw_to_plot):
        if chw_val not in results_for_bag:
            print(f"  No data found for cHW = {chw_val}. Skipping.")
            continue
        data = results_for_bag[chw_val]
        y_true = data['y_true']
        y_pred = data['y_pred']
        if len(np.unique(y_true)) < 2:
            print(f"  Skipping cHW={chw_val} for plotting, only one class present in y_true.")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[idx], lw=2.5,
                 label=f'$c_{{HW}} = {chw_val:+.1f}$ (AUC: {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--', label='Chance (AUC = 0.500)')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title(f'ROC Curves for Selected $c_{{HW}}$ Values (Bag Size: {bag_size})', fontsize=18)
    plt.legend(loc="lower right", fontsize=16, frameon=True, title="$c_{HW}$ Configurations")
    plt.tick_params(labelsize=12)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, which='both', linestyle=':', alpha=0.7)
    plt.tight_layout()
    # plt.savefig(f'0_param_roc_{BAG_SIZE_EXPECTED[i]}.png', bbox_inches='tight')  # Tight bounding box
    plt.show()
    plt.close()
    
    i += 1
# %%

