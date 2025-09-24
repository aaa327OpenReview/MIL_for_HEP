# %%
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


pd.set_option('display.max_columns', None)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "True"

#%%

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("GPU:", gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

# %%


# %%

df = pd.read_parquet(SOME_PATH, 
                     engine='pyarrow', 
                    )

# %%

CHW_VALS_TO_PLOT = np.array([0, 0.1, 5, 10])

features_to_plot = [
    'l0_eta',
    'q0_eta',
    'd_phi_qq',
    'd_phi_ll',
    # 'pt_qq',
    # 'pt_ll',
    # 'met_et',
    'q0_pt',
    'l0_pt',
    
]

feature_labels = {
    'l0_pt': r'$p_{T, \ell_0}$',
    'l1_pt': r'$p_{T, \ell_1}$',
    'q0_pt': r'$p_{T, q_0}$',
    'q1_pt': r'$p_{T, q_1}$',
    'l0_eta': r'$\eta_{\ell_0}$',
    'l1_eta': r'$\eta_{\ell_1}$',
    'q0_eta': r'$\eta_{q_0}$',
    'q1_eta': r'$\eta_{q_1}$',
    'met_et': r'$E_T^{\text{miss}}$',
    'met_phi': r'$\phi^{\text{miss}}$',
    'm_ll': r'$m_{\ell\ell}$',
    'm_qq': r'$m_{qq}$',
    'pt_ll': r'$p_{T, \ell\ell}$',
    'pt_qq': r'$p_{T, qq}$',
    'd_phi_ll': r'$\Delta\phi_{\ell\ell}$',
    'd_phi_qq': r'$\Delta\phi_{qq}$'
}

cHW_values = CHW_VALS_TO_PLOT



plt.style.use('seaborn-v0_8-whitegrid')

color_map = {
    0: 'black',  
    0.1: 'blue',
    5: 'red',
    10: 'green'
}

linestyle_map = {
    0: '-',
    0.1: '-',
    5: '-',
    10: '-'
}

fig, axes = plt.subplots(3, 2, figsize=(16, 22))
axes = axes.flatten()

for i, feature in enumerate(features_to_plot):
    ax = axes[i]
    sm_data = df[df['cHW'] == 0][feature]
    if sm_data.min() < 0:
        cutoff = np.percentile(np.abs(sm_data), 99.99) 
        ax.set_xlim(-cutoff, cutoff)
    else:
        cutoff = np.percentile(sm_data, 99.99)
        ax.set_xlim(0, cutoff)
    for chw in cHW_values:
        subset = df[df['cHW'] == chw]
        ax.hist(subset[feature], 
                 bins=50, 
                 histtype='step', 
                 linewidth=2.5,
                 density=True, 
                 label=f'$c_{{HW}} = {chw}$',
                 color=color_map.get(chw, 'gray'),      
                 linestyle=linestyle_map.get(chw, '-')) 

    pretty_label = feature_labels.get(feature, feature)
    ax.set_xlabel(pretty_label, fontsize=25)
    ax.tick_params(labelsize=16)
    
    if i % 2 == 0:
        ax.set_ylabel('Normalized Frequency', fontsize=25)

for i in range(len(features_to_plot), len(axes)):
    axes[i].axis('off')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right', title='$c_{HW}$ Value', fontsize=21, bbox_to_anchor=(0.99, 0.07))
fig.suptitle('Kinematic Distributions for Different $c_{HW}$ Values', fontsize=30, y=0.97)
plt.tight_layout(rect=[0, 0, 1, 0.97]) 
plt.show()
# %%
