# Dataset Generation and Preprocessing

This document details the pipeline used to generate the High-Energy Physics (HEP) collision data used in the paper. The pipeline consists of three main stages:

1.  **Monte Carlo Simulation**: Using MadGraph5\_aMC@NLO and SMEFTsim to generate raw events in the form of .root files.
2.  **Feature Extraction**: Using ROOT and ExRootAnalysis to calculate kinematic features and export to .csv files.
3.  **Preprocessing**: Using Python to clean, encode, aggregate, and compress the data into parquet file.

## 1\. Prerequisites and Installation

### Software Requirements

  * **MadGraph5\_aMC@NLO** (v3.6.2 recommended)
  * **ROOT** (Cern Root)
  * **Python** (with `pandas`, `uproot`, `pyarrow`)

### Installation Steps

1.  **Download MadGraph5:**

    ```bash
    wget https://launchpad.net/mg5amcnlo/3.0/3.6.x/+download/MG5_aMC_v3.6.2.tar.gz
    tar -xfz MG5_aMC_v3.6.2.tar.gz
    ```

2.  **Install SMEFTsim Models:**
    Download the UFO models from the [SMEFTsim repository](https://github.com/SMEFTsim/SMEFTsim) and copy them to the MadGraph models directory.

    ```bash
    cp -r SMEFTsim-main/UFO_models/* MG5_aMC_v3_6_2/models
    ```

3.  **Install ExRootAnalysis:**
    Launch the MadGraph console and install ExRootAnalysis.

    ```bash
    cd MG5_aMC_v3_6_2
    ./bin/mg5_aMC
    install ExRootAnalysis
    ```

    > **Note on Compilation:** Based on your version of Linux, you might have to fix the Makefile, i.e. in ExRootAnalysis/Makefile change:

    >   * `CXXFLAGS += $(ROOTCFLAGS) -Wno-write-strings -D_FILE_OFFSET_BITS=64 -DDROP_CGAL -I.` to `CXXFLAGS += $(ROOTCFLAGS) -Wno-write-strings -D_FILE_OFFSET_BITS=64 -DDROP_CGAL -I. -I/usr/include/tirpc`
    >   * `LIBS = $(ROOTLIBS)` to `LIBS = $(ROOTLIBS) -ltirpc`

    > Then compile standalone in ExRootAnalysis directory with `make`. 

-----

## 2\. Monte Carlo Event Generation

We generate two distinct sets of events:

1.  **Background Events:** Standard Model $ZZ$ production ($ZZ \to \ell\ell \nu\nu$) where the Wilson coefficient is zero ($c_{HW}=0$).
2.  **Signal Events:** Higgs production ($H \to WW \to \ell\nu\ell\nu$) where the Wilson coefficient $c_{HW}$ is varied across a specific range.

### 2.1. Background Event Generation


1.  **Generate Process Code:**
    Launch the MadGraph console and generate the directory for the background process.

    ```bash
    cd MG5_aMC_v3_6_2
    ./bin/mg5_aMC

    import model SMEFTsim_top_MwScheme_UFO-massless
    define vl = ve vm vt
    define vl~ = ve~ vm~ vt~
    generate u d > u d z z QCD=0 NP=0 NPcHW=0, (z > e+ e-), (z > vl vl~) QCD=0 NP=0 NPcHW=0

    # Output to a specific directory
    output SM_Background_ud_ZZ_SMEFT_NP0 # Or a similar descriptive name
    exit
    ```

2.  **Run Event Generation:**
    Configure the number of events, and launch the run.

    ```bash
    cd SM_Background_ud_ZZ_SMEFT_NP0

    # (Optional) Edit Cards/run_card.dat to set 'nevents' (e.g., 1000000)

    # Launch generation
    ./bin/generate_events
    ```



### 2.2. Signal Event Generation

We first generate the process code, and then use a shell script to automatically generate events for various values of $c_{HW}$.

1.  **Generate Process Code:**
    Launch MadGraph to create the signal process directory.

    ```bash
    cd MG5_aMC_v3_6_2
    ./bin/mg5_aMC

    import model SMEFTsim_top_MwScheme_UFO-massless

    generate u d > u d h $$ w+ w- / z a QCD=0 NP=1 NPcHW=1, h > e+ ve e- ve~ / z QCD=0 NP=1 NPcHW=1

    output cHW-prod
    exit
    ```

2.  **Automated Mass Production:**
    We use the `generate_signal_events_root_file.sh` script to automate the parameter scan. This script iterates through a list of $c_{HW}$ values, modifies the `Cards/param_card.dat`, and triggers event generation for each value.

      * **Preparation:**  Change `nevents` parameter to desired number of events in the `Cards/run_card.dat` file. (Double check all parameters except $c_{HW}$ is 0 in `param_card.dat`)


    ```bash
    cd cHW-prod

    # Run the automation script
    ./generate_signal_events_root_file.sh
    ```

    **What it does:**

      * Iterates through $c_{HW}$ values (e.g., from -10.0 to 10.0).
      * Updates the `chw` block in `param_card.dat`.
      * Executes `./bin/generate_events` using multi-core processing.
      * Produces distinct ROOT files for each parameter value.

-----

## 3\. Feature Extraction (ROOT to CSV)

Raw ROOT files are processed to extract low-level 4-vector data and calculate high-level engineered features (e.g., $m_{\ell\ell}$, $\Delta\phi_{jj}$, $MET$). This is can be done interactively within the ROOT shell.

* **Configure:** Edit `root_to_csv.C` file to insert paths to the input ROOT files and output CSV filenames.


```bash
# Launch ROOT
root -l
# 1. Execute rootlogon to load ExRootAnalysis libraries
root [0] .x rootlogon.C
# 2. Load the analysis script (defines the 'ana' function)
root [1] .L root_to_csv.C
# 3. Run the main analysis function
root [2] ana()
```

-----

## 4\. Preprocessing and Compression

The final step converts the individual CSV files into a single, machine-learning-ready Parquet dataset.

* **Configure:** Edit `csv_preprocessing_and_to_parquet.py` file to insert necessary file paths.

```bash
# Run the script
python csv_preprocessing_and_to_parquet.py
```
