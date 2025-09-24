# MIL_for_HEP

> **Anonymous** code repository for the paper **"Increasing Information Extraction in Low-Signal Regimes via Multiple Instance Learning"**

---

## Overview

This repository contains the code used for the experiments in the submission. It is organized to allow reproduction of the training runs and of the figures reported in the paper while keeping the repository anonymous for double-blind review.

## Requirements

- **Python** 3.8+ (we recommend 3.10)
- `pip` and a virtual environment manager (venv, virtualenv, or conda)

Create and activate a virtual environment (example using `venv` on macOS/Linux):

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Repository structure

```
MIL_for_HEP/
├─ training/                 # Training scripts used to run experiments
├─ analysis_and_plots/       # Scripts to reproduce figures and tables
├─ src/                      # Helper functions, and YAML config files
├─ requirements.txt
└─ README.md
```
