# Evolutionary Search Based on Graph Neural Networks for Road-Closure Optimization

This repository contains the SUMO networks, experiment configurations, Python code, and execution scripts used for the experiments in the manuscript:

> **Evolutionary Search Based on Graph Neural Networks for Road-Closure Optimization**  
> Submitted to *Evolutionary Intelligence*.

The code implements a learning-guided evolutionary search framework for road-closure optimization in transportation networks. A Graph Neural Network (GNN) policy estimates edge-level closure-risk scores, and these scores are used to guide selected operations of a Genetic Algorithm (GA), while the GA remains the main stochastic search engine.

The current experiment workflow is driven by `experiments.json` and executed through `run_experiments.py`, so experiments can be run without manually editing scripts for each network, OD setting, demand level, guidance mode, or random seed.

---

## Overview

The repository supports the following experimental workflow:

1. Generate policy-training datasets with SUMO simulations.
2. Merge policy-dataset parts for each problem instance.
3. Train a GATv2-based policy GNN for each instance.
4. Run GA optimization under four guidance modes:
   - `none`: GA baseline without GNN guidance
   - `init`: GNN guidance during initialization only
   - `mutation`: GNN guidance during mutation only
   - `both`: GNN guidance during both initialization and mutation
5. Analyze the final best-so-far average travel time (ATT) across random seeds.

The experiments reported in the manuscript cover 39 problem instances derived from five road networks and multiple demand regimes.

---

## Repository structure

The repository should have the following layout:

```text
.
├── README.md
├── environment.yml             # Conda environment specification, if used
├── requirements.txt            # Optional pip requirements, if used instead of Conda
├── experiments.json            # Experiment definitions
├── run_experiments.py          # Main experiment runner
├── run_pipeline.ps1            # PowerShell wrapper around run_experiments.py
├── merge_policy_parts.py       # Merges generated policy-dataset parts
│
├── netA/                       # SUMO files for network A
│   ├── network_v2.net.xml
│   ├── sim_1od_x0.8.sumocfg
│   ├── sim_1od_x1.0.sumocfg
│   ├── sim_1od_x1.2.sumocfg
│   ├── sim_2od_x0.8.sumocfg
│   ├── sim_2od_x1.0.sumocfg
│   ├── sim_2od_x1.2.sumocfg
│   ├── sim_3od_x0.8.sumocfg
│   ├── sim_3od_x1.0.sumocfg
│   ├── sim_3od_x1.2.sumocfg
│   └── [route/additional files referenced by the .sumocfg files]
│
├── netB/                       # SUMO files for network B
│   ├── netB.net.xml
│   ├── sim_1od_x0.8.sumocfg
│   ├── sim_1od_x1.0.sumocfg
│   ├── sim_1od_x1.2.sumocfg
│   ├── sim_2od_x0.8.sumocfg
│   ├── sim_2od_x1.0.sumocfg
│   ├── sim_2od_x1.2.sumocfg
│   ├── sim_3od_x0.8.sumocfg
│   ├── sim_3od_x1.0.sumocfg
│   ├── sim_3od_x1.2.sumocfg
│   └── [route/additional files referenced by the .sumocfg files]
│
├── netC/                       # SUMO files for network C
│   ├── netC.net.xml
│   ├── sim_1od_x0.8.sumocfg
│   ├── sim_1od_x1.0.sumocfg
│   ├── sim_1od_x1.2.sumocfg
│   ├── sim_2od_x0.8.sumocfg
│   ├── sim_2od_x1.0.sumocfg
│   ├── sim_2od_x1.2.sumocfg
│   ├── sim_3od_x0.8.sumocfg
│   ├── sim_3od_x1.0.sumocfg
│   ├── sim_3od_x1.2.sumocfg
│   └── [route/additional files referenced by the .sumocfg files]
│
├── netD/                       # SUMO files for network D
│   ├── netD.net.xml
│   ├── sim_1od_x0.8.sumocfg
│   ├── sim_1od_x1.0.sumocfg
│   ├── sim_1od_x1.2.sumocfg
│   ├── sim_2od_x0.8.sumocfg
│   ├── sim_2od_x1.0.sumocfg
│   ├── sim_2od_x1.2.sumocfg
│   ├── sim_3od_x0.8.sumocfg
│   ├── sim_3od_x1.0.sumocfg
│   ├── sim_3od_x1.2.sumocfg
│   └── [route/additional files referenced by the .sumocfg files]
│
├── netJ/                       # SUMO files for network J / Wildau scenario
│   ├── Netzmodell2.net.xml
│   ├── wildau_6od_x0.8.sumocfg
│   ├── wildau_6od_x1.0.sumocfg
│   ├── wildau_6od_x1.2.sumocfg
│   └── [route/additional files referenced by the .sumocfg files]
│
├── py/
│   ├── make_policy_dataset.py
│   ├── train_policy_gnn.py
│   ├── ga_edge_closure_gnn_policy.py
│   ├── analyze_runs.py
│   └── [other Python modules imported by these scripts]
│
└── results/                    # Optional summarized results used in the manuscript
    └── README.md
```

Generated folders such as `output_policy_*` and `runs_*` are not required before running the pipeline. They are created during execution.

---

## Requirements

The experiments were developed and tested in a Windows-based SUMO workflow.

- Windows 10/11
- Python 3.x; the manuscript experiments used Python 3.13.9
- Conda or another Python environment manager
- Eclipse SUMO; the manuscript experiments used SUMO 1.24.0
- `sumo` executable available from the command line, or a valid full path to `sumo.exe`
- PyTorch; the manuscript experiments used PyTorch 2.5.1
- PyTorch Geometric; the manuscript experiments used PyTorch Geometric 2.7.0

GPU acceleration is not required. The experiments in the manuscript were conducted without GPU acceleration.

---

## Environment setup

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate psps
```

### Option 2: pip

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install SUMO separately and verify that it is available:

```bash
sumo --version
```

---

## Important configuration note

Before running the experiments on a new machine, check `experiments.json`.

The local development version may contain machine-specific paths such as:

```json
"root": "D:/PSPS",
"sumo_bin": "C:/PROGRA~2/Eclipse/Sumo/bin/sumo.exe"
```

For a portable public repository, these should be changed to values appropriate for the local checkout, for example:

```json
"root": ".",
"sumo_bin": "sumo"
```

Alternatively, keep `sumo_bin` as the full path to `sumo.exe` if SUMO is not available on `PATH`.

---

## Experiment configuration

All experiments are defined in `experiments.json`. Each experiment specifies:

- experiment name
- SUMO configuration file (`sumocfg`)
- SUMO network file (`net`)
- minimum and maximum number of closed edges (`kmin`, `kmax`)
- output folder for generated policy data (`out_policy`)
- trained policy model path (`policy_model`)
- GA results folder (`runs_root`)
- GA guidance modes: `none`, `init`, `mutation`, and `both`
- random seeds 0--9
- population size, generation count, and parallel job count
- demand size
- protected edges
- dataset, SUMO, and training seeds

The current configuration contains 39 experiment definitions:

```text
Network A: 9 experiments  (1OD/2OD/3OD × demand multipliers 0.8/1.0/1.2, kmax = 4)
Network B: 9 experiments  (1OD/2OD/3OD × demand multipliers 0.8/1.0/1.2, kmax = 9)
Network C: 9 experiments  (1OD/2OD/3OD × demand multipliers 0.8/1.0/1.2, kmax = 9)
Network D: 9 experiments  (1OD/2OD/3OD × demand multipliers 0.8/1.0/1.2, kmax = 9)
Network J: 3 experiments  (6OD × demand multipliers 0.8/1.0/1.2, kmax = 9)
```

Example experiment names:

```text
A_k4_2od_x1.0
B_k9_3od_x1.2
C_k9_2od_x0.8
D_k9_1od_x1.0
J_k9_6od_x1.0
```

---

## Main experimental settings

The manuscript experiments use the following settings:

```text
Population size: 24
Number of generations: 15
Random seeds per mode: 10, using seeds 0--9
Guidance modes: none, init, mutation, both
Crossover: uniform crossover
Crossover rate: 0.9
Mutation: bit-flip mutation with separated closing/opening moves
Mutation rate: 0.1
Maximum closure budget: floor(|E_cand| / 10)
Objective: average travel time (ATT) from SUMO tripinfo output
Penalty value for invalid simulations: 1e12
Parallel SUMO evaluations: up to 8 processes
```

The policy dataset for each instance contains 1,000 random closure combinations, generated as 8 parts with 125 samples per part. The policy GNN is trained once per instance and then used to guide GA initialization and/or mutation depending on the selected guidance mode.

---

## Running the experiments

### Full pipeline

From a PowerShell prompt in the repository root:

```powershell
conda activate psps
.\run_pipeline.ps1 -Name A_k4_2od_x1.0
```

This runs the default pipeline:

```text
dataset -> merge -> train -> ga -> analyze
```

The same run can also be launched directly with Python:

```bash
python run_experiments.py --config experiments.json --name A_k4_2od_x1.0
```

### Dry run

To print the commands without executing them:

```powershell
.\run_pipeline.ps1 -Name A_k4_2od_x1.0 -DryRun
```

or:

```bash
python run_experiments.py --config experiments.json --name A_k4_2od_x1.0 --dry-run
```

### Run selected steps only

For example, to run only GA and analysis using an existing trained policy model:

```powershell
.\run_pipeline.ps1 -Name A_k4_2od_x1.0 -Steps ga,analyze
```

or:

```bash
python run_experiments.py --config experiments.json --name A_k4_2od_x1.0 --steps ga analyze
```

---

## Pipeline details

The pipeline consists of five steps.

### 1. Dataset generation

```text
py/make_policy_dataset.py
```

This step generates policy-training data using the SUMO configuration and network specified for the selected experiment. By default, each experiment uses 8 parts and 125 samples per part.

### 2. Merge

```text
merge_policy_parts.py
```

This step merges the generated dataset parts into the experiment's `out_policy` folder.

### 3. Train

```text
py/train_policy_gnn.py
```

This step trains the GNN policy and saves the trained model to the experiment's `policy_model` path, usually:

```text
output_policy_<experiment_name>/policy_model.pt
```

### 4. GA / GA+GNN optimization

```text
py/ga_edge_closure_gnn_policy.py
```

This step runs the GA under four modes:

```text
none       # GA baseline without GNN guidance
init       # GNN guidance during initialization
mutation   # GNN guidance during mutation
both       # GNN guidance during both initialization and mutation
```

For each experiment, the default configuration runs seeds 0--9.

### 5. Analysis

```text
py/analyze_runs.py
```

This step analyzes the run folders and writes the analysis log to the selected experiment's `runs_root` folder.

---

## Output folders

The pipeline creates experiment-specific output folders.

Examples:

```text
output_policy_A_k4_2od_x1.0/
  part0/
  part1/
  ...
  policy_model.pt
  train.log
  merge.log

runs_A_k4_2od_x1.0/
  none_seed0/
  init_seed0/
  mutation_seed0/
  both_seed0/
  ...
  analyze.log
```

These folders can be large. They should normally be excluded from Git unless the repository is intended to archive the complete experimental outputs. For publication, full run outputs and trained model files can be stored in a persistent external archive or institutional storage, with the corresponding access information added only after it becomes stable.

---

## Reproducing the manuscript results

To reproduce one scenario from scratch:

```bash
python run_experiments.py --config experiments.json --name A_k4_2od_x1.0
```

To reproduce all scenarios, run the same command for each experiment name in `experiments.json`.

A minimal manual sequence is:

```bash
python run_experiments.py --config experiments.json --name A_k4_2od_x1.0 --steps dataset merge train
python run_experiments.py --config experiments.json --name A_k4_2od_x1.0 --steps ga analyze
```

To analyze an already completed run folder manually:

```bash
python py/analyze_runs.py --root runs_A_k4_2od_x1.0 --modes none init mutation both --select best --jobs 4
```

---

## Data availability

The repository contains the experiment runner, configuration file, Python source code, and SUMO network/configuration files required to reproduce the experiments. The experiment definitions in `experiments.json` specify the network files, SUMO configuration files, demand sizes, protected edges, random seeds, GA parameters, GNN policy paths, and output folders for all reported scenarios.

For full reproducibility, each `net*/` folder must include not only the `.net.xml` and `.sumocfg` files, but also all route, trip, additional, or other files referenced inside the `.sumocfg` files.

Large generated outputs, including `output_policy_*`, `runs_*`, trained model files (`*.pt`), and complete log files, may be stored outside the Git repository if necessary. If these files are archived externally, describe the archive location in the manuscript or in a release note associated with the final repository version.

---

## Code availability

The code associated with the manuscript is provided in this repository. Because repository locations, release tags, version identifiers, and archive records may change before or after publication, this README does not include fixed external access details. Add those details only in the final manuscript, repository release, or data-availability record when they are stable.

---

## Suggested `.gitignore`

```gitignore
__pycache__/
*.pyc
.venv/
venv/

output_policy_*/
runs_*/
*.log

*.pt
*.pth
*.ckpt

.DS_Store
.vscode/
.idea/
```

If trained models or run outputs are intentionally archived in the repository, remove the corresponding patterns from `.gitignore`.

---

## Citation

If you use this code or data, please cite the associated manuscript once the final bibliographic information is available:

```text
Kim, S., Park, J., and Jang, B. Evolutionary Search Based on Graph Neural Networks for Road-Closure Optimization. Submitted to Evolutionary Intelligence.
```

After publication, replace this temporary citation with the final journal citation.
