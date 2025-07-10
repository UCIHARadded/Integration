# 🔬 Integration Framework for Domain Generalization with GNNs, SHAP, and Curriculum Learning

## 🧠 Overview

This repository provides an integrated framework for research in Domain Generalization (DG) using Graph Neural Networks (GNNs), SHAP-based interpretability, and Curriculum Learning strategies. It extends the DIVERSIFY architecture with automation of latent domain estimation, curriculum learning and SHAP evaluation tools. The framework supports continual learning setups, clustering-based domain estimation, and perturbation-based interpretability evaluation.

## 🚀 Core Pipelines

### 🔧 Training Pipeline

The training phase includes data loading, clustering-based domain estimation, curriculum sorting, model training with an optional CNN backbone, and domain-specific optimization.

        train.py --> datautil/ --> alg/ --> loss/ --> network/

Training Pipeline Illustration:

      ┌────────────┐
      │  Raw Data  │
      └────┬───────┘
           │
           ▼
      ┌──────────────┐     ┌────────────────────┐
      │ Domain Split │<───▶│ Cluster Estimation │
      └────┬─────────┘     └─────────┬──────────┘
           ▼                         ▼
      ┌──────────────┐      ┌──────────────────┐
      │ Curriculum   │────▶ │ Model Training   │
      │ Learning     │      │      (CNN)       │
      └──────────────┘      └────────┬─────────┘
                                     ▼
                            ┌──────────────────┐
                            │ Save Trained     │
                            │ Weights & Logs   │
                            └──────────────────┘

📊 Evaluation Pipeline

Evaluation supports test-time inference, SHAP-based explainability metrics (flip rate, coherence, AOPC), and generalization gap measurements across domains.

Evaluation Pipeline Illustration:

      ┌────────────┐
      │ Test Data  │
      └────┬───────┘
           ▼
      ┌────────────────────┐
      │ Load Trained Model │
      └────┬───────────────┘
           ▼
      ┌────────────────────────────┐
      │ Inference & SHAP Analysis  │
      └────────────┬───────────────┘
                   ▼
         ┌────────────────────┐
         │ Explanation Metrics│
         └────────────────────┘

✨ Key Features

    ✅ Automated Latent Domain Estimation using clustering.

    ✅ Curriculum Learning for sequential and staged domain training.

    ✅ SHAP-based Evaluations: Flip Rate, AOPC, Coherence, Sparsity.

    ✅ GNN Integration: Uses graph-based backbones like GCN.

    ✅ Cross-domain Testing with configurable datasets.

    ✅ Extensible Pipeline for novel DG/DA experiments.

📁 File Structure

Integration-main/
├── alg/                        # Core domain generalization algorithms
│   ├── algs/                   # Specific algorithm implementations (e.g. diversify.py)
│   ├── alg.py                  # Main algorithm controller
│   └── modelopera.py           # Model operations
├── datautil/                   # Data loaders, curriculum sorting, clustering
│   ├── actdata/                # Activity recognition dataset utilities
│   ├── getcurriculumloader.py
│   ├── getdataloader_single.py
│   └── cluster.py              # Clustering logic for domain estimation
├── loss/                       # Custom loss functions
│   └── common_loss.py
├── network/                    # Model architectures
│   ├── act_network.py          # CNN for activity data
│   ├── Adver_network.py        # Adversarial components
│   └── common_network.py       # Base models
├── utils/                      # Utility functions and argument parsing
│   ├── params.py
│   └── util.py
├── shap_utils.py               # SHAP evaluation metrics
├── train.py                    # Entry point for training
├── env.yml                     # Conda environment spec
└── README.md                   # This file

📊 Supported Datasets

Currently supports activity recognition datasets, especially cross-subject and cross-location setups via:

    actdata/cross_people.py

    UCI HAR and WESAD datasets (with preprocessing expected in CSV format)

▶️ How to Run
📦 Setup

conda env create -f env.yml
conda activate diversify_env

🏋️‍♂️ Training

python train.py \
  --data_dir ./data/ \
  --task act \
  --algorithm diversify \
  --latent_domain_num 3 \
  --max_epoch 20 \
  --local_epoch 2 \
  --enable_shap True

🧪 Evaluation

After training, use the same script with --eval flag (if implemented) or embed evaluation in train.py by enabling SHAP.
📂 Outputs and Artifacts

    ./checkpoints/: Trained model weights.

    ./logs/: Training loss and accuracy logs.

    ./results/: SHAP metrics, AOPC plots, domain generalization reports.

Typical evaluation output includes:
Metric	Description
Flip Rate	Change in prediction after key feature masking
AOPC	Confidence drop curve area
Coherence	Agreement across similar inputs
Sparsity	Minimum features required to explain
🔍 In-depth Analysis

    Latent Domain Clustering: Uses KMeans-based latent splitting via datautil/cluster.py.

    Curriculum Training: Orders samples by feature norm or confidence before training.

    SHAP Explanations: shap_utils.py implements post-hoc perturbation evaluations and ranking metrics.

    GNN Support: Drop-in support via network/common_network.py using PyTorch Geometric-style layers.

    Cross-Domain Testing: Evaluates generalization across unseen person IDs or locations.

📜 License

This project is licensed under the MIT License.

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy...
