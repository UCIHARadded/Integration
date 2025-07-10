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

### 📊 Evaluation Pipeline

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

## ✨ Key Features

            ✅ Automated Latent Domain Estimation using clustering.
        
            ✅ Curriculum Learning for sequential and staged domain training.
        
            ✅ SHAP-based Evaluations: Flip Rate, AOPC, Coherence, Sparsity.
        
            ✅ Cross-domain Testing with configurable datasets.
        
            ✅ Extensible Pipeline for novel DG/DA experiments.
            

## 📁 File Structure

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

## 📊 Supported Datasets

Currently supports activity recognition datasets, especially cross-subject and cross-location setups via:

    actdata/cross_people.py

    EMG dataset (with preprocessing expected in CSV format)

Direct Link- https://wjdcloud.blob.core.windows.net/dataset/diversity_emg.zip

## ▶️ How to Run

### 📦 Setup

conda env create -f env.yml
conda activate diversify_env

### Dataset download 

        # Download the dataset
        !wget https://wjdcloud.blob.core.windows.net/dataset/diversity_emg.zip
        !unzip diversity_emg.zip && mv emg data/
        
        # Create necessary directories
        !mkdir -p ./data/train_output/act/
        
        !mkdir -p ./data/emg
        !mv emg/* ./data/emg

### 🏋️‍♂️ Training


                python train.py --data_dir ./data/ --task cross_people --test_envs 0 --dataset emg --algorithm diversify --alpha1 1.0 --alpha 1.0 --lam 0.0 --local_epoch 3 --max_epoch 50 --lr 0.01 --output ./train_output --automated_k --curriculum --CL_PHASE_EPOCHS 20 --enable_shap
                
                python train.py --data_dir ./data/ --task cross_people --test_envs 1 --dataset emg --algorithm diversify --alpha1 0.1 --alpha 10.0 --lam 0.0 --local_epoch 2 --max_epoch 15 --lr 0.01 --output ./data/train_output1 --automated_k --curriculum --CL_PHASE_EPOCHS 10 --enable_shap
                
                python train.py --data_dir ./data/ --task cross_people --test_envs 2 --dataset emg --algorithm diversify --alpha1 0.5 --alpha 21.5 --lam 0.0 --local_epoch 1 --max_epoch 150 --lr 0.01 --output ./data/train_output2 --automated_k --curriculum --CL_PHASE_EPOCHS 2 --enable_shap
                
                python train.py --data_dir ./data/ --task cross_people --test_envs 3 --dataset emg --algorithm diversify --alpha1 5.0 --alpha 0.1 --lam 0.0 --local_epoch 5 --max_epoch 30 --lr 0.01 --output ./data/train_output3 --automated_k --curriculum --CL_PHASE_EPOCHS 5 --enable_shap



### 🧪 Evaluation

After training, use the same script with --eval flag (if implemented) or embed evaluation in train.py by enabling SHAP.


## 📂 Outputs and Artifacts

    ./checkpoints/: Trained model weights.

    ./logs/: Training loss and accuracy logs.

    ./results/: SHAP metrics, AOPC plots, domain generalization reports.

Typical evaluation output includes:

                Metric	                   Description
                Flip Rate	Change in prediction after key feature masking
                AOPC	        Confidence drop curve area
                Coherence	Agreement across similar inputs
                Sparsity	Minimum features required to explain
                
## 🔍 In-depth Analysis

    Latent Domain Clustering: Uses KMeans-based latent splitting via datautil/cluster.py.

    Curriculum Training: Orders samples by feature norm or confidence before training.

    SHAP Explanations: shap_utils.py implements post-hoc perturbation evaluations and ranking metrics.

    Cross-Domain Testing: Evaluates generalization across unseen person IDs or locations.

## 📜 License

This project is free for academic and commercial use with attribution.

            @misc{Integration2025,
              title={Integration Framework for Domain Generalization with GNNs, SHAP, and Curriculum Learning},
              author={Rishabh Gupta et al.},
              year={2025},
              note={https://github.com/UCIHARadded/Integration}
            }

## Contact

E-mail- rishabhgupta8218@gmail.com
