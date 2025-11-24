## Centrality-based Masking in GraphMAE2

This repository implements centrality-based masking strategies (specifically PageRank and Degree Centrality) for GraphMAE2, a Self-Supervised Masked Graph Autoencoder.

By prioritizing the masking of structurally important nodes (e.g., hubs or influential nodes), we aim to create a more challenging and robust pre-training task, forcing the model to learn deeper structural dependencies.

### Features

PageRank Initial Masking: Mask input nodes based on their PageRank scores. High PageRank = Higher probability of being masked.

Degree Centrality Initial Masking: Mask input nodes based on their degree. High Degree = Higher probability of being masked.

AUC-ROC Evaluation: Integrated AUC-ROC metric calculation alongside standard Accuracy for node classification tasks.

### File Structure

The core logic modifications are contained in the following files:

models/edcoder.py: Implements the encoding_mask_noise function with new masking strategies (PageRank/Degree).

models/finetune.py: Updates the evaluation loop to compute and return AUC-ROC scores.

utils.py: Helper function compute_auc for calculating AUC for binary and multi-class classification.

main_full_batch.py: Main training script updated to log and print AUC metrics.

### Installation

Follow the installation instructions from the original GraphMAE2 repository.

You will also need networkx for PageRank calculation:

pip install networkx scikit-learn


### Usage

1. PageRank Masking (Recommended)

To run pre-training with PageRank-based initial masking:

python main_full_batch.py --dataset cora --encoder gat --decoder gat --seed 0 --device 0 --mask_method pagerank


2. Degree Centrality Masking

To run pre-training with Degree-based initial masking:

python main_full_batch.py --dataset cora --encoder gat --decoder gat --seed 0 --device 0 --mask_method degree


3. Random Masking (Baseline)

To run the standard GraphMAE2 with random masking:

python main_full_batch.py --dataset cora --encoder gat --decoder gat --seed 0 --device 0 --mask_method random


(Note: Ensure your edcoder.py logic supports the degree or random keys if you switch between them manually, or stick to the provided version which defaults to PageRank if specified).

### Output

The script will output training progress and final evaluation metrics:

# final_acc: 0.8350±0.0000
# early-stopping_acc: 0.8410±0.0000
# final_auc: 0.9120±0.0000
# early-stopping_auc: 0.9150±0.0000


### Reference

This work is based on GraphMAE2: A Decoding-Enhanced Masked Self-Supervised Graph Learner.
