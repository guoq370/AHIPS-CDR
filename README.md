# AHIPS-CDR: Attention-Driven HIN Embedding with Independent Parallel Strategy for Course-Job Cross-Domain Recommendation

This repository contains the official implementation of the paper:  
*“Bridging Education and Employment: An Attention-Driven Cross-Domain Intelligent Recommendation System with Independent Parallel Strategy”* (submitted to Expert Systems With Applications).

## 📋 Overview

AHIPS-CDR is a cross-domain recommendation framework that addresses semantic conflict and feature cancellation in course-job heterogeneous information networks (HINs). The model introduces an **Independent Parallel Strategy (IPS)** to separate intra-domain and inter-domain representation learning, and uses **dual-level attention** to adaptively weight meta-path embeddings.

## 🔧 Requirements

- Python 3.10
- PyTorch 2.0
- DGL 1.1
- transformers (Hugging Face)
- numpy, scipy, pandas, scikit-learn

## Install all dependencies:
```bash
pip install -r requirements.txt
bash

## 📁 Repository Structure
- Path	Description
- data/	Preprocessing scripts for Coursera & LinkedIn datasets, skill alignment (BERT + community detection)
- src/	Core implementation of AHIPS-CDR: HIN builder, Metapath2Vec++, attention, IPS, enhanced MF
- baselines/	11 baseline methods (MF, FM, HeteRec, FMGrank, HGT, RCDR, HGCL, SEAGULL, MI-DTCDR, KPAR, HAGO)
- config/	Hyperparameter YAML files (embedding dim, learning rate, batch size, etc.)
- scripts/	Training, evaluation, ablation, and sparsity experiments
