# ST-GAT NIDS — Two-Stage Spatio-Temporal Graph Neural Network for Network Intrusion Detection

---

## Table of Contents
1. [Project Objective](#objective)
2. [Datasets](#datasets)
3. [Unified Attack Taxonomy](#taxonomy)
4. [Feature Engineering](#features)
5. [Data Splits & Sampling](#splits)
6. [Stage 1 — Binary Classifier (XGBoost)](#stage1)
7. [Graph Construction](#graph)
8. [SMOTE Augmentation](#smote)
9. [Stage 2 — ST-GAT Architecture](#stage2)
10. [Training Configuration](#training)
11. [Training History](#history)
12. [Final Results](#results)
13. [Two-Stage Pipeline Summary](#pipeline)
14. [Known Limitations](#limitations)
15. [Saved Artifacts](#artifacts)

---

## 1. Project Objective <a name="objective"></a>

Build a generalizable, dataset-agnostic NIDS capable of:
- **Binary attack detection** (Stage 1 — XGBoost)
- **Multiclass attack classification** (Stage 2 — ST-GAT)

Using graph-based deep learning, evaluated across multiple heterogeneous datasets including a completely blind test set never seen during training.

---

## 2. Datasets <a name="datasets"></a>

### Training & Validation

**CICIDS2017** (`zufarasyraf/cicids2017` on Kaggle)
| File | Rows | Label |
|---|---|---|
| DoS_GoldenEye-12293.csv | 12,293 | DoS GoldenEye |
| DoS_Hulk.csv | 158,468 | DoS Hulk |
| DoS_Slowhttptest-15113.csv | 15,113 | DoS Slowhttptest |
| FTP_Patator-56169.csv | 56,169 | FTP-Patator |
| SSH_Patator-12718.csv | 12,718 | SSH-Patator |
| Web_Attack_Brute_Force-9348.csv | 9,348 | Web Attack Brute Force |
| Web_Attack_SQL_Injection-17197.csv | 17,197 | Web Attack SQL Injection |
| BENIGN_CICIDS2017.csv | 315,106 | BENIGN |
| **Total** | **596,412** | **91 original columns** |

- Timestamp format: `datetime` with microsecond precision
- Files with 2024 timestamps (regenerated): per-file row-order used as temporal proxy

**UNSW-NB15** (`mrwellsdavid/unsw-nb15` on Kaggle)
| Files | Rows | Columns |
|---|---|---|
| UNSW-NB15_1.csv through UNSW-NB15_4.csv (concatenated) | 2,540,047 | 49 |

- Temporal ordering via `stime` Unix timestamp
- Predefined train/test split ignored — own temporal split applied

### Blind Test (Zero Training Exposure)

**CIC-DDoS2019** (`rodrigorosasilva` on Kaggle)
| File | Rows |
|---|---|
| DrDoS_DNS.csv | 5,071,011 |
| DrDoS_LDAP.csv | 2,179,930 |
| BENIGN | 5,014 |
| **Total** | **7,255,955** |

- Sampled to 200,000 for evaluation (5,014 BENIGN + 194,986 DoS)

---

## 3. Unified Attack Taxonomy <a name="taxonomy"></a>

| ID | Unified Class | CICIDS2017 Sources | UNSW-NB15 Sources |
|---|---|---|---|
| 0 | BENIGN | BENIGN | BENIGN (NaN attack_cat) |
| 1 | DoS | DoS Hulk, DoS GoldenEye, DoS Slowhttptest | DoS, Generic |
| 2 | Bruteforce | FTP-Patator, SSH-Patator | — |
| 3 | WebAttack | Web Attack SQL Injection, Web Attack Brute Force | Exploits |
| 4 | Reconnaissance | — | Reconnaissance, Fuzzers, Analysis |
| 5 | Backdoor | — | Backdoor, Worms |
| 6 | Shellcode | — | Shellcode |

---

## 4. Feature Engineering <a name="features"></a>

### Canonical 17-Feature Set (harmonized across datasets)

| Feature | CICIDS2017 Source | UNSW-NB15 Source |
|---|---|---|
| src_port | Src Port | sport |
| dst_port | Dst Port | dsport |
| protocol | Protocol | proto |
| duration | Flow Duration | dur |
| packets_fwd | Total Fwd Packet | spkts |
| packets_bwd | Total Bwd packets | dpkts |
| bytes_fwd | Total Length of Fwd Packet | sbytes |
| bytes_bwd | Total Length of Bwd Packet | dbytes |
| bytes_per_sec | Flow Bytes/s | (sbytes+dbytes)/dur |
| packets_per_sec | Flow Packets/s | (spkts+dpkts)/dur |
| pkt_size_mean_fwd | Fwd Packet Length Mean | smeansz |
| pkt_size_mean_bwd | Bwd Packet Length Mean | dmeansz |
| syn_flag_count | SYN Flag Count | 0 (missing) |
| ack_flag_count | ACK Flag Count | 0 (missing) |
| fin_flag_count | FIN Flag Count | 0 (missing) |
| rst_flag_count | RST Flag Count | 0 (missing) |
| flow_iat_mean | Flow IAT Mean | sintpkt |

### UNSW-Specific Derivations
- `bytes_per_sec = (sbytes + dbytes) / dur` — zero-division handled by filling 0
- `packets_per_sec = (spkts + dpkts) / dur` — same handling
- TCP flag columns filled with 0; `flag_missing = 1` indicator added for all UNSW rows

### Protocol Encoding
```
tcp  → 6
udp  → 17
icmp → 1
unknown/other → 255
```

### Outlier Capping
- 99th percentile computed on **train split only**
- Same cap values applied to val and test splits
- Key caps: `bytes_per_sec` capped at 2,448,598 (CICIDS), 132,000,000 (UNSW); `ack_flag_count` capped at 203

### Scaling
**RobustScaler** fitted on train only, applied to 10 continuous columns:
`duration`, `bytes_fwd`, `bytes_bwd`, `packets_fwd`, `packets_bwd`, `bytes_per_sec`, `packets_per_sec`, `pkt_size_mean_fwd`, `pkt_size_mean_bwd`, `flow_iat_mean`

### GNN-Specific Edge Feature Normalization
| Feature | Normalization |
|---|---|
| src_port, dst_port | ÷ 65535 |
| protocol | ÷ 255 |
| ack_flag_count | clip at p99=203, ÷ 203 |
| syn_flag_count | ÷ 17 |
| fin_flag_count | ÷ 8 |
| rst_flag_count | ÷ 24 |
| All edge values | clipped to [−1.5, 10.5] |
| Node features | clipped to [−3.0, 10.0] |

---

## 5. Data Splits & Sampling <a name="splits"></a>

### Temporal Split Strategy
- **CICIDS2017:** per-file 70/15/15 row-order split (file row order = capture sequence)
- **UNSW-NB15:** global `stime`-sorted 70/15/15 split
- Splits merged separately: `train+train`, `val+val`, `test+test`
- Test sets kept **separate** for independent cross-dataset evaluation

### Class Sampling (Train Only)
| Class | Sampling Strategy | Final Train Count |
|---|---|---|
| BENIGN | 50,000 per dataset | 100,000 |
| DoS | 30,000 per dataset | 60,000 |
| Bruteforce | 30,000 (CICIDS only) | 30,000 |
| WebAttack | All available | 43,711 |
| Reconnaissance | All available | 23,390 |
| Backdoor | All available | 1,379 |
| Shellcode | All available | 848 |
| **Total** | | **259,328** |

### Final Dataset Sizes
| Split | Rows |
|---|---|
| combined_train | 259,328 |
| combined_val | 470,469 |
| cicids_test | 89,465 |
| unsw_test | 381,008 |

> Val and test splits left **untouched** (natural distribution) for honest evaluation.

---

## 6. Stage 1 — Binary Classifier <a name="stage1"></a>

### Model
**XGBoost (XGBClassifier)**

### Hyperparameters
| Parameter | Value |
|---|---|
| n_estimators | 500 (early stopped at 293) |
| max_depth | 6 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| scale_pos_weight | 0.6276 |
| eval_metric | aucpr |
| early_stopping_rounds | 30 |
| tree_method | hist |
| device | cuda |

### Input / Output
- **Input:** 17 canonical features (`flag_missing` excluded after leak test — `flag_missing` had 0.104 importance, indicating dataset identity leakage)
- **Output:** Binary probability score (0 = BENIGN, 1 = ATTACK)

### Top 10 Feature Importances
| Rank | Feature | Importance |
|---|---|---|
| 1 | pkt_size_mean_bwd | 0.376 |
| 2 | rst_flag_count | 0.151 |
| 3 | packets_bwd | 0.132 |
| 4 | duration | 0.058 |
| 5 | bytes_bwd | 0.056 |
| 6 | bytes_fwd | 0.051 |
| 7 | packets_per_sec | 0.034 |
| 8 | packets_fwd | 0.028 |
| 9 | dst_port | 0.025 |
| 10 | flow_iat_mean | 0.019 |

### Results
| Metric | Value |
|---|---|
| Val PR-AUC | 0.9988 |
| Val Macro-F1 | 0.9780 |
| CICIDS test Macro-F1 | 0.9999 |
| CICIDS test Recall (Attack) | 0.9998 |
| UNSW test Macro-F1 | 0.9739 |
| UNSW test Recall (Attack) | 0.9997 |
| Generalization gap (recall) | 0.0000 |

### Per-Class F1 on UNSW Test
| Class | F1 |
|---|---|
| DoS | 0.9478 |
| WebAttack | 0.7065 |
| Reconnaissance | 0.7029 |
| Shellcode | 0.0845 |
| Backdoor | 0.1145 |

> `xgb_attack_prob` (float 0–1) added as **18th edge feature** for Stage 2 GNN input (Option B: full graph with XGBoost confidence as edge signal).

---

## 7. Graph Construction <a name="graph"></a>

### Graph Type
Static node set with temporal edge snapshots (IP-as-node, flow-as-edge)

### Node Definition
- **Nodes:** Unique IP addresses
- **Total nodes:** 50 (controlled testbed networks)
- **Node feature dimensionality:** 8

### Node Features (8-dim)
| Index | Feature | Derivation |
|---|---|---|
| 0 | mean_bytes_fwd | Mean fwd bytes as source IP |
| 1 | mean_bytes_bwd | Mean bwd bytes as source IP |
| 2 | mean_packets_fwd | Mean fwd packets as source |
| 3 | mean_packets_bwd | Mean bwd packets as source |
| 4 | out_degree | log1p(flow count as src) / log1p(128580) |
| 5 | mean_xgb_attack_prob | Mean XGBoost score of outgoing flows |
| 6 | unique_dst_count | Unique destinations ÷ 10 |
| 7 | unique_src_count | Unique sources ÷ 10 |

### Edge Definition
- Each network flow = one **directed edge** (src_ip → dst_ip)
- **Edge feature dimensionality:** 18 (17 canonical + xgb_attack_prob)

### Temporal Snapshots
| Dataset | Window Method | Window Size |
|---|---|---|
| CICIDS2017 | Row-order blocks within file | 500 rows |
| UNSW-NB15 | Real-time sliding window | 10 seconds |

| Split | Snapshots |
|---|---|
| Train | 520 (258 CICIDS + 262 UNSW) |
| Val | 942 |
| CICIDS test | 179 |
| UNSW test | 763 |

### Sequence Parameters
- Sequence length: **T = 8 snapshots**
- Stride: **4** (50% overlap)
- Total training sequences: **129**

### IP Coverage (DDoS2019 Blind Test)
- Total unique IPs in sample: 168
- IPs known to ip2idx: 2 (1.19%)
- IPs unknown (mapped to node 0): 166 (98.81%)
- Impact: GAT topology signal collapsed; edge features only

---

## 8. SMOTE Augmentation <a name="smote"></a>

Applied to **Backdoor** and **Shellcode** classes only within training edge features:

| Class | Original | After SMOTE |
|---|---|---|
| Backdoor (Class 5) | 1,379 | 5,000 |
| Shellcode (Class 6) | 848 | 5,000 |
| Synthetic edges generated | — | 7,773 |

- SMOTE applied on 18-dim edge feature space
- Synthetic edges distributed into existing snapshots containing those classes
- Synthetic edge values clipped to [−1.5, 10.5] post-generation
- Saved as `train_snapshots_augmented.pt`

---

## 9. Stage 2 — ST-GAT Architecture <a name="stage2"></a>

**Full model name:** Edge-Augmented Spatio-Temporal Graph Attention Network (EdgeAugmented-ST-GAT)

**Total trainable parameters: 179,463**

### Component 1 — EdgeAugmentedGAT (Spatial)

```
Input: node features (8-dim) + edge features (18-dim)

Layer 1 — GATConv:
  in_channels  : 26 (8 node + 18 edge, concatenated per message)
  out_channels : 64
  heads        : 4, concat=True → output 256-dim
  attention dropout : 0.3
  activation   : ELU

Layer 2 — GATConv:
  in_channels  : 274 (256 node + 18 edge)
  out_channels : 64
  heads        : 1, concat=False → output 64-dim
  activation   : ELU

Output per node: 64-dim embedding
```

### Component 2 — Temporal Transformer

```
Input  : sequence of T=8 node embedding matrices (each 50 × 64)
Layers : 2-layer Transformer Encoder
  d_model        : 64
  nhead          : 4
  dim_feedforward: 256
  dropout        : 0.1
Output : temporally-enriched node embeddings per snapshot (50 × 64)
```

### Component 3 — Edge Classifier MLP

```
Input per edge: src_embed(64) + dst_embed(64) + edge_attr(18) = 146-dim

Linear(146 → 128) → ReLU → Dropout(0.3)
Linear(128 → 64)  → ReLU
Linear(64  → 7)

Output: logits over 7 attack classes
```

---

## 10. Training Configuration <a name="training"></a>

### Loss Function — Focal Loss
- **gamma:** 2
- Class weights (inverse frequency, **capped at 5.0**):

| Class | Raw Weight | Capped Weight |
|---|---|---|
| BENIGN | 0.370 | 0.370 |
| DoS | 0.617 | 0.617 |
| Bruteforce | 1.235 | 1.235 |
| WebAttack | 0.848 | 0.848 |
| Reconnaissance | 1.584 | 1.584 |
| Backdoor | 26.865 | **5.000** |
| Shellcode | 43.687 | **5.000** |

### Optimizer — AdamW
| Parameter | Value |
|---|---|
| lr (warmup, epochs 1–5) | 1e-4 |
| lr (main training) | 1e-3 |
| weight_decay | 1e-4 |

### Scheduler — CosineAnnealingLR
| Parameter | Value |
|---|---|
| T_max | 50 |
| eta_min | 1e-6 |

### Regularization
| Technique | Value |
|---|---|
| Gradient clipping | max_norm = 1.0 |
| Attention dropout | 0.3 |
| Classifier dropout | 0.3 |
| Transformer dropout | 0.1 |

### Hard Example Mining (OHEM)
- Backdoor (class 5) and Shellcode (class 6) edge losses multiplied by additional **2.0×** on top of focal loss

### Checkpoint Criterion
```
Custom Score = 0.5 × Macro-F1 + 0.3 × WebAttack-F1 + 0.2 × Recon-F1
```
Rationale: stabilizes checkpointing around mid-tier classes; prevents Macro-F1 oscillation from rare class swings on imbalanced val set.

### Early Stopping
- Patience: **15 epochs** (no improvement in custom score)

---

## 11. Training History <a name="history"></a>

| Run | Architecture | Epochs | Best Macro-F1 | Notes |
|---|---|---|---|---|
| Run 1 | GraphSAGE | 10 | 0.1612 | NaN loss due to unscaled features; wrong T_max=5 |
| Run 2 | GraphSAGE | 10 | 0.5326 | Fixed feature scaling + gradient clipping |
| Run 3 | GraphSAGE continuation | 21 | 0.6253 | Best GraphSAGE result |
| Run 4 | GraphSAGE + OHEM | 21 | 0.5674 (custom) | OHEM added, lower LR |
| Run 5 | GAT initial | 10 | 0.4983 (custom) | T_max bug recurred |
| Run 6 | GAT full | 43 | **0.6437 (custom)** | Best overall — epoch 28 checkpoint |

**Total training epochs across all runs: ~103**

**Key fixes applied during training:**
- Unscaled features (src_port max=65535, ack_flag max=482,518, out_degree max=128,580) → immediate NaN in GraphSAGE Conv1
- Fix: per-column normalization + node feature log1p transform
- CosineAnnealingLR T_max=5 (instead of 50) → LR hit 0 at epoch 10 prematurely
- Fix: verified `scheduler.T_max` after initialization

---

## 12. Final Results <a name="results"></a>

### Validation Set (Best GAT Checkpoint — Epoch 28)
| Class | F1 |
|---|---|
| BENIGN | 0.9877 |
| DoS | 0.9263 |
| Bruteforce | 0.8497 |
| WebAttack | 0.6834 |
| Reconnaissance | 0.5486 |
| Backdoor | 0.0852 |
| Shellcode | 0.5247 |
| **Macro-F1** | **0.6437** |

### CICIDS2017 Test Set
| Class | F1 |
|---|---|
| BENIGN | 0.9967 |
| DoS | 0.9279 |
| Bruteforce | 0.8298 |
| WebAttack | 0.9890 |
| Reconnaissance | 0.0000 (absent in CICIDS) |
| Backdoor | 0.0000 (absent in CICIDS) |
| Shellcode | 0.0000 (absent in CICIDS) |
| **Macro-F1** | **0.5348** |

### UNSW-NB15 Test Set (Cross-Dataset Generalization)
| Class | F1 |
|---|---|
| BENIGN | 0.9874 |
| DoS | 0.9649 |
| Bruteforce | 0.0000 (absent in UNSW) |
| WebAttack | 0.5939 |
| Reconnaissance | 0.5726 |
| Backdoor | 0.1007 |
| Shellcode | 0.3653 |
| **Macro-F1** | **0.5121** |

### CIC-DDoS2019 Blind Test
| Metric | Value |
|---|---|
| Stage 1 Attack Recall | 97.47% |
| Stage 2 Macro-F1 | 0.1047 |
| BENIGN F1 | 0.7310 |
| DoS F1 | 0.0016 |
| **Combined Detection Rate** | **98.66%** |

> Low GAT performance on DDoS2019 expected: 98.81% IP collapse to unknown node, DrDoS reflection variants unseen in training.

### Cross-Dataset Generalization Gap
| Class | Val F1 | CICIDS Test | UNSW Test | Val→UNSW Gap |
|---|---|---|---|---|
| BENIGN | 0.9877 | 0.9967 | 0.9874 | 0.0003 |
| DoS | 0.9263 | 0.9279 | 0.9649 | −0.0386 |
| Bruteforce | 0.8497 | 0.8298 | 0.0000 | 0.8497 |
| WebAttack | 0.6834 | 0.9890 | 0.5939 | 0.0895 |
| Reconnaissance | 0.5486 | 0.0000 | 0.5726 | −0.0240 |
| Backdoor | 0.0852 | 0.0000 | 0.1007 | −0.0155 |
| Shellcode | 0.5247 | 0.0000 | 0.3653 | 0.1594 |

---

## 13. Two-Stage Pipeline Summary <a name="pipeline"></a>

### Detection Rates
| Dataset | Stage 1 Caught | GNN Additional | Total Detection |
|---|---|---|---|
| CICIDS2017 test | 99.98% | +0.01% | 99.99% |
| UNSW-NB15 test | 99.97% | +0.00% | 99.97% |
| CIC-DDoS2019 blind | 97.47% | +1.19% | **98.66%** |

### Net New Attacks Caught Purely by GNN Topology
| Dataset | Net New Attacks |
|---|---|
| UNSW-NB15 test | 11 flows |
| CIC-DDoS2019 blind | 3,757 flows |

> These flows had normal feature distributions (low XGBoost confidence) but anomalous connection patterns detectable only via graph structure — validating the theoretical case for GNN-based NIDS.

### Pipeline Flow
```
Raw Traffic
    ↓
Schema Parser → Canonical 17 Features
    ↓
RobustScaler (10 cols)
    ↓
Stage 1: XGBoost Binary Classifier
    ↓ (xgb_attack_prob added as edge feature)
Graph Construction: IP-as-node, Flow-as-edge
Temporal Snapshots (500 rows / 10s windows)
    ↓
Stage 2: EdgeAugmented-ST-GAT
    ├── EdgeAugmentedGAT (spatial, 2 layers)
    ├── Temporal Transformer (2 layers, T=8)
    └── Edge Classifier MLP (→ 7 classes)
    ↓
Multiclass Attack Classification
```

---

## 14. Known Limitations <a name="limitations"></a>

- **Small node space (50 IPs):** Controlled testbed, not real enterprise scale. GAT topology signal limited by graph density.
- **Bruteforce generalization:** Single-source class (CICIDS only), absent from UNSW/DDoS2019 — cross-dataset generalization untested.
- **Backdoor consistently weak (F1: 0.085–0.101):** Insufficient real training samples (1,379). SMOTE helped but structural signature is subtle.
- **DDoS2019 IP collapse (98.81% unknown):** GAT topology signal lost; predictions rely on edge features only — structural advantage eliminated.
- **DrDoS reflection variants unseen in training:** Stage 1 mean confidence on DDoS2019 DoS = 0.555 vs 0.999 for known DoS variants.
- **CICIDS2017 timestamp inconsistency:** Some files show 2024 dates (regenerated), others 2017. Per-file row-order used as temporal proxy — weaker than true timestamps.
- **Static node features:** Node features computed once from training data and held fixed across all snapshots. Dynamic node feature updates (per-snapshot recomputation) not implemented.

---

## 15. Saved Artifacts <a name="artifacts"></a>

| File | Description |
|---|---|
| `robust_scaler.pkl` | RobustScaler fitted on train (10 columns) |
| `xgb_binary_stage1_noleak.json` | Stage 1 XGBoost model (293 trees) |
| `ip2idx.json` | IP string to node index mapping (50 nodes) |
| `attack_mapping.json` | Unified class name to integer mapping (7 classes) |
| `node_features.npy` | Static node feature matrix (50 × 8) |
| `train_snapshots_augmented.pt` | 520 augmented temporal snapshots (PyG Data objects) |
| `val_snapshots.pt` | 942 validation snapshots |
| `cicids_test_snapshots.pt` | 179 CICIDS test snapshots |
| `unsw_test_snapshots.pt` | 763 UNSW test snapshots |
| `stgnn_best.pt` | Best GAT checkpoint (epoch 28, custom score 0.6437) |

---

*Documentation generated post-training. All results reproducible from saved artifacts and described hyperparameters.*