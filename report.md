# Technical Report: Spatio-Temporal Graph Neural Network for Network Intrusion Detection System (NIDS)

## 1. Executive Summary

A robust **Edge-Aware Graph Neural Network (GNN)** pipeline was developed for multi-class network intrusion detection using **Spatio-Temporal Graph Snapshots**. Unlike traditional tabular approaches, this framework models network traffic as graphs, enabling the detection of attack intent through structural connectivity patterns and temporal behavior.

The proposed approach captures attack signatures such as:

* **Starburst Topology** → Denial of Service (DoS/DDoS)
* **Spiderweb Topology** → Network Probing/Scanning
* **Point-to-Point Topology** → Brute Force Attacks

By leveraging graph topology alongside traffic-flow characteristics, the model learns both behavioral and structural attack indicators.

---

# 2. Data Transformation Pipeline

The raw network traffic data undergoes a three-stage transformation process to convert imbalanced flow records into normalized graph snapshots suitable for GNN training.

## Stage 1: Multi-Class Feature Extraction & Mapping

**Script:** `multiclass_pipeline.py`

### Label Normalization

Raw attack labels from multiple datasets are mapped into a unified six-class taxonomy:

| Class ID | Category              |
| -------- | --------------------- |
| 0        | Benign                |
| 1        | DoS / DDoS            |
| 2        | Probe                 |
| 3        | Brute Force           |
| 4        | Web Attack            |
| 5        | Infiltration / Botnet |

### Graph Construction

Network traffic is transformed into directed graphs:

* **Nodes** = IP Addresses (network endpoints)
* **Edges** = Traffic flows between endpoints

### Temporal Windowing

Traffic records are segmented into fixed-length:

* **Window Size:** 60 seconds

Each window forms an independent graph snapshot while preserving temporal ordering.

### Dataset Balancing

To prevent model bias toward benign traffic:

* Temporal windows are sampled using a **50/50 balancing strategy**
* Equal representation of benign and malicious activity is maintained

---

## Stage 2: Statistical Normalization

**Scripts:**

* `gat_training.py`
* `graph_scaler.py`

### Feature Cleaning

Only high-signal edge features are retained:

| Feature       | Description                   |
| ------------- | ----------------------------- |
| duration      | Flow duration                 |
| tot_fwd_pkts  | Total forward packets         |
| tot_fwd_bytes | Total forward bytes           |
| protocol      | Transport protocol identifier |

### Standardization

Global **Z-Score Normalization** is applied:

[
z = \frac{x-\mu}{\sigma}
]

where:

* μ = dataset mean
* σ = dataset standard deviation

Benefits:

* Prevents byte-count features from dominating learning
* Stabilizes optimization
* Improves convergence during training

---

# 3. Graph Architecture & Technical Specifications

## Graph Representation

### Nodes

Represent network endpoints (IP addresses).

Node features are initialized as:

```text
[1.0]
```

This structural identity vector allows the model to learn node importance through graph connectivity.

### Edges

Represent directed traffic flows between endpoints.

### Edge Features

```text
[
  duration,
  tot_fwd_pkts,
  tot_fwd_bytes,
  protocol
]
```

### Prediction Granularity

The model performs:

**Edge-Level Classification**

Each network flow is classified independently into one of the six attack categories.

---

## Neural Network Architecture

### Model: EdgeAwareGAT

The architecture is based on Graph Attention Networks v2 (GATv2).

#### Core Components

* Multiple GATv2 convolution layers
* 4 attention heads
* Edge-aware attention mechanism
* Multi-class edge classifier

### Edge-Aware Attention

Unlike standard GAT models, attention calculations explicitly incorporate edge attributes:

```text
edge_dim = 4
```

This enables the network to evaluate neighboring nodes using both:

* Graph topology
* Traffic-flow characteristics

As a result, attention weights reflect network behavior rather than connectivity alone.

---

## Loss Function

### Weighted Focal Loss

The model employs a custom:

```text
WeightedFocalLoss
```

to address class imbalance.

#### Advantages

* Reduces dominance of frequent classes
* Focuses learning on difficult samples
* Improves detection of rare attacks

Particularly beneficial for:

* Infiltration
* Botnet
* Low-frequency attack classes

---

# 4. Final Dataset Composition

After preprocessing, the dataset consists of graph snapshots optimized for GNN training.

## Class Distribution

| Class ID | Label                 |
| -------- | --------------------- |
| 0        | Benign                |
| 1        | DoS / DDoS            |
| 2        | Probe                 |
| 3        | Brute Force           |
| 4        | Web Attack            |
| 5        | Infiltration / Botnet |

---

## Snapshot Format

Each graph snapshot is stored as a compressed `.npz` file containing:

| Field      | Description                 |
| ---------- | --------------------------- |
| edge_index | Graph topology              |
| edge_attr  | Edge feature matrix         |
| edge_label | Edge classification targets |
| num_nodes  | Number of graph nodes       |

---

# 5. Model Training Workflow

## Step 1: Prepare Raw Data

Place CICIDS2017 and/or UNSW-NB15 CSV files in the data directory.

```text
data/
├── CICIDS2017/
└── UNSW-NB15/
```

---

## Step 2: Generate Graph Snapshots

Run:

```bash
python multiclass_pipeline.py
```

Output:

```text
balanced_multiclass_graphs/
```

containing balanced spatio-temporal graph snapshots.

---

## Step 3: Train the GNN

Run:

```bash
python gat_training.py
```

The training pipeline automatically:

* Loads graph snapshots
* Applies normalization
* Constructs graph batches
* Performs attention-based message passing
* Computes edge-level predictions

---

# 6. Reproduction Artifacts

The following files are required to fully reproduce the workflow.

| File                     | Purpose                                                |
| ------------------------ | ------------------------------------------------------ |
| `multiclass_pipeline.py` | Converts raw CSV traffic into balanced graph snapshots |
| `gat_training.py`        | Defines the EdgeAwareGAT model and executes training   |
| `scaler_dictionary.npy`  | Stores feature-scaling metadata for inference          |
| `edge_scaler_config.pt`  | Contains normalization parameters (mean/std)           |

---

# 7. Inference Pipeline

During deployment:

1. Incoming traffic is converted into graph snapshots.

2. Features are normalized using:

   * `scaler_dictionary.npy`
   * `edge_scaler_config.pt`

3. Graphs are passed through the trained EdgeAwareGAT model.

4. Each edge receives a prediction:

```text
Benign
DoS/DDoS
Probe
Brute Force
Web Attack
Infiltration/Botnet
```

This guarantees identical preprocessing during training and inference.

---

# 8. Validation & Observations

Visualization of generated graph snapshots (`graphsnap.png`) demonstrates that different attack categories exhibit distinguishable structural signatures.

Examples include:

| Attack Type    | Structural Pattern       |
| -------------- | ------------------------ |
| DoS/DDoS       | Starburst                |
| Probe/Scan     | Spiderweb                |
| Brute Force    | Point-to-Point           |
| Benign Traffic | Distributed Connectivity |

These observations validate the hypothesis that combining graph topology with flow-level features enables more effective intrusion detection than traditional tabular approaches.

---

# 9. Conclusion

The proposed Spatio-Temporal Graph Neural Network framework successfully transforms raw network traffic into graph representations and performs edge-level intrusion detection using an Edge-Aware GAT architecture.

Key achievements include:

* Unified 6-class attack taxonomy
* Temporal graph snapshot generation
* Edge-aware attention mechanisms
* Global feature normalization
* Focal-loss-based imbalance handling
* Edge-level multi-class classification

The resulting pipeline provides a scalable and reproducible foundation for next-generation Network Intrusion Detection Systems (NIDS), capable of leveraging both network behavior and structural attack patterns for enhanced detection performance.

