# CICIDS2017 GNN Intrusion Detection System - Development Log

## Project Goal

Develop a **Graph Neural Network (GNN)** based Intrusion Detection System using the **CICIDS2017** dataset.

The project is divided into multiple stages:

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Graph Construction
6. Binary Intrusion Detection
7. Attack Classification using GNN

---

# Stage 1 — Data Collection

Download the **MachineLearningCVE** version of the CICIDS2017 dataset from Kaggle.

The dataset consists of eight CSV files:

* Monday
* Tuesday
* Wednesday
* Thursday Morning
* Thursday Afternoon
* Friday Morning
* Friday Afternoon
* Friday DDoS

Merge all CSV files into one dataset.

---

# Dataset Overview

The dataset contains over **2.8 million network flows** extracted using **CICFlowMeter**.

It contains over **80 flow features** including:

* Flow Duration
* Packet Counts
* Packet Lengths
* IAT Statistics
* TCP Flags
* Flow Bytes/s
* Flow Packets/s
* Label

---

# Stage 2 — Data Cleaning

Cleaning steps:

* Merge all CSV files
* Strip whitespace from column names
* Remove duplicate rows
* Replace infinite values
* Remove NaN values
* Save cleaned dataset as a new CSV

Example output:

```
CICIDS2017_Cleaned.csv
```

---

# Manual IP Assignment

Since the MachineLearningCVE dataset removes the original IP addresses, manually assign source and destination IPs according to the official CICIDS2017 documentation.

## Network Topology

### Firewall

* Public: 205.174.165.80
* Internal: 172.16.0.1

### DNS Server

* 192.168.10.3

### Attackers

| Machine | IP             |
| ------- | -------------- |
| Kali    | 205.174.165.73 |
| Windows | 205.174.165.69 |
| Windows | 205.174.165.70 |
| Windows | 205.174.165.71 |

### Victim Machines

| Machine              | IP            |
| -------------------- | ------------- |
| Ubuntu Web Server    | 192.168.10.50 |
| Ubuntu Server        | 192.168.10.51 |
| Ubuntu 14.4 (32-bit) | 192.168.10.19 |
| Ubuntu 14.4 (64-bit) | 192.168.10.17 |
| Ubuntu 16.4 (32-bit) | 192.168.10.16 |
| Ubuntu 16.4 (64-bit) | 192.168.10.12 |
| Windows 7            | 192.168.10.9  |
| Windows 8.1          | 192.168.10.5  |
| Windows Vista        | 192.168.10.8  |
| Windows 10 (32-bit)  | 192.168.10.14 |
| Windows 10 (64-bit)  | 192.168.10.15 |
| macOS                | 192.168.10.25 |

---

# Attack Schedule

## Monday

* Benign traffic only

---

## Tuesday

### FTP Patator

Time:

09:20–10:20

Attacker:

205.174.165.73

Victim:

192.168.10.50

---

### SSH Patator

Time:

14:00–15:00

Attacker:

205.174.165.73

Victim:

192.168.10.50

---

## Wednesday

Attacks:

* DoS Slowloris
* DoS SlowHTTPTest
* DoS Hulk
* DoS GoldenEye

Attacker:

205.174.165.73

Victim:

192.168.10.50

---

### Heartbleed

Attacker:

205.174.165.73

Victim:

192.168.10.51

---

## Thursday Morning

Attacks:

* Web Attack - Brute Force
* Web Attack - XSS
* Web Attack - SQL Injection

Victim:

192.168.10.50

---

## Thursday Afternoon

### Infiltration

Victims

* Windows Vista
* macOS

---

## Friday Morning

### Botnet

Victims

* Windows 10
* Windows 7
* Windows Vista
* Windows 8

---

## Friday Afternoon

### Port Scan

Victim:

Ubuntu Web Server

---

### DDoS LOIT

Attackers

* 205.174.165.69
* 205.174.165.70
* 205.174.165.71

Victim

192.168.10.50

---

# Memory Efficient Workflow

After creating the merged dataset:

1. Delete all temporary DataFrames
2. Invoke garbage collector
3. Reload only the merged CSV
4. Continue processing

---

# Clean Dataset

Save as

```
CICIDS2017_Cleaned.csv
```

---

# EDA Plan

Create a directory:

```
data_analysis/
```

Save every visualization as PNG.

Suggested charts:

## 1. Binary Class Distribution

BENIGN vs ATTACK

---

## 2. Multi-class Distribution

Attack-wise counts.

Display exact counts on each bar.

---

## 3. Correlation Heatmap

Numeric features only.

---

## 4. Boxplots

Show feature distributions and outliers.

---

## 5. Density Plots

Distribution of important numerical features.

---

## 6. Pairplot

Use only important features.

Random sample from the ML-ready dataset.

---

## 7. Histograms

Important numerical features.

---

## 8. Missing Values Heatmap

Visualize missing data.

---

## 9. Feature Statistics

Generate descriptive statistics.

---

# Feature Selection

After correlation analysis:

Dropped 17 highly correlated features:

* Total Backward Packets
* Total Length of Bwd Packets
* Fwd IAT Total
* Fwd IAT Max
* SYN Flag Count
* CWE Flag Count
* ECE Flag Count
* Average Packet Size
* Avg Fwd Segment Size
* Avg Bwd Segment Size
* Fwd Header Length.1
* Subflow Fwd Packets
* Subflow Fwd Bytes
* Subflow Bwd Packets
* Subflow Bwd Bytes
* Idle Max
* Idle Min

Final dataset shape:

```
(2,827,876, 56)
```

---

# Class Distribution

| Label                      |   Samples |
| -------------------------- | --------: |
| BENIGN                     | 2,271,320 |
| DoS Hulk                   |   230,124 |
| PortScan                   |   158,804 |
| DDoS                       |   128,025 |
| DoS GoldenEye              |    10,293 |
| FTP-Patator                |     7,935 |
| SSH-Patator                |     5,897 |
| DoS Slowloris              |     5,796 |
| DoS Slowhttptest           |     5,499 |
| Bot                        |     1,956 |
| Web Attack - Brute Force   |     1,507 |
| Web Attack - XSS           |       652 |
| Infiltration               |        36 |
| Web Attack - SQL Injection |        21 |
| Heartbleed                 |        11 |

---

# Scaling Observation

Original CSV size:

```
980 MB
```

Scaled CSV size:

```
~3 GB
```

Reason:

* CSV stores floating-point values as text.
* StandardScaler converts integers into long decimal values.
* More characters per value increase file size.

---

# Recommended Pipeline

## Stage 1

Binary Classification

```
Benign
↓

Attack?
```

Possible models:

* Random Forest
* XGBoost
* LightGBM
* MLP

Output:

```
Benign
Attack
```

---

## Stage 2

Only attack samples are passed to the GNN.

```
Attack

↓

Graph Construction

↓

Graph Neural Network

↓

Attack Classification
```

Attack classes include:

* DoS
* DDoS
* Bot
* PortScan
* Web Attack
* Infiltration
* Heartbleed

---

# Advantages

* Faster inference
* Reduced GNN computation
* Better scalability
* Lower memory usage
* Improved attack classification accuracy

---

# Future Work

* Graph construction using Source IP and Destination IP
* Node feature engineering
* Edge feature engineering
* PyTorch Geometric implementation
* GAT / GraphSAGE / GCN comparison
* Hierarchical IDS evaluation
