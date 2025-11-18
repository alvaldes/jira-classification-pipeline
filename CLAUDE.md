# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a machine learning pipeline for software defect prediction using JIRA datasets. The project implements the HCBOU (Hybrid Cluster-Based Oversampling and Undersampling) method for multiclass imbalanced classification to predict software bugs and defects in various software projects.

## Repository Structure

```
├── datasets/           # CSV files with software metrics and defect data
│   ├── activemq-5.0.0.csv
│   ├── groovy-1_6_BETA_1.csv
│   ├── hbase-0.94.0.csv
│   └── ...other project datasets
├── preprocessing/      # Jupyter notebooks for data preprocessing
│   ├── groovy-1_5_7.ipynb
│   ├── hbase-0.94.0.ipynb
│   └── ...other preprocessing notebooks
├── class-blance/      # HCBOU balancing implementation notebooks
│   ├── activemq-5.0.0.ipynb
│   └── activemq-5.0.0-hcbou.ipynb
├── utils/             # Utility code and analysis tools
│   └── HCBOU Code.ipynb
└── papers/            # Research papers and references
```

## Core Technologies

- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and metrics
- **Imbalanced-learn**: Specialized tools for imbalanced datasets (SMOTE, ClusterCentroids)
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter Notebooks**: Development environment for all analysis

## Key Features & Algorithms

### HCBOU Implementation
The repository implements the Hybrid Cluster-Based Oversampling and Undersampling (HCBOU) method:

1. **Majority Class Balancing**: Uses ClusterCentroids with MiniBatchKMeans for undersampling
2. **Minority Class Balancing**: Uses clustering + SMOTE for intelligent oversampling
3. **Hyperparameters**:
   - `maxclusterMaj`: Maximum clusters for majority class (typically 8-10)
   - `maxclusterMin`: Maximum clusters for minority class (typically 6)
   - `KSMOTE`: K-neighbors for SMOTE (typically 2-3)
   - `MinClusterObs`: Minimum observations per cluster (typically 3-5)

### Data Pipeline
1. **Data Loading**: CSV files with software metrics (LOC, cyclomatic complexity, etc.)
2. **Preprocessing**: StandardScaler normalization, missing value handling
3. **Class Balancing**: HCBOU implementation for imbalanced binary/multiclass problems
4. **Feature Selection**: Decision tree-based feature importance
5. **Model Training**: Random Forest, AdaBoost with OneVsRest/OneVsOne strategies
6. **Evaluation**: Multiple metrics including ROC-AUC, F1-score, geometric mean

## Dataset Structure

Each dataset contains software metrics with these key columns:
- **File**: Source file path
- **Code Metrics**: CountLineCode, AvgCyclomatic, MaxCyclomatic, etc.
- **Process Metrics**: COMM, ADEV, DDEV, Added_lines, Del_lines
- **Target Variables**: RealBug (boolean), RealBugCount (integer)

## Common Development Tasks

### Running Analysis on a New Dataset
```python
# Load and preprocess data
df = pd.read_csv('path/to/dataset.csv')
df = df.drop(columns=['HeuBug', 'HeuBugCount', 'RealBugCount'])
X = df.drop(columns=['RealBug'])
y = df['RealBug']

# Train/test split and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=['float64', 'int64']))

# Apply HCBOU balancing (see utils/HCBOU Code.ipynb for full implementation)
```

### HCBOU Configuration
Key hyperparameters to adjust based on dataset size:
```python
ClassNumber = 2  # Binary or multiclass
maxclusterMaj = 8   # Max clusters for majority class
maxclusterMin = 6   # Max clusters for minority class
KSMOTE = 3         # SMOTE neighbors
MinClusterObs = 5  # Min observations per cluster
```

### Evaluation Pipeline
```python
# Multiple evaluation strategies
OvR_clf = OneVsRestClassifier(RandomForestClassifier())
OvO_clf = OneVsOneClassifier(AdaBoostClassifier())

# Key metrics
- accuracy_score, f1_score, precision_score, recall_score
- geometric_mean_score (for imbalanced data)
- roc_auc_score_multiclass (custom implementation)
- cohen_kappa_score, hamming_loss
```

## File Conventions

- **Notebooks**: Use descriptive names like `project-version.ipynb`
- **HCBOU Implementation**: Separate notebooks for balancing (`*-hcbou.ipynb`)
- **Data Files**: CSV format with standardized column names
- **Results**: Excel files with structured evaluation metrics

## Important Notes

- All datasets are already preprocessed and cleaned
- No dependency management files (requirements.txt) exist - dependencies must be installed manually
- The HCBOU method requires careful hyperparameter tuning for each dataset
- Use the utils/HCBOU Code.ipynb as the reference implementation
- Results are typically saved to Excel workbooks with multiple worksheets for different algorithms

## Research Context

This work is based on academic research in software defect prediction, specifically focusing on:
- Imbalanced classification problems in software engineering
- Hybrid sampling techniques for multiclass datasets
- Evaluation of machine learning models on real-world software projects