# Handwritten Digit Recognition Using Machine Learning Techniques

A comprehensive implementation comparing multiple machine learning algorithms for handwritten digit classification using the DIDA dataset.

## 📋 Project Overview

This project implements and compares four different machine learning approaches for digit recognition:
- Multi-layer Perceptron Neural Network
- Linear Regression (One-vs-All)
- Logistic Regression
- Gaussian Naive Bayes

## 📊 Dataset

**DIDA Dataset (10k version)**
- Source: https://didadataset.github.io/DIDA/
- Size: 10,000 samples
- Image format: 28×28 grayscale images (784 features)
- Classes: 10 digits (0-9)
- Split: 80% training (8,000), 20% testing (2,000)

## 🗂️ Project Structure

```
.
├── data_loading_and_preprocessing.ipynb    # Data preprocessing pipeline
├── Linear_for_project.ipynb                # Linear Regression implementation
├── logistic_regression_from_scratch.ipynb  # Logistic Regression from scratch
├── Neural Network (1).py                   # MLP Neural Network
├── Gaussian_Naive_Bayes.ipynb             # Naive Bayes classifier
├── preprocessed_data.pkl                   # Preprocessed dataset (generated)
└── README.md                               # This file
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib pillow
```

### Data Preprocessing

1. **Load and preprocess the dataset:**
```python
# Run data_loading_and_preprocessing.ipynb
# This will:
# - Load images from the 10000/ folder structure
# - Normalize pixel values to [0,1]
# - Split into train/test sets
# - Save to preprocessed_data.pkl
```

**Key preprocessing steps:**
- Pixel normalization: [0, 255] → [0, 1]
- Image inversion: Black on white → White on black
- Train/test split: 80/20 with stratification
- Data shape: (samples, 784)

## 🤖 Model Implementations

### 1. Multi-Layer Perceptron Neural Network

**File:** `Neural Network (1).py`

**Architecture:**
- Input layer: 784 neurons
- Hidden layers: (128, 64) neurons with ReLU activation
- Output layer: 10 neurons (softmax)
- Optimizer: Adam
- Early stopping enabled

**Hyperparameters:**
- Learning rate: 0.001
- Max epochs: 100
- Batch size: 32
- L2 regularization: 0.0001

**Performance:**
```
Accuracy:  ~85-87%
Precision: 0.85-0.87
Recall:    0.85-0.87
F1-Score:  0.85-0.87
Training time: ~50-80 seconds
```

**Key Features:**
- Cross-validation (5-fold)
- Hyperparameter sensitivity analysis
- Per-class accuracy breakdown

---

### 2. Linear Regression (One-vs-All)

**File:** `Linear_for_project.ipynb`

**Implementation:**
- Custom Batch Gradient Descent (BGD)
- One-vs-All multi-class strategy
- 10 binary classifiers (one per digit)

**Hyperparameters:**
- Learning rate (alpha): 0.0055
- Iterations: 1000
- Early stopping based on cost convergence

**Performance:**
```
Accuracy:  ~72-74%
Precision: 0.72-0.74
Recall:    0.72-0.74
F1-Score:  0.72-0.74
Training time: ~90-100 seconds
```

**Key Features:**
- Cost function monitoring
- Hyperparameter tuning (alpha testing)
- Cross-validation ready

---

### 3. Logistic Regression (From Scratch)

**File:** `logistic_regression_from_scratch.ipynb`

**Implementation:**
- Custom sigmoid activation
- Gradient descent optimization
- One-vs-All strategy

**Hyperparameters:**
- Learning rate: 0.1
- Iterations: 1000
- Convergence tolerance for early stopping

**Performance:**
```
Accuracy:  ~74-76%
Precision: 0.74-0.76
Recall:    0.74-0.76
F1-Score:  0.74-0.76
Training time: ~170-180 seconds
```

**Key Features:**
- Cost tracking per iteration
- Hyperparameter sensitivity analysis
- 5-fold cross-validation
- Training time analysis

---

### 4. Gaussian Naive Bayes

**File:** `Gaussian_Naive_Bayes.ipynb`

**Implementation:**
- Scikit-learn GaussianNB
- With and without PCA (80% variance)

**Hyperparameters:**
- Variance smoothing: 1e-9
- PCA components: 0.80 (80% variance retained)

**Performance:**

**Without PCA:**
```
Accuracy:  ~51-53%
Precision: 0.54
Recall:    0.51
F1-Score:  0.51
```

**With PCA:**
```
Accuracy:  ~62-64%
Precision: 0.66
Recall:    0.63
F1-Score:  0.64
Training time: <1 second
```

**Key Features:**
- PCA dimensionality reduction
- 5-fold cross-validation
- Extremely fast training

---

## 📈 Model Comparison

### Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **Neural Network** | **85-87%** | **0.85-0.87** | **0.85-0.87** | **0.85-0.87** | 50-80s |
| **Logistic Regression** | 74-76% | 0.74-0.76 | 0.74-0.76 | 0.74-0.76 | 170-180s |
| **Linear Regression** | 72-74% | 0.72-0.74 | 0.72-0.74 | 0.72-0.74 | 90-100s |
| **Naive Bayes (PCA)** | 62-64% | 0.66 | 0.63 | 0.64 | <1s |
| **Naive Bayes (No PCA)** | 51-53% | 0.54 | 0.51 | 0.51 | <1s |

### Key Insights

**1. Best Overall Performance:**
- **Winner:** Neural Network (MLP)
- **Reason:** Complex feature extraction through multiple layers
- **Trade-off:** Longer training time but significantly better accuracy

**2. Best Speed-to-Accuracy Ratio:**
- **Winner:** Linear Regression
- **Reason:** Reasonable accuracy (~73%) with moderate training time
- **Use case:** Good baseline model for quick prototyping

**3. Fastest Training:**
- **Winner:** Gaussian Naive Bayes
- **Reason:** Probabilistic model with simple calculations
- **Trade-off:** Lower accuracy, not suitable for production

**4. Most Consistent:**
- **Winner:** Neural Network
- **Reason:** Low variance across cross-validation folds
- **Cross-validation std:** ~0.01-0.02

### Strengths & Weaknesses

**Neural Network (MLP):**
- ✅ Best accuracy and F1-score
- ✅ Robust feature learning
- ✅ Good generalization
- ❌ Longer training time
- ❌ More hyperparameters to tune
- ❌ Requires more computational resources

**Logistic Regression:**
- ✅ Solid performance (~75% accuracy)
- ✅ Interpretable model
- ✅ Well-suited for multi-class problems
- ❌ Slower convergence
- ❌ Limited by linear decision boundaries

**Linear Regression:**
- ✅ Fast training relative to accuracy
- ✅ Simple implementation
- ✅ Good for baseline comparison
- ❌ Not designed for classification
- ❌ Lower accuracy than logistic regression

**Naive Bayes:**
- ✅ Extremely fast training (<1s)
- ✅ PCA significantly improves performance (+10-12%)
- ❌ Strong independence assumption
- ❌ Lowest accuracy overall
- ❌ Not suitable for complex patterns

### Hyperparameter Impact

**Neural Network:**
- Architecture (128, 64): Balanced complexity vs. overfitting
- Learning rate 0.001: Stable convergence
- Batch size 32: Good trade-off between speed and stability

**Logistic Regression:**
- Alpha = 0.1: Optimal convergence rate
- Lower alpha (0.05): Slower but more stable
- Higher alpha (0.2): Faster but less stable

**Linear Regression:**
- Alpha = 0.0055: Best balance found through grid search
- Cost monitoring: Early stopping prevents overfitting

## 🔬 Evaluation Metrics

All models are evaluated using:

1. **Accuracy**: Overall correctness
2. **Precision**: True positives / (True positives + False positives)
3. **Recall**: True positives / (True positives + False negatives)
4. **F1-Score**: Harmonic mean of precision and recall
5. **Confusion Matrix**: Per-class error analysis
6. **Cross-Validation**: 5-fold CV for generalization assessment

## 📝 Usage Guide

### 1. Preprocessing (Required First)

```python
# Run: data_loading_and_preprocessing.ipynb
# Generates: preprocessed_data.pkl
```

### 2. Train Models

**Neural Network:**
```python
# Run: Neural Network (1).py
python "Neural Network (1).py"
```

**Linear Regression:**
```python
# Run all cells in: Linear_for_project.ipynb
```

**Logistic Regression:**
```python
# Run all cells in: logistic_regression_from_scratch.ipynb
```

**Naive Bayes:**
```python
# Run all cells in: Gaussian_Naive_Bayes.ipynb
```

### 3. Load Preprocessed Data (in your code)

```python
import pickle

with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
    
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
```

## 🎯 Results & Analysis

### Cross-Validation Results

**Neural Network (5-fold):**
- Mean accuracy: 85.5% (±1.5%)
- Consistent performance across folds

**Logistic Regression (5-fold):**
- Mean accuracy: 69.8% (±1.4%)
- Good stability

**Linear Regression:**
- Manual train/test split
- ~73% test accuracy

**Naive Bayes (5-fold with PCA):**
- Mean accuracy: 62.8% (±0.6%)
- Very consistent but limited by model assumptions

### Confusion Matrix Analysis

Common misclassifications across all models:
- **3 ↔ 5**: Similar curved shapes
- **4 ↔ 9**: Overlapping features
- **7 ↔ 1**: Similar vertical strokes

## 🔮 Real-World Applicability

### Recommended Use Cases

**Neural Network:**
- Production digit recognition systems
- High-accuracy requirements
- Sufficient computational resources available

**Logistic Regression:**
- Embedded systems with moderate resources
- When interpretability is important
- Real-time applications with acceptable accuracy

**Linear Regression:**
- Quick baseline implementations
- Educational purposes
- Resource-constrained environments

**Naive Bayes:**
- Rapid prototyping
- When extreme speed is critical
- Low-stakes applications

### Scalability Considerations

1. **Dataset Size:**
   - Neural Network: Scales well with more data
   - Linear/Logistic Regression: Linear scaling
   - Naive Bayes: Very efficient even with large datasets

2. **Feature Dimensionality:**
   - PCA can reduce computational cost
   - Neural networks handle high dimensions naturally
   - Naive Bayes benefits significantly from PCA

3. **Inference Speed:**
   - All models: Fast prediction (<1ms per sample)
   - Naive Bayes: Fastest
   - Neural Network: Slightly slower but negligible

## 🤝 Contributing

This is a graduate-level academic project. Contributions following the course requirements are welcome.

## 📄 License

Academic use only. Part of a Machine Learning course project.

## 👥 Author

Graduate-level Machine Learning course project

---

## 📚 References

- DIDA Dataset: https://didadataset.github.io/DIDA/
- Scikit-learn Documentation: https://scikit-learn.org/
- Neural Networks and Deep Learning (Michael Nielsen)

---

**Note:** All hyperparameters and architectures were selected based on empirical testing and cross-validation results to optimize the balance between accuracy and computational efficiency.