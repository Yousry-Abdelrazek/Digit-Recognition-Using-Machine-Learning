import pickle
import numpy as np
import pandas as pd
import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def load_preprocessed_data(input_file='preprocessed_data.pkl'):

    print(f"Loading preprocessed data from {input_file}...")
    
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print("Preprocessed data loaded successfully!")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test
# ============================================================
print("="*70)
print(" DATA LOADING AND PREPROCESSING")
print("="*70)
X_train, X_test, y_train, y_test = load_preprocessed_data()


# Display data information
print (f"x-train {X_train.shape[0]}")
print (f"x-test {X_test.shape[0]}")

print(f"num of classes: {len(np.unique(y_train))}")
print(f"Data split: {X_train.shape[0]/(X_train.shape[0]+X_test.shape[0])*100:.1f}% train, {X_test.shape[0]/(X_train.shape[0]+X_test.shape[0])*100:.1f}% test")
print()

# ==========================================================
print("="*70)
print(" MLP MODEL TRAINING FUNCTION")
print("="*70)

def train_mlp(X_train, y_train, hidden_layers=(128, 64), learning_rate=0.001,
              max_epochs=100, batch_size=32, verbose=True):
  
    if verbose:
        print(f"Hyperparameters:")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Max epochs: {max_epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Activation: ReLU")
        print(f"  - Solver: Adam optimizer")
        print ("Start training ...")

    
    # Start timing
    start_time = time.time()
    
    # Initialize MLP model
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers, 
        activation='relu',                
        solver='adam',                    
        alpha=0.0001,                       # L2 regularization parameter
        batch_size=batch_size,             
        learning_rate_init=learning_rate,   
        max_iter=max_epochs,                
        random_state=42,                    
        early_stopping=True,                
        validation_fraction=0.1,           
        n_iter_no_change=10,                
        verbose=False                    
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Calculate training time
    training_time = time.time() - start_time
    
    if verbose:
        print(f"\nTraining completed!")
        print(f"  - Actual iterations: {model.n_iter_}")
        print(f"  - Training time: {training_time:.2f} seconds")
        
        
        if hasattr(model, 'loss_'):
            print(f"  - Final loss: {model.loss_:.6f}")
    
    return model, training_time


# ============================================================
print("="*70)
print(" TRAINING MLP WITH DEFAULT HYPERPARAMETERS")
print("="*70)

mlp_model, mlp_training_time = train_mlp(
    X_train, 
    y_train,
    hidden_layers=(128, 64),
    learning_rate=0.001,
    max_epochs=100,
    batch_size=32
)
print()


# ============================================================
print("="*70)
print(" MODEL EVALUATION ON TEST SET")
print("="*70)

# Make predictions on test set
y_pred = mlp_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Display metrics
print("Performance Metrics:")
print(f"  - Accuracy:  {accuracy*100:.2f}%")
print(f"  - Precision: {precision:.4f}")
print(f"  - Recall:    {recall:.4f}")
print(f"  - F1-Score:  {f1:.4f}")
print()








#CONFUSION MATRIX GENERATION
# ============================================================
print("="*70)
print("CONFUSION MATRIX")
print("="*70)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix (rows=actual, columns=predicted):")
print(cm)
print()

# Calculate per-class accuracy
print("Per-class Accuracy:")
for i in range(10):
    class_accuracy = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
    print(f"  Digit {i}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
print()








#  K-FOLD CROSS-VALIDATION (K=5)
# ============================================================
print("="*70)
print(" 5-FOLD CROSS-VALIDATION")
print("="*70)

# Create a fresh model for cross-validation
cv_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=32,
    learning_rate_init=0.001,
    max_iter=100,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10
)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(cv_model, X_train, y_train, cv=5, scoring='accuracy')

# Display cross-validation results
print(f"\nCross-Validation Results (k=5):")
print(f"  - Fold scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"  - Mean accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print(f"  - Standard deviation: {cv_scores.std():.4f}")
print(f"  - Min accuracy: {cv_scores.min():.4f}")
print(f"  - Max accuracy: {cv_scores.max():.4f}")
print()

# HYPERPARAMETER SENSITIVITY ANALYSIS
# ============================================================
print("="*70)
print("HYPERPARAMETER SENSITIVITY ANALYSIS")
print("="*70)

# Test different architectures
architectures = [(64, 32), (128, 64), (256, 128), (128, 64, 32)]
print("Testing different neural network architectures:")
print("-" * 70)

arch_results = []
for arch in architectures:
    model, train_time = train_mlp(
        X_train, y_train, 
        hidden_layers=arch, 
        learning_rate=0.001,
        max_epochs=100,
        batch_size=32,
        verbose=False
    )
    y_pred_arch = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_arch)
    arch_results.append((arch, acc, train_time))
    print(f"Architecture {arch}: Accuracy = {acc:.4f}, Time = {train_time:.2f}s")

print()

# Test different learning rates
learning_rates = [0.0001, 0.001, 0.01]
print("Testing different learning rates:")
print("-" * 70)

lr_results = []
for lr in learning_rates:
    model, train_time = train_mlp(
        X_train, y_train,
        hidden_layers=(128, 64),
        learning_rate=lr,
        max_epochs=100,
        batch_size=32,
        verbose=False
    )
    y_pred_lr = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_lr)
    lr_results.append((lr, acc, train_time))
    print(f"Learning rate {lr}: Accuracy = {acc:.4f}, Time = {train_time:.2f}s")
    
    


# alphas = [0, 0.00001, 0.0001, 0.001, 0.01]
# print("Testing different alpha values:")
# print("-" * 70)

# for alpha_val in alphas:
#     model = MLPClassifier(
#         hidden_layer_sizes=(128, 64),
#         solver='adam',
#         alpha=alpha_val,
#         learning_rate_init=0.001,
#         max_iter=100,
#         random_state=42
#     )
#     model.fit(X_train_norm, Y_train)
#     train_acc = model.score(X_train_norm, Y_train)
#     test_acc = model.score(X_test_norm, Y_test)
    
#     print(f"Alpha {alpha_val}: Train={train_acc:.4f}, Test={test_acc:.4f}")


print()

# Test different batch sizes
batch_sizes = [16, 32, 64, 128]
print("Testing different batch sizes:")
print("-" * 70)

batch_results = []
for bs in batch_sizes:
    model, train_time = train_mlp(
        X_train, y_train,
        hidden_layers=(128, 64),
        learning_rate=0.001,
        max_epochs=100,
        batch_size=bs,
        verbose=False
    )
    y_pred_bs = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_bs)
    batch_results.append((bs, acc, train_time))
    print(f"Batch size {bs}: Accuracy = {acc:.4f}, Time = {train_time:.2f}s")

print()


# FINAL result
# ============================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print("MLP Neural Network - Complete Analysis Summary")
print("-" * 70)
print(f"Dataset: DIDA (10k samples)")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"\nBest Model Configuration:")
print(f"  - Architecture: (128, 64)")
print(f"  - Learning rate: 0.001")
print(f"  - Batch size: 32")
print(f"  - Training time: {mlp_training_time:.2f}s")
print(f"\nTest Set Performance:")
print(f"  - Accuracy: {accuracy:.4f}")
print(f"  - Precision: {precision:.4f}")
print(f"  - Recall: {recall:.4f}")
print(f"  - F1-Score: {f1:.4f}")
print(f"\nCross-Validation (5-fold):")
print(f"  - Mean accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print("\n" + "="*70)
print("Analysis complete! All results saved.")
print("="*70)