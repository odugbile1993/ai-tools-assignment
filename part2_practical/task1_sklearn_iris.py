"""
Task 1: Iris Classification with Decision Trees
Dataset: Iris Species Dataset
Goal: Predict iris species using decision tree classifier
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
print("Loading Iris dataset...")
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"Dataset shape: {X.shape}")
print(f"Features: {feature_names}")
print(f"Target classes: {target_names}")

# Check for missing values
print(f"\nMissing values in features: {np.isnan(X).sum()}")

# Exploratory Data Analysis
plt.figure(figsize=(12, 8))

# Feature distributions
plt.subplot(2, 2, 1)
for i in range(3):
    plt.hist(X[y == i, 0], alpha=0.7, label=target_names[i])
plt.xlabel(feature_names[0])
plt.ylabel('Frequency')
plt.legend()
plt.title('Sepal Length Distribution')

plt.subplot(2, 2, 2)
for i in range(3):
    plt.hist(X[y == i, 1], alpha=0.7, label=target_names[i])
plt.xlabel(feature_names[1])
plt.ylabel('Frequency')
plt.legend()
plt.title('Sepal Width Distribution')

plt.subplot(2, 2, 3)
for i in range(3):
    plt.hist(X[y == i, 2], alpha=0.7, label=target_names[i])
plt.xlabel(feature_names[2])
plt.ylabel('Frequency')
plt.legend()
plt.title('Petal Length Distribution')

plt.subplot(2, 2, 4)
for i in range(3):
    plt.hist(X[y == i, 3], alpha=0.7, label=target_names[i])
plt.xlabel(feature_names[3])
plt.ylabel('Frequency')
plt.legend()
plt.title('Petal Width Distribution')

plt.tight_layout()
plt.savefig('iris_distributions.png')
plt.show()

# Preprocessing: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train Decision Tree classifier
print("\nTraining Decision Tree Classifier...")
dt_classifier = DecisionTreeClassifier(
    max_depth=3, 
    random_state=42,
    criterion='gini'
)
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred = dt_classifier.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print("\n" + "="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Feature importance
feature_importance = dt_classifier.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names)
plt.title('Feature Importance in Decision Tree')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

print("\nFeature Importance:")
for name, importance in zip(feature_names, feature_importance):
    print(f"{name}: {importance:.4f}")

# Visualize decision tree (simplified)
from sklearn.tree import plot_tree

plt.figure(figsize=(15, 10))
plot_tree(dt_classifier, 
          feature_names=feature_names,
          class_names=target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree Visualization')
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Task 1 completed successfully!")
