# Part 3: Ethics & Optimization

## 1. Ethical Considerations

### Potential Biases in MNIST Model

**Data Bias:**
- MNIST contains primarily Western/Arabic numerals
- Handwriting styles may not represent global diversity
- Limited representation of different writing instruments and surfaces

**Model Bias:**
- May perform poorly on handwritten digits from different cultural backgrounds
- Potential issues with digits written by people with motor disabilities

### Mitigation Strategies

**TensorFlow Fairness Indicators:**
```python
# Example fairness analysis
from tensorflow_model_analysis import FairnessIndicators

# Would analyze performance across different subgroups
# Check for equalized odds, demographic parity

Data Augmentation:

Add rotated, skewed, and noisy versions of digits

Include digits from diverse writing styles

Biases in Amazon Reviews Model
Language Bias:

Model trained primarily on English reviews

Cultural differences in expression not captured

Potential bias against non-native English speakers

Product Bias:

Over-representation of popular brands


# Custom rules for fairness
from spacy.language import Language

@Language.component("fairness_check")
def fairness_check(doc):
    # Implement custom fairness rules
    # Check for biased language patterns
    # Flag potential unfair comparisons
    return doc
2. Troubleshooting Challenge
Buggy Code Analysis
Common TensorFlow Errors and Fixes:

Dimension Mismatches:

Use tf.reshape() or tf.expand_dims()

Ensure consistent input shapes

Incorrect Loss Functions:

Use sparse_categorical_crossentropy for integer labels

Use categorical_crossentropy for one-hot encoded labels

Gradient Issues:

Check learning rate

Use gradient clipping for stability

Sample Debugging Solution

# Original buggy code (example)
def buggy_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)  # Missing activation
    ])
    model.compile(optimizer='adam', loss='mse')  # Wrong loss for classification
    return model

# Fixed version
def fixed_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # Added flatten
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # Correct activation
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # Correct loss
                  metrics=['accuracy'])
    return model

Optimization Recommendations
Model Optimization:

Use mixed precision training

Implement model pruning

Apply quantization for deployment

Performance Optimization:

Use TensorFlow Dataset API for efficient data loading

Implement caching and prefetching

Utilize GPU acceleration

Fairness Optimization:

Regular bias auditing

Diverse training data collection

Transparent model documentation



















Under-representation of products from smaller companies

spaCy Rule-Based Mitigation
