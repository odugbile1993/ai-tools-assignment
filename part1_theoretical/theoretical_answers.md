# Part 1: Theoretical Understanding

## Q1: TensorFlow vs PyTorch

**Primary Differences:**

1. **Computational Graph:**
   - TensorFlow: Static graph (define-and-run)
   - PyTorch: Dynamic graph (define-by-run)

2. **Debugging:**
   - TensorFlow: Requires TensorBoard or special tools
   - PyTorch: Standard Python debugging with pdb

3. **API Design:**
   - TensorFlow: Multiple API levels (Keras, Estimators)
   - PyTorch: Pythonic, object-oriented approach

4. **Deployment:**
   - TensorFlow: Better production deployment (TF Serving)
   - PyTorch: Catching up with TorchServe

**When to choose:**

- **TensorFlow:** Production systems, mobile deployment, large-scale distributed training
- **PyTorch:** Research, rapid prototyping, when Pythonic syntax is preferred

## Q2: Jupyter Notebooks Use Cases

1. **Exploratory Data Analysis:** Interactive data visualization and statistical analysis
2. **Model Prototyping:** Quick iteration and experimentation with different architectures

## Q3: spaCy vs Basic String Operations

spaCy provides:
- Pre-trained statistical models for accurate NER
- Linguistic annotations (POS tagging, dependency parsing)
- Efficient tokenization handling edge cases
- Built-in word vectors and similarity measures
- Production-ready pipeline architecture

## Comparative Analysis: Scikit-learn vs TensorFlow

| Aspect | Scikit-learn | TensorFlow |
|--------|-------------|------------|
| Target Applications | Classical ML algorithms | Deep learning, neural networks |
| Ease of Use | Very beginner-friendly | Steeper learning curve |
| Community Support | Extensive for traditional ML | Largest for deep learning |
| Performance | Optimized for small-medium data | GPU acceleration for large data |
| Deployment | Simple models | Comprehensive serving options |
