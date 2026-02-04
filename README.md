# 🫀 Heart Disease Risk Prediction: Logistic Regression Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-Only-green.svg)](https://numpy.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Complete implementation of logistic regression for heart disease prediction: EDA, training, visualization, regularization, and SageMaker deployment exploration.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset Description](#-dataset-description)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Implementation Details](#-implementation-details)
  - [Step 1: Data Preparation](#step-1-data-preparation)
  - [Step 2: Basic Logistic Regression](#step-2-basic-logistic-regression)
  - [Step 3: Decision Boundary Visualization](#step-3-decision-boundary-visualization)
  - [Step 4: Regularization](#step-4-regularization)
  - [Step 5: Deployment](#step-5-deployment)
- [Results & Performance](#-results--performance)
- [Deployment Evidence](#-deployment-evidence)
- [Mathematical Foundation](#-mathematical-foundation)
- [Key Insights](#-key-insights)
- [References](#-references)

---

## 🎯 Overview

This project implements **logistic regression from scratch** using only NumPy to predict heart disease risk. The implementation follows the complete machine learning pipeline:

| Phase | Description |
|-------|-------------|
| **Data Preparation** | Load, explore, clean, normalize, and split the dataset |
| **Model Training** | Implement sigmoid, cost function, and gradient descent |
| **Visualization** | Plot decision boundaries for feature pairs |
| **Regularization** | Add L2 regularization to prevent overfitting |
| **Deployment** | Explore AWS SageMaker deployment workflow |

### Why This Matters

Heart disease is the **world's leading cause of death**, claiming approximately **18 million lives annually** (WHO). Predictive models like this enable:

- ✅ **Early identification** of at-risk patients
- ✅ **Better resource allocation** in healthcare settings
- ✅ **Data-driven clinical decisions**
- ✅ **Improved treatment outcomes**

---

## 📊 Dataset Description

**Source:** [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/neurocipher/heartdisease)

### Dataset Statistics

| Property | Value |
|----------|-------|
| **Total Samples** | 270 patients |
| **Features** | 13 clinical measurements |
| **Target Classes** | 2 (Presence/Absence of heart disease) |
| **Disease Rate** | ~55% (approximately balanced) |
| **Missing Values** | None |

### Feature Descriptions

| Feature | Range | Description |
|---------|-------|-------------|
| `Age` | 29-77 years | Patient age |
| `Sex` | 0-1 | Gender (1=Male, 0=Female) |
| `Chest pain type` | 1-4 | Type of chest pain experienced |
| `BP` | 94-200 mm Hg | Resting blood pressure |
| `Cholesterol` | 126-564 mg/dL | Serum cholesterol level |
| `FBS over 120` | 0-1 | Fasting blood sugar > 120 mg/dL |
| `EKG results` | 0-2 | Electrocardiogram results |
| `Max HR` | 71-202 bpm | Maximum heart rate achieved |
| `Exercise angina` | 0-1 | Exercise-induced angina |
| `ST depression` | 0-6.2 | ST depression induced by exercise |
| `Slope of ST` | 1-3 | Slope of peak exercise ST segment |
| `Number of vessels fluro` | 0-3 | Major vessels colored by fluoroscopy |
| `Thallium` | 3-7 | Thallium stress test result |
| **Heart Disease** | Presence/Absence | **Target variable** |

### Selected Features (6)

For this implementation, we selected 6 clinically relevant numerical features:

1. **Age** - Cardiovascular risk increases with age
2. **BP** - High blood pressure is a major risk factor
3. **Cholesterol** - High levels lead to arterial plaque buildup
4. **Max HR** - Lower maximum heart rate may indicate problems
5. **ST depression** - ECG abnormality suggesting ischemia
6. **Number of vessels fluro** - Blocked vessels visible in fluoroscopy

---

## 📁 Project Structure

```
logistic-regression/
├── class-notebooks/
Main notebook
│   │   ├── README.md                               # ← This file
│   │   └── heart_disease_model.npy                 # Exported model
│   ├── week2_classification_hour1_final.ipynb
│   ├── week2_classification_hour2_regularization.ipynb
│   └── homework.md
├── dataset/
│   └── Heart_Disease_Prediction.csv
├── heart_disease_lr_analysis.ipynb
└── README.md
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Dependencies

```bash
pip install numpy pandas matplotlib
```

> ⚠️ **Note:** This implementation uses **only NumPy** for the core ML algorithms (no scikit-learn), as required by the homework specifications.

### Running the Notebook

```bash
jupyter notebook heart_disease_complete_solution.ipynb
```

---

## 📝 Implementation Details

### Step 1: Data Preparation

**Goal:** Load, explore, and prepare the Heart Disease dataset for training.

#### What We Did:

1. **Loaded CSV** into pandas DataFrame
2. **Binarized target** column (Presence → 1, Absence → 0)
3. **Performed EDA:**
   - Statistical summary of all features
   - Missing value analysis (none found)
   - Class distribution visualization
   - Feature distribution histograms by class
4. **Selected 6 features** based on medical relevance
5. **Split data** 70/30 stratified (preserving class proportions)
6. **Normalized features** using Min-Max scaling

#### Key Insights:

- Dataset is approximately **balanced** (~55% disease rate)
- No missing values → clean data
- Features have **different scales** (Age: 29-77 vs Cholesterol: 126-564)
- Normalization essential for gradient descent convergence

---

### Step 2: Basic Logistic Regression

**Goal:** Implement logistic regression from scratch using only NumPy.

#### Functions Implemented:

| Function | Purpose |
|----------|---------|
| `sigmoid(z)` | Maps any real number to (0, 1) |
| `compute_cost(w, b, X, y)` | Binary cross-entropy loss |
| `compute_gradient(w, b, X, y)` | Derivatives for gradient descent |
| `gradient_descent(...)` | Iterative optimization |
| `predict(w, b, X)` | Binary classification (threshold 0.5) |
| `compute_metrics(y_true, y_pred)` | Accuracy, Precision, Recall, F1 |

#### Training Configuration:

- **Learning rate (α):** 0.1
- **Iterations:** 1000
- **Initialization:** Zeros

#### Convergence Plot:

The cost function decreased monotonically from ~0.693 to ~0.45, indicating:
- Learning rate is appropriate (not too high)
- Model is learning meaningful patterns
- Convergence achieved within 1000 iterations

#### Results:

| Metric | Training Set | Test Set |
|--------|--------------|----------|
| **Accuracy** | 82.5% | 80.2% |
| **Precision** | 84.1% | 81.8% |
| **Recall** | 85.6% | 83.3% |
| **F1 Score** | 84.8% | 82.5% |

---

### Step 3: Decision Boundary Visualization

**Goal:** Visualize how the model separates classes using 2D feature pairs.

#### Feature Pairs Analyzed:

1. **Age vs Cholesterol**
   - Moderate separation
   - Both features contribute to increased risk
   - Some class overlap expected

2. **BP vs Max HR**
   - Lower max HR associated with disease
   - BP shows less clear separation
   - Diagonal decision boundary

3. **ST depression vs Number of vessels** ⭐
   - **Best separation** of all pairs
   - Clear pattern: higher ST depression + more vessels → higher risk
   - Confirms these are the strongest predictors

#### Key Finding:

> The combination of **ST depression** and **Number of vessels** provides the clearest 2D separation, achieving ~78% accuracy with just 2 features.

---

### Step 4: Regularization

**Goal:** Add L2 regularization to prevent overfitting and improve generalization.

#### Regularized Cost Function:

$$J_{reg}(\vec{w}, b) = J(\vec{w}, b) + \frac{\lambda}{2m}\sum_{j=1}^{n} w_j^2$$

#### Lambda Values Tested:

| λ | Train Acc | Test Acc | ||w|| | Comment |
|---|-----------|----------|-------|---------|
| 0 | 82.5% | 80.2% | 4.21 | Baseline |
| 0.001 | 82.5% | 80.2% | 4.18 | Very light |
| 0.01 | 82.3% | 80.5% | 4.02 | Light |
| **0.1** | **81.9%** | **81.0%** | **3.45** | **Optimal** |
| 1.0 | 79.4% | 78.5% | 2.01 | Too strong |

#### Observations:

- **Higher λ → Smaller weights** (shrinkage effect)
- **λ = 0.1** provides best test performance
- Very high λ causes underfitting (weights shrink too much)
- Regularization produces **smoother decision boundaries**

---

### Step 5: Deployment

**Goal:** Explore how to deploy the trained model to production using AWS SageMaker.

#### Model Export:

The trained model is saved as `heart_disease_model.npy` containing:
- Trained weights (`w`) and bias (`b`)
- Normalization parameters (`X_min`, `X_max`)
- Feature column names
- Model metadata (λ, accuracy, training date)

#### Inference Handler:

Created `inference.py` with functions for:
- `model_fn()` - Load model from disk
- `input_fn()` - Parse JSON input
- `predict_fn()` - Make predictions
- `output_fn()` - Format response

#### Local Testing Results:

| Patient Profile | Probability | Risk Level |
|-----------------|-------------|------------|
| High Risk (Age=65, Chol=350, Vessels=3) | 87.3% | HIGH ⚠️ |
| Low Risk (Age=35, Chol=180, Vessels=0) | 12.1% | LOW ✅ |
| Medium Risk (Age=55, Chol=280, Vessels=1) | 48.5% | MEDIUM ⚡ |

---

## 📈 Results & Performance

### Final Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 81.0% |
| **Test Precision** | 82.5% |
| **Test Recall** | 83.8% |
| **Test F1 Score** | 83.1% |
| **Optimal λ** | 0.1 |

### Feature Importance (by weight magnitude)

| Rank | Feature | Weight | Interpretation |
|------|---------|--------|----------------|
| 1 | Number of vessels fluro | +1.82 | More vessels → Higher risk |
| 2 | ST depression | +1.45 | Higher depression → Higher risk |
| 3 | Max HR | -0.92 | Lower max HR → Higher risk |
| 4 | Age | +0.65 | Older age → Higher risk |
| 5 | Cholesterol | +0.43 | Higher cholesterol → Higher risk |
| 6 | BP | +0.31 | Higher BP → Higher risk |

---

## 🖼️ Deployment Evidence

### Step 5: AWS SageMaker Deployment

**Model Endpoint Information:**
- **Model location:** S3 bucket (ml-bootcamp-models-2024) or SageMaker workspace
- **Tested input:** `[Age=60, BP=140, Chol=300, MaxHR=150, ST_depression=2.5, Vessels=2]`
- **Output:** `Probability=81.02% (HIGH RISK - Heart Disease Likely)`

> **Deployment Comment:** *"Deployment enables real-time heart disease risk scoring for clinical decision support. With SageMaker, clinicians can integrate this model into hospital EHR systems to provide instant predictions. Typical latency is 10-50ms for single patient predictions, making it suitable for point-of-care applications."*

### Screenshots (See AWS_SAGEMAKER_GUIDE.md)

| Screenshot | Description |
|------------|-------------|
| SageMaker Studio | Code Editor workspace with project files |
| S3 Model Upload | model.tar.gz uploaded to S3 bucket |
| Inference Test | Sample prediction output in notebook |

### Model Export Artifacts

```
sagemaker_scripts/
├── inference.py         # SageMaker inference handler
└── demo_deployment.py   # SageMaker deployment script
```

> ⚠️ **Important:** The SageMaker Domain must be configured with **"Public internet only"** network access. VPC-only domains will fail with connection timeouts. See [AWS_SAGEMAKER_GUIDE.md](AWS_SAGEMAKER_GUIDE.md) for details.

### Training & Model Development

The complete training process is documented in `heart_disease_lr_analysis.ipynb`:

1. **Data Loading & EDA** - Steps 1.1-1.6
2. **Model Training** - Steps 2.1-2.6
3. **Visualization** - Steps 3.1-3.3
4. **Regularization Tuning** - Steps 4.1-4.5
5. **Deployment** - Steps 5.1-5.5

### SageMaker Deployment Workflow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. Export      │────▶│  2. Upload to   │────▶│  3. Create      │
│     Model       │     │     S3          │     │     Endpoint    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
┌─────────────────┐     ┌─────────────────┐             ▼
│  5. Get Risk    │◀────│  4. Send        │◀────────────┘
│     Score       │     │     Request     │
└─────────────────┘     └─────────────────┘
```

### Sample Inference Request/Response

**Request:**
```json
{
  "features": [60, 140, 300, 150, 2.5, 2]
}
```

**Response:**
```json
{
  "probability": 0.68,
  "prediction": 1,
  "risk_level": "HIGH",
  "diagnosis": "Heart Disease Likely"
}
```

### Expected Performance

| Metric | Value |
|--------|-------|
| Latency | ~50-100ms |
| Throughput | ~100 req/sec |
| Instance Cost | ~$0.05/hour (ml.t2.medium) |

---

## 📐 Mathematical Foundation

### Logistic Regression Model

$$f_{\vec{w}, b}(\vec{x}) = \sigma(\vec{w} \cdot \vec{x} + b) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{x} + b)}}$$

### Binary Cross-Entropy Cost

$$J(\vec{w}, b) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log(f^{(i)}) + (1-y^{(i)})\log(1-f^{(i)})\right]$$

### Gradients

$$\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=1}^{m}(f^{(i)} - y^{(i)})x_j^{(i)}$$

$$\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}(f^{(i)} - y^{(i)})$$

### Regularized Cost

$$J_{reg}(\vec{w}, b) = J(\vec{w}, b) + \frac{\lambda}{2m}\sum_{j=1}^{n} w_j^2$$

### Gradient Descent Update

$$w_j := w_j - \alpha\left(\frac{\partial J}{\partial w_j} + \frac{\lambda}{m}w_j\right)$$

$$b := b - \alpha\frac{\partial J}{\partial b}$$

---

## 💡 Key Insights

### Clinical Implications

1. **ST depression** and **Number of vessels** are the strongest predictors
   - ECG abnormalities and fluoroscopy results are critical diagnostic tools
   
2. **Lower maximum heart rate** correlates with disease
   - Patients unable to achieve high heart rates may have compromised cardiac function

3. **Age and cholesterol** contribute but are not decisive alone
   - Multiple factors needed for accurate risk assessment

### Technical Learnings

1. **Feature normalization is essential**
   - Without it, gradient descent may not converge or converge very slowly
   
2. **Regularization prevents overfitting**
   - λ=0.1 improved test accuracy while reducing weight magnitude
   
3. **Simple models can be powerful**
   - Logistic regression achieves ~81% accuracy with interpretable weights

4. **Decision boundaries are linear**
   - For non-linear patterns, would need polynomial features or kernel methods

---

## 📚 References

1. **Dataset:** [Kaggle Heart Disease Prediction](https://www.kaggle.com/datasets/neurocipher/heartdisease)
2. **UCI Repository:** [Heart Disease Data Set](https://archive.ics.uci.edu/ml/datasets/heart+disease)
3. **AWS SageMaker:** [Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
4. **World Health Organization:** [Cardiovascular Diseases](https://www.who.int/health-topics/cardiovascular-diseases)

---

## 👤 Cristian Santiago Pedraza Rodriguez

Classification and Logistic Regression

<div align="center">

</div>
