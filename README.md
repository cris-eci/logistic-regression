# Heart Disease Risk Prediction: Logistic Regression Analysis

## 📋 Exercise Summary

This project implements **logistic regression from scratch** for heart disease prediction using Python (NumPy, Pandas, Matplotlib). The analysis includes comprehensive exploratory data analysis (EDA), model training with visualization of decision boundaries, L2 regularization techniques, and exploration of deployment strategies using Amazon SageMaker.

**Key Features:**
- 📊 **Step 1**: Data loading, preprocessing, and stratified train/test split
- 🔄 **Step 2**: Custom logistic regression implementation (sigmoid, cost function, gradient descent)  
- 📊 **Step 3**: Decision boundary visualization with multiple feature pairs
- 🎯 **Step 4**: L2 regularization with hyperparameter tuning
- ☁️ **Step 5**: Amazon SageMaker deployment exploration

---

## 📊 Dataset Description

**Source:** [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/neurocipher/heartdisease)  
**Original Repository:** UCI Machine Learning Repository

### Dataset Statistics
- **Total Samples:** 270 patients
- **Features:** 14 clinical measurements
- **Target Distribution:** ~44% disease presence rate (120 positive cases, 150 negative cases)
- **Missing Values:** None detected

### Feature Overview
| Feature | Description | Range | Type |
|---------|-------------|--------|------|
| `Age` | Patient age | 29-71 years | Numerical |
| `Sex` | Gender | 0=Female, 1=Male | Categorical |
| `Chest pain type` | Chest pain classification | 1-4 scale | Categorical |
| `BP` | Blood Pressure | 94-200 mmHg | Numerical |
| `Cholesterol` | Serum cholesterol | 126-564 mg/dL | Numerical |
| `FBS over 120` | Fasting blood sugar > 120 mg/dl | 0=False, 1=True | Binary |
| `EKG results` | Electrocardiogram results | 0-2 scale | Categorical |
| `Max HR` | Maximum heart rate achieved | 95-202 bpm | Numerical |
| `Exercise angina` | Exercise induced angina | 0=No, 1=Yes | Binary |
| `ST depression` | ST depression induced by exercise | 0.0-6.2 | Numerical |
| `Slope of ST` | Slope of peak exercise ST segment | 1-3 scale | Categorical |
| `Number of vessels fluro` | Vessels colored by fluoroscopy | 0-3 | Numerical |
| `Thallium` | Thallium stress test result | 3=Normal, 6=Fixed, 7=Reversible | Categorical |
| `Heart Disease` | **Target Variable** | Presence/Absence | Binary |

---

## 🚀 Step 1: Data Preparation 

### 1.1 Data Loading and Initial Exploration

Successfully loaded the Heart Disease Prediction dataset using pandas. Initial analysis revealed:
- **Dataset Shape:** 270 samples × 14 features
- **No missing values** detected across all columns
- **Target column** successfully identified: "Heart Disease" (text values)

### 1.2 Target Binarization

Converted target variable from text to numerical format:
```
Original values:
- "Presence" → 1 (positive class - disease present)  
- "Absence"  → 0 (negative class - no disease)

Final distribution:
- No disease (0): 150 samples (55.6%)
- Disease (1):    120 samples (44.4%)
```

**Conclusion:** Dataset is reasonably balanced (~44% disease rate), reducing the need for special imbalanced data techniques.

### 1.3 Exploratory Data Analysis (EDA)

#### Statistical Summary
Key insights from numerical features:
- **Age range:** 29-71 years (mean: 54.4 years)
- **Blood Pressure:** 94-200 mmHg (mean: 131.3 mmHg)  
- **Cholesterol:** 126-564 mg/dL (mean: 249.7 mg/dL)
- **Max Heart Rate:** 95-202 bpm (mean: 149.7 bpm)
- **ST Depression:** 0.0-6.2 units (mean: 1.05 units)

#### Data Quality Assessment
- ✅ **No missing values** in any column
- ✅ **Appropriate data types** (numerical features properly recognized)
- ✅ **Reasonable value ranges** (no obvious outliers requiring removal)

### 1.4 Feature Selection

Selected **6 key numerical features** based on medical relevance:

| Selected Feature | Medical Significance | Range |
|------------------|---------------------|-------|
| `Age` | Older patients higher risk | [29.0, 71.0] |
| `BP` | Blood pressure indicator | [94.0, 200.0] |
| `Cholesterol` | Cardiovascular risk factor | [126.0, 564.0] |
| `Max HR` | Heart function capacity | [95.0, 202.0] |
| `ST depression` | ECG abnormality measure | [0.0, 6.2] |
| `Number of vessels fluro` | Arterial blockage count | [0.0, 3.0] |

**Rationale:** These features represent core cardiovascular health indicators commonly used in clinical risk assessment.

### 1.5 Stratified Train/Test Split (70/30)

Implemented **custom stratified splitting** to preserve class distribution:

```
Split Results:
- Training set: 189 samples (70%)
- Test set:     81 samples (30%)

Disease rate verification:
- Training: 44.4% (preserved)
- Test:     44.4% (preserved)
```

**Implementation Details:**
- Used manual stratification (no scikit-learn dependency)
- Random seed: 42 (for reproducibility)
- Class proportions maintained within ±1% between splits

### 1.6 Feature Normalization

Applied **Min-Max normalization** to scale all features to [0, 1] range:

**Normalization Formula:** `(x - min) / (max - min)`

**Before normalization (training data ranges):**
- Age: [29.0, 77.0]
- BP: [94.0, 200.0]  
- Cholesterol: [126.0, 564.0]
- Max HR: [71.0, 202.0]
- ST depression: [0.0, 6.2]
- Number of vessels fluro: [0.0, 3.0]

**After normalization:**
- All training features: [0.0, 1.0] range achieved
- Test set normalization: Applied same training statistics (proper ML practice)

**Critical Detail:** Normalization parameters computed **only from training data** to prevent data leakage.

## 📚 References

1. **Dataset Source:** [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/neurocipher/heartdisease)
2. **Original Data:** UCI Machine Learning Repository
3. **Medical Context:** WHO reports heart disease as leading cause of death globally (~18M deaths/year)

---

## 👨‍💻 Author - Cristian Santiago Pedraza Rodriguez

**Course:**  Enterprise architectures(AREP)

**Institution:** Universidad Escuela Colombiana de Ingeniería Julio Garavito

**Semester:** Eight Semester

---

*This README will be updated as additional steps are completed.*