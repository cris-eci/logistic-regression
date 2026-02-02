Heart Disease Risk Prediction: Logistic Regression Homework
Introductory Context
Heart disease is the world's leading cause of death, claiming approximately 18 million lives each year, as reported by the World Health Organization. Predictive models like logistic regression can enable early identification of at-risk patients by analyzing clinical features such as age, cholesterol, and blood pressure. This not only improves treatment outcomes but also optimizes resource allocation in healthcare settings. In this homework, you'll implement logistic regression on the Heart Disease Dataset—a real-world UCI repository collection of 303 patient records with 14 features and a binary target (1 for disease presence, 0 for absence). You'll train models, visualize boundaries, apply regularization, and explore deployment via Amazon SageMaker to mimic a production pipeline.

Homework Instructions
Complete this in a Jupyter notebook, implementing functions from class theory (e.g., sigmoid, cost, GD). Use NumPy, Pandas, and Matplotlib—no scikit-learn for core training. Emphasize exploration: Tune parameters, interpret results, and document findings.

Step 1: Load and Prepare the Dataset
Download from Kaggle: Kaggle is a popular online platform for data science enthusiasts, hosting datasets, competitions, and notebooks—think of it as GitHub for data and ML projects (free to join at kaggle.com). To access the Heart Disease Dataset, visit https://www.kaggle.com/datasets/neurocipher/heartdisease. Sign up/log in, click "Download" (or "Download API" if using CLI), and save the CSV file (e.g., heart.csv or similar—check the dataset page for exact filename).
Load into Pandas; binarize the target column (e.g., map to 1=disease presence, 0=absence).
EDA: Summarize stats, handle missing/outliers, plot class distribution.
Prep: 70/30 train/test split (stratified); normalize numerical features. Select ≥6 features (e.g., Age, Cholesterol, BP, Max HR, ST Depression, Vessels).
Reporting: Markdown summary of data insights/preprocessing (e.g., "Downloaded from Kaggle; 303 samples, ~55% disease rate").

Step 2: Implement Basic Logistic Regression
Sigmoid, cost (binary cross-entropy), GD (gradients, updates; track costs).
Train on full train set (α~0.01, 1000+ iters). Plot cost vs. iterations.
Predict (threshold 0.5); evaluate acc/precision/recall/F1 on train/test.
Reporting: Cost plot + metrics table. Comment on convergence/interpretations (e.g., w coefficients).

Step 3: Visualize Decision Boundaries
Select ≥3 feature pairs (e.g., Age-Cholesterol, BP-Max HR, ST Depression-Vessels).
For each: Subset to 2D, train model, plot boundary line + scatter (true labels).
Discuss separability/nonlinearity.
Reporting: ≥3 plots. Markdown: Insights per pair (e.g., "Clear divide at chol>250").

Step 4: Repeat with Regularization
Add L2 to cost/gradients (λ/(2m)||w||²; dw += (λ/m)w).
Tune λ ([0, 0.001, 0.01, 0.1, 1]); retrain full model + pairs.
Re-plot costs/boundaries (one pair: unreg vs. reg). Re-eval metrics/||w||.
Reporting: λ-metrics table + plots. Markdown: "Optimal λ=[val] improves [metric] by [val]%."

Step 5: Explore Deployment in Amazon SageMaker
Export best model (w/b as NumPy array).
In SageMaker (use free tier/Studio): Create notebook instance; upload/run your notebook for training. Explore docs to build/deploy a simple endpoint (e.g., via script: load data, train, save model; create inference handler for patient inputs → prob output).
Test: Invoke endpoint with sample (e.g., Age=60, Chol=300); capture response.
Self-Guided: Follow AWS tutorials (search "SageMaker logistic regression endpoint")—experiment with instance types, monitoring.
Reporting: Notebook section: High-level steps + sample output. Comment: "Deployment enables [e.g., real-time risk scoring]; latency [val]ms."

Deliverables
Jupyter Notebook (heart_disease_lr_analysis.ipynb): End-to-end executable; markdown for steps, inline comments, all plots/tables. Final insights section.
README.md: Repo overview:
Exercise Summary: "Implements logistic regression for heart disease prediction: EDA, training/viz, reg, SageMaker deployment."
Dataset Description: "Kaggle Heart Disease (303 patients; features: Age 29-77, Chol 112-564 mg/dL, etc.; ~55% presence rate). Downloaded from https://www.kaggle.com/datasets/neurocipher/heartdisease."
Deployment Evidence: Describe process; embed ≥3 images (screenshots: training job status, endpoint config, inference response). "Model at [endpoint ARN]; tested input [Age=60, Chol=300] → Output: Prob=0.68 (high risk)."
GitHub Repo: Create public/private repo; include notebook, README.md, CSV, images. Share link in submission (e.g., "Repo: github.com/[user]/heart-disease-lr").
Submission and Evaluation
Submit: GitHub repo link via moodle.
Criteria (100 pts): EDA (10), Implementation (35), Viz/Analysis (20), Reg (15), Deployment/Repo (15), Clarity (5).