# Machine Learning predicting diabetes risk from demographic and health factors

By predicting **diabetes status** (0 = non-diabetic, 1 = diabetic) from a combination of demographic, lifestyle, and health-related features, we can identify individuals at higher risk earlier and enable proactive healthcare interventions. The dataset includes variables such as age, gender, BMI, hypertension, heart disease, and smoking history, alongside biomarkers like HbA1c and blood glucose levels.

The target of this project is the binary variable **`diabetes`**, indicating whether a person is diagnosed with diabetes or not.

This project supports **SDG 3 (Good Health and Well-being)** by promoting early detection and personalised prevention strategies, and **SDG 10 (Reduced Inequalities)** by showing how data-driven approaches can help make healthcare more accessible and equitable across populations.

**Impact**: By modelling patterns of diabetes risk, healthcare providers can improve patient outcomes through targeted interventions, encourage lifestyle changes before severe complications occur, and reduce the long-term economic and societal burden of diabetes.

[Dataset info, index and stakeholder report](https://www.notion.so/IC2-Classification-Machine-Learning-Predicting-diabetes-risk-from-demographic-and-health-factors-26698c6768cd80b19de1d7ce6b864816?source=copy_link)

---

## Hypothesis
We assume that simple patient information (e.g., **age, BMI, HbA1c, smoking history, comorbidities**) contains enough signal to estimate diabetes risk.  

- If the model can accurately classify diabetes, it can be used as an **early-warning tool** in routine care.  
- Adjusting thresholds and class weights allows us to balance **recall** (catching more diabetics) and **precision** (reducing false alarms).  

---

## Dataset Info
- **Source:** [Diabetes Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)  
- **Content:** 99,991 patient records × 13 columns (demographics, comorbidities, biomarkers, lifestyle).  
- **Target variable:** `diabetes` (binary: 1 = diabetic, 0 = non-diabetic).  
- **Features:**  
  - **Demographics & lifestyle:** age, gender, smoking history  
  - **Comorbidities:** hypertension, heart disease  
  - **Biomarkers:** HbA1c level, blood glucose level  
  - **Derived features:** age groups, BMI categories, age×BMI interaction, simple risk score  

---

## Notebook Index
### 01. IC2_Classification_Diabetes_Preprocessing.ipynb
- Load & inspect raw dataset (shape, target balance: 8.5% diabetics vs 91.5% non-diabetics).  
- Clean categorical values (gender, smoking history).  
- Feature engineering (age group, BMI category, age×BMI interaction, risk score).  
- Handle outliers (e.g., extreme BMI values).  
- Save cleaned dataset.  

### 02. IC2_Classification_Diabetes_ML.ipynb
- Load preprocessed data.  
- Create **Variants A, B, C** to study leakage risks (glucose & HbA1c).  
- Train/test split with stratification.  
- Preprocessing pipelines (numeric + categorical).  
- Models tested: Logistic Regression, Decision Tree, Random Forest (balanced + SMOTE).  
- Handle class imbalance with **class weights** and **SMOTE oversampling**.  
- Threshold tuning to optimize **precision vs recall trade-off**.  
- Evaluate models with accuracy, precision, recall, F1, ROC AUC + confusion matrices.  
- Select best model (Random Forest with `{0:1,1:2}` weights, tuned threshold ≈ 0.575).  
- Save final model + metadata.  
- Visualize results (ROC, PR curve, confusion matrix, threshold analysis).  

---

## Final Results
After testing multiple classification models, the **Random Forest with class weight {0:1, 1:2} and tuned threshold (≈0.575)** delivered the best balance.  

### Test set performance (final model):
- **Precision ≈ 0.90** → 9 out of 10 flagged patients are truly diabetic.  
- **Recall ≈ 0.50** → detects about half of true diabetics.  
- **F1-score ≈ 0.65** → strong balance between precision and recall.  
- **ROC AUC ≈ 0.93** → reliable separation of risk vs. no risk.  

### Comparison (Variant B, Random Forest only):
| Model            | Accuracy | Precision | Recall | F1    | ROC AUC |
|------------------|----------|-----------|--------|-------|---------|
| RF balanced      | 0.9318   | 0.5991    | 0.5980 | 0.5985| 0.9313  |
| RF SMOTE         | 0.9182   | 0.5147    | 0.6474 | 0.5735| 0.9307  |
| RF tuned (final) | 0.9321   | **0.8956**| 0.5050 | **0.6458** | 0.9321  |

**Takeaway:**  
- RF balanced gives a clean balance.  
- RF SMOTE increases recall but adds more false alarms.  
- RF tuned with class weights + threshold is the most practical option, combining **high precision** with **usable recall**.  

---

## Limitations, Bias & Ethical Reflection
- **Bias risks:** subgroup imbalance (age, sex, lifestyle), diagnosis bias across clinics.  
- **Mitigation:** subgroup performance monitoring, threshold calibration per clinic, periodic retraining.  
- **Limitations:** recall is only ~50%, so some diabetics are still missed. More features (family history, lifestyle, meds) could improve results.  

---

## Takeaway
This project shows how **routine patient data** can be transformed into a **practical early-warning system** for diabetes.  
The model does not replace doctors but helps them **focus resources on those most at risk**, improving preventive care and reducing long-term costs.  
