import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load the dataset (fixed file path issue)
df = pd.read_csv(r"E:\Prostrate cancer prediction\prostate_cancer_prediction.csv")
print(df.head(3))  # Display first 3 rows

# Check class distribution
print(df['Follow_Up_Required'].value_counts())

# Dataset information
print(df.info())

# Drop the 'Patient_ID' column
df.drop(columns=['Patient_ID'], inplace=True)

# Encode categorical columns
enc = LabelEncoder()
categorical_cols = [
    'Family_History', 'Race_African_Ancestry', 'DRE_Result', 'Biopsy_Result',
    'Difficulty_Urinating', 'Weak_Urine_Flow', 'Blood_in_Urine', 'Pelvic_Pain',
    'Back_Pain', 'Erectile_Dysfunction', 'Cancer_Stage', 'Treatment_Recommended',
    'Survival_5_Years', 'Exercise_Regularly', 'Healthy_Diet', 'Smoking_History',
    'Alcohol_Consumption', 'Hypertension', 'Diabetes', 'Cholesterol_Level',
    'Follow_Up_Required', 'Genetic_Risk_Factors', 'Previous_Cancer_History', 'Early_Detection'
]

for col in categorical_cols:
    df[col] = enc.fit_transform(df[col])

# Fix issue with duplicate column names
x = df.drop(columns=['Early_Detection'])  # Use only one version
y = df['Early_Detection']

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=19, train_size=0.80)

# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=150, max_depth=3, random_state=42)
model = rfc.fit(x_train, y_train)
print("Random Forest Score:", model.score(x_test, y_test))

# XGBoost Classifier (fixed learning_rate value)
xgb = XGBClassifier(learning_rate=0.01, random_state=42)
xgb_model = xgb.fit(x_train, y_train)
print("XGBoost Score:", xgb_model.score(x_test, y_test))

# Cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=19)
scores = cross_val_score(model, x, y, cv=kfold)
print("Cross-validation Mean Score:", scores.mean())

# Check class distribution
print(y.value_counts())

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import numpy as np

# Define the parameter grid for Random Forest
rf_params = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5)
}

# Define the parameter grid for XGBoost
xgb_params = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20),
    'learning_rate': np.linspace(0.01, 0.3, 10),
    'subsample': np.linspace(0.5, 1.0, 10)
}

# Perform Randomized Search CV for Random Forest
rf_search = RandomizedSearchCV(RandomForestClassifier(), rf_params, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
rf_search.fit(x_train, y_train)
best_rf = rf_search.best_estimator_

# Perform Randomized Search CV for XGBoost
xgb_search = RandomizedSearchCV(XGBClassifier(), xgb_params, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
xgb_search.fit(x_train, y_train)
best_xgb = xgb_search.best_estimator_

# Print the best parameters
print("Best Random Forest Parameters:", rf_search.best_params_)
print("Best XGBoost Parameters:", xgb_search.best_params_)

# Evaluate models
print("Optimized Random Forest Accuracy:", best_rf.score(x_test, y_test))
print("Optimized XGBoost Accuracy:", best_xgb.score(x_test, y_test))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Generate confusion matrix
y_pred_rf = best_rf.predict(x_test)
y_pred_xgb = best_xgb.predict(x_test)

# Display confusion matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp_rf.plot()
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Display confusion matrix for XGBoost
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb)
disp_xgb.plot()
plt.title("Confusion Matrix - XGBoost")
plt.show()


importances_rf = best_rf.feature_importances_
importances_xgb = best_xgb.feature_importances_

# Convert to DataFrame for better visualization
import pandas as pd
feat_importance_rf = pd.DataFrame({'Feature': x.columns, 'Importance': importances_rf})
feat_importance_xgb = pd.DataFrame({'Feature': x.columns, 'Importance': importances_xgb})

# Sort and plot the top features
feat_importance_rf.sort_values(by="Importance", ascending=False).head(10).plot(kind="bar", x="Feature", y="Importance", title="Feature Importance - Random Forest", legend=False)
plt.show()

feat_importance_xgb.sort_values(by="Importance", ascending=False).head(10).plot(kind="bar", x="Feature", y="Importance", title="Feature Importance - XGBoost", legend=False)
plt.show()


from sklearn.metrics import roc_curve, auc

# Get probabilities for the positive class
y_probs_rf = best_rf.predict_proba(x_test)[:, 1]
y_probs_xgb = best_xgb.predict_proba(x_test)[:, 1]

# Compute ROC curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_probs_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_probs_xgb)

# Compute AUC
auc_rf = auc(fpr_rf, tpr_rf)
auc_xgb = auc(fpr_xgb, tpr_xgb)

# Plot ROC curve
plt.figure(figsize=(8,6))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.2f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {auc_xgb:.2f})", linestyle='dashed')
plt.plot([0, 1], [0, 1], 'k--')  # Random guessing line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


#predoction
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Function to encode user input
def encode_user_input(user_input):
    enc = LabelEncoder()
    # Encode categorical features as done during training (you should apply the same transformations)
    encoded_input = user_input.copy()
    categorical_columns = ['Family_History', 'Race_African_Ancestry', 'DRE_Result', 
                           'Biopsy_Result', 'Difficulty_Urinating', 'Weak_Urine_Flow', 
                           'Blood_in_Urine', 'Pelvic_Pain', 'Back_Pain', 'Erectile_Dysfunction', 
                           'Cancer_Stage', 'Treatment_Recommended', 'Survival_5_Years', 
                           'Exercise_Regularly', 'Healthy_Diet', 'Smoking_History', 
                           'Alcohol_Consumption', 'Hypertension', 'Diabetes', 'Cholesterol_Level', 
                           'Follow_Up_Required', 'Genetic_Risk_Factors', 'Previous_Cancer_History']
    
    for col in categorical_columns:
        encoded_input[col] = enc.fit_transform(encoded_input[col])
        
    return encoded_input

# Function for interactive prediction
def interactive_prediction(model):
    print("Please enter the following details:")
    
    # Take input from user (example values)
    user_input = {
        'Age': int(input("Age: ")),
        'Family_History': input("Family History (Yes/No): "),
        'Race_African_Ancestry': input("Race African Ancestry (Yes/No): "),
        'PSA_Level': float(input("PSA Level: ")),
        'DRE_Result': input("DRE Result (Normal/Abnormal): "),
        # Add other features here in a similar manner
        # Example: 'Blood_in_Urine': input("Blood in Urine (Yes/No): ")
    }
    
    # Create a DataFrame for the user input
    user_input_df = pd.DataFrame([user_input])
    
    # Encode the input
    encoded_user_input = encode_user_input(user_input_df)
    
    # Get the prediction
    prediction = model.predict(encoded_user_input)
    
    # Output the prediction
    if prediction == 1:
        print("The model predicts: 'Follow-up Required'.")
    else:
        print("The model predicts: 'No Follow-up Required'.")

# Example usage with the Random Forest model
interactive_prediction(best_rf)