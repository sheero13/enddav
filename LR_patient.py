# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = pd.read_csv(r"archive (1)\train_df.csv") 

# # Check if 'Patient_ID' exists and drop it, as it's not a predictive feature.
# if 'Patient_ID' in df.columns:
#     df = df.drop('Patient_ID', axis=1)

# print("Dataset preview:")
# print(df.head(), "\n")

# # --- Data Preprocessing ---

# # 1. Handle Categorical Columns using Label Encoding
# # The columns 'gender', 'primary_di', and 'discharge_to' are categorical.
# categorical_cols = ['gender', 'primary_diagnosis', 'discharge_to']
# le = LabelEncoder()
# for col in categorical_cols:
#     df[col] = le.fit_transform(df[col])

# # No other non-predictive columns were identified for dropping in the train_df.csv image

# X = df.drop('readmitted', axis=1)
# y = df['readmitted']

# # 2. Scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns) 

# # 3. Train-Test Split (Splitting the loaded training data for validation)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

# # --- Model Training and Evaluation ---
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # --- Feature Interpretation (Coefficients) ---
# coefficients = pd.DataFrame({
#     'Feature': X.columns,
#     'Coefficient': model.coef_[0]
# }).sort_values(by='Coefficient', ascending=False)

# print("\nFeature Importance (Logistic Regression Coefficients):")
# print(coefficients)

# # Visualization
# plt.figure(figsize=(10,6))
# sns.barplot(x='Coefficient', y='Feature', data=coefficients, palette='RdBu_r') 
# plt.title('Influence on Patient Readmission Probability (Logistic Regression Coefficients)')
# plt.xlabel('Coefficient Value (Higher = Increased Readmission Likelihood)')
# plt.show()

# # Problem Description
# # Build a logistic regression model to predict the probability of patient readmission (target 'readmitted') based on demographic, diagnosis, procedure, and clinical history features present in 'train_df.csv'. Identify the strongest influencing factors using the model's standardized coefficients.

# # Dataset Description
# # Dataset (from train_df.csv) includes features: age, gender, primary_di, num_procedu, days_in_ho, comorbidi (comorbidity_score), discharge_to, and readmitted (target: 1 = readmitted, 0 = not readmitted).

# # Inference
# # High readmission risk is strongly linked to the patient's primary diagnosis (primary_diagnosis), age, and the discharge location (discharge_to).
# # Interestingly, longer hospital stays (days_in_hospital) and a higher comorbidity count (comorbidity_score) show negative coefficients, suggesting they are protective factors, which may indicate that the hospital provides successful intensive discharge support for the most complex cases.
# # Model achieved an accuracy of 83% but struggled to predict the readmitted class (precision/recall of 0.00 for class 1). This is common with highly imbalanced data (826 vs 174).
# # Key features (Risk Factors): primary_diagnosis (+), age (+), discharge_to (+).
# # Key features (Protective Factors): gender (−), days_in_hospital (−), comorbidity_score (−).

# # Result
# # Patients whose readmission risk is highest are those whose profile is captured by the combination of their primary diagnosis, age, and discharge plan. The model highlights that the nature of the condition and patient demographic are the strongest external drivers of readmission.

# # Strongest Risk Factors:
# # Primary Diagnosis
# # Age
# # Discharge Destination

# # To improve prediction performance, hospitals should collect more granular data on the severity of the primary condition and the quality of post-discharge care.


#----------------------------------------------------------------------------------------------------------------------------
# general
#----------------------------------------------------------------------------------------------------------------------------

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns

# # --- Configuration ---
# # You MUST change these two variables for any new dataset.
# TARGET_COLUMN = 'readmitted' # Replace with your dataset's target column name (e.g., 'churn', 'dropout', 'fraud')
# FILE_PATH = "your_new_dataset.csv" # Replace with the path to your CSV file

# # --- Data Loading (Robust) ---
# try:
#     # Use na_values to handle common string representations of missing data
#     df = pd.read_csv(FILE_PATH, na_values=['', 'NA', 'N/A', ' ', '?'])
# except FileNotFoundError:
#     print(f"Error: File not found at {FILE_PATH}. Please check the path and filename.")
#     exit()

# print("Dataset preview:")
# print(df.head(), "\n")
# print(f"Target column status check: {TARGET_COLUMN} unique values: {df[TARGET_COLUMN].unique()}\n")

# # --- Data Preprocessing: Handling Messy Data (Universal) ---

# # 1. Feature Engineering from Date Columns
# date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

# # Attempt to convert 'object' columns that look like dates
# for col in df.select_dtypes(include=['object']).columns:
#     try:
#         # If conversion works, replace the original column
#         df[col] = pd.to_datetime(df[col], errors='raise')
#         date_cols.append(col)
#     except (ValueError, TypeError):
#         pass

# if date_cols:
#     print(f"Found Date Columns: {date_cols}. Creating 'days_since_start' features.")
    
#     # Calculate a common reference point (e.g., the earliest date)
#     reference_date = df[date_cols].min().min()
    
#     for col in date_cols:
#         # Convert date column to datetime objects (handling errors by setting to NaT)
#         df[col] = pd.to_datetime(df[col], errors='coerce')
        
#         # Create a new feature: Days elapsed since the reference date
#         new_col_name = f'days_since_{col[:5]}' # e.g., days_since_regis
#         df[new_col_name] = (df[col] - reference_date).dt.days

#     # Drop the original date columns
#     df = df.drop(columns=date_cols)

# # 2. Handle Target Column (Must be cleaned and binary integer)
# # This handles string targets (like 'Yes'/'No', 'Readmitted'/'Not') or float targets.
# if df[TARGET_COLUMN].dtype == 'object':
#     print("Target column is string (object). Mapping to 0/1.")
#     # Identify the positive class by frequency or assume the first unique value is 1
#     target_map = {df[TARGET_COLUMN].value_counts().index[0]: 0} # Most frequent class is 0 (Non-event)
#     for value in df[TARGET_COLUMN].unique():
#         if value not in target_map:
#             target_map[value] = 1 # The rest are 1 (Event)
#             break 
            
#     # If there are only two unique values, assign 1 to the minority class.
#     unique_values = df[TARGET_COLUMN].dropna().unique()
#     if len(unique_values) == 2:
#         minority_class = df[TARGET_COLUMN].value_counts().idxmin()
#         target_map = {minority_class: 1}
#         for value in unique_values:
#             if value != minority_class:
#                 target_map[value] = 0
                
#     df[TARGET_COLUMN] = df[TARGET_COLUMN].map(target_map)
    
# # Ensure the target column is a clean integer (1 or 0)
# df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna(0).astype(int)

# # 3. Handle Categorical Features (Strings/Objects)
# categorical_cols = df.select_dtypes(include='object').columns.tolist()
# if categorical_cols:
#     print(f"Found Categorical Columns: {categorical_cols}. Applying One-Hot Encoding.")
#     # Fill NaN/missing values in categorical columns with 'Missing' before encoding
#     for col in categorical_cols:
#         df[col] = df[col].fillna('Missing')
#     df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# # 4. Handle Remaining Numerical Missing Values
# numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
# # Exclude the target variable and any potential ID columns from imputation
# numerical_cols_to_impute = [col for col in numerical_cols if col != TARGET_COLUMN and 'id' not in col.lower()]

# if numerical_cols_to_impute:
#     print(f"Imputing {len(numerical_cols_to_impute)} numerical columns using the median.")
#     for col in numerical_cols_to_impute:
#         # Convert to numeric first (handles any leftover non-numeric noise)
#         df[col] = pd.to_numeric(df[col], errors='coerce')
#         # Impute missing values with the median (robust against outliers)
#         df[col] = df[col].fillna(df[col].median())

# # --- Model Building ---
# # Identify non-feature columns to drop (IDs, original messy columns)
# cols_to_exclude = [TARGET_COLUMN] + [col for col in df.columns if 'id' in col.lower()]
# X = df.drop(cols_to_exclude, axis=1, errors='ignore')
# y = df[TARGET_COLUMN]

# # Check for single-class split failure (if target is still single class after cleaning)
# if len(y.unique()) < 2:
#     print("\nMODEL FAILED: After cleaning, the target column still contains only one class. Cannot train model.")
#     exit()

# # Scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# # Train-Test Split (Stratified ensures both 0 and 1 classes are present in the training set)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)

# # Model Training
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print("\n--- Model Evaluation ---")
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # --- Feature Interpretation (Coefficients) ---
# coefficients = pd.DataFrame({
#     'Feature': X.columns,
#     'Coefficient': model.coef_[0]
# }).sort_values(by='Coefficient', ascending=False)

# print("\nFeature Importance (Logistic Regression Coefficients):")
# print(coefficients)

# # Visualization
# plt.figure(figsize=(10, min(20, len(coefficients) * 0.4))) # Dynamically sized plot
# sns.barplot(x='Coefficient', y='Feature', data=coefficients, palette='RdBu_r') 
# plt.title(f'Influence on {TARGET_COLUMN} Probability (Logistic Regression Coefficients)')
# plt.xlabel('Coefficient Value (Higher = Increased Likelihood)')
# plt.show()

# # Problem Description
# # Build a generalized logistic regression model to predict the binary outcome specified by TARGET_COLUMN. The code automatically handles and cleans date columns (by converting to days elapsed), string/categorical features (using One-Hot Encoding), and imputes missing numerical data (using the median). The primary objective is to identify features that are the strongest risk factors (positive coefficients) and protective factors (negative coefficients).

# # Dataset Description
# # The script is designed to accept any single-file CSV. It assumes the target is binary and the remaining columns are features, handling date, numeric, and categorical types robustly.

# # Inference
# # The model's interpretation relies entirely on the final coefficients: features with the largest positive coefficients are the strongest risk factors, and features with the largest negative coefficients are the strongest protective factors. This generalized code provides the structural framework for rigorous analysis regardless of the specific domain (healthcare, finance, education).

# # Result
# # A clean, scaled dataset is fed into a Logistic Regression model to produce interpretable coefficients. The output provides a direct ranking of features by their influence on the event probability.

# # Strongest Risk Factors:
# # Features with the largest positive coefficients (e.g., High Age, Specific Location_A)

# # Strongest Protective Factors:
# # Features with the largest negative coefficients (e.g., High Visits, Specific Education_B)

# # Actionable Insights:
# # Interventions should be targeted at modifying behaviors or environments related to the top positive coefficient features, while leveraging the factors associated with the strongest negative coefficients.
