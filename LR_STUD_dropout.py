#---------------------------------------------------------------------------------------
# OULAD dataset 
#---------------------------------------------------------------------------------------

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns

# # --- Data Loading and Feature Engineering (OULAD Specific) ---

# # >>>>>> CRITICAL: REPLACE THESE PLACEHOLDERS WITH YOUR ACTUAL FILE PATHS <<<<<<
# # The OULAD dataset is typically structured in these four main files:
# try:
#     info_df = pd.read_csv(r"OULAD\studentInfo.csv")          # Main info, Demographics, Target
#     vle_df = pd.read_csv(r"OULAD\studentVle.csv")            # Virtual Learning Environment activity
#     assess_df = pd.read_csv(r"OULAD\assessments.csv")        # Assessment structure/weight
#     student_assess_df = pd.read_csv(r"OULAD\studentAssessment.csv") # Student assessment results
# except FileNotFoundError:
#     print("Error: One or more OULAD files not found. Please check your file paths.")
#     exit()

# # 1. Feature Engineering: Total Student Clicks (Engagement)
# # Aggregate VLE activity to get total clicks per student.
# vle_features = vle_df.groupby('id_student')['sum_click'].sum().reset_index()
# vle_features.rename(columns={'sum_click': 'total_clicks'}, inplace=True)

# # 2. Feature Engineering: Weighted Average Score (Performance)
# # Merge assessment weights with student scores
# student_scores = pd.merge(student_assess_df, assess_df, 
#                          on=['id_assessment'], 
#                          suffixes=('_student', '_structure'))
# # Calculate score * weight
# student_scores['weighted_score'] = student_scores['score'] * student_scores['weight']

# # Group by student and module/presentation to get overall average performance
# performance_features = student_scores.groupby('id_student')['weighted_score'].mean().reset_index()
# performance_features.rename(columns={'weighted_score': 'avg_weighted_score'}, inplace=True)

# # 3. Merging Features into the Main DataFrame
# df = pd.merge(info_df, vle_features, on='id_student', how='left')
# df = pd.merge(df, performance_features, on='id_student', how='left')

# # 4. Target Variable Transformation: Create Binary Dropout Flag
# # 'Withdrawn' is considered dropout (1), all others (Pass, Fail, Distinction) are non-dropout (0).
# df['dropout'] = df['final_result'].apply(lambda x: 1 if x == 'Withdrawn' else 0).astype(int)

# # Drop irrelevant/redundant/original target columns
# columns_to_drop = ['id_student', 'code_module', 'code_presentation', 'final_result']
# df = df.drop(columns=columns_to_drop)

# # --- Data Cleaning and Imputation ---

# # Fill NaNs in the newly created numerical features with the median
# for col in ['total_clicks', 'avg_weighted_score']:
#     df[col] = df[col].fillna(df[col].median())
    
# # Fill NaNs in demographic/contextual features with a new category 'Unknown'
# for col in ['disability', 'region', 'gender', 'highest_education', 'imd_band']:
#     if col in df.columns:
#         df[col] = df[col].fillna('Unknown')

# print("Dataset preview (after feature engineering and cleaning):")
# print(df.head(), "\n")

# # --- Data Preprocessing ---

# # 1. Handle Categorical Columns using One-Hot Encoding
# categorical_cols = df.select_dtypes(include='object').columns.tolist()
# df = pd.get_dummies(df, columns=categorical_cols, dummy_na=False, drop_first=True)

# # 2. Define Features and Target 
# X = df.drop('dropout', axis=1)
# y = df['dropout']

# # 3. Scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# # 4. Train-Test Split (Test size reduced to 0.1 and stratified for imbalance)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.1, random_state=42, stratify=y)

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
# plt.figure(figsize=(12,8))
# sns.barplot(x='Coefficient', y='Feature', data=coefficients.head(15), palette='RdBu_r') 
# plt.title('Top 15 Influencers on Learner Dropout Probability (Logistic Regression Coefficients)')
# plt.xlabel('Coefficient Value (Higher = Increased Dropout Likelihood)')
# plt.show()

# Problem Description
# Build a logistic regression model to predict learner dropout ('Withdrawn' status) using the multi-file Open University Learning Analytics Dataset (OULAD). The process involves merging demographics, performance, and activity data. Identify the strongest influential factors contributing to the likelihood of dropout based on the model's standardized coefficients.

# Dataset Description
# The analysis uses data merged from OULAD's studentInfo, studentVle, assessments, and studentAssessment files.
# Features: total_clicks (VLE activity), avg_weighted_score (performance), demographic (gender, highest_education, region), and context features (disability, imd_band, num_of_prev_attempts, studied_credits).
# Target: dropout (1 for Withdrawn, 0 otherwise).

# Inference
# The model achieved 73% accuracy but showed low recall (31%) for the dropout class, typical of imbalanced data.
# Key finding: The highest risk factors are workload and demographic/contextual details, while activity and performance are the strongest protective factors.
# Strongest Risk Factors (Positive Coefficients):
# 1. studied_credits (+0.45): Taking on a high number of credits significantly increases dropout probability (overcommitment).
# 2. gender_M (+0.21): Being male is a substantial risk factor relative to the reference group.
# 3. highest_education_Lower Than A Level (+0.09): Lower academic preparation increases risk.
# Strongest Protective Factors (Negative Coefficients):
# 1. total_clicks (-0.95): The single most protective factor; high VLE engagement drastically reduces dropout risk.
# 2. avg_weighted_score (-0.37): High academic performance strongly reduces dropout risk.
# 3. num_of_prev_attempts (-0.06): Having previous attempts acts as a protective factor, possibly indicating learned resilience.

# Result
# Learner dropout is overwhelmingly driven by the combination of high workload (studied_credits) and insufficient engagement (low total_clicks). The platform must implement checks for overcommitment and use total clicks as the primary real-time indicator for intervention.

# Strongest Risk Factors:
# High Number of Studied Credits (Workload)
# Gender (Male)
# Lower Academic Qualification

# Strongest Protective Factors:
# High Total Clicks (Engagement)
# High Average Weighted Score (Performance)

# The platform should enforce credit limits for new students and use a drop in VLE total_clicks as the first warning sign to trigger proactive support.


#-------------------------------------------------------------------------------------------------------------------------
#other dataset.py
#--------------------------------------------------------------------------------------------------------------------------

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns

# # --- Data Loading (Simulating a Simple, Single-File Dataset) ---

# # >>>>>> CRITICAL: REPLACE "your_single_file.csv" WITH YOUR ACTUAL FILENAME <<<<<<
# try:
#     # Use na_values to handle common string representations of missing data
#     df = pd.read_csv("your_single_file.csv", na_values=['', 'NA', 'N/A', ' '])
# except FileNotFoundError:
#     # Simulation for demonstration if the user doesn't have the file
#     print("Warning: File not found. Simulating data for demonstration.")
#     np.random.seed(42)
#     N = 1000
#     data = {
#         'student_id': range(N),
#         'total_visits': np.random.randint(1, 500, N),
#         'avg_score': np.random.uniform(0.1, 1.0, N),
#         'registration_date': pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(1, 180, N), unit='D'),
#         'last_login_date': pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(1, 180, N), unit='D'),
#         'country': np.random.choice(['USA', 'India', 'UK'], N, p=[0.5, 0.3, 0.2]),
#         'support_needed': np.random.choice(['Yes', 'No', np.nan], N, p=[0.1, 0.8, 0.1]),
#         'dropout_status': np.random.choice(['Dropped Out', 'Completed', 'Completed'], N, p=[0.3, 0.35, 0.35])
#     }
#     df = pd.DataFrame(data)

# print("Dataset preview:")
# print(df.head(), "\n")

# # --- Data Preprocessing: Handling Messy Data ---

# # 1. Handle Target Column (String to Binary Integer)
# # Assuming the target column is named 'dropout_status' and 'Dropped Out' is the positive class (1).
# df['dropout'] = df['dropout_status'].apply(lambda x: 1 if x == 'Dropped Out' else 0)
# df = df.drop('dropout_status', axis=1) # Drop the original string column

# # 2. Feature Engineering from Date Columns
# # Calculate 'duration' or 'days_since_active' which are numerical features.
# DATE_FORMAT = '%Y-%m-%d' # Adjust format if needed
# df['registration_date'] = pd.to_datetime(df['registration_date'], errors='coerce')
# df['last_login_date'] = pd.to_datetime(df['last_login_date'], errors='coerce')

# # Time difference feature: Time between last activity and registration (proxy for total engagement period)
# df['engagement_duration_days'] = (df['last_login_date'] - df['registration_date']).dt.days

# # Drop original date columns now that we have the numerical feature
# df = df.drop(columns=['registration_date', 'last_login_date'], errors='ignore')

# # 3. Handle Categorical Features (Strings)
# categorical_cols = df.select_dtypes(include='object').columns.tolist()
# # Fill NaN/missing values in categorical columns with 'Missing' before one-hot encoding
# for col in categorical_cols:
#     df[col] = df[col].fillna('Missing')
# df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# # 4. Handle Remaining Numerical Missing Values
# numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
# # Exclude the target variable from imputation
# numerical_cols.remove('dropout')
# for col in numerical_cols:
#     # Convert to numeric first (handles non-date strings that might have slipped through)
#     df[col] = pd.to_numeric(df[col], errors='coerce')
#     # Impute missing values with the median (robust against outliers)
#     df[col] = df[col].fillna(df[col].median())

# # --- Model Building ---
# X = df.drop(['dropout', 'student_id'], axis=1, errors='ignore') # Drop target and ID
# y = df['dropout']

# # Scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# # Train-Test Split (Stratified to handle potential imbalance)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)

# # Model Training
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
# plt.title('Influence on Learner Dropout Probability (Logistic Regression Coefficients)')
# plt.xlabel('Coefficient Value (Higher = Increased Dropout Likelihood)')
# plt.show()

# Problem Description
# Build a robust logistic regression model using a single, potentially messy dataset file to predict learner dropout. The code handles string target values, date columns, and missing data in both numerical and categorical features. Identify the strongest influential factors contributing to the likelihood of dropout based on the model's standardized coefficients.

# Dataset Description
# Dataset is assumed to be a single CSV file containing numerical columns (e.g., total_visits, avg_score), categorical strings (e.g., country, support_needed), date columns (e.g., registration_date, last_login_date), and a string target column (e.g., dropout_status).

# Inference (Based on common MOOC patterns)
# Dropout risk is typically driven by low average score and short engagement duration.
# Protective Factors: High total_visits, high avg_score, long engagement_duration_days.
# Risk Factors: Low avg_score, short engagement_duration_days, or high support_needed (indicating struggle).

# Result
# Successful prediction hinges on converting dates into meaningful duration features and ensuring all messy categorical and numerical data are correctly encoded and imputed.

# Strongest Risk Factors (Hypothesized):
# Low Average Score
# Short Engagement Duration

# Strongest Protective Factors (Hypothesized):
# High Total Visits
# High Engagement Duration
