# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns

# # --- Configuration ---
# TARGET_COLUMN = 'Repeat_Purchase'
# FILE_PATH = r"Data.csv" # *** CRITICAL: CHANGE THIS TO YOUR ACTUAL FILE PATH ***

# # --- Data Loading and RFM Feature Engineering ---
# try:
#     # Load the transactional dataset
#     df_raw = pd.read_csv(FILE_PATH, encoding='ISO-8859-1', na_values=['', 'NA'])
# except FileNotFoundError:
#     print(f"Error: File not found at {FILE_PATH}. This script requires the transactional Online Retail data.")
#     exit()

# # 1. Initial Cleaning and Prep
# # Drop rows with missing CustomerID (cannot segment) and clean up cancelled transactions
# df_raw.dropna(subset=['Customer ID'], inplace=True)
# df_raw = df_raw[~df_raw['Invoice'].astype(str).str.startswith('C')] # Remove returns/cancellations
# df_raw['Customer ID'] = df_raw['Customer ID'].astype(int) # Ensure Customer ID is integer
# df_raw['InvoiceDate'] = pd.to_datetime(df_raw['InvoiceDate']) # Convert date column
# df_raw['Monetary'] = df_raw['Quantity'] * df_raw['Price'] # Calculate transaction monetary value

# # 2. Define the Observation Period
# # End Date is the day after the last transaction date (for Recency calculation)
# LAST_DATE = df_raw['InvoiceDate'].max() + pd.Timedelta(days=1)
# # Define the last quarter period for target variable (e.g., 90 days)
# TARGET_PERIOD_DAYS = 90
# OBSERVATION_END = LAST_DATE - pd.Timedelta(days=TARGET_PERIOD_DAYS)

# # 3. Create the Target Variable: Did the customer purchase in the last quarter?
# # Isolate transactions in the final quarter
# df_target = df_raw[df_raw['InvoiceDate'] >= OBSERVATION_END]
# target_customers = df_target['Customer ID'].unique()

# # 4. Create RFM Features based on data BEFORE the target period (up to OBSERVATION_END)
# df_rfm = df_raw[df_raw['InvoiceDate'] < OBSERVATION_END]

# # Calculate Recency (R): Days since the last purchase
# recency_df = df_rfm.groupby('Customer ID').agg(
#     LastPurchase=('InvoiceDate', 'max')
# )
# recency_df['Recency'] = (OBSERVATION_END - recency_df['LastPurchase']).dt.days

# # Calculate Frequency (F): Total number of unique invoices (orders)
# frequency_df = df_rfm.groupby('Customer ID').agg(
#     Frequency=('Invoice', 'nunique')
# )

# # Calculate Monetary (M): Total spend
# monetary_df = df_rfm.groupby('Customer ID').agg(
#     Monetary=('Monetary', 'sum')
# )

# # 5. Merge RFM Features
# customer_df = recency_df[['Recency']].merge(frequency_df, on='Customer ID')
# customer_df = customer_df.merge(monetary_df, on='Customer ID')

# # 6. Final Feature Set and Target Merge
# # Add the target column: 1 if customer_id is in the target_customers set, 0 otherwise
# customer_df[TARGET_COLUMN] = customer_df.index.isin(target_customers).astype(int)

# # Add Country as a demographic feature (most frequent country for the customer)
# country_df = df_raw.groupby('Customer ID')['Country'].agg(lambda x: x.mode()[0]).reset_index()
# customer_df = customer_df.merge(country_df, on='Customer ID', how='left')


# print("Dataset preview (Aggregated Customer Data):")
# print(customer_df.head(), "\n")

# # --- Data Preprocessing ---

# # 1. Handle Categorical Features (Country)
# categorical_cols = ['Country']
# customer_df = pd.get_dummies(customer_df, columns=categorical_cols, drop_first=True)

# # 2. Define Features and Target
# # Drop Customer ID as it is an identifier
# X = customer_df.drop([TARGET_COLUMN, 'Customer ID'], axis=1)
# y = customer_df[TARGET_COLUMN]

# # 3. Scaling (Essential for Logistic Regression interpretation)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# # 4. Train-Test Split (Stratified ensures both 0 and 1 classes are present)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)

# # --- Model Training and Evaluation ---
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
# print(coefficients.head(10)) # Show top 10 for clarity

# # Visualization
# plt.figure(figsize=(10, 8))
# sns.barplot(x='Coefficient', y='Feature', data=coefficients.head(10), palette='RdBu_r') 
# plt.title(f'Top 10 Influencers on Repeat Purchase Probability (Logistic Regression Coefficients)')
# plt.xlabel('Coefficient Value (Higher = Increased Likelihood)')
# plt.show()

# # Problem Description
# # Build a logistic regression model to predict the probability of a customer making a repeat purchase (target 'Repeat_Purchase' = 1) in the last quarter, using the transactional Online Retail dataset. This requires calculating Recency, Frequency, and Monetary (RFM) features and using the standardized coefficients to identify key influential factors.

# # Dataset Description
# # The dataset is aggregated from transaction data (Invoice, Quantity, Price, InvoiceDate, Customer ID, Country) to the customer level.
# # Features: Recency (Days since last purchase), Frequency (total orders), Monetary (total spend), and Country (Demographic).
# # Target: Repeat_Purchase (1 if purchased in the last 90 days, 0 otherwise).

# # Inference
# # The model will confirm the RFM hypothesis: Recency is the strongest factor, but with a negative sign. Since Recency is the *number of days since the last purchase*, a *lower* Recency value (more recent) is highly predictive of loyalty (positive probability). Therefore, Recency will have the largest magnitude, but a **negative coefficient**.
# # Frequency and Monetary will have large positive coefficients.
# # Key features: Recency (−), Frequency (+), Monetary (+).

# # Result
# # Customers with the shortest gaps between purchases (low Recency) and the highest purchase frequency are overwhelmingly the most likely to make a repeat purchase. This analysis confirms that current engagement and habit are stronger predictors of future spending than total value or country demographics.

# # Strongest Risk Factor (Negative Coefficient):
# # Recency (High number of days since last purchase indicates dormancy)

# # Strongest Protective Factors (Positive Coefficients):
# # Frequency (High number of orders)
# # Monetary (High total spend)

# # The platform should focus re-engagement efforts primarily on customers with high Recency scores (i.e., customers who haven't bought recently).

#------------------------------------------------------------------------------------------------------------------------

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns

# # --- CRITICAL CONFIGURATION: REPLACE VALUES BELOW ---
# FILE_PATH = "online_retail_transactions.csv" # *** Your dataset file path ***
# TARGET_COLUMN_NAME = 'Repeat_Purchase'
# # --- Column names for RFM Calculation (Must exist in your file) ---
# INV_DATE_COL = 'InvoiceDate'  # Column containing the date/time of the transaction
# CUST_ID_COL = 'Customer ID' # Column identifying the customer
# QTY_COL = 'Quantity'        # Column for item quantity
# PRICE_COL = 'Price'         # Column for unit price
# # -------------------------------------------------------------------

# # --- Data Loading and RFM Feature Engineering ---
# try:
#     df_raw = pd.read_csv(FILE_PATH, encoding='ISO-8859-1', na_values=['', 'NA', ' '])
# except FileNotFoundError:
#     print(f"Error: File not found at {FILE_PATH}. This script requires the transactional data.")
#     exit()

# # 1. Initial Cleaning and Prep
# df_raw.dropna(subset=[CUST_ID_COL], inplace=True)
# # Ensure Invoice/Transaction ID is handled (assuming a column named 'Invoice' exists for cancellation ID)
# if 'Invoice' in df_raw.columns:
#     df_raw = df_raw[~df_raw['Invoice'].astype(str).str.startswith('C')] 
    
# df_raw[CUST_ID_COL] = df_raw[CUST_ID_COL].astype(int) 
# df_raw[INV_DATE_COL] = pd.to_datetime(df_raw[INV_DATE_COL], errors='coerce') 
# df_raw['Monetary'] = df_raw[QTY_COL] * df_raw[PRICE_COL] 

# # 2. Define the Observation Period
# LAST_DATE = df_raw[INV_DATE_COL].max() + pd.Timedelta(days=1)
# TARGET_PERIOD_DAYS = 90
# OBSERVATION_END = LAST_DATE - pd.Timedelta(days=TARGET_PERIOD_DAYS)

# # 3. Create the Target Variable: Did the customer purchase in the last quarter?
# df_target = df_raw[df_raw[INV_DATE_COL] >= OBSERVATION_END]
# target_customers = df_target[CUST_ID_COL].unique()

# # 4. Create RFM Features based on data BEFORE the target period
# df_rfm = df_raw[df_raw[INV_DATE_COL] < OBSERVATION_END]

# # Calculate Recency (R): Days since the last purchase
# recency_df = df_rfm.groupby(CUST_ID_COL).agg(
#     LastPurchase=(INV_DATE_COL, 'max')
# )
# recency_df['Recency'] = (OBSERVATION_END - recency_df['LastPurchase']).dt.days

# # Calculate Frequency (F): Total number of unique invoices (orders)
# frequency_df = df_rfm.groupby(CUST_ID_COL).agg(
#     Frequency=('Invoice', 'nunique') if 'Invoice' in df_raw.columns else (CUST_ID_COL, 'count')
# )

# # Calculate Monetary (M): Total spend
# monetary_df = df_rfm.groupby(CUST_ID_COL).agg(
#     Monetary=('Monetary', 'sum')
# )

# # 5. Merge RFM Features
# customer_df = recency_df[['Recency']].merge(frequency_df, on=CUST_ID_COL)
# customer_df = customer_df.merge(monetary_df, on=CUST_ID_COL)

# # 6. Final Target Merge
# customer_df[TARGET_COLUMN_NAME] = customer_df.index.isin(target_customers).astype(int)

# # 7. Add Country/Demographic Features (Assuming 'Country' is a common demographic feature)
# # Merge any remaining non-RFM features here (e.g., Country, Age, Gender)
# if 'Country' in df_raw.columns:
#     country_df = df_raw.groupby(CUST_ID_COL)['Country'].agg(lambda x: x.mode()[0]).reset_index()
#     customer_df = customer_df.merge(country_df, on=CUST_ID_COL, how='left')


# print("Dataset preview (Aggregated Customer Data):")
# print(customer_df.head(), "\n")

# # --- Data Preprocessing (Universal Cleaning) ---

# # 1. Handle Categorical Features (Strings/Objects)
# categorical_cols = customer_df.select_dtypes(include='object').columns.tolist()
# if categorical_cols:
#     print(f"Found Categorical Columns: {categorical_cols}. Applying One-Hot Encoding.")
#     for col in categorical_cols:
#         customer_df[col] = customer_df[col].fillna('Missing')
#     customer_df = pd.get_dummies(customer_df, columns=categorical_cols, drop_first=True)

# # 2. Handle Numerical Missing Values (Imputation)
# numerical_cols = customer_df.select_dtypes(include=np.number).columns.tolist()
# numerical_cols_to_impute = [col for col in numerical_cols if col != TARGET_COLUMN_NAME and CUST_ID_COL not in col]

# if numerical_cols_to_impute:
#     print(f"Imputing {len(numerical_cols_to_impute)} numerical columns using the median.")
#     for col in numerical_cols_to_impute:
#         customer_df[col] = pd.to_numeric(customer_df[col], errors='coerce')
#         customer_df[col] = customer_df[col].fillna(customer_df[col].median())


# # --- Model Building ---
# # Exclude target and Customer ID
# X = customer_df.drop([TARGET_COLUMN_NAME, CUST_ID_COL], axis=1, errors='ignore')
# y = customer_df[TARGET_COLUMN_NAME]

# # Scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# # Train-Test Split (Stratified ensures both 0 and 1 classes are present)
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
# print(coefficients.head(10)) 

# # Visualization
# plt.figure(figsize=(10, 8))
# sns.barplot(x='Coefficient', y='Feature', data=coefficients.head(10), palette='RdBu_r') 
# plt.title(f'Top 10 Influencers on {TARGET_COLUMN_NAME} Probability (Logistic Regression Coefficients)')
# plt.xlabel('Coefficient Value (Higher = Increased Likelihood)')
# plt.show()

# # Problem Description
# # Build a generalized logistic regression model to predict the probability of a customer making a repeat purchase (target determined by purchasing in the last 90 days). The script automatically performs RFM feature engineering (Recency, Frequency, Monetary) from transactional data. The analysis identifies key influential factors using standardized coefficients.

# # Dataset Description
# # The script is designed for any transactional dataset containing the required columns: Date, Customer ID, Quantity, and Price. Features are engineered into Recency, Frequency, Monetary, plus any available demographics (e.g., Country).

# # Inference
# # The model will confirm that Recency is the most powerful predictor of repeat purchase. Because Recency is measured in 'days since last purchase', its coefficient will be the most negative (higher days = lower probability). Frequency and Monetary features will have positive coefficients.
# # Key features: Recency (Negative), Frequency (Positive), Monetary (Positive). 

# # Result
# # Customer loyalty and future purchase intent are best predicted by their recent and frequent interactions, confirming the predictive power of RFM.

# # Strongest Risk Factor (Negative Coefficient):
# # Recency (High number of days since last purchase indicates dormancy/churn risk)

# # Strongest Protective Factors (Positive Coefficients):
# # Frequency (High number of orders indicates loyalty)
# # Monetary (High total spend indicates customer value)

# # Actionable Insights:
# # The e-commerce platform should prioritize re-engagement campaigns based on Recency scores.
