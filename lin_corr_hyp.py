# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr, linregress

# df = pd.read_csv(r"Hours and Scores.csv")

# print("Dataset Loaded Successfully!\n")
# print(df.head())

# X = df['Hours']
# y = df['Scores']

# r, p_value = pearsonr(X, y)
# print(f"\nCorrelation Coefficient (r): {r:.3f}")
# print(f"P-value: {p_value:.5f}")

# alpha = 0.05
# if p_value < alpha:
#     print("Reject H₀: Significant correlation between study hours and exam scores.")
# else:
#     print("Fail to reject H₀: No significant correlation found.")

# slope, intercept, r_value, p_val, std_err = linregress(X, y)
# print(f"\nRegression Equation: Exam_Score = {intercept:.2f} + {slope:.2f} * Study_Hours")
# print(f"R-squared: {r_value**2:.3f}")
# print(f"P-value (Regression): {p_val:.5f}")

# plt.scatter(X, y, color='blue', label='Data Points')
# plt.plot(X, intercept + slope * X, color='red', label='Regression Line')
# plt.title('Study Hours vs Exam Scores')
# plt.xlabel('Study Hours')
# plt.ylabel('Exam Scores')
# plt.legend()
# plt.show()

#Interpretation
# Correlation:
# r = 0.997 → Very strong positive correlation.
# p < 0.05 → Statistically significant relationship.

# Regression:
# Regression equation:
# Exam Score = 35 + 5 × Study Hours
# R² = 0.994 → 99.4% of the variation in exam scores is explained by study hours.
# p < 0.05 → The slope is significant.

# Conclusion:
# Reject the Null Hypothesis (H₀).
# There is a strong, significant positive relationship between study hours and exam scores.
# As study time increases, exam performance tends to increase.
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr, linregress

# # 1. File and Column Names
# # --- Update the file path for your CSV file ---
# FILE_PATH = "your_data_file.csv" 

# # --- Update the column names exactly as they appear in your CSV ---
# X_COLUMN = "Independent Variable Column Name" # e.g., 'Study_Hours', 'Weekly_Exercise_Hours'
# Y_COLUMN = "Dependent Variable Column Name"  # e.g., 'Exam_Score', 'Resting_Heart_Rate'

# # 2. Contextual Labels (For Hypotheses, Plots, and Interpretation)
# CONTEXT_X = "Weekly Exercise Hours"  # Descriptive name for X-axis variable
# CONTEXT_Y = "Resting Heart Rate"   # Descriptive name for Y-axis variable

# # 3. Statistical Parameters
# ALPHA = 0.05 # Chosen significance level (commonly 0.05)


# # --- 1. Data Loading ---
# try:
#     df = pd.read_csv(FILE_PATH)
#     print(f"Dataset Loaded Successfully from: {FILE_PATH}\n")
#     print(df.head())
    
#     X = df[X_COLUMN]
#     Y = df[Y_COLUMN]

# except FileNotFoundError:
#     print(f"ERROR: File not found at '{FILE_PATH}'. Please check the file path.")
#     exit()
# except KeyError as e:
#     print(f"ERROR: Column {e} not found. Check that X_COLUMN and Y_COLUMN match the CSV headers.")
#     exit()
# except Exception as e:
#     print(f"An unexpected error occurred during data loading: {e}")
#     exit()


# # --- 2. Hypothesis Formulation ---
# print("\n--- HYPOTHESES ---")
# print(f"Null Hypothesis (H₀): There is no significant linear relationship (ρ = 0) between {CONTEXT_X} and {CONTEXT_Y}.")
# print(f"Alternative Hypothesis (Hₐ): There is a significant linear relationship (ρ ≠ 0) between {CONTEXT_X} and {CONTEXT_Y}.")


# # --- 3. Correlation Analysis ---
# r, p_value_corr = pearsonr(X, Y)
# print("\n--- CORRELATION ANALYSIS ---")
# print(f"Correlation Coefficient (r): {r:.4f}")
# print(f"P-value (Correlation): {p_value_corr:.5f}")

# # Correlation Decision
# print(f"\nDecision at α = {ALPHA}:")
# if p_value_corr < ALPHA:
#     print(f"Reject H₀: The correlation is statistically significant.")
#     is_significant = True
# else:
#     print(f"Fail to reject H₀: The correlation is not statistically significant.")
#     is_significant = False


# # --- 4. Simple Linear Regression ---
# print("\n--- SIMPLE LINEAR REGRESSION ---")

# if is_significant or True: # Always perform regression if requested, but check significance
    
#     slope, intercept, r_value_linregress, p_value_reg, std_err = linregress(X, Y)
#     r_squared = r_value_linregress**2

#     print(f"Regression Equation: {CONTEXT_Y} = {intercept:.4f} + ({slope:.4f}) * {CONTEXT_X}")
#     print(f"R-squared (R²): {r_squared:.4f}")
#     print(f"P-value (Slope Significance): {p_value_reg:.5f}")

#     # Slope Decision
#     print(f"\nSlope Decision at α = {ALPHA}:")
#     if p_value_reg < ALPHA:
#         print(f"The slope ({slope:.4f}) is statistically significant.")
#     else:
#         print(f"The slope ({slope:.4f}) is NOT statistically significant.")

#     # --- 5. Visualization ---
#     plt.figure(figsize=(9, 6))
#     plt.scatter(X, Y, color='blue', label='Data Points')
#     plt.plot(X, intercept + slope * X, color='red', label='Regression Line')
#     plt.title(f'{CONTEXT_X} vs. {CONTEXT_Y}')
#     plt.xlabel(CONTEXT_X)
#     plt.ylabel(CONTEXT_Y)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # --- 6. Interpretation and Conclusion ---
# print("\n--- INTERPRETATION & CONCLUSION ---")

# strength = 'no linear'
# if abs(r) >= 0.8: strength = 'very strong'
# elif abs(r) >= 0.6: strength = 'strong'
# elif abs(r) >= 0.3: strength = 'moderate'
# elif abs(r) > 0: strength = 'weak'

# direction = 'positive' if r > 0 else 'negative' if r < 0 else 'zero'
# direction_text = f'{direction} (r = {r:.4f})'

# print(f"1. Correlation: There is a {strength}, {direction_text} linear relationship.")
# if is_significant:
#     print(f"2. Significance: Since P-value ({p_value_corr:.5f}) < α ({ALPHA}), we REJECT H₀.")
#     print(f"3. Regression: The model explains {r_squared*100:.2f}% of the variation in {CONTEXT_Y}.")
#     print(f"4. Conclusion: A significant change in {CONTEXT_Y} ({slope:.4f} units) is associated with a one-unit increase in {CONTEXT_X}.")
# else:
#     print(f"2. Significance: Since P-value ({p_value_corr:.5f}) ≥ α ({ALPHA}), we FAIL TO REJECT H₀.")
#     print(f"3. Conclusion: There is no statistical evidence to support a linear relationship between {CONTEXT_X} and {CONTEXT_Y}.")
#-----------------------------------------------------------------------------------------------------------


# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr, linregress
# import numpy as np # Needed for array/list handling

# # --- Data Definition from Paper Lists ---
# # Replace these lists with the data given in your exam paper
# heart_rate_list = [75, 72, 68, 65, 63, 61, 58, 55, 53, 50]
# exercise_time_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20] 

# # Create DataFrame
# data = pd.DataFrame({
#     'Weekly_Exercise_Hours': exercise_time_list,
#     'Resting_Heart_Rate': heart_rate_list
# })

# print("Dataset Created Successfully!\n")
# print(data.head())

# # Define variables for analysis
# X = data['Weekly_Exercise_Hours']
# Y = data['Resting_Heart_Rate']

# # --- Correlation Analysis ---
# r, p_value = pearsonr(X, Y)
# print(f"\nCorrelation Coefficient (r): {r:.3f}")
# print(f"P-value: {p_value:.5f}")

# alpha = 0.05
# print("\n--- Hypothesis Test for Correlation ---")
# if p_value < alpha:
#     print("Reject H₀: Significant correlation between weekly exercise hours and resting heart rate.")
# else:
#     print("Fail to reject H₀: No significant correlation found.")

# # --- Simple Linear Regression ---
# # Note: linregress automatically calculates the p-value for the slope (p_val) 
# slope, intercept, r_value_linregress, p_val, std_err = linregress(X, Y)

# print("\n--- Regression Results ---")
# print(f"Regression Equation: Resting_Heart_Rate = {intercept:.2f} + {slope:.2f} * Weekly_Exercise_Hours")
# print(f"R-squared (R²): {r_value_linregress**2:.3f}")
# print(f"P-value (Slope Significance): {p_val:.5f}")

# # --- Visualization ---
# plt.figure(figsize=(8, 6))
# plt.scatter(X, Y, color='darkgreen', label='Data Points')
# plt.plot(X, intercept + slope * X, color='red', label='Regression Line')
# plt.title('Weekly Exercise Hours vs. Resting Heart Rate')
# plt.xlabel('Weekly Exercise Hours (Hours)')
# plt.ylabel('Resting Heart Rate (BPM)')
# plt.legend()
# plt.grid(True)
# plt.show()

# # --- Interpretation and Conclusion ---
# print("\n# --- Interpretation ---")

# # Correlation:
# print("# Correlation:")
# print(f"# r = {r:.3f} → Strong negative correlation. More exercise leads to a lower heart rate.")
# print(f"# p = {p_value:.5f} < 0.05 → Statistically significant relationship.")

# # Regression:
# print("\n# Regression:")
# print(f"# Regression Equation: Resting_Heart_Rate = {intercept:.2f} + ({slope:.2f}) * Weekly_Exercise_Hours")
# print(f"# Slope ({slope:.2f}): For every additional hour of exercise, the heart rate is predicted to decrease by {abs(slope):.2f} BPM.")
# print(f"# R² = {r_value_linregress**2:.3f} → {r_value_linregress**2 * 100:.1f}% of the variation in resting heart rate is explained by exercise hours.")
# print(f"# p < 0.05 → The slope (relationship) is statistically significant.")

# # Conclusion:
# print("\n# Conclusion:")
# print("# Reject the Null Hypothesis (H₀: ρ = 0).")
# print("# There is a strong, significant negative relationship between weekly exercise hours and resting heart rate.")
