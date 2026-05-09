import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("creditcard.csv")

# Show first rows
print(data.head())

# Dataset shape
print("Dataset Shape:", data.shape)

# Fraud count
fraud_count = data['Class'].value_counts()

print("\nFraud Count:")
print(fraud_count)

# Plot fraud vs normal transactions
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=data)

plt.title("Fraud vs Normal Transactions")
plt.xticks([0,1], ['Normal', 'Fraud'])

plt.show()

# Plot transaction amount distribution
plt.figure(figsize=(10,5))

sns.histplot(data['Amount'], bins=50)

plt.title("Transaction Amount Distribution")

plt.show()