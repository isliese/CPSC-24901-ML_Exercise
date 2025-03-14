from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd

# Load the Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Plot the trained decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree for Iris Dataset")
plt.show()

# Loan Application Decision Tree (Hypothetical Data)
data = {
    "Credit Score": [750, 650, 600, 700, 720, 580, 680, 690, 710, 620],
    "Income": [50000, 40000, 30000, 45000, 48000, 28000, 42000, 43000, 47000, 32000],
    "Approved": [1, 1, 0, 1, 1, 0, 1, 1, 1, 0]
}

df = pd.DataFrame(data)
X_loan = df[["Credit Score", "Income"]]
y_loan = df["Approved"]

# Train decision tree on loan data
clf_loan = DecisionTreeClassifier()
clf_loan.fit(X_loan, y_loan)

# Plot the decision tree for loan application
plt.figure(figsize=(10, 6))
plot_tree(clf_loan, feature_names=["Credit Score", "Income"], class_names=["Rejected", "Approved"], filled=True)
plt.title("Decision Tree for Loan Application")
plt.show()

