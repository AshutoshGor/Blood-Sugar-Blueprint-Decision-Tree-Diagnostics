import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree
import seaborn as sns
from sklearn.tree import export_text


# Load the dataset
data = pd.read_csv("diabetes.csv")

# Check the first few rows of the data
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Get statistics of the data
print(data.describe())

# Plot the graph for data visualization
sns.pairplot(data, hue='Outcome')
plt.show()

# Splitting the data for testing & training
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)


# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier
clf.fit(X_train, y_train)


# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", report)

# Decision Tree model
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=X.columns.tolist(), class_names=['No Diabetes', 'Diabetes'])

plt.xlim(0, 20)
plt.ylim(0, 5)

plt.show()

# Generate the tree structure as text
tree_text = export_text(clf, feature_names=X.columns.tolist())

# Save the tree structure as a text file
with open("decision_tree.txt", "w") as text_file:
    text_file.write(tree_text)
