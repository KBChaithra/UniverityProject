import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

# Load the dataset from the .smiles file
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('../../data/NR-AR-LBD_train.smiles', sep='\t', names=['Molecules', 'Label'])

# Convert SMILES strings to Morgan fingerprints
def smiles_to_morgan_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2)  # Change radius as needed
    return fingerprint

# Extract Morgan fingerprints and labels
fingerprints = []
binary_labels = []

for index, row in data.iterrows():
    fingerprint = smiles_to_morgan_fingerprint(row['Molecules'])
    if fingerprint is not None:
        fingerprints.append(fingerprint)
        binary_labels.append(row['Label'])

# Convert Morgan fingerprints to numpy array
X = np.array(fingerprints)

# Convert binary labels to numpy array
y = np.array(binary_labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest Classifier (you can choose other classifiers)
#clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize and train a Support Vector Classification (SVC) model
clf = SVC(kernel='rbf', random_state=42)

# Initialize and train a Decision Tree Classifier model
#clf = DecisionTreeClassifier(random_state=42)

# Initialize and train a GradientBoostingClassifier
#clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the classification model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'F1-score: {f1:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

