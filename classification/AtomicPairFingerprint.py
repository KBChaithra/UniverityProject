import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

# Define a function to convert SMILES to Atom Pair fingerprints
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def smiles_to_atom_pair_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fingerprint = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol)
    return fingerprint

# Load the dataset from the .smiles file
data = pd.read_csv('../../data/NR-AR-LBD_train.smiles', sep='\t', names=['Molecules', 'Label'])

# Convert SMILES strings to Atom Pair fingerprints
data['AtomPairFingerprint'] = data['Molecules'].apply(smiles_to_atom_pair_fingerprint)

# Convert Atom Pair fingerprints to numerical arrays
fp_matrix = np.array([list(map(int, x.ToBitString())) for x in data['AtomPairFingerprint']])

# Extract labels and convert to a NumPy array
y = data['Label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(fp_matrix, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest Classifier (you can choose other classifiers)
#clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize and train a Support Vector Classification (SVC) model
#clf = SVC(kernel='rbf', random_state=42)

# Initialize and train a Decision Tree Classifier model
clf = DecisionTreeClassifier(random_state=42)

# Initialize and train a GradientBoostingClassifier model
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
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')

# Calculate and print the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
