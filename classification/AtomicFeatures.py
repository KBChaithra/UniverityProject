import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define a function to extract atomic features for an atom
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error


def extract_atomic_features(atom):
    atomic_features = {
        'atomic_number': float(atom.GetAtomicNum()),
        'atomic_mass': float(atom.GetMass()),
        'atomic_charge': float(atom.GetFormalCharge()),
        'hybridization_state': atom.GetHybridization(),
        'valence_electrons': float(atom.GetTotalValence()),
        'num_non_hydrogen_neighbors': len([bond for bond in atom.GetBonds() if not bond.GetBeginAtom().GetSymbol() == 'H']),
    }
    return atomic_features

# Load the dataset from the .smiles file
data = pd.read_csv('../../data/NR-AR-LBD_train.smiles', sep='\t', names=['Molecules', 'Label'])

# Convert SMILES strings to RDKit molecules
data['Molecule'] = data['Molecules'].apply(Chem.MolFromSmiles)

# Find the maximum number of atoms in a molecule
max_atoms = max(len(molecule.GetAtoms()) for molecule in data['Molecule'])

# Create an empty NumPy array filled with zeros
X = np.zeros((len(data), max_atoms, 6), dtype=np.float32)

# Generate atomic features for each molecule and fill the array with padding
for i, molecule in enumerate(data['Molecule']):
    atomic_features_molecule = [extract_atomic_features(atom) for atom in molecule.GetAtoms()]
    num_atoms = len(atomic_features_molecule)
    X[i, :num_atoms, :] = [list(atomic_features.values()) for atomic_features in atomic_features_molecule]

# Extract labels and convert to a NumPy array
y = data['Label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest Classifier (you can choose other classifiers)
#clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize and train a Support Vector Classification (SVC) model
#clf = SVC(kernel='rbf', random_state=42)

# Initialize and train a Decision Tree Classifier model
clf = DecisionTreeClassifier(random_state=42)

# Initialize and train a GradientBoostingClassifier model
#clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

clf.fit(X_train.reshape(X_train.shape[0], -1), y_train)



# Make predictions on the test set
y_pred = clf.predict(X_test.reshape(X_test.shape[0], -1))

# Evaluate the classification model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'F1-score: {f1:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')

# Calculate and print the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
