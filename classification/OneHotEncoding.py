import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

# Load the dataset
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('../../data/NR-AR-LBD_train.smiles', sep='\t', names=['SMILES', 'Label'])

# Define a function to convert SMILES to one-hot encoding
def smiles_to_one_hot(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * max_length  # Return an all-zero vector for invalid SMILES
    else:
        atom_list = [atom.GetSymbol() for atom in mol.GetAtoms()]
        one_hot = [0] * max_length
        for i, atom in enumerate(atom_list):
            if i >= max_length:
                break  # Stop if we reach max_length
            # Create a unique integer identifier for each unique atom symbol
            if atom not in atom_id_mapping:
                atom_id_mapping[atom] = len(atom_id_mapping) + 1
            atom_id = atom_id_mapping[atom]
            one_hot[i] = atom_id
        return one_hot

# Initialize a dictionary to map atom symbols to unique integer identifiers
atom_id_mapping = {}

# Determine the maximum length for one-hot encoding
max_length = max(len(Chem.MolFromSmiles(smiles).GetAtoms()) for smiles in data['SMILES'])

# Apply one-hot encoding to SMILES strings
data['OneHot'] = data['SMILES'].apply(smiles_to_one_hot)

# Convert one-hot encoding to numerical arrays
X = np.array(data['OneHot'].tolist())

# Convert labels to a NumPy array
y = data['Label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest Classifier (or another classifier of your choice)
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
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)


print(f'Accuracy: {accuracy:.4f}')
print(f'F1-score: {f1:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
