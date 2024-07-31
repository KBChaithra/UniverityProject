import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

# Load the dataset from the .smiles file
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('../../data/NR-AR-LBD_train.smiles', sep='\t', names=['Molecules', 'Label'])

# Calculate molecular descriptors for each molecule
def calculate_molecular_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Calculate a set of molecular descriptors
    descriptors = {
        'MolWt': Descriptors.MolWt(mol),  # Molecular weight
        'LogP': Descriptors.MolLogP(mol),  # Hydrophobicity
        'TPSA': Descriptors.TPSA(mol),  # Polar surface area
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),  # Hydrogen bond acceptors
        'NumHDonors': Descriptors.NumHDonors(mol),  # Hydrogen bond donors
    }

    return descriptors

# Extract molecular descriptors and labels
descriptor_list = []
binary_labels = []

for index, row in data.iterrows():
    descriptors = calculate_molecular_descriptors(row['Molecules'])
    if descriptors is not None:
        descriptor_list.append(descriptors)
        binary_labels.append(row['Label'])

# Convert the list of descriptors to a DataFrame
descriptor_df = pd.DataFrame(descriptor_list)

# Fill any missing descriptor values with zeros (or you can use other strategies)
descriptor_df = descriptor_df.fillna(0)

# Convert the descriptor DataFrame and labels to numpy arrays
X = descriptor_df.values
y = np.array(binary_labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest Classifier (you can choose other classifiers)
#clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize and train a Support Vector Classification (SVC) model
#clf = SVC(kernel='rbf', random_state=42)

# Initialize and train a Decision Tree Classifier model
clf = DecisionTreeClassifier(random_state=42)

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
