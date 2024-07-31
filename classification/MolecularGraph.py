import pandas as pd
import networkx as nx
import numpy as np
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, recall_score, precision_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('../../data/NR-AR-LBD_train.smiles', sep='\t', names=['SMILES', 'Label'])

# Define a function to convert SMILES to a molecular graph
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Create a molecular graph
    graph = nx.Graph()

    for atom in mol.GetAtoms():
        atom_id = atom.GetIdx()
        atom_symbol = atom.GetSymbol()
        graph.add_node(atom_id, symbol=atom_symbol)

    for bond in mol.GetBonds():
        start_atom_id = bond.GetBeginAtom().GetIdx()
        end_atom_id = bond.GetEndAtom().GetIdx()
        bond_type = bond.GetBondTypeAsDouble()
        graph.add_edge(start_atom_id, end_atom_id, bond_type=bond_type)

    return graph

# Convert SMILES strings to molecular graphs
data['Graph'] = data['SMILES'].apply(smiles_to_graph)

# Extract graph-based features (you may need to use more advanced methods here)
# Define a function to extract graph-based features
def extract_graph_features(graph):
    if graph is None:
        return None

    # Calculate basic graph properties
    num_nodes = len(graph.nodes)
    num_edges = len(graph.edges)
    average_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0

    # Convert features to a NumPy array
    features = np.array([num_nodes, num_edges, average_degree])

    return features

data['GraphFeatures'] = data['Graph'].apply(extract_graph_features)

# Remove rows with missing features
data = data.dropna(subset=['GraphFeatures'])

# Convert graph features to a suitable format (e.g., NumPy array)
X = np.array(data['GraphFeatures'].tolist())
y = data['Label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest Classifier
#clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize and train a Support Vector Classification (SVC) model
clf = SVC(kernel='rbf', random_state=42)

# Initialize and train a Decision Tree Classifier model
#clf = DecisionTreeClassifier(random_state=42)

# Initialize and train a GradientBoostingClassifier model
#clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the classification model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Calculate F1-score
f1 = f1_score(y_test, y_pred)
print(f'F1-Score: {f1:.4f}')

# Calculate Precision
precision = precision_score(y_test, y_pred)
print(f'Precision: {precision:.4f}')

# Calculate Recall
recall = recall_score(y_test, y_pred)
print(f'Recall: {recall:.4f}')

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")