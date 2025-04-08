import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from preprocess import load_protein_data, load_data

# this class implements a secondary structure predictor using a Random Forest classifier
# it uses a sliding window approach and one-hot encoding to train and make predictions
class RFProteinSecondaryStructurePredictor:
    # initialize the predictor with default paths and sliding window size
    def __init__(self, window_size=7, model_path="models/random_forest_model.pkl", encoder_path="encoders/rf_encoder.pkl"):
        self.window_size = window_size
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.model = None
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore") # encoder for amino acids
        self.structure_map = {'c': 0, 'h': 1, 'e': 2} # mapping structure labels to integers
        self.inverse_structure_map = {v: k for k, v in self.structure_map.items()}  # reverse mapping for predictions
    
    # pads the input sequence on both sides with 'P' to support sliding windows at the edges
    # 'P' (proline) is used as padding because it has minimal impact on protein folding and structure predictions
    def pad_sequence(self, seq):
        pad_size = self.window_size // 2
        return ['P'] * pad_size + list(seq) + ['P'] * pad_size

    # generates sliding windows from amino acid sequences
    def create_sliding_windows(self, sequences, structures):
        X_windowed, y_windowed = [], []
        
        for seq, structure in zip(sequences, structures):
            padded_seq = self.pad_sequence(seq)
            for i in range(len(seq)):
                window = padded_seq[i:i + self.window_size]  
                X_windowed.append(window)
                y_windowed.append(structure[i]) 
                
        return X_windowed, y_windowed

    # fits a one-hot encoder on a flat list of amino acids and saves it to disk
    def fit_encoder(self, flat_amino_acids):
        self.encoder.fit(np.array(flat_amino_acids).reshape(-1, 1))
        joblib.dump(self.encoder, self.encoder_path)

    # encodes a list of sliding windows using the fitted one-hot encoder
    def encode_windows(self, windows):        
        flat = [aa for window in windows for aa in window] 
        encoded = self.encoder.transform(np.array(flat).reshape(-1, 1))  
        return encoded.reshape(len(windows), -1)

    # trains the Random Forest model on a protein training dataset
    # loads the data, creates sliding windows, encodes features, fits the model, and saves it
    def train(self, train_file):
        train_df = load_protein_data(train_file)
        sequences, structures = load_data(train_df)
        
        X_windowed, y_windowed = self.create_sliding_windows(sequences, structures)
        X_flat = [aa for window in X_windowed for aa in window]  
        
        self.fit_encoder(X_flat)
        X_encoded_windowed = self.encode_windows(X_windowed)
        
        self.model = RandomForestClassifier(n_estimators=3520, max_depth=15, random_state=42)
        self.model.fit(X_encoded_windowed, y_windowed)
        
        joblib.dump(self.model, self.model_path)

    # loads the pre-trained model and encoder from disk
    def load_model(self):
        self.model = joblib.load(self.model_path)
        self.encoder = joblib.load(self.encoder_path)

    # predicts the secondary structure for a single amino acid sequence
    def predict(self, sequence):
        if self.model is None:
            self.load_model()

        padded_seq = self.pad_sequence(sequence)
        X_new = [padded_seq[i:i + self.window_size] for i in range(len(sequence))]
        X_encoded_new = self.encode_windows(X_new)
        predicted_structure = self.model.predict(X_encoded_new)
        
        return ''.join(str(label) for label in predicted_structure)

    # evaluates model performance on a labeled test dataset
    def evaluate(self, test_file):
        
        test_df = load_protein_data(test_file)
        sequences, structures = load_data(test_df)
        
        X_windowed, y_windowed = self.create_sliding_windows(sequences, structures)
        X_encoded_windows = self.encode_windows(X_windowed)
        
        y_encoded = np.array([str(label) for label in y_windowed])
        
        y_pred = self.model.predict(X_encoded_windows)
        accuracy = accuracy_score(y_encoded, y_pred)
        
        return accuracy