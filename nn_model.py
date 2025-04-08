import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score
from preprocess import load_protein_data, load_data

# this class implements a protein secondary structure predictor using an LSTM neural network.
# it encodes sequences, trains a model, and predicts secondary structures (helix, sheet, coil).
class ProteinStructurePredictor:
    # define the amino acids in a string and create a dictionary mapping each amino acid to a unique integer
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    amino_acid_to_int = {aa: i+1 for i, aa in enumerate(amino_acids)}
    
    # initialize the predictor with default paths and regularization strength (l2_lambda)
    def __init__(self, l2_lambda=0.01, model_path="models/neural_network_model.h5", encoder_path="encoders/nn_encoder.pkl"):
        self.l2_lambda = l2_lambda  
        self.model = None  
        self.label_encoder = LabelEncoder()  
        self.inverse_structure_map = {0: 'c', 1: 'h', 2: 'e'}  # mapping of encoded labels back to structure types
        self.model_path = model_path  
        self.encoder_path = encoder_path 

    # loads protein data from a file, processes it, and encodes the sequences and structures for training or prediction
    def prosess_data(self, filename):
        df = load_protein_data(filename)  
        sequences, structures = load_data(df)  
        padded_sequences, padded_structures = self.encode_sequences(sequences, structures)  
        return padded_sequences, padded_structures, df  

    # encode amino acid sequences as integers
    # sequences are padded to the same length and structures are encoded using a label encoder
    def encode_sequences(self, sequences, structures):
        max_length = max(len(seq) for seq in sequences)  

        encoded_sequences = [[self.amino_acid_to_int.get(aa, 0) for aa in seq] for seq in sequences]
        padded_sequences = keras.preprocessing.sequence.pad_sequences(encoded_sequences, maxlen=max_length, padding='post')  # Pad sequences

        flat_structures = [s for sublist in structures for s in sublist]
        self.label_encoder.fit(flat_structures)  
        
        encoded_structures = [self.label_encoder.transform(seq) for seq in structures]
        padded_structures = keras.preprocessing.sequence.pad_sequences(encoded_structures, maxlen=max_length, padding='post')  # Pad structures

        return np.array(padded_sequences), np.array(padded_structures)

    # creates and compiles a neural network model
    # model consists of an embedding layer, an LSTM layer, a dropout layer, and a dense output layer
    def create_model(self, input_dimension, output_dimension):
        model = keras.Sequential([
            keras.layers.Embedding(input_dim=input_dimension + 1, output_dim=64, input_length=None), 
            keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=l2(self.l2_lambda)), 
            keras.layers.Dropout(0.3), 
            keras.layers.Dense(output_dimension, activation='softmax', kernel_regularizer=l2(self.l2_lambda))  
        ])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
        return model

    # trains the model on a protein training dataset
    # saves the trained model and label encoder to disk
    def train(self, train_file):
        padded_sequences, padded_structures, df = self.prosess_data(train_file)
        self.model = self.create_model(input_dimension=20, output_dimension=len(self.label_encoder.classes_))  
        self.model.fit(padded_sequences, padded_structures, epochs=10, batch_size=32) 
        self.model.save(self.model_path)  
        joblib.dump(self.label_encoder, self.encoder_path)  

    # predicts the secondary structure for a single amino acid sequence
    def predict_structure(self, seq, use_inverse_transform=True):
        processed_sequence = [self.amino_acid_to_int.get(aa, 0) for aa in seq]  
        processed_sequence = np.array(processed_sequence).reshape(1, -1)  

        predicted_structure = self.model.predict(processed_sequence) 
        predicted_labels = np.argmax(predicted_structure, axis=-1)[0]  

        # convert the predicted labels back to structure names using the label encoder or inverse map
        if use_inverse_transform:
            predicted_struct = ''.join(self.label_encoder.inverse_transform(predicted_labels).astype(str))
        else:
            predicted_struct = ''.join(self.inverse_structure_map[label] for label in predicted_labels)
        
        return predicted_struct

    # loads a pre-trained model and encoder from disk
    def load_model(self):
        self.model = load_model(self.model_path)  
        self.label_encoder = joblib.load(self.encoder_path) 

    # evaluates model performance on a labeled test dataset
    def evaluate(self, test_file):
        df_test = load_protein_data(test_file) 
        sequences_test, structures_test = load_data(df_test) 
        padded_sequences_test, padded_structures_test = self.encode_sequences(sequences_test, structures_test)  

        y_test = padded_structures_test  
        y_pred = self.model.predict(padded_sequences_test) 
        y_pred_labels = np.argmax(y_pred, axis=-1) 

        accuracy = accuracy_score(y_test.flatten(), y_pred_labels.flatten())  
        return accuracy 
