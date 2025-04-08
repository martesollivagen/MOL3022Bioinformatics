from nn_model import ProteinStructurePredictor

# test the Neural Network model
def main():
    nn_predictor = ProteinStructurePredictor()

    try:
        nn_predictor.load_model()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Model not found. Training the model...")
        padded_sequences, padded_structures, _ = nn_predictor.load_data("dataset/protein-secondary-structure.train")
        nn_predictor.train(padded_sequences, padded_structures)

    test_sequence = "MKTIIALSYIFCLVFADYKDDDDK"
    
    predicted_structure_labels, _ = nn_predictor.predict_structure(test_sequence)
    print(f"Predicted Secondary Structure: {''.join(predicted_structure_labels)}")

    accuracy = nn_predictor.evaluate("dataset/protein-secondary-structure.test")
    print(f"NN Model Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
