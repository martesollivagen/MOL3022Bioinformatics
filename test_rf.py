from rf_model import RFProteinSecondaryStructurePredictor

# test the Random Forest model
def main():
    rf_predictor = RFProteinSecondaryStructurePredictor()

    try:
        rf_predictor.load_model()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Model not found. Training the model...")
        rf_predictor.train("dataset/protein-secondary-structure.train", "dataset/protein-secondary-structure.test")

    test_sequence = "MKTIIALSYIFCLVFADYKDDDDK"
    
    predicted_structure = rf_predictor.predict(test_sequence)
    print(f"Predicted Secondary Structure: {predicted_structure}")

    accuracy = rf_predictor.evaluate("dataset/protein-secondary-structure.test")
    print(f"RF Model Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
