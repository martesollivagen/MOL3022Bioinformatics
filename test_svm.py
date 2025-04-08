from svm_model import SVMProteinSecondaryStructurePredictor  

# test the SVM model
def main():
    svm_predictor = SVMProteinSecondaryStructurePredictor()

    try:
        svm_predictor.load_model()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Model not found. Training the model...")
        svm_predictor.train("dataset/protein-secondary-structure.train", "dataset/protein-secondary-structure.test")
    
    test_sequence = "MKTIIALSYIFCLVFADYKDDDDK"
    
    predicted_structure = svm_predictor.predict(test_sequence)
    print(f"Predicted Structures: {predicted_structure}")

    accuracy = svm_predictor.evaluate("dataset/protein-secondary-structure.test")
    print(f"SVM Model Accuracy: {accuracy}")

if __name__ == "__main__":
    main()