import tkinter as tk
from tkinter import messagebox
from svm_model import SVMProteinSecondaryStructurePredictor
from rf_model import RFProteinSecondaryStructurePredictor
from nn_model import ProteinStructurePredictor
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 

# class to create a graphical user interface for predicting protein structures
class ProteinStructurePredictorGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Protein Structure Predictor")
        self.geometry("900x800")

        # initialize the predictors
        self.svm_predictor = SVMProteinSecondaryStructurePredictor()
        self.rf_predictor = RFProteinSecondaryStructurePredictor()
        self.nn_predictor = ProteinStructurePredictor()

        # load models
        self.load_models()

        # create GUI elements
        self.create_widgets()

    # load pre-trained models or train them if they are not found
    def load_models(self):
        try:
            self.svm_predictor.load_model()
            self.rf_predictor.load_model()
            self.nn_predictor.load_model()
            print("Models loaded successfully.")
        except FileNotFoundError:
            print("Model not found. Training the models...")
            self.svm_predictor.train("dataset/protein-secondary-structure.train")
            self.rf_predictor.train("dataset/protein-secondary-structure.train")
            self.nn_predictor.train("dataset/protein-secondary-structure.train")

    # create and place the GUI widgets on the window
    def create_widgets(self):
        self.sequence_label = tk.Label(self, text="Enter Protein Sequence:")
        self.sequence_label.pack(pady=10)

        self.sequence_entry = tk.Entry(self, width=50)
        self.sequence_entry.pack(pady=10)

        self.predict_button = tk.Button(self, text="Predict", command=self.predict_structure)
        self.predict_button.pack(pady=10)

        # labels to show the predicted structure and model accuracy
        self.svm_label = tk.Label(self, text="SVM Predicted Structure: ")
        self.svm_label.pack(pady=5)

        self.nn_label = tk.Label(self, text="NN Predicted Structure: ")
        self.nn_label.pack(pady=5)

        self.rf_label = tk.Label(self, text="RF Predicted Structure: ")
        self.rf_label.pack(pady=5)

        self.svm_accuracy_label = tk.Label(self, text="SVM Model Accuracy: ")
        self.svm_accuracy_label.pack(pady=5)

        self.nn_accuracy_label = tk.Label(self, text="NN Model Accuracy: ")
        self.nn_accuracy_label.pack(pady=5)

        self.rf_accuracy_label = tk.Label(self, text="RF Model Accuracy: ")
        self.rf_accuracy_label.pack(pady=5)

        # frame to display the plot of model accuracies
        self.plot_frame = tk.Frame(self)
        self.plot_frame.pack(pady=20)

    # display a bar plot comparing training and testing accuracy of the models
    def show_accuracy(self, nn_train, nn_test, rf_train, rf_test, svm_train, svm_test):
        models = ['Neural Network', 'Random Forest', 'SVM']
        train_accuracy = [nn_train, rf_train, svm_train]  
        test_accuracy = [nn_test, rf_test, svm_test]   

        x = np.arange(len(models))  
        width = 0.3 

        fig, ax = plt.subplots(figsize=(8, 5))

        bars1 = ax.bar(x - width/2, train_accuracy, width, label='Training Accuracy', color='blue')
        bars2 = ax.bar(x + width/2, test_accuracy, width, label='Test Accuracy', color='orange')

        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy on Training vs. Test Dataset')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1)
        ax.legend()

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), 
                            textcoords="offset points",
                            ha='center', va='bottom')

        for widget in self.plot_frame.winfo_children():
            widget.destroy() 

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    # predict the structure of the entered protein sequence
    def predict_structure(self):
        protein_sequence = self.sequence_entry.get()

        if not protein_sequence:
            messagebox.showerror("Error", "Please enter a protein sequence.")
            return

        try:
            svm_predicted_structure = self.svm_predictor.predict(protein_sequence)
            nn_predicted_structure = self.nn_predictor.predict_structure(protein_sequence)
            rf_predicted_structure = self.rf_predictor.predict(protein_sequence)

            svm_accuracy = self.svm_predictor.evaluate("dataset/protein-secondary-structure.test")
            nn_accuracy = self.nn_predictor.evaluate("dataset/protein-secondary-structure.test")
            rf_accuracy = self.rf_predictor.evaluate("dataset/protein-secondary-structure.test")

            # these values represent the accuracy of each model on the training dataset
            # currently, these values are hard-coded as placeholders
            # if the models are retrained or updated, the code below can be uncommented 
            svm_accuracy_train = 0.65 #self.svm_predictor.evaluate("dataset/protein-secondary-structure.train")
            nn_accuracy_train = 0.86 #self.nn_predictor.evaluate("dataset/protein-secondary-structure.train")
            rf_accuracy_train = 0.62 #self.rf_predictor.evaluate("dataset/protein-secondary-structure.train")

            # update the GUI with predictions and accuracies
            self.svm_label.config(text=f"SVM Predicted Structure: {svm_predicted_structure}")
            self.nn_label.config(text=f"NN Predicted Structure: {''.join(nn_predicted_structure)}")
            self.rf_label.config(text=f"RF Predicted Structure: {rf_predicted_structure}")
            self.svm_accuracy_label.config(text=f"SVM Model Accuracy: {svm_accuracy:.2f}")
            self.nn_accuracy_label.config(text=f"NN Model Accuracy: {nn_accuracy:.2f}")
            self.rf_accuracy_label.config(text=f"RF Model Accuracy: {rf_accuracy:.2f}")

            self.show_accuracy(nn_accuracy_train, nn_accuracy, rf_accuracy_train, rf_accuracy, svm_accuracy_train, svm_accuracy)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


if __name__ == "__main__":
    app = ProteinStructurePredictorGUI()
    app.mainloop()
