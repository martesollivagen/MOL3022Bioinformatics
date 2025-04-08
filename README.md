# MOL3022 - Bioinformatics

## Overview

This is a tool for predicting protein secondary structure, while also comparing different machine learning models. It lets the user input a protein sequence, and gives the predicted secondary structure and accuracy from three different models; neural network, random forest and support vector machine (SVM). 

## Run Locally

To run this project locally, follow the steps below:

- Ensure Python 3.7 or later is installed.
- Clone the repository:  
  `git clone https://github.com/martesollivagen/MOL3022Bioinformatics.git`
- Install dependencies using `pip`:

  ```bash
  pip install numpy pandas scikit-learn matplotlib tensorflow keras

- If all packages cannot be installed directly, create a virtual environment (as shown below) and then run the `pip install` command above:
1. create environment
    ```bash
    python -m venv env
    ```

2. activate environment:
    ```bash
    source env/bin/activate # on macOS/Linux
    ```
    or
    ```bash
    env\Scripts\activate # on Windows
    ```