import pandas as pd

# loads and cleans protein structure data from a file
# converting datasets to a pandas DateFrame was used for inspecting the datasets and easily applying transformations 
# (e.g., replacing "_" with "c")
def load_protein_data(filename):

    # reads file and filters out comments, placeholder tokens, and the final "<end>" marker
    def read_clean_file(filename):
        with open(filename, "r") as file:
            lines = [line.strip() for line in file 
                     if line.strip() and not line.startswith(("#", "<>", "<end>"))]
        return lines

    data = read_clean_file(filename)

    # split each line into amino acid and structure, and store in DataFrame
    df = pd.DataFrame([line.split() for line in data], columns=['Amino_Acid', 'Structure'])

    # replace "_" with "c" (coil structure) for same structure as 'h' and 'e'
    df["Structure"] = df["Structure"].replace("_", "c")

    return df

# converts a flat DataFrame into lists of amino acid sequences and corresponding structures
def load_data(df):
    sequences, structures = [], []
    temp_seq, temp_struct = [], []

    # group rows into sequences based on the "end" marker
    for _, row in df.iterrows():
        if row["Amino_Acid"].lower() == "end":
            if temp_seq:
                sequences.append(temp_seq)
                structures.append(temp_struct)
            temp_seq, temp_struct = [], []
        else:
            temp_seq.append(row["Amino_Acid"])
            temp_struct.append(row["Structure"])
    
    # add final sequence if the file didn't end with "end"
    if temp_seq:
        sequences.append(temp_seq)
        structures.append(temp_struct)

    return sequences, structures
