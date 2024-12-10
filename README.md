# RepAs
This is the implementation of "RepAs: Iterative Refinement for Predicting Polyadenylation Site Usage", SAC, 2025.

## Local Environment Setup
conda create --n `name` python=3.12 \
conda activate `name`

## Dependencies
Our model was developed in Windows with the following packages:
- pandas
- torch == 2.3.1
- cuda == 12.1
- torch_geometric

## Arguments
train_path: Path to training set \
val_path: Path to validation set \
test_path: Path to testing set \
model_path: Path for saving model \
hidden_size: Size of hidden layers \
k: Number of nucleotides in the k-mers \
lamda: Regularization parameter

## Training
To train the model, please run `python main.py`

## Results
After the script completes, a file named `results.txt` is created containing the poly(A) site id, true usage value and predicted usage value for each poly(A) site of the testing set.
