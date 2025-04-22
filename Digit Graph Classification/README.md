
## File Structure
We use two kinds of model in this project
ImprovedCNN:
- `improvedCNN.ipynb`: The script to train and evaluate ImprovedCNN model.

Resnet Based Transformer:
- `train_main.ipynb`: The main script to train and evaluate the ResNet model.
- `model_resnet_se_transformer.py`: Implementation of the ResNet based Transformer model with the ability to choose desire architecture.
- `engine_main.py`: Utility functions for data loading, training, and evaluation.
- `customdatasets.py`: Use dataset with some data augumentation.
- `requirements.txt`: List of Python packages and versions used in this project.

## Hyperparameters

### Training Parameters:
- `--batch_size`: Specifies the batch size for training. The number of samples that will be processed in each iteration.
- `--epochs`: Specifies the number of training epochs. An epoch is one complete pass through the entire training dataset.
- `--start_epoch`: Specifies the starting epoch for training.
- `--model`: Specifies the name of the model architecture to train (e.g., 'resnet_18').

## How to use
For ImprovedCNN model. Open `CNN.ipynb` and run it.

For Resnet Based Transformer model. Open `train_main.ipynb` and run it.
