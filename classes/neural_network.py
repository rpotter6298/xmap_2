import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Define the SimpleRNN model with Softmax activation
class SimpleRNN(nn.Module):
    """
    A simple Recurrent Neural Network (RNN) for multi-class classification.
    """
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)  # Softmax activation for multi-class
    
    def forward(self, x):
        """
        Forward pass of the RNN.
        """
        rnn_out, _ = self.rnn(x)
        output = self.fc(rnn_out[:, -1, :])  # Use the output from the last time step
        output = self.softmax(output)         # Softmax activation for multi-class classification
        return output

class NeuralNetworkModel:
    """
    A class to handle loading data, preprocessing, training, and evaluating a simple RNN model.
    """
    def __init__(self, data_path, features, target, hidden_size=64, n_layers=1, lr=0.001, batch_size=16, num_epochs=50, random_state=42, target_names = []):
        self.data_path = data_path
        self.features = features
        self.target = target
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.random_state = random_state
        self.target_names = target_names

        # Placeholders for data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Placeholders for tensors
        self.X_train_tensor = None
        self.X_test_tensor = None
        self.y_train_tensor = None
        self.y_test_tensor = None

        # DataLoader
        self.train_dataloader = None

        # Model, loss, optimizer
        self.model = None
        self.criterion = None
        self.optimizer = None

        # Label Encoder
        self.label_encoder = LabelEncoder()

        # Load and preprocess data
        self.load_data()
        self.preprocess_data()

        # Split the data
        self.split_data()

        # Convert to tensors
        self.convert_to_tensors()

        # Create DataLoader
        self.create_dataloader()

        # Define model
        self.define_model()

        # Define loss function and optimizer
        self.define_loss_optimizer()

    def load_data(self):
        """Loads the dataset from a CSV file."""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully from {self.data_path}. Shape: {self.data.shape}")
        except FileNotFoundError:
            print(f"File not found at {self.data_path}. Please check the path and try again.")
            raise

    def preprocess_data(self):
        """
        Selects relevant features and target, encodes target labels, reshapes data for RNN input.
        """
        # Encode target labels
        try:
            self.data['GPA_encoded'] = self.label_encoder.fit_transform(self.data[self.target])
            print(f"Target '{self.target}' encoded successfully.")
            print(f"Classes: {self.label_encoder.classes_}")
        except KeyError as e:
            print(f"KeyError: {e}. Please check if the target column exists in the dataset.")
            raise

        # Select features and target
        try:
            X = self.data[self.features].values
            y = self.data['GPA_encoded'].values
            print(f"Selected features: {self.features}")
            print(f"Selected target: 'GPA_encoded'")
        except KeyError as e:
            print(f"KeyError: {e}. Please check if the specified columns exist in the dataset.")
            raise

        # Reshape X for RNN: (samples, time steps, features)
        # Since we have only one time point, the time step dimension will be 1
        X = X.reshape(X.shape[0], 1, X.shape[1])
        print(f"Data reshaped for RNN. New shape of X: {X.shape}")

        self.X = X
        self.y = y

    def split_data(self):
        """
        Splits the data into training and test sets.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state, stratify=self.y
        )
        print(f"Data split into training and test sets.")
        print(f"Training set size: {self.X_train.shape[0]} samples")
        print(f"Test set size: {self.X_test.shape[0]} samples")

    def convert_to_tensors(self):
        """
        Converts NumPy arrays to PyTorch tensors.
        """
        self.X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        self.X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(self.y_train, dtype=torch.long)  # Changed to long for CrossEntropyLoss
        self.y_test_tensor = torch.tensor(self.y_test, dtype=torch.long)
        print("Converted NumPy arrays to PyTorch tensors.")

    def create_dataloader(self):
        """
        Creates a DataLoader for the training data.
        """
        train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        print("Training DataLoader created.")

    def define_model(self):
        """
        Defines the SimpleRNN model.
        """
        input_size = self.X_train.shape[2]  # Number of features at each time step
        output_size = len(self.label_encoder.classes_)  # Number of classes
        self.model = SimpleRNN(input_size, self.hidden_size, output_size, self.n_layers)
        print(f"SimpleRNN model defined with input_size={input_size}, hidden_size={self.hidden_size}, n_layers={self.n_layers}, output_size={output_size}")

    def define_loss_optimizer(self):
        """
        Defines the loss function and optimizer.
        """
        self.criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss for multi-class classification
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        print(f"Loss function (CrossEntropyLoss) and optimizer (Adam) defined with learning rate={self.lr}")

    def train_model(self):
        """
        Trains the RNN model.
        """
        self.model.train()
        print("Starting training...")
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for batch_X, batch_y in self.train_dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.train_dataloader)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
        print("Training completed.")

    def evaluate_model(self):
        """
        Evaluates the model on the test set and plots the ROC curve for each class.
        """
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(self.X_test_tensor)
            _, predicted = torch.max(test_outputs, 1)
            predicted = predicted.numpy()
            y_true = self.y_test
            y_pred = predicted

            # Calculate overall accuracy
            accuracy = np.mean(y_pred == y_true)
            print(f"Test Accuracy: {accuracy:.4f}")

            # Calculate ROC AUC for each class
            # One-hot encode the true labels
            y_true_onehot = np.zeros((y_true.size, len(self.label_encoder.classes_)))
            y_true_onehot[np.arange(y_true.size), y_true] = 1

            # Apply softmax to get probabilities
            y_probs = test_outputs.numpy()

            auc_scores = {}
            for i, class_label in enumerate(self.label_encoder.classes_):
                auc = roc_auc_score(y_true_onehot[:, i], y_probs[:, i])
                auc_scores[class_label] = auc
                print(f"Test AUC for class '{class_label}': {auc:.4f}")

            # Plot ROC curves for each class
            plt.figure(figsize=(10, 8))
            for i, class_label in enumerate(self.label_encoder.classes_):
                fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_probs[:, i])
                plt.plot(fpr, tpr, label=f"{class_label} (AUC = {auc_scores[class_label]:.2f})")

            plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curves for Multi-Class Classification")
            plt.legend(loc="lower right")
            plt.show()

            # Print Classification Report

            print("Classification Report:")
            print(classification_report(y_true, y_pred, target_names=self.target_names))

            # Plot Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.colorbar()
            tick_marks = np.arange(len(self.label_encoder.classes_))
            plt.xticks(tick_marks, self.label_encoder.classes_, rotation=45)
            plt.yticks(tick_marks, self.label_encoder.classes_)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")

            # Loop over data dimensions and create text annotations.
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.show()

    def run(self):
        """
        Runs the full training and evaluation pipeline.
        """
        self.train_model()
        self.evaluate_model()
