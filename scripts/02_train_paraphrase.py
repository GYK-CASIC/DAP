import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_scheduler
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import logging
import os
from datetime import datetime
from tqdm import tqdm

# Global hyperparameter settings
ROUND =                    # Adversarial round
DATA_START =               # Data_start_point
DATA_END =                 # Data_end_point
CHECKPOINT_PATH =          # Checkpoint_path
PRETRAINED_MODEL_PATH =    # Pretrained_model_path


DATA_RANGE = f'{DATA_START}_{DATA_END}'
TRAIN_DATA_PATH = f'./data_procedure/{ROUND}round/rl_{ROUND}round_{DATA_START}_{DATA_END}_rl.json'
SEED = 42
BATCH_SIZE_TRAIN = 8
BATCH_SIZE_VALID = 16
LEARNING_RATE = 1e-5
NUM_EPOCHS = 2


# Set random seed for reproducibility across libraries
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Get current time and set up output directory with timestamp
def get_output_dir():
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
    output_dir = f'./output/rl/{current_time}_{DATA_RANGE}_rl'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# Dataset class with format detection to handle different text types
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        """
        Initialize dataset by loading data and preparing texts with labels.
        
        Args:
            file_path (str): Path to JSON data file.
            tokenizer: Tokenizer for text processing.
            max_length (int): Maximum sequence length after tokenization.
        """
        with open(file_path, 'r') as file:
            data = json.load(file)
        self.texts = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        for entry in data:
            # Add original human-written text (label 0)
            self.texts.append(entry['original_text'])
            self.labels.append(0)  # 0 for original (human-written)
            
            # Detect if 'ai_generated_text' or 'paraphrased_text' is present for AI-generated text
            if 'paraphrased_text' in entry:
                self.texts.append(entry['paraphrased_text'])
                self.labels.append(1)  # 1 for paraphrased AI text
            elif 'ai_generated_text' in entry:
                self.texts.append(entry['ai_generated_text'])
                self.labels.append(1)  # 1 for AI-generated text
            else:
                raise ValueError(f"Entry {entry['id']} has neither 'ai_generated_text' nor 'paraphrased_text'.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get tokenized text and corresponding label for a given index.
        
        Returns:
            Dictionary containing input_ids, attention_mask, and labels.
        """
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0), 
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label)
        }

# Custom RoBERTa model for sequence classification
class CustomRoberta(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaForSequenceClassification.from_pretrained(PRETRAINED_MODEL_PATH, num_labels=2)

    def forward(self, input_ids, attention_mask, labels):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token indices.
            attention_mask: Mask for padding tokens.
            labels: True labels for loss calculation.
            
        Returns:
            Model outputs containing loss and logits.
        """
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output

# Calculate accuracy for human-written texts (label=0) at a given threshold
def get_human_accuracy_for_threshold(probabilities, labels, threshold):
    """
    Calculate accuracy for classifying human-written texts (label=0) using a threshold.
    
    Args:
        probabilities: List/array of probabilities for AI-generated class (P(AI|text)).
        labels: Ground truth labels (0=human, 1=AI).
        threshold: Threshold above which text is classified as AI (1), else human (0).
        
    Returns:
        Accuracy percentage for human-written texts.
    """
    human_total = 0
    human_correct = 0
    for prob, label in zip(probabilities, labels):
        if label == 0:  # Human-written text
            human_total += 1
            # Predict human (0) if probability <= threshold, else AI (1)
            pred_label = 0 if prob <= threshold else 1
            if pred_label == label:
                human_correct += 1
    return (human_correct / human_total * 100) if human_total > 0 else 0

# Search for optimal threshold in [0,1] with 0.0001 steps to get closest to 99% accuracy on human texts
def find_optimal_threshold(probabilities, labels):
    """
    Find the optimal threshold that makes accuracy on human texts (label=0) closest to 99%.
    
    Args:
        probabilities: List/array of probabilities for AI-generated class.
        labels: Ground truth labels.
        
    Returns:
        Optimal threshold value.
    """
    best_threshold = 0.0
    best_diff = float('inf')  # Stores the smallest difference from 99%
    for threshold in np.arange(0.0, 1.0001, 0.0001):
        accuracy = get_human_accuracy_for_threshold(probabilities, labels, threshold)
        diff = abs(accuracy - 99)
        if diff < best_diff:
            best_diff = diff
            best_threshold = threshold
    return best_threshold

# Inference function: process dataset once, return predicted probabilities (AI), true labels, and average loss
def evaluate(model, data_loader, device):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The trained model.
        data_loader: DataLoader for evaluation data.
        device: Computation device (CPU/GPU).
        
    Returns:
        Array of AI probabilities, array of true labels, and average loss.
    """
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(** batch)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            # Probability of being AI-generated (second class in softmax)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            labels = batch['labels'].cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels)

    avg_loss = total_loss / len(data_loader)
    return np.array(all_probs), np.array(all_labels), avg_loss

# Calculate accuracy metrics for original/human texts, AI texts, and overall based on a given threshold
def compute_metrics(probabilities, labels, threshold):
    """
    Calculate accuracy metrics for original texts, AI texts, and overall.
    
    Returns:
        (Accuracy on original texts, Accuracy on AI texts, Overall accuracy)
    """
    correct_original = 0
    total_original = 0
    correct_ai = 0
    total_ai = 0
    correct_total = 0
    total_total = 0

    for prob, label in zip(probabilities, labels):
        pred_label = 1 if prob > threshold else 0  # Predict AI if prob > threshold
        # Update counts for original texts (label=0)
        if label == 0:
            total_original += 1
            if pred_label == label:
                correct_original += 1
        # Update counts for AI texts (label=1)
        else:
            total_ai += 1
            if pred_label == label:
                correct_ai += 1
        # Update overall counts
        total_total += 1
        if pred_label == label:
            correct_total += 1

    # Calculate accuracy percentages
    acc_original = (correct_original / total_original * 100) if total_original > 0 else 0.0
    acc_ai = (correct_ai / total_ai * 100) if total_ai > 0 else 0.0
    acc_total = (correct_total / total_total * 100) if total_total > 0 else 0.0

    return acc_original, acc_ai, acc_total

def main():
    set_seed(SEED)
    
    # Create timestamped output directory for current training session
    session_output_dir = get_output_dir()
    
    # Set up logging to session-specific log file
    log_file = os.path.join(session_output_dir, "main.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize tokenizer and dataset
    tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
    train_dataset = TextDataset(TRAIN_DATA_PATH, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    # Set up computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomRoberta().to(device)
    
    # Load pre-trained model checkpoint if provided
    if CHECKPOINT_PATH:
        model.load_state_dict(torch.load(CHECKPOINT_PATH))

    # Configure optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    num_training_steps = NUM_EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear", 
        optimizer=optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps
    )

    # Log training parameters
    logging.info("Training Parameters:")
    logging.info(f"Learning Rate: {LEARNING_RATE}")
    logging.info(f"Batch Size (train): {BATCH_SIZE_TRAIN}")
    logging.info(f"Batch Size (valid): {BATCH_SIZE_VALID}")
    logging.info(f"Number of Epochs: {NUM_EPOCHS}")
    logging.info(f"Optimizer: AdamW")
    logging.info(f"Scheduler: Linear Scheduler")
    logging.info(f"Checkpoint: {CHECKPOINT_PATH}")

    for epoch in range(NUM_EPOCHS):
        # --------------------------
        # 1) Training phase
        # --------------------------
        model.train()
        for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Log training progress every 5 steps
            if step % 5 == 0:
                logging.info(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.item():.6f}")
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.6f}")

        # 2) Save model checkpoint after each epoch
        model_save_path = os.path.join(session_output_dir, f"model_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved model at the end of epoch {epoch+1} in {session_output_dir}")

if __name__ == "__main__":
    main()
