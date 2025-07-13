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

# ==== Hyperparameter Settings ==== 
ROUND =                    # Adversarial round
DATA_START =               # Data_start_point
DATA_END =                 # Data_end_point
CHECKPOINT_PATH =          # Checkpoint_path
PRETRAINED_MODEL_PATH =    # Pretrained_model_path

DATA_RANGE = f'{DATA_START}_{DATA_END}'
TRAIN_FILE = f'./data_procedure/{ROUND}round/rl_{ROUND}round_{DATA_START}_{DATA_END}_finetune.json'
OUTPUT_DIR = './output/finetune' 
SEED = 42 
BATCH_SIZE_TRAIN = 8 
BATCH_SIZE_VALID = 16 
LEARNING_RATE = 1e-5 
NUM_EPOCHS = 2

# ==== Check and Create Output Directory ==== 
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set random seeds for reproducibility
def set_seed(seed): 
    """Set random seeds for reproducibility across multiple libraries."""
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)

# Create a timestamped output subdirectory for the current training session
def create_output_subdir(base_dir, data_range): 
    """Create a timestamped subdirectory for the current training session."""
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M') 
    subdir_name = f"{current_time}_{data_range}_finetune" 
    subdir = os.path.join(base_dir, subdir_name) 
    os.makedirs(subdir, exist_ok=True) 
    return subdir

# Dataset class for text classification
class TextDataset(Dataset):
    """Dataset class for loading text classification data."""
    def __init__(self, file_path, tokenizer, max_length=512, is_paraphrase=False):
        """
        Initialize the dataset.
        
        Args:
            file_path (str): Path to the JSON data file.
            tokenizer: Tokenizer to process the text.
            max_length (int): Maximum sequence length after tokenization.
            is_paraphrase (bool): If True, use paraphrased text instead of AI-generated text.
        """
        with open(file_path, 'r') as file:
            data = json.load(file)
        self.texts = []
        self.labels = []
        for entry in data:
            if is_paraphrase:
                # Add original text (label 0) and paraphrased text (label 1)
                self.texts.append(entry['original_text'])
                self.labels.append(0)  # 0 represents human
                self.texts.append(entry['paraphrased_text'])
                self.labels.append(1)  # 1 represents AI
            else:
                # Add original text (label 0) and AI-generated text (label 1)
                self.texts.append(entry['original_text'])
                self.labels.append(0)  # 0 represents human
                self.texts.append(entry['ai_generated_text'])
                self.labels.append(1)  # 1 represents AI
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """Return tokenized text and label for the given index."""
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0), 
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label)
        }

# Custom model wrapper for RoBERTa classification
class CustomRoberta(nn.Module):
    """Custom wrapper for RoBERTa model for sequence classification."""
    def __init__(self):
        super().__init__()
        self.roberta = RobertaForSequenceClassification.from_pretrained(PRETRAINED_MODEL_PATH, num_labels=2)

    def forward(self, input_ids, attention_mask, labels):
        """Forward pass through the model."""
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output

# Helper function to calculate human text accuracy at a given threshold
def get_human_accuracy_for_threshold(probabilities, labels, threshold):
    """
    Calculate accuracy for human-written texts at a given threshold.
    
    Args:
        probabilities (list): List of predicted probabilities for being AI-generated.
        labels (list): List of true labels (0 for human, 1 for AI).
        threshold (float): Threshold for classifying as AI-generated.
        
    Returns:
        float: Accuracy percentage for human-written texts.
    """
    human_total = 0
    human_correct = 0
    for prob, label in zip(probabilities, labels):
        if label == 0:  # Human-written text
            human_total += 1
            if prob <= threshold:  # Correctly classified as human
                human_correct += 1
    return (human_correct / human_total * 100) if human_total > 0 else 0

# Function to find the optimal threshold for classifying human texts
def find_optimal_threshold(probabilities, labels):
    """
    Find the optimal threshold that achieves closest to 99% accuracy on human texts.
    
    Args:
        probabilities (list): List of predicted probabilities for being AI-generated.
        labels (list): List of true labels (0 for human, 1 for AI).
        
    Returns:
        float: Optimal threshold value.
    """
    best_threshold = 0.0
    best_accuracy = 0.0
    # Search for threshold in small increments from 0.0 to 1.0
    for threshold in np.arange(0.0, 1.0001, 0.0001):
        accuracy = get_human_accuracy_for_threshold(probabilities, labels, threshold)
        if abs(accuracy - 99) < abs(best_accuracy - 99):
            best_accuracy = accuracy
            best_threshold = threshold
    return best_threshold

# Main training and validation function
def main(): 
    """Main training and validation loop."""
    set_seed(SEED)
    
    # Create timestamped output directory for this training session
    session_output_dir = create_output_subdir(OUTPUT_DIR, DATA_RANGE)
    
    # Configure logging to output to both console and file
    log_file = os.path.join(session_output_dir, "main.log") 
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Initialize tokenizer and datasets
    tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL_PATH) 
    train_dataset = TextDataset(TRAIN_FILE, tokenizer) 

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True) 
    
    # Set up device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = CustomRoberta().to(device)
    
    # Load model checkpoint if provided
    if CHECKPOINT_PATH:
        model.load_state_dict(torch.load(CHECKPOINT_PATH))
        logging.info(f"Loaded model checkpoint from {CHECKPOINT_PATH}")

    # Configure optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    num_training_steps = NUM_EPOCHS * len(train_loader) 
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Log training parameters
    logging.info("Training Parameters:") 
    logging.info(f"Learning Rate: {LEARNING_RATE}") 
    logging.info(f"Batch Size (train): {BATCH_SIZE_TRAIN}") 
    logging.info(f"Batch Size (valid): {BATCH_SIZE_VALID}") 
    logging.info(f"Number of Epochs: {NUM_EPOCHS}") 
    logging.info(f"Optimizer: AdamW") 
    logging.info(f"Scheduler: Linear Scheduler") 
    logging.info(f"Checkpoint: {CHECKPOINT_PATH}")

    # Training loop
    for epoch in range(NUM_EPOCHS): 
        model.train() 
        for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")): 
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()} 
            
            # Forward pass
            outputs = model(**batch) 
            loss = outputs.loss 
            
            # Backward pass and optimization
            loss.backward() 
            optimizer.step() 
            lr_scheduler.step() 
            optimizer.zero_grad()

            # Log training progress
            if step % 5 == 0: 
                logging.info(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.item()}") 
                print(f"Step {step}, Loss: {loss.item()}")

        # Save model checkpoint after each epoch
        model_save_path = os.path.join(session_output_dir, f"model_epoch{epoch+1}.pth") 
        torch.save(model.state_dict(), model_save_path) 
        logging.info(f"Saved model at the end of epoch {epoch+1} in {session_output_dir}")
        print(f"Saved model at the end of epoch {epoch+1} in {session_output_dir}")

if __name__ == "__main__": 
    main()
