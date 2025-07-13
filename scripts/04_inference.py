import json
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

# Configuration
class Config:
    """Configuration class for model inference"""
    model_path = ''        # Path to the saved model
    test_file_path = ''    # Path to the test dataset
    tokenizer_path = ''    # Path to the tokenizer
    batch_size = 32
    max_length = 512
    output_dir = "./output/inference"
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Inference dataset class
class InferenceTextDataset(Dataset):
    """Dataset class for inference on text classification tasks"""
    def __init__(self, file_path, tokenizer, text_key, max_length=512):
        """
        Initialize the dataset for inference
        
        Args:
            file_path (str): Path to the JSON dataset file
            tokenizer: Tokenizer to process text
            text_key (str): Key in JSON to retrieve text data
            max_length (int): Maximum sequence length after tokenization
        """
        with open(file_path, 'r') as file:
            data = json.load(file)
        self.texts = []
        self.labels = []
        self.ids = []
        
        for idx, entry in enumerate(data):
            self.texts.append(entry[text_key])
            # Original text is labeled as 0, generated/paraphrased text as 1
            self.labels.append(1 if text_key != 'original_text' else 0)
            self.ids.append(idx)  # Unique identifier for each text
            
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """Return tokenized text, label, and metadata for a given index"""
        text = self.texts[idx]
        label = self.labels[idx]
        text_id = self.ids[idx]
        
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
            'labels': torch.tensor(label),
            'text': text,
            'id': text_id
        }

# Custom Roberta model class
class CustomRoberta(torch.nn.Module):
    """Custom Roberta model class for sequence classification"""
    def __init__(self):
        super().__init__()
        self.roberta = RobertaForSequenceClassification.from_pretrained(
            Config.tokenizer_path, 
            num_labels=2
        )

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass through the model"""
        return self.roberta(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )

# Inference function that returns confidence scores and AUROC calculation results
def inference(model, test_loader, device):
    """
    Run inference on the test dataset
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run inference on (CPU/GPU)
        
    Returns:
        Tuple containing true labels, predicted probabilities, texts, and IDs
    """
    model.eval()
    all_labels = []
    all_probabilities = []
    all_texts = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference Progress"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            texts = batch['text']
            ids = batch['id']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Probability of being AI-generated (second class)
            probabilities = F.softmax(outputs.logits, dim=-1)[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_texts.extend(texts)
            all_ids.extend(ids)

    return all_labels, all_probabilities, all_texts, all_ids

# Find the optimal threshold that results in a 1% error rate for human texts
def find_threshold_for_human_error(labels, probabilities):
    """
    Find the optimal threshold that results in a 1% error rate for human texts
    
    Args:
        labels: True labels (0 for human, 1 for AI)
        probabilities: Predicted probabilities for being AI-generated
        
    Returns:
        Optimal threshold value
    """
    # Generate 100,000 thresholds between 0 and 1
    thresholds = np.linspace(0, 1, 100000)
    for threshold in thresholds:
        predictions = (np.array(probabilities) > threshold).astype(int)
        human_labels = np.array(labels) == 0
        human_errors = (predictions[human_labels] != 0).sum()
        human_error_rate = human_errors / human_labels.sum() if human_labels.sum() > 0 else 0
        
        if human_error_rate <= 0.01:
            return threshold
            
    return 1.0  # Default to maximum threshold if no suitable threshold found

# Save file of misclassified cases
def save_wrong_cases(texts, labels, predictions, probabilities, ids, output_file):
    """
    Save misclassified cases to a file
    
    Args:
        texts: List of texts
        labels: True labels
        predictions: Predicted labels
        probabilities: Predicted probabilities
        ids: Text IDs
        output_file: Path to save the JSON file
    """
    wrong_cases = [
        {
            "id": int(text_id),
            "text": text,
            "label": int(label),
            "prediction": int(prediction),
            "confidence": float(probability)
        }
        for text, label, prediction, probability, text_id in zip(texts, labels, predictions, probabilities, ids)
        if label != prediction
    ]
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(wrong_cases, f, indent=4)

# Calculate and save performance metrics
def calculate_and_save_metrics(labels, probabilities, threshold, output_file):
    """
    Calculate and save performance metrics to a file
    
    Args:
        labels: True labels
        probabilities: Predicted probabilities
        threshold: Threshold for classification
        output_file: Path to save the metrics
    """
    predictions = (np.array(probabilities) > threshold).astype(int)
    
    # Separate human and AI samples
    human_labels = np.array(labels) == 0
    ai_labels = np.array(labels) == 1
    
    # Calculate metrics
    metrics = {
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "overall_accuracy": float((predictions == labels).mean()),
        "human_accuracy": float((predictions[human_labels] == 0).mean()) if human_labels.sum() > 0 else 0,
        "ai_accuracy": float((predictions[ai_labels] == 1).mean()) if ai_labels.sum() > 0 else 0,
        "human_fpr": float((predictions[human_labels] != 0).mean()) if human_labels.sum() > 0 else 0,
        "ai_fnr": float((predictions[ai_labels] != 1).mean()) if ai_labels.sum() > 0 else 0,
        "confusion_matrix": {
            "tp": int(((predictions == 1) & (np.array(labels) == 1)).sum()),
            "tn": int(((predictions == 0) & (np.array(labels) == 0)).sum()),
            "fp": int(((predictions == 1) & (np.array(labels) == 0)).sum()),
            "fn": int(((predictions == 0) & (np.array(labels) == 1)).sum())
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

# Main function
def main():
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(Config.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(Config.tokenizer_path)
    model = CustomRoberta().to(Config.device)
    
    # Load model checkpoint
    model.load_state_dict(torch.load(Config.model_path, map_location=Config.device))
    print(f"Loaded model from {Config.model_path}")
    
    # Prepare datasets
    print("Preparing datasets...")
    original_dataset = InferenceTextDataset(Config.test_file_path, tokenizer, text_key="original_text")
    ai_dataset = InferenceTextDataset(Config.test_file_path, tokenizer, text_key="ai_generated_text")
    paraphrase_dataset = InferenceTextDataset(Config.test_file_path, tokenizer, text_key="paraphrased_text")
    
    # Combine datasets for inference
    combined_dataset = ConcatDataset([original_dataset, ai_dataset])
    combined_paraphrase_dataset = ConcatDataset([original_dataset, paraphrase_dataset])
    
    # Create data loaders
    combined_loader = DataLoader(combined_dataset, batch_size=Config.batch_size, shuffle=False)
    combined_paraphrase_loader = DataLoader(combined_paraphrase_dataset, batch_size=Config.batch_size, shuffle=False)
    
    # 1. Evaluate on original_text and ai_generated_text
    print("\nEvaluating on original_text and ai_generated_text...")
    combined_labels, combined_probabilities, combined_texts, combined_ids = inference(
        model, combined_loader, Config.device
    )
    
    # Calculate AUROC
    combined_auroc = roc_auc_score(combined_labels, combined_probabilities)
    
    # Find optimal threshold for 1% human error rate
    optimal_threshold = find_threshold_for_human_error(combined_labels, combined_probabilities)
    
    # Separate AI samples and calculate accuracy
    ai_start_idx = len(original_dataset)
    ai_probabilities = combined_probabilities[ai_start_idx:]
    ai_predictions = (np.array(ai_probabilities) > optimal_threshold).astype(int)
    ai_accuracy = (ai_predictions == 1).mean() * 100
    
    # Save metrics and misclassified cases
    calculate_and_save_metrics(
        combined_labels, 
        combined_probabilities, 
        optimal_threshold,
        os.path.join(output_dir, "metrics_ai.json")
    )
    
    # 2. Evaluate on original_text and paraphrased_text
    print("\nEvaluating on original_text and paraphrased_text...")
    paraphrase_labels, paraphrase_probabilities, paraphrase_texts, paraphrase_ids = inference(
        model, combined_paraphrase_loader, Config.device
    )
    
    # Calculate AUROC
    paraphrase_auroc = roc_auc_score(paraphrase_labels, paraphrase_probabilities)
    
    # Separate paraphrase samples and calculate accuracy
    paraphrase_start_idx = len(original_dataset)
    paraphrase_probabilities = paraphrase_probabilities[paraphrase_start_idx:]
    paraphrase_predictions = (np.array(paraphrase_probabilities) > optimal_threshold).astype(int)
    paraphrase_accuracy = (paraphrase_predictions == 1).mean() * 100
    
    # Save metrics and misclassified cases
    calculate_and_save_metrics(
        paraphrase_labels, 
        paraphrase_probabilities, 
        optimal_threshold,
        os.path.join(output_dir, "metrics_paraphrase.json")
    )

    
    # Print results
    print("\n" + "="*50)
    print(f"Optimal threshold for 1% human error rate: {optimal_threshold:.4f}")
    print("="*50)
    print(f"AI text accuracy at 1% FPR: {ai_accuracy:.4f}%")
    print(f"Paraphrased text accuracy at 1% FPR: {paraphrase_accuracy:.4f}%")
    print("="*50)
    print(f"AUROC for AI-Human classification: {combined_auroc:.4f}")
    print(f"AUROC for Paraphrased-Human classification: {paraphrase_auroc:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
