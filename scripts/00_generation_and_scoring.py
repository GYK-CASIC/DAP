import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, RobertaTokenizer, RobertaForSequenceClassification
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import numpy as np
from lexical_diversity import lex_div as ld
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import transformers
import sys


config = {
    "DPO_PATH": "../dataset/tweepfake/dpo.json",                  # Directory for saving DPO datasets
    "MODEL_ID": "../model/generator_model_path",                  # Identifier of the base model
    "TOKENIZER_PATH": "../model/detector_model_path",             # Path to the detector’s tokenizer
    "MAX_NEW_TOKENS": 150,                                        # Maximum tokens to generate per sequence
    "TEMPERATURE": 1.3,                                           # Sampling temperature
    "TOP_P": 0.95,                                                # Nucleus-sampling probability (top-p)
    "TOP_K": 100,                                                 # Top-k sampling size
    "NUM_RETURN_SEQUENCES": 100,                                  # Number of sequences to generate per prompt
    "INPUT_FILE": "../dataset/tweepfake/train_rl.json",           # Path to train_rl.json
    "OUTPUT_FILE_TEMPLATE": "../dataset/tweepfake/inter.json",    # Template path for saving intermediate data
    "BATCH_SIZE": 32,                                             # Generation batch size
    "MAX_LENGTH": 512,                                            # Maximum total sequence length
}


class RLProcessor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lora_model = None
        self.base_model = None
        self.tokenizer = None
        self.initialized = False

    def compare_params(self, params_before, params_after): 
        """Compare parameters between two models and count how many parameters have changed."""
        modified_count = 0
        total_params = len(params_before)

        for (name_before, param_before), (name_after, param_after) in zip(params_before, params_after):
            if not torch.equal(param_before, param_after):
                print(f"Parameter {name_before} has been modified.")
                modified_count += 1

        print(f"Total parameters: {total_params}")
        print(f"Modified parameters: {modified_count}")

        if modified_count > 0:
            return True, modified_count
        else:
            return False, modified_count

    def initialize_model(self):
        """Initialize the base model and load LoRA weights if provided."""
        lora_path = self.config.get('LORA_PATH', None)

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config['MODEL_ID'],
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.config['MODEL_ID'])

        if lora_path:
            peft_config = PeftConfig.from_pretrained(lora_path)
            self.lora_model = PeftModel.from_pretrained(
                self.base_model,
                lora_path,
                config=peft_config
            )
            print(f"Loaded Lora model from {lora_path}")

        else:
            self.lora_model = self.base_model

        self.initialized = True

    def _ensure_path(self, path):
        """Ensure the directory for the given path exists; create if not."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def paraphrase_texts(self, input_path, output_path):
        """Generate paraphrased versions of input texts using the initialized model."""
        self.initialize_model()

        generator = pipeline(
            "text-generation",
            model=self.lora_model,
            model_kwargs={"torch_dtype": torch.bfloat16},
            tokenizer=self.tokenizer,
            device_map="auto",
            max_new_tokens=self.config['MAX_NEW_TOKENS'],
            temperature=self.config['TEMPERATURE'],
            top_p=self.config['TOP_P'],
            top_k=self.config['TOP_K'],
            do_sample=True,
            num_return_sequences=self.config['NUM_RETURN_SEQUENCES']
        )

        with open(input_path, 'r') as f:
            data = json.load(f)

        # Index adjustment (since original data starts at ID 401)
        start = self.config['DATA_START'] - 401
        end = self.config['DATA_END'] - 400
        data = data[start:end]

        # Define termination tokens
        terminators = [
            generator.tokenizer.eos_token_id,
            generator.tokenizer.convert_tokens_to_ids("<|eot_id|>") if "<|eot_id|>" in generator.tokenizer.get_vocab() else None
        ]
        terminators = [t for t in terminators if t is not None]  # Remove None values
        processed_data = []
        for entry in tqdm(data, desc="Paraphrasing"):
            messages = self._create_prompt(entry['ai_generated_text'])

            prompt = generator.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            outputs = generator(
                prompt,
                max_new_tokens=self.config['MAX_NEW_TOKENS'],
                eos_token_id=terminators,
                do_sample=True,
                temperature=self.config['TEMPERATURE'],
                top_p=self.config['TOP_P'],
                top_k=self.config['TOP_K'],
                num_return_sequences=self.config['NUM_RETURN_SEQUENCES']
            )

            result = {
                "id": entry['id'],
                "original_text": entry['original_text'],
                "ai_generated_text": entry['ai_generated_text']
            }

            for i, output in enumerate(outputs, 1):
                generated = output["generated_text"][len(prompt):].strip()
                result[f"paraphrased_{i}th_ai_text"] = generated

            processed_data.append(result)

        self._ensure_path(output_path)
        with open(output_path, 'w') as f:
            json.dump(processed_data, f, indent=4)
    

    def _create_prompt(self, text):
        """Create a prompt for paraphrasing AI-generated text to be more human-like."""
        return [
            {"role": "user", "content":
             f"Please rewrite the AI-generated text to make it sound more human-like:{text}\n1. Paraphrase: Use synonyms, adjust vocabulary, and restructure sentences for improved flow and logical coherence.\n2. Segment: Split longer sentences into shorter ones, and rearrange sentence order where appropriate.\n3. Diversify: Maintain the original meaning and text length while enhancing diversity.\n4. Stealth: Ensure the result is undetectable by AI content detectors.\n5. Brevity: Only return the paraphrased text without additional comments or redundancy."}
        ]

    def filter_data(self, input_path, output_path):
        """Filter out entries with newline characters in their values."""
        with open(input_path, 'r') as f:
            data = json.load(f)

        for entry in data:
            # Remove keys whose values contain newline characters
            keys_to_remove = [k for k, v in entry.items() if isinstance(v, str) and '\n' in v]
            for k in keys_to_remove:
                del entry[k]

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)

    def calculate_length_features(self, input_path):
        """Calculate and print length-related statistics (max, min, average, variance) for paraphrased texts."""
        with open(input_path, 'r') as f:
            data = json.load(f)

        word_counts = {}
        for i in range(1, 101):
            key = f"paraphrased_{i}th_ai_text"
            word_counts[key] = []

        for entry in data:
            for i in range(1, 101):
                key = f"paraphrased_{i}th_ai_text"
                if key in entry:
                    word_counts[key].append(len(entry[key].split()))

        total = 0
        for i in range(1, 101):
            key = f"paraphrased_{i}th_ai_text"
            if word_counts[key]:
                max_words = np.max(word_counts[key])
                min_words = np.min(word_counts[key])
                avg_words = np.mean(word_counts[key])
                variance_words = np.var(word_counts[key])

                print(f"\n{key} Word Counts:")
                print(f"Max word count: {max_words}")
                print(f"Min word count: {min_words}")
                print(f"Average word count: {avg_words}")
                print(f"Variance: {variance_words}")

                total = total + avg_words

        total = total / 100
        print(f"All Average word count: {total}")

    def pre_inference(self, input_path, ai_correct_path, ai_incorrect_path, human_correct_path, human_incorrect_path):
        """Perform inference using a detector model to classify texts as human or AI-generated, and save results."""
        with open(input_path, 'r') as f:
            data = json.load(f)

        class InferenceTextDataset(Dataset):
            """Dataset class for loading texts and labels for inference."""
            def __init__(self, file_path, tokenizer, max_length=512):
                with open(file_path, 'r') as file:
                    self.data = json.load(file)

                self.texts = []
                self.labels = []
                self.ids = []
                self.ths = []
                self.sentences = []

                for entry in self.data:
                    # Original text is human-written (label 0)
                    self.texts.append(entry['original_text'])
                    self.labels.append(0)
                    self.ids.append(entry['id'])
                    self.ths.append('original')
                    self.sentences.append(entry['original_text'])

                    # Paraphrased texts are AI-generated (label 1)
                    for i in range(1, 101):
                        paraphrased_key = f'paraphrased_{i}th_ai_text'
                        if paraphrased_key in entry:
                            self.texts.append(entry[paraphrased_key])
                            self.labels.append(1)
                            self.ids.append(entry['id'])
                            self.ths.append(f'{i}th')
                            self.sentences.append(entry[paraphrased_key])

                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = self.texts[idx]
                label = self.labels[idx]
                encoding = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                          return_tensors='pt')
                return {'input_ids': encoding['input_ids'].squeeze(0),
                        'attention_mask': encoding['attention_mask'].squeeze(0),
                        'labels': torch.tensor(label),
                        'id': self.ids[idx],
                        'th': self.ths[idx],
                        'sentence': self.sentences[idx]}

        tokenizer = RobertaTokenizer.from_pretrained(self.config['TOKENIZER_PATH'])
        test_dataset = InferenceTextDataset(input_path, tokenizer, self.config['MAX_LENGTH'])

        class CustomRoberta(nn.Module):
            """Custom wrapper for Roberta model for sequence classification."""
            def __init__(self):
                super().__init__()
                self.roberta = RobertaForSequenceClassification.from_pretrained(self.config['TOKENIZER_PATH'], num_labels=2)

            def forward(self, input_ids, attention_mask, labels=None):
                output = self.roberta(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                return output

        def collect_probabilities(model, test_loader, device):
            """Collect prediction probabilities and true labels from the model."""
            model.eval()
            all_probabilities = []
            all_labels = []
            all_data = []

            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Collecting Probabilities"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    probabilities = F.softmax(outputs.logits, dim=-1)
                    ai_probabilities = probabilities[:, 1].cpu().numpy()  # Probability of being AI-generated

                    all_probabilities.extend(ai_probabilities)
                    all_labels.extend(labels.cpu().numpy())
                    all_data.extend(zip(batch['id'], batch['th'], batch['sentence']))

            return all_probabilities, all_labels, all_data

        def get_human_accuracy_for_threshold(probabilities, labels, threshold):
            """Calculate accuracy for classifying human-written texts at a given threshold."""
            human_total = 0
            human_correct = 0

            for prob, label in zip(probabilities, labels):
                if label == 0:  # Human-written
                    human_total += 1
                    predicted_label = 1 if prob > threshold else 0
                    if predicted_label == 0:  # Correctly classified as human
                        human_correct += 1

            human_accuracy = (human_correct / human_total * 100) if human_total > 0 else 0
            return human_accuracy


        def find_optimal_threshold(probabilities, labels):
            """Find the threshold that achieves ~99% accuracy on human-written texts."""
            best_threshold = 0.0
            best_human_accuracy = 0.0

            # Search thresholds from 0.0 to 1.0 in 0.001 increments
            for threshold in [i * 0.001 for i in range(1001)]:
                human_accuracy = get_human_accuracy_for_threshold(probabilities, labels, threshold)
                if abs(human_accuracy - 99) < abs(best_human_accuracy - 99):
                    best_human_accuracy = human_accuracy
                    best_threshold = threshold

            return best_threshold


        def classify_and_save_predictions(model, test_loader, device, optimal_threshold, all_probabilities, all_labels,
                                          all_data):
            """Classify texts using the optimal threshold and save results into correct/incorrect categories."""
            model.eval()

            ai_correct_predictions = []
            ai_incorrect_predictions = []
            human_correct_predictions = []
            human_incorrect_predictions = []

            human_total = 0
            human_correct = 0
            ai_total = 0
            ai_correct = 0

            for i, (prob, label, data) in enumerate(zip(all_probabilities, all_labels, all_data)):
                id, th, sentence = data
                predicted_label = 1 if prob > optimal_threshold else 0  # 1=AI, 0=human

                if label == 0:  # Human-written
                    human_total += 1
                    if predicted_label == label:  # Correct classification
                        human_correct += 1
                        human_correct_predictions.append({
                            'id': int(id),
                            'original_label': int(label),
                            'predicted_label': int(predicted_label),
                            'ai_probability': float(prob),
                            'th': th,
                            'sentence': sentence
                        })
                    else:  # Incorrect classification
                        human_incorrect_predictions.append({
                            'id': int(id),
                            'original_label': int(label),
                            'predicted_label': int(predicted_label),
                            'ai_probability': float(prob),
                            'th': th,
                            'sentence': sentence
                        })
                else:  # AI-generated
                    ai_total += 1
                    if predicted_label == label:  # Correct classification
                        ai_correct += 1
                        ai_correct_predictions.append({
                            'id': int(id),
                            'original_label': int(label),
                            'predicted_label': int(predicted_label),
                            'ai_probability': float(prob),
                            'th': th,
                            'sentence': sentence
                        })
                    else:  # Incorrect classification
                        ai_incorrect_predictions.append({
                            'id': int(id),
                            'original_label': int(label),
                            'predicted_label': int(predicted_label),
                            'ai_probability': float(prob),
                            'th': th,
                            'sentence': sentence
                        })

            human_accuracy = (human_correct / human_total * 100) if human_total > 0 else 0
            ai_accuracy = (ai_correct / ai_total * 100) if ai_total > 0 else 0

            def save_predictions(predictions, file_path):
                """Helper to save predictions to JSON."""
                with open(file_path, 'w') as f:
                    json.dump(predictions, f, indent=4)

            save_predictions(ai_correct_predictions, ai_correct_path)
            save_predictions(ai_incorrect_predictions, ai_incorrect_path)
            save_predictions(human_correct_predictions, human_correct_path)
            save_predictions(human_incorrect_predictions, human_incorrect_path)

            return human_accuracy, ai_accuracy

        test_loader = DataLoader(test_dataset, batch_size=self.config['BATCH_SIZE'], shuffle=False)

        model = CustomRoberta().to(self.device)
        if self.config['INFERENCE_MODEL_PATH']:
            model.load_state_dict(torch.load(self.config['INFERENCE_MODEL_PATH'], map_location=self.device))

        all_probabilities, all_labels, all_data = collect_probabilities(model, test_loader, self.device)

        optimal_threshold = find_optimal_threshold(all_probabilities, all_labels)
        print(f"Optimal Threshold: {optimal_threshold:.4f}")

        human_accuracy, ai_accuracy = classify_and_save_predictions(model, test_loader, self.device, optimal_threshold,
                                                                    all_probabilities, all_labels, all_data)

        print(f"Human Text Accuracy: {human_accuracy:.2f}%")
        print(f"AI Text Accuracy: {ai_accuracy:.2f}%")
        print("=" * 50)
        print(f'The checkpoint is {self.config["INFERENCE_MODEL_PATH"]}')

    def calculate_diversity(self, correct_file_path, incorrect_file_path, output_dir):
        """Calculate lexical diversity metrics and generate comparison plots."""
        os.makedirs(output_dir, exist_ok=True)

        with open(correct_file_path, 'r') as f:
            correct_data = json.load(f)

        with open(incorrect_file_path, 'r') as f:
            incorrect_data = json.load(f)

        metrics = {
            'Simple TTR': {'correct': [], 'incorrect': []}  # Type-Token Ratio
        }

        def calculate_metrics(data, metrics_dict, label, field):
            """Calculate diversity metrics for texts and store results."""
            for entry in data:
                sentence = entry[field]
                flt = ld.flemmatize(sentence)  # Lemmatize for better TTR calculation

                metrics_dict['Simple TTR'][label].append(ld.ttr(flt))

        calculate_metrics(correct_data, metrics, 'correct', 'sentence')
        calculate_metrics(incorrect_data, metrics, 'incorrect', 'sentence')

        for metric_name, data_dict in metrics.items():
            plt.figure(figsize=(12, 6))

            # Create histograms with 20 bins between 0 and 1
            counts_correct, bins_correct = np.histogram(data_dict['correct'], bins=np.linspace(0, 1, 21))
            counts_incorrect, bins_incorrect = np.histogram(data_dict['incorrect'], bins=np.linspace(0, 1, 21))

            total_correct = len(data_dict['correct'])
            total_incorrect = len(data_dict['incorrect'])

            # Normalize by total number of texts
            normalized_correct = counts_correct / total_correct if total_correct > 0 else counts_correct
            normalized_incorrect = counts_incorrect / total_incorrect if total_incorrect > 0 else counts_incorrect

            plt.plot(bins_correct[:-1], normalized_correct, marker='o', label='Correct AI Text', color='blue')
            plt.plot(bins_incorrect[:-1], normalized_incorrect, marker='o', label='Incorrect AI Text', color='orange')

            plt.title(f'Lexical Diversity Measure: {metric_name}')
            plt.xlabel('Diversity Measure Range')
            plt.ylabel('Normalized Number of Texts')
            plt.legend()
            plt.grid()
            plt.tight_layout()

            output_file_path = os.path.join(output_dir, f'{metric_name}.pdf')
            plt.savefig(output_file_path)
            plt.close()

        print("All graphs have been saved to the specified path.")

    def calculate_scores(self, incorrect_input_path, incorrect_output_path, correct_input_path, correct_output_path):
        """Calculate deception, length, diversity, and total scores for texts."""
        # Load AI-generated data for reference
        with open(self.config['OUTPUT_FILE_TEMPLATE'].format(round=self.config['ROUND'], DATA_START=self.config['DATA_START'], DATA_END=self.config['DATA_END']), 'r') as f:
            ai_generated_data = json.load(f)

        ai_generated_dict = {entry['id']: entry for entry in ai_generated_data}


        def calculate_ttr(sentence):
            """Calculate Type-Token Ratio (TTR) for a sentence."""
            flt = ld.flemmatize(sentence)
            return ld.ttr(flt)


        def calculate_length_score(generated_len, sentence_len):
            """Calculate score based on length similarity to original AI-generated text."""
            if sentence_len == 0:
                return 0
            deviation = abs(generated_len - sentence_len) / sentence_len
            length_score = max(0, 1 - deviation)  # Higher = more similar length
            return length_score

        def process_data(input_path, output_path):
            """Process input data to compute and add scores."""
            with open(input_path, 'r') as f:
                data = json.load(f)

            for entry in tqdm(data, desc="Scoring data"):
                id = entry['id']
                sentence = entry['sentence']
                ai_probability = entry['ai_probability']

                # Score 1: Deception (lower AI probability = better deception)
                deception_score = 1 - ai_probability

                # Score 2: Length similarity to original AI-generated text
                ai_generated_text = ai_generated_dict[id]['ai_generated_text']
                length_score = calculate_length_score(len(ai_generated_text), len(sentence))

                # Score 3: Diversity (TTR difference from original AI text)
                sentence_ttr = calculate_ttr(sentence)
                ai_generated_ttr = calculate_ttr(ai_generated_text)
                diversity_score = sentence_ttr - ai_generated_ttr  # Higher = more diverse

                # Total score with weighted components
                total_score = 0.4 * deception_score + 0.2 * length_score + 0.4 * diversity_score

                # Add scores to entry
                entry['deception_score'] = 0.4 * deception_score
                entry['length_score'] = 0.2 * length_score
                entry['diversity_score'] = 0.4 * diversity_score
                entry['total_score'] = total_score

            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Scores saved to {output_path}")

        process_data(incorrect_input_path, incorrect_output_path)
        process_data(correct_input_path, correct_output_path)

    def merge_scores(self, incorrect_score_path, correct_score_path, merged_score_path):
        """Merge scores from correct and incorrect classifications, keeping top/bottom performers per ID."""
        with open(incorrect_score_path, 'r') as f:
            incorrect_data = json.load(f)

        with open(correct_score_path, 'r') as f:
            correct_data = json.load(f)

        result_dict = {}

        # Collect all unique IDs
        all_ids = set([entry['id'] for entry in incorrect_data + correct_data])
        for id_num in tqdm(all_ids, desc="Processing IDs"):
            # Get all incorrect entries for this ID (highest total score is best)
            incorrect_entries = [entry for entry in incorrect_data if entry['id'] == id_num]
            # Get all correct entries for this ID (lowest total score is worst)
            correct_entries = [entry for entry in correct_data if entry['id'] == id_num]

            if incorrect_entries:
                max_incorrect_entry = max(incorrect_entries, key=lambda x: x['total_score'])
            else:
                max_incorrect_entry = None
                print(f"ID {id_num} has no incorrectly classified data.")

            if correct_entries:
                min_correct_entry = min(correct_entries, key=lambda x: x['total_score'])
            else:
                min_correct_entry = None
                print(f"ID {id_num} has no correctly classified data.")

            result_dict[id_num] = {
                'max_incorrect': max_incorrect_entry,
                'min_correct': min_correct_entry
            }

        with open(merged_score_path, 'w') as f:
            json.dump(result_dict, f, indent=4)

        print(f"Merged scores saved to {merged_score_path}")

    def prepare_dpo_data(self, merge_file_path, filter_file_path, output_path):
        """Prepare dataset for Direct Preference Optimization (DPO) with chosen/rejected pairs."""
        # Load merged score data
        with open(merge_file_path, 'r') as f:
            merge_data = json.load(f)

        # Load filtered data to get original AI-generated texts
        with open(filter_file_path, 'r') as f:
            filter_data = json.load(f)

        # Map ID to original AI-generated text
        filter_dict = {entry['id']: entry.get('ai_generated_text', None) for entry in filter_data if
                       'ai_generated_text' in entry}

        result_dataset = []

        for id_num, entries in merge_data.items():
            id_num = int(id_num)  # Convert ID to integer

            # Get best (incorrectly classified) and worst (correctly classified) paraphrases
            max_incorrect_sentence = entries['max_incorrect']['sentence'] if entries['max_incorrect'] else None
            min_correct_sentence = entries['min_correct']['sentence'] if entries['min_correct'] else None

            # Get original AI-generated text as input
            input_text = filter_dict.get(id_num, None)

            # Only include valid entries with all required components
            if max_incorrect_sentence and min_correct_sentence and input_text:
                entry = {
                    "instruction": "Please rewrite the AI-generated text to make it sound more human-like:{text}\n1. Paraphrase: Use synonyms, adjust vocabulary, and restructure sentences for improved flow and logical coherence.\n2. Segment: Split longer sentences into shorter ones, and rearrange sentence order where appropriate.\n3. Diversify: Maintain the original meaning and text length while enhancing diversity.\n4. Stealth: Ensure the result is undetectable by AI content detectors.\n5. Brevity: Only return the paraphrased text without additional comments or redundancy.\n",
                    "input": input_text,
                    "chosen": max_incorrect_sentence,  # Better paraphrase (undetected)
                    "rejected": min_correct_sentence   # Worse paraphrase (detected)
                }
                result_dataset.append(entry)

        # Save to output path
        with open(output_path, 'w') as f:
            json.dump(result_dataset, f, indent=4)
        
        # Also save to configured DPO path
        dpo_path = self.config['DPO_PATH'].format(round=self.config['ROUND'], DATA_START=self.config['DATA_START'], DATA_END=self.config['DATA_END'])
        with open(dpo_path, 'w') as f:
            json.dump(result_dataset, f, indent=4)

        print(f"Dataset generated and saved to {output_path}")

    def prepare_finetune_data(self, merge_file_path, input_file, output_path):
        """Prepare dataset for fine-tuning with original and AI-generated texts."""
        # Load merged score data
        with open(merge_file_path, 'r') as f:
            merge_data = json.load(f)

        # Load original training data
        with open(input_file, 'r') as f:
            train_data = json.load(f)

        # Map ID to (original text, AI-generated text)
        train_dict = {entry['id']: (entry['original_text'], entry['ai_generated_text']) for entry in train_data if
                      'id' in entry}

        result_dataset = []

        # Extract relevant data for fine-tuning
        for id_num, entries in merge_data.items():
            id_num = int(id_num)  # Convert ID to integer

            # Get original and AI-generated texts from training data
            original_ai_texts = train_dict.get(id_num, None)

            if original_ai_texts:
                original_text, ai_generated_text = original_ai_texts
                entry = {
                    "id": id_num,
                    "original_text": original_text,
                    "ai_generated_text": ai_generated_text
                }
                result_dataset.append(entry)

        # Save dataset
        with open(output_path, 'w') as f:
            json.dump(result_dataset, f, indent=4)

        print(f"Dataset generated and saved to {output_path}")

    def prepare_rl_data(self, merge_file_path, input_file, output_path):
        """Prepare dataset for reinforcement learning with paraphrased texts."""
        # Load merged score data
        with open(merge_file_path, 'r') as f:
            merge_data = json.load(f)

        # Load original training data
        with open(input_file, 'r') as f:
            train_data = json.load(f)

        # Map ID to original human-written text
        train_dict = {entry['id']: entry['original_text'] for entry in train_data if
                      'id' in entry}

        result_dataset = []

        # Extract best paraphrases and original texts
        for id_num, entries in merge_data.items():
            id_num = int(id_num)  # Convert ID to integer

            # Get best paraphrase (incorrectly classified)
            paraphrased_text = entries['max_incorrect']['sentence'] if entries['max_incorrect'] else None

            # Get original human-written text
            original_text = train_dict.get(id_num, None)

            # Include valid entries
            if paraphrased_text and original_text:
                entry = {
                    "id": id_num,
                    "original_text": original_text,
                    "paraphrased_text": paraphrased_text
                }
                result_dataset.append(entry)

        # Save dataset
        with open(output_path, 'w') as f:
            json.dump(result_dataset, f, indent=4)

        print(f"Dataset generated and saved to {output_path}")

    def run_full_pipeline(self):
        """Execute the complete pipeline from paraphrasing to dataset preparation."""
        # Get configuration parameters
        round_num = self.config['ROUND']
        input_path = self.config['INPUT_FILE']
        DATA_START = self.config['DATA_START']
        DATA_END = self.config['DATA_END']
        output_path = self.config['OUTPUT_FILE_TEMPLATE'].format(round=round_num,DATA_START=DATA_START, DATA_END=DATA_END)
        output_dir = self.config['DIVERSITY_OUTPUT_DIR'].format(round=round_num)

        # Define paths for intermediate outputs
        filtered_output_path = output_path.replace('_paraphrase_initial', '_filter')
        ai_correct_path = output_path.replace('_paraphrase_initial', '_ai_correct')
        ai_incorrect_path = output_path.replace('_paraphrase_initial', '_ai_incorrect')
        human_correct_path = output_path.replace('_paraphrase_initial', '_human_correct')
        human_incorrect_path = output_path.replace('_paraphrase_initial', '_human_incorrect')
        incorrect_score_path = output_path.replace('_paraphrase_initial', '_ai_incorrect_score')
        correct_score_path = output_path.replace('_paraphrase_initial', '_ai_correct_score')
        merged_score_path = output_path.replace('_paraphrase_initial', '_ai_score_merge')
        dpo_output_path = output_path.replace('_paraphrase_initial', '_dpo')
        finetune_output_path = output_path.replace('_paraphrase_initial', '_finetune')
        rl_output_path = output_path.replace('_paraphrase_initial', '_rl')

        print("Processing single task")

        # 1. Generate paraphrases
        self.paraphrase_texts(input_path, output_path)

        # 2. Filter out invalid entries
        self.filter_data(output_path, filtered_output_path)

        # 3. Analyze length features of paraphrases
        self.calculate_length_features(filtered_output_path)

        # 4. Run detector inference and classify texts
        self.pre_inference(filtered_output_path, ai_correct_path, ai_incorrect_path, human_correct_path, human_incorrect_path)

        # 5. Calculate and visualize lexical diversity
        self.calculate_diversity(ai_correct_path, ai_incorrect_path, output_dir)

        # 6. Compute scores for paraphrases
        self.calculate_scores(ai_incorrect_path, incorrect_score_path, ai_correct_path, correct_score_path)

        # 7. Merge scores to identify best/worst paraphrases
        self.merge_scores(incorrect_score_path, correct_score_path, merged_score_path)

        # 8. Prepare DPO training dataset
        self.prepare_dpo_data(merged_score_path, filtered_output_path, dpo_output_path)

        # 9. Prepare fine-tuning dataset
        self.prepare_finetune_data(merged_score_path, input_path, finetune_output_path)

        # 10. Prepare reinforcement learning dataset
        self.prepare_rl_data(merged_score_path, input_path, rl_output_path)

if __name__ == "__main__":
    processor = RLProcessor(config)
    processor.run_full_pipeline()
