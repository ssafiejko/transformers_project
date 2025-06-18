import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model, HubertModel
from pathlib import Path, PurePosixPath
import warnings
import random
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, TensorDataset


warnings.filterwarnings("ignore")
TARGET_LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
SILENCE_LABEL = 'silence'
UNKNOWN_LABEL = 'unknown'

class CustomAudioDataset(Dataset):
    """
    Custom dataset for loading audio files and their corresponding labels.
    Args:
        file_paths (list): List of audio file paths.
        sample_rate (int): Sample rate for audio files.
        batch_size (int): Batch size for DataLoader.
        classes (list): List of class names.
        
    """
    def __init__(self, file_paths, sample_rate, batch_size):
        self.label_to_idx = {label: idx for idx, label in enumerate(TARGET_LABELS + [SILENCE_LABEL, UNKNOWN_LABEL])}
        self.file_paths = file_paths
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.max_length = sample_rate

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = Path(self.file_paths[idx])
        label_str = path.parent.name
        
        # Map to 'unknown' if not in target labels or silence
        if label_str not in self.label_to_idx:
            label_str = UNKNOWN_LABEL
        
        label = self.label_to_idx[label_str]

        try:
            waveform, sr = torchaudio.load(path)
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            
            # Ensure waveform is mono
            if waveform.dim() > 1 and waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                
            if waveform.size(1) > self.max_length:
                waveform = waveform[:, :self.max_length]
            else:
                padding = self.max_length - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, padding))
                
            return waveform.squeeze(0), label
        except Exception as e:
            print(f"Error loading {path}: {str(e)}")
            return torch.zeros(self.max_length), 0

def load_data_lists(audio_dir, val_txt, test_txt):
    """
    Load the dataset file lists for training, validation, and testing.
    
    Args:
        audio_dir (str): Root directory containing the audio files.
        val_txt (str): Path to validation list file.
        test_txt (str): Path to testing list file.
    Returns:
        tuple: (train_files, val_files, test_files) where each is a list of file paths
               and a second tuple with label counts for debugging
    """
    # Load validation and test files from txt files
    with open(val_txt) as f:
        val_files = set(line.strip() for line in f)
    with open(test_txt) as f:
        test_files = set(line.strip() for line in f)

    # Initialize lists
    train_files = []
    val_files_full = []
    test_files_full = []
    
    # For debugging label counts
    train_label_counts = {label: 0 for label in TARGET_LABELS + [SILENCE_LABEL, UNKNOWN_LABEL]}
    val_label_counts = {label: 0 for label in TARGET_LABELS + [SILENCE_LABEL, UNKNOWN_LABEL]}
    test_label_counts = {label: 0 for label in TARGET_LABELS + [SILENCE_LABEL, UNKNOWN_LABEL]}

    # Process target labels
    for label in TARGET_LABELS:
        label_dir = os.path.join(audio_dir, label)
        if os.path.exists(label_dir):
            for wav_file in Path(label_dir).rglob('*.wav'):
                rel_path = PurePosixPath(label) / wav_file.name
                rel_path = str(rel_path) 
                full_path = str(wav_file)
                
                if rel_path in val_files:
                    val_files_full.append(full_path)
                    val_label_counts[label] += 1
                elif rel_path in test_files:
                    test_files_full.append(full_path)
                    test_label_counts[label] += 1
                else:
                    train_files.append(full_path)
                    train_label_counts[label] += 1

    # Process silence (from silence folder)
    silence_dir = os.path.join(audio_dir, SILENCE_LABEL)
    if os.path.exists(silence_dir):
        for wav_file in Path(silence_dir).rglob('*.wav'):
            rel_path = PurePosixPath(SILENCE_LABEL) / wav_file.name
            rel_path = str(rel_path)
            full_path = str(wav_file)
            
            if rel_path in val_files:
                val_files_full.append(full_path)
                val_label_counts[SILENCE_LABEL] += 1
            elif rel_path in test_files:
                test_files_full.append(full_path)
                test_label_counts[SILENCE_LABEL] += 1
            else:
                train_files.append(full_path)
                train_label_counts[SILENCE_LABEL] += 1

    # Process unknown (from other folders)
    for item in os.listdir(audio_dir):
        item_path = os.path.join(audio_dir, item)
        if os.path.isdir(item_path) and item not in TARGET_LABELS and item != SILENCE_LABEL:
            for wav_file in Path(item_path).rglob('*.wav'):
                rel_path = PurePosixPath(item) / wav_file.name
                rel_path = str(rel_path)
                full_path = str(wav_file)
                
                if rel_path in val_files:
                    val_files_full.append(full_path)
                    val_label_counts[UNKNOWN_LABEL] += 1
                elif rel_path in test_files:
                    test_files_full.append(full_path)
                    test_label_counts[UNKNOWN_LABEL] += 1
                else:
                    train_files.append(full_path)
                    train_label_counts[UNKNOWN_LABEL] += 1

    return train_files, val_files_full, test_files_full, train_label_counts, val_label_counts, test_label_counts

def load_data(audio_dir, val_txt, test_txt, batch_size, sample_rate):
    """
    Load the dataset and create DataLoader instances for training, validation, and testing.
    """
    train_files, val_files, test_files, train_counts, val_counts, test_counts = load_data_lists(
        audio_dir, val_txt, test_txt)

    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    print(f"Test samples: {len(test_files)}")
    
    print("Label distribution in training set:")
    for label, count in train_counts.items():
        print(f"{label}: {count}")
        
    train_dataset = CustomAudioDataset(train_files, sample_rate, batch_size)
    val_dataset = CustomAudioDataset(val_files, sample_rate, batch_size)
    test_dataset = CustomAudioDataset(test_files, sample_rate, batch_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

    return train_loader, val_loader, test_loader

def load_feature_tensor_data(audio_dir, val_txt, test_txt, batch_size, sample_rate, model_name="facebook/wav2vec2-base"):
    print('Loading data from drive...')
    train_loader, val_loader, test_loader = load_data(audio_dir, val_txt, test_txt, 1, sample_rate)
    print('Transforming train to tensor...')
    train_loader = waveform_to_tensor_loader(train_loader, batch_size, shuffle=True, model_name=model_name)
    print('Transforming val to tensor...')
    val_loader = waveform_to_tensor_loader(val_loader, batch_size, shuffle=False, model_name=model_name)
    print('Transforming test to tensor...')
    test_loader = waveform_to_tensor_loader(test_loader, batch_size, shuffle=False, model_name=model_name)
    print('Complete!')
    return train_loader, val_loader, test_loader

def load_feature_tensor_data_full(audio_dir, val_txt, test_txt, batch_size, sample_rate, model_name="facebook/wav2vec2-base"):
    print('Loading data from drive...')
    train_loader, val_loader, test_loader = load_data(audio_dir, val_txt, test_txt, 1, sample_rate)
    print('Transforming train to tensor...')
    train_loader = waveform_to_tensor_loader_full_output(train_loader, batch_size, shuffle=True, model_name=model_name)
    print('Transforming val to tensor...')
    val_loader = waveform_to_tensor_loader_full_output(val_loader, batch_size, shuffle=False, model_name=model_name)
    print('Transforming test to tensor...')
    test_loader = waveform_to_tensor_loader_full_output(test_loader, batch_size, shuffle=False, model_name=model_name)
    print('Complete!')
    return train_loader, val_loader, test_loader  

def waveform_to_tensor_loader(waveform_loader, batch_size=8, shuffle=False, model_name="facebook/wav2vec2-base"):
    device = torch.device("mps" if torch.backends.mps.is_available()
                                   else "cuda" if torch.cuda.is_available()
                                   else "cpu")

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    if "hubert" in model_name.lower():
        model = HubertModel.from_pretrained(model_name).to(device)
    else:
        model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for waveforms, labels in tqdm(waveform_loader):
            waveforms = waveforms.to(device)

            # If mono channel: [B, 1, T] → [B, T]
            waveforms = waveforms.squeeze()
            if waveforms.ndim == 1:
                waveforms = waveforms.unsqueeze(0)
            inputs = processor(waveforms, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(device)

            input_values = input_values.squeeze()
            if input_values.ndim == 1:
                input_values = input_values.unsqueeze(0)
            outputs = model(input_values)
            features = outputs.last_hidden_state.mean(dim=1)  # [B, 768]

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    features_tensor = torch.cat(all_features)
    labels_tensor = torch.cat(all_labels)

    dataset = TensorDataset(features_tensor, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

def waveform_to_tensor_loader_full_output(waveform_loader, batch_size=8, shuffle=False, model_name="facebook/wav2vec2-base"):
    device = torch.device("mps" if torch.backends.mps.is_available()
                                   else "cuda" if torch.cuda.is_available()
                                   else "cpu")

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    if "hubert" in model_name.lower():
        model = HubertModel.from_pretrained(model_name).to(device)
    else:
        model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for waveforms, labels in tqdm(waveform_loader):
            waveforms = waveforms.to(device)

            # If mono channel: [B, 1, T] → [B, T]
            waveforms = waveforms.squeeze()
            if waveforms.ndim == 1:
                waveforms = waveforms.unsqueeze(0)
            inputs = processor(waveforms, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(device)

            input_values = input_values.squeeze()
            if input_values.ndim == 1:
                input_values = input_values.unsqueeze(0)
            outputs = model(input_values)
            features = outputs.last_hidden_state  # [B, T, 768]


            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    features_tensor = torch.cat(all_features)
    labels_tensor = torch.cat(all_labels)

    dataset = TensorDataset(features_tensor, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_tensor_loader(loader, path):
    features = []
    labels = []
    for x, y in loader:
        features.append(x)
        labels.append(y)
    features = torch.cat(features)
    labels = torch.cat(labels)
    torch.save({'features': features, 'labels': labels}, path)


def load_tensor_dataset(file_path, batch_size=32, shuffle=False):
    data = torch.load(file_path)
    dataset = TensorDataset(data['features'], data['labels'])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)