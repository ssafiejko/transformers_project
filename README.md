# Audio Classification with Transformers
This repository contains state-of-the-art transformer-based models for audio classification tasks, with support for both custom architectures and pretrained models from HuggingFace.

##  Features

- **Multiple Model Architectures**:
  - Custom Audio Transformer (ViT-style)
  - Pretrained Wav2Vec2/HuBERT with classification heads
  - Traditional RNN/LSTM baselines
  - Simple MLP for feature classification

- **Complete Training Pipeline**:
  - End-to-end training utilities
  - Comprehensive metrics tracking (accuracy, F1, precision, recall)
  - Model saving and evaluation

- **Audio Processing**:
  - Mel spectrogram extraction
  - Audio patching for transformer input
  - Pretrained feature extraction

### Model Architectures
#### 1. Custom Audio Transofrmer
from custom_transformer import AudioTransformer

```python
model = AudioTransformer(
    sample_rate=16000,
    n_mels=64,
    patch_time=6,
    patch_freq=16,
    emb_dim=128,
    num_heads=4,
    num_layers=4,
    num_classes=10
)
```
#### 2. Pretrained Models
```python
from pretrained_models import Wav2Vec2Classifier, HubertClassifier

wav2vec2 = Wav2Vec2Classifier(num_classes=10)
hubert = HubertClassifier(num_classes=10)
```

#### 3. Sequence Models
```python
from pretrained_models import LSTMClassifier, RNNClassifier

lstm = LSTMClassifier(input_dim=768, num_classes=10)
rnn = RNNClassifier(input_dim=768, num_classes=10)
```

### Training
```python
from trainer import Trainer
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

trainer = Trainer(
    model_instance=model,
    optimizer=Adam(model.parameters(), lr=1e-4),
    criterion=CrossEntropyLoss(),
    train_loader=train_loader,
    valid_loader=val_loader,
    test_loader=test_loader
)

# Train for 10 epochs
trainer.train(n_epochs=10)
```
#### Advanced Training options:
```python
# Train multiple instances with different seeds
trainer.train_multiple(
    n=3,           # Number of runs
    n_epochs=10,
    log=True,
    save_model=True
)

# Access training history
print(trainer.history['val_acc'])  # Validation accuracy per epoch
```
### Data Processing
#### Loading Audio Data
```python
from utils import load_data

train_loader, val_loader, test_loader = load_data(
    audio_dir='path/to/audio',
    val_txt='validation_list.txt',
    test_txt='testing_list.txt',
    batch_size=32,
    sample_rate=16000
)
```
#### Extracting Features
```python
from utils import load_feature_tensor_data

# Extract Wav2Vec2 features
train_loader, val_loader, test_loader = load_feature_tensor_data(
    audio_dir='path/to/audio',
    val_txt='validation_list.txt',
    test_txt='testing_list.txt',
    batch_size=32,
    sample_rate=16000,
    model_name="facebook/wav2vec2-base"
)
```

### Evaluation
After training, the Trainer automatically evaluates on the test set:

```python
Test Loss: 0.1234, Test Acc: 0.9567, 
Test F1: 0.9532, Test Precision: 0.9541, 
Test Recall: 0.9528
```

### Configuration
Key parameters for the custom transformer:

```python
{
    'sample_rate': 16000,    # Audio sample rate
    'n_mels': 64,            # Mel bands for spectrogram
    'patch_time': 6,         # Time dimension patch size
    'patch_freq': 16,        # Frequency dimension patch size
    'emb_dim': 128,          # Transformer embedding dimension
    'num_heads': 4,          # Attention heads
    'num_layers': 4,         # Transformer layers
    'hidden_dim': 256,       # Feed-forward dimension
    'num_classes': 10        # Output classes
}
```
### Pretrained Models:
The repository supports these pretrained models from HuggingFace:

- facebook/wav2vec2-base
- facebook/hubert-base-ls960
