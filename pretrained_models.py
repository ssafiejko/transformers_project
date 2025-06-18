from transformers import Wav2Vec2Model, Wav2Vec2Config
from transformers import HubertModel, HubertConfig
from torch import nn
import torch
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config

class Wav2Vec2Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Load config first to avoid version issues
        config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec2 = Wav2Vec2Model(config)
        
        # Freeze feature extractor
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        outputs = self.wav2vec2(x)
        x = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(x)
    
class HubertClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        config = HubertConfig.from_pretrained("facebook/hubert-base-ls960")
        self.hubert = HubertModel(config)

        for param in self.hubert.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        outputs = self.hubert(x)
        x = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(x)
    

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_layers=2, num_classes=10):
        super().__init__()
        self.bidirectional = False
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=self.bidirectional)
        lstm_out_dim = hidden_dim * (2 if self.bidirectional else 1)
        self.classifier = nn.Linear(lstm_out_dim, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):  # x: [B, T, 768]
        lstm_out, _ = self.lstm(x)  # [B, T, 2*hidden_dim]
        out = lstm_out[:, -1, :]  # Last time step
        out = self.dropout(out)
        return self.classifier(out)


class RNNClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_layers=2, num_classes=10):
        super().__init__()
        self.bidirectional = False
        self.rnn = nn.RNN(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            bidirectional=self.bidirectional,
            nonlinearity='tanh'
        )
        self.dropout = nn.Dropout(0.3)
        lstm_out_dim = hidden_dim * (2 if self.bidirectional else 1)

        self.classifier = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x):  # x: [B, T, 768]
        rnn_out, _ = self.rnn(x)  # [B, T, 2*hidden_dim]
        out = rnn_out[:, -1, :]  # Last time step
        out = self.dropout(out)
        return self.classifier(out)