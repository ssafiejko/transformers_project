import torch
import pickle
import copy
import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

class Trainer:
    def __init__(self, model_instance, optimizer, criterion, train_loader, valid_loader, test_loader):
        self.device = torch.device("mps" if torch.backends.mps.is_available()
                                   else "cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.model_instance = model_instance.to(self.device)
        self.model = copy.deepcopy(model_instance.to(self.device))
        self.optimizer_base = optimizer
        self.optimizer = optimizer.__class__(self.model.parameters(), **optimizer.defaults)
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.history = {
            'train_acc': [],
            'val_acc': [],
            'train_f1_score': [],
            'val_f1_score': [],
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': [],
            'train_loss': [],
            'val_loss': []
        }

        self.best_val_acc = 0
        self.latest_iteration = 0

        try:
            self.model_name = model_instance.name
        except Exception:
            self.model_name = model_instance.__class__.__name__
            print(f'No model name found. Using {self.model_name}.')

    def train_step(self):
        self.model.train()
        train_loss, train_correct = 0.0, 0
        all_labels = []
        all_preds = []

        for x, y in tqdm.tqdm(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() * x.size(0)
            _, preds = torch.max(out, 1)
            train_correct += torch.sum(preds == y)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        train_loss /= len(self.train_loader.dataset)
        train_acc = train_correct.float() / len(self.train_loader.dataset)
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        train_precision = precision_score(all_labels, all_preds, average='macro')
        train_recall = recall_score(all_labels, all_preds, average='macro')

        return train_loss, train_acc, train_f1, train_precision, train_recall

    def validation_step(self, save):
        self.model.eval()
        val_loss, val_correct = 0.0, 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for x, y in self.valid_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)

                val_loss += loss.item() * x.size(0)
                _, preds = torch.max(out, 1)
                val_correct += torch.sum(preds == y)
                all_labels.extend(y.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        val_loss /= len(self.valid_loader.dataset)
        val_acc = val_correct.float() / len(self.valid_loader.dataset)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_precision = precision_score(all_labels, all_preds, average='macro')
        val_recall = recall_score(all_labels, all_preds, average='macro')

        if save and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            #torch.save(self.model.state_dict(), f'{self.model_name}_{self.latest_iteration+1}.pth')
        return val_loss, val_acc, val_f1, val_precision, val_recall

    def train_log(self, train_loss, train_acc, train_f1, train_precision, train_recall,
                  val_loss, val_acc, val_f1, val_precision, val_recall):
        self.history['train_acc'].append(train_acc.cpu().numpy())
        self.history['val_acc'].append(val_acc.cpu().numpy())
        self.history['train_f1_score'].append(train_f1)
        self.history['val_f1_score'].append(val_f1)
        self.history['train_precision'].append(train_precision)
        self.history['val_precision'].append(val_precision)
        self.history['train_recall'].append(train_recall)
        self.history['val_recall'].append(val_recall)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    def save(self):
        with open(f'{self.model_name}_training_history_{self.latest_iteration+1}.pkl', 'wb') as f:
            pickle.dump(self.history, f)
            print(f'Training history saved as {self.model_name}_training_history_{self.latest_iteration+1}.pkl')

    def train(self, n_epochs=10, log=True, save_model=True, save_history=True):
        for epoch in range(n_epochs):
            print(f'Epoch {epoch+1}/{n_epochs}')
            train_loss, train_acc, train_f1, train_precision, train_recall = self.train_step()
            val_loss, val_acc, val_f1, val_precision, val_recall = self.validation_step(save_model)
            if log:
                self.train_log(train_loss, train_acc, train_f1, train_precision, train_recall,
                               val_loss, val_acc, val_f1, val_precision, val_recall)
        if save_history:
            self.save()
        self.test()

    def test(self):
        self.model.eval()
        test_loss, test_correct = 0.0, 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)

                test_loss += loss.item() * x.size(0)
                _, preds = torch.max(out, 1)
                test_correct += torch.sum(preds == y)
                all_labels.extend(y.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        test_loss /= len(self.test_loader.dataset)
        test_acc = test_correct.float() / len(self.test_loader.dataset)
        test_f1 = f1_score(all_labels, all_preds, average='macro')
        test_precision = precision_score(all_labels, all_preds, average='macro')
        test_recall = recall_score(all_labels, all_preds, average='macro')

        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, '
              f'Test F1: {test_f1:.4f}, Test Precision: {test_precision:.4f}, '
              f'Test Recall: {test_recall:.4f}')

        output_filename = f'{self.model_name}_test_predictions_{self.latest_iteration+1}.txt'
        with open(output_filename, 'w') as f:
            f.write('y_true\ty_pred\n')
            for label, pred in zip(all_labels, all_preds):
                f.write(f'{label}\t{pred}\n')
        print(f'Predictions saved to {output_filename}')

        return test_loss, test_acc, test_f1, test_precision, test_recall


    def train_multiple(self, n=3, **trainkwargs):
        print(f'Training multiple {self.model_name} models ({n} times)...')
        for i in range(n):
            self.best_val_acc = 0
            torch.manual_seed(i)
            self.latest_iteration = i
            self.model = copy.deepcopy(self.model_instance)
            self.optimizer = self.optimizer_base.__class__(self.model.parameters(), **self.optimizer_base.defaults)
            self.train(**trainkwargs)
        print('Training complete.')
