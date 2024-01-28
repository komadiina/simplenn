import torch as T
import torch.nn as nn
import numpy as np

class NeuralNet(nn.Module):
    def __init__(self, layers=[], loss_fn=None, device='cpu'):
        super(NeuralNet, self).__init__()
        
        self.layers: nn.Sequential = nn.Sequential(*layers)
        self.loss_fn = loss_fn
        self.device = device
        self.to(device)

    def forward(self, x):
        return self.layers(x)

    def train_model(self, x, y, optimizer, epochs=1000, early_stopping=True, early_stopping_epochs=3):
        self.train()  

        epoch_loss_increase = 0
        previous_epoch_loss = 0
        minimum_loss = 0

        for epoch in range(epochs):
            x = x.to(self.device)
            y = y.to(self.device)
            y_pred = self.layers(x)
            loss = self.loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose:
                print(f'Loss: {loss.item()}')
            
            if early_stopping:
                if loss.item() > previous_epoch_loss:
                    epoch_loss_increase += 1
                    
                    if epoch_loss_increase == early_stopping_epochs:
                        break
                elif loss.item() < previous_epoch_loss:
                    minimum_loss = loss.item()
                    epoch_loss_increase = 0
                else:
                    epoch_loss_increase = 0

            previous_epoch_loss = loss.item()
        
        return minimum_loss

    def test_model(self, x, y, verbose=True):
        self.eval()  
        total_loss = 0
        predictions = np.array([])

        with T.no_grad():
            for i in range(len(x)):
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.layers(x)
                predictions = np.append(predictions, y_pred)
                loss = self.loss_fn(y_pred, y)
                total_loss += loss.item()
        return predictions

    def predict(self, x):
        self.eval()  
        return self.layers(x)
