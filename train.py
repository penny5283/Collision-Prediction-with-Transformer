import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from glob import glob
from datetime import datetime


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
print(f'Using device: {device}')


class Model(nn.Module):
    def __init__(self, dim, dim_out, window_size, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(Model, self).__init__()
        self.input_mlp = nn.Linear(dim, d_model)
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation='gelu', norm_first=True)
        self.transformer_encoder = nn.Model(
            encoder_layer, num_layers)
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model * window_size, 16),
            nn.GELU(),
            nn.Linear(16, dim_out))

    def forward(self, x):

        x = self.input_mlp(x)
        x = self.transformer_encoder(x)
        x = x.view(x.size(0), -1)
        for layer in self.output_mlp:
            x = layer(x)
        return x


def train(save_path, num_layers):
    print(save_path)
    
    with open(save_path, 'rb') as f:
        X_train, y_train, X_test, y_test = np.load(f, allow_pickle=True)
        
    save_path = save_path[:2] + 'layer_' + str(num_layers) + '_' + save_path[2:]
    # z-score normalization
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    # mean = np.mean(X_test, axis=0)
    # std = np.std(X_test, axis=0)
    X_test = (X_test - mean) / std

    only_total = False
    is_remove_adj = False

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = y_train.astype(np.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = y_test.astype(np.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # use total
    def calc_total(tensor):
        sum_indices = tensor[:, :, 1:5].sum(dim=-1, keepdim=True)

        # Concatenate the result with the remaining parts of the tensor
        # Keep parts before index 1 and after index 4
        tensor_before = tensor[:, :, :1]
        tensor_after = tensor[:, :, 5:]

        # Concatenate along the last dimension
        result_tensor = torch.cat(
            [tensor_before, sum_indices, tensor_after], dim=-1)
        return result_tensor

    def remove_adj(tensor, feature_idx=2):
        tensor_before = tensor[:, :, :feature_idx]
        tensor_after = tensor[:, :, -3:]

        # Concatenate along the last dimension
        result_tensor = torch.cat([tensor_before, tensor_after], dim=-1)
        return result_tensor

    if only_total:
        X_train = calc_total(X_train)
        y_train = y_train.sum(dim=1).reshape(-1, 1)
        X_test = calc_total(X_test)
        y_test = y_test.sum(dim=1).reshape(-1, 1)

    if is_remove_adj:
        if only_total:
            feature_idx = 2
        else:
            feature_idx = 5
        X_train = remove_adj(X_train, feature_idx)
        X_test = remove_adj(X_test, feature_idx)

    # Hyperparameters
    batch_size = 1024
    seq_len = X_train.shape[1]
    dim = X_train.shape[2]
    dim_out = y_train.shape[1]

    d_model = 64
    nhead = 8
    # num_layers = 2
    dim_feedforward = d_model * 4
    dropout = 0.1

    num_epochs = 160 if num_layers <=2 else 210
    lr = 1e-4
    weight_decay = 5e-2

    # Create the transformer encoder
    model = Model(
        dim, dim_out, seq_len, d_model, nhead, num_layers, dim_feedforward, dropout).to(device)

    # Create dataloader
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)

    losses = []
    eval_losses = []
    eval_step = 10
    best_eval_loss = float('inf')
    model_states = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            # Move batches to GPU if available
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output.squeeze(), batch_y.squeeze()
                             if only_total else batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Evaluate the model every eval_step epochs
        if (epoch + 1) % eval_step == 0:
            model.eval()
            eval_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in test_dataloader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    output = model(batch_x)
                    loss = criterion(
                        output.squeeze(), batch_y.squeeze() if only_total else batch_y)
                    eval_loss += loss.item()

            eval_loss /= len(test_dataloader)
            eval_losses.append(eval_loss)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                model_states = model.state_dict()

            now = datetime.now().strftime("%H:%M:%S")
            print(f"{now} ({COUNT+1}/{TOTAL}) Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss / len(dataloader):.4f}, Evaluation Loss: {eval_loss:.4f}")
        else:
            now = datetime.now().strftime("%H:%M:%S")
            print(
                f"{now} ({COUNT+1}/{TOTAL}) Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss / len(dataloader):.4f}")

        losses.append(epoch_loss / len(dataloader))


    # Prediction
    model.load_state_dict(model_states)
    model.to(device)
    model.eval()

    with torch.no_grad():
        y_pred = model(X_test.to(device)).cpu()
        rounded_prediction = torch.round(y_pred)  # Round to nearest integer

    mse = torch.round(criterion(rounded_prediction, y_test), decimals=4)

    print('MSE: {:.4f}'.format(mse))
    print('RMSE: {:.4f}'.format(torch.sqrt(mse)))

    # Plot the loss curve
    train_loss = losses[10:]
    eval_loss = eval_losses[1:]
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot([i*eval_step for i in range(len(eval_loss)) if eval_losses[i] is not None],
             [loss for loss in eval_loss if loss is not None], label='Evaluation Loss', marker='o')

    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    # plt.show()
    plt.savefig(save_path[:-4] + '_model_{:.4}'.format(mse) + '.png')

    torch.save(model_states, save_path[:-4] +
               '_model_{:.4}'.format(mse) + '.pth')


if __name__ == '__main__':
    save_paths = sorted(glob('./*.npy'))
    COUNT = 0
    TOTAL = len(save_paths)
    print(save_paths)
    for save_path in tqdm(save_paths):
        for num_layer in [4]:
        # for num_layer in [1, 2, 4, 6]:
            train(save_path, num_layer)
            COUNT += 1
