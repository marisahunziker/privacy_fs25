from itertools import product
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from opacus import PrivacyEngine


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MatrixFactorizationModel(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim=32, dropout_rate=0.1):
        super(MatrixFactorizationModel, self).__init__()

        # Latent factors
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)

        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, user_id, movie_id):
        # Pure matrix factorization
        user_latent = self.user_embedding(user_id) # [batch_size, embedding_dim]
        movie_latent = self.movie_embedding(movie_id) # [batch_size, embedding_dim]

        # Apply dropout
        user_latent = self.dropout(user_latent)

        # Final prediction (dot product)
        prediction = (user_latent * movie_latent).sum(dim=-1)  # Dot product for prediction

        return prediction

    






class MovieDataset(Dataset):
    """
    A PyTorch Dataset for loading movie recommendation data.
    """
    def __init__(self, path):
        self.dataset = torch.load(path)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        user_id, movie_id, rating, gender, age, occupation, zip_code = self.dataset[idx]
        
        return {
            "user_id": user_id,
            "movie_id": movie_id,
            "rating": rating,
        }


    



def train(model, train_loader, val_loader, test_loader,
        criterion, optimizer,
        model_base_path, losses_base_path, metrics_base_path, num_epochs=10):
    
    train_losses = []
    val_losses = []

    iteration = 0

    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()

        for data in tqdm(train_loader):
            user_id = data['user_id'].to(device)
            movie_id = data['movie_id'].to(device)
            rating = data['rating'].to(device)

            # Forward + Backward + Optimize
            outputs = model(user_id, movie_id)
            loss = criterion(outputs, rating.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * rating.size(0)
            iteration += 1

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in tqdm(val_loader):
                user_id = data['user_id'].to(device)
                movie_id = data['movie_id'].to(device)
                rating = data['rating'].to(device)

                outputs = model(user_id, movie_id)
                loss = criterion(outputs, rating.float())
                val_loss += loss.item() * rating.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Validation Loss: {avg_val_loss:.6f}")
        
        if (epoch + 1) >= 50 and (epoch + 1) % 5 == 0:
            checkpoint_model_path = f"{model_base_path}_epoch{epoch+1}.pt"
            checkpoint_losses_path = f"{losses_base_path}_epoch{epoch+1}.pt"
            checkpoint_metrics_path = f"{metrics_base_path}_epoch{epoch+1}.pt"

            torch.save(model.state_dict(), checkpoint_model_path)
            torch.save({'train_loss': train_losses, 'val_loss': val_losses}, checkpoint_losses_path)

            # Evaluate and save metrics
            mse, mae, rmse = evaluate(model, test_loader)
            metrics = {
                'epoch': epoch + 1,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
            }
            torch.save(metrics, checkpoint_metrics_path)

            print(f"Checkpoint saved at epoch {epoch+1}: {checkpoint_model_path}")

    




    
def evaluate(model, test_loader):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            user_id = data['user_id'].to(device)
            movie_id = data['movie_id'].to(device)
            rating = data['rating'].float().to(device)

            outputs = model(user_id, movie_id)
            predictions.append(outputs.cpu().numpy())
            targets.append(rating.cpu().numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mse)

    return mse, mae, rmse




def main():
    print("Non-private recommender system")
    
    # Define experimental configs
    learning_rate = 0.01
    weight_decay = 1e-4
    batch_size = 128

    num_epochs = 300
    criterion = nn.MSELoss()

    print("Loading datasets...")
    train_dataset = MovieDataset('datasets/train_dataset.pt')
    val_dataset = MovieDataset('datasets/val_dataset.pt')
    test_dataset = MovieDataset('datasets/test_dataset.pt')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    metadata = torch.load('datasets/metadata.pt')
    n_users = metadata['n_users']
    n_movies = metadata['n_movies']


    model = MatrixFactorizationModel(n_users=n_users, n_movies=n_movies).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print("Starting training...")

    model_path = 'models/np/model_np_no_metadata'
    losses_path = 'losses/np/losses_np_no_metadata'
    metrics_path = 'metrics/np/metrics_np_no_metadata'
    train(model, train_loader, val_loader, test_loader, criterion, optimizer, model_path, losses_path, metrics_path, num_epochs=num_epochs)

    print("Training completed.")

    return 1


if __name__ == "__main__":
    main()

