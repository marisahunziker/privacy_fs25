import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

"""
This script implements a non-private recommender system using matrix factorization without metadata.
It is used to evaluate the impact of metadata on the performance of the recommender system.
"""



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MatrixFactorizationModel(nn.Module):
    """
    MatrixFactorizationModel implements a simple matrix factorization approach for collaborative filtering in recommendation systems.

    Args:
        n_users (int): Number of unique users in the dataset.
        n_movies (int): Number of unique movies/items in the dataset.
        embedding_dim (int, optional): Dimensionality of the latent embedding vectors for users and movies. Default is 32.
        dropout_rate (float, optional): Dropout rate applied to user embeddings during training to prevent overfitting. Default is 0.1.

    Attributes:
        user_embedding (nn.Embedding): Embedding layer mapping user IDs to latent vectors.
        movie_embedding (nn.Embedding): Embedding layer mapping movie IDs to latent vectors.
        dropout (nn.Dropout): Dropout layer applied to user embeddings.

    Methods:
        forward(user_id, movie_id):
            Computes the predicted rating or interaction score for a given pair of user and movie IDs.
            Args:
                user_id (Tensor): Tensor of user IDs.
                movie_id (Tensor): Tensor of movie IDs.
            Returns:
                Tensor: Predicted scores for each user-movie pair, computed as the dot product of their latent vectors.
    """
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
    A PyTorch Dataset for loading movie rating data.
    Args:
        path (str): Path to the file containing the dataset, expected to be a torch-serialized object.
    Attributes:
        dataset (list): Loaded dataset, where each entry is expected to be a tuple containing
            (user_id, movie_id, rating, gender, age, occupation, zip_code).
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Retrieves a sample by index, returning a dictionary with keys:
            'user_id', 'movie_id', and 'rating'.
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
    """
    Trains a recommendation model using the provided data loaders, loss function, and optimizer.
    Tracks and saves training and validation losses, and periodically checkpoints the model and evaluation metrics.
    Args:
        model (torch.nn.Module): The recommendation model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset, used for evaluation.
        criterion (torch.nn.Module): Loss function to optimize.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        model_base_path (str): Base file path for saving model checkpoints.
        losses_base_path (str): Base file path for saving loss checkpoints.
        metrics_base_path (str): Base file path for saving evaluation metrics.
        num_epochs (int, optional): Number of training epochs. Default is 10.
    Returns:
        None
    Side Effects:
        - Prints training and validation loss after each epoch.
        - Saves model checkpoints, loss history, and evaluation metrics every 5 epochs after epoch 50.
    """
    
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
    """
    Evaluates the performance of a recommendation model on a test dataset.

    Args:
        model (torch.nn.Module): The recommendation model to be evaluated.
        test_loader (torch.utils.data.DataLoader): DataLoader providing the test dataset.

    Returns:
        tuple: A tuple containing the following evaluation metrics:
            - mse (float): Mean Squared Error between predicted and true ratings.
            - mae (float): Mean Absolute Error between predicted and true ratings.
            - rmse (float): Root Mean Squared Error between predicted and true ratings.
    """
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
    """
    Main function to train a non-private recommender system without metadata.
    This function sets up the experimental configuration, loads the datasets and metadata,
    initializes the matrix factorization model, optimizer, and loss criterion, and then
    trains the model using the provided training, validation, and test data loaders.
    The trained model, training losses, and evaluation metrics are saved to specified paths.
    Returns:
        int: Returns 1 upon successful completion of training.
    """
    print("Non-private recommender system without metadata")
    
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

