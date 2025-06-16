from itertools import product
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

"""
This script implements a non-private recommender system using matrix factorization with user and movie embeddings,
augmented by user metadata embeddings.
It is used to experiment with different configurations of learning rates and weight decays,
and to evaluate the model's performance on a movie recommendation task.
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MatrixFactorizationModel(nn.Module):
    """
    MatrixFactorizationModel implements a matrix factorization-based recommender system with user and movie embeddings,
    augmented by user metadata embeddings (gender, age, occupation, zip code).

    Args:
        n_users (int): Number of unique users.
        n_movies (int): Number of unique movies.
        n_genders (int, optional): Number of unique gender categories. Default is 2.
        n_ages (int, optional): Number of unique age categories. Default is 7.
        n_occupations (int, optional): Number of unique occupation categories. Default is 21.
        n_zip_codes (int, optional): Number of unique zip code categories. Default is 100.
        embedding_dim (int, optional): Dimension of the latent factors for users and movies. Default is 32.
        metadata_dim (int, optional): Dimension of the embeddings for each metadata feature. Default is 8.
        dropout_rate (float, optional): Dropout rate applied after combining user latent and metadata embeddings. Default is 0.1.

    Forward Args:
        user_id (Tensor): Tensor of user IDs, shape [batch_size].
        movie_id (Tensor): Tensor of movie IDs, shape [batch_size].
        gender (Tensor): Tensor of gender indices, shape [batch_size].
        age (Tensor): Tensor of age indices, shape [batch_size].
        occupation (Tensor): Tensor of occupation indices, shape [batch_size].
        zip_code (Tensor): Tensor of zip code indices, shape [batch_size].

    Returns:
        Tensor: Predicted ratings or scores, shape [batch_size].
    """
    def __init__(self, n_users, n_movies, n_genders=2, n_ages=7, n_occupations=21, n_zip_codes=100, embedding_dim=32, metadata_dim=8, dropout_rate=0.1):
        super(MatrixFactorizationModel, self).__init__()

        # Latent factors
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)

        # Metadata embeddings
        self.gender_embedding = nn.Embedding(n_genders, metadata_dim)
        self.age_embedding = nn.Embedding(n_ages, metadata_dim)
        self.occupation_embedding = nn.Embedding(n_occupations, metadata_dim)
        self.zip_code_embedding = nn.Embedding(n_zip_codes, metadata_dim)

        # Fully connected layer after concatenating user latent + metadata
        self.fc_user = nn.Linear(embedding_dim + 4 * metadata_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, user_id, movie_id, gender, age, occupation, zip_code):
        # Matrix factorization part
        user_latent = self.user_embedding(user_id) # [batch_size, embedding_dim]
        movie_latent = self.movie_embedding(movie_id) # [batch_size, embedding_dim]

        # Metadata embeddings
        gender_latent = self.gender_embedding(gender)
        age_latent = self.age_embedding(age)
        occupation_latent = self.occupation_embedding(occupation)
        zip_code_latent = self.zip_code_embedding(zip_code)

        # Combine user latent factors with metadata
        user_metadata = torch.cat([user_latent, gender_latent, age_latent, occupation_latent, zip_code_latent], dim=-1)
        user_metadata = self.fc_user(user_metadata)  # [batch_size, embedding_dim]
        user_metadata = self.dropout(user_metadata)  # Apply dropout

        # Final prediction
        prediction = (user_metadata * movie_latent).sum(dim=-1)  # Dot product for prediction

        return prediction
    






class MovieDataset(Dataset):
    """
    A PyTorch Dataset for loading movie recommendation data.
    Args:
        path (str): Path to the file containing the dataset, expected to be a torch-serialized object.
    Attributes:
        dataset (list): List of tuples, each containing user and movie information.
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Retrieves a sample by index, returning a dictionary with keys:
            - 'user_id': Identifier for the user.
            - 'movie_id': Identifier for the movie.
            - 'rating': The rating given by the user.
            - 'gender': The gender of the user.
            - 'age': The age of the user.
            - 'occupation': The occupation of the user.
            - 'zip_code': The zip code of the user.
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
            "gender": gender,
            "age": age,
            "occupation": occupation,
            "zip_code": zip_code
        }

    



def train(model, train_loader, val_loader, test_loader,
        criterion, optimizer, experiment_id,
        model_base_path, losses_base_path, metrics_base_path, num_epochs=10):
    """
    Trains a recommendation model using the provided data loaders, loss function, and optimizer.
    Args:
        model (torch.nn.Module): The recommendation model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset (used for evaluation).
        criterion (callable): Loss function to optimize.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        experiment_id (str): Identifier for the current experiment (used in saved metrics).
        model_base_path (str): Base file path for saving model checkpoints.
        losses_base_path (str): Base file path for saving loss checkpoints.
        metrics_base_path (str): Base file path for saving evaluation metrics.
        num_epochs (int, optional): Number of training epochs. Default is 10.
    Returns:
        None
    Side Effects:
        - Trains the model and prints training/validation loss per epoch.
        - Saves model checkpoints, loss history, and evaluation metrics at specified intervals.
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
            gender = data['gender'].to(device)
            age = data['age'].to(device)
            occupation = data['occupation'].to(device)
            zip_code = data['zip_code'].to(device)

            # Forward + Backward + Optimize
            outputs = model(user_id, movie_id, gender, age, occupation, zip_code)
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
                gender = data['gender'].to(device)
                age = data['age'].to(device)
                occupation = data['occupation'].to(device)
                zip_code = data['zip_code'].to(device)

                outputs = model(user_id, movie_id, gender, age, occupation, zip_code)
                loss = criterion(outputs, rating.float())
                val_loss += loss.item() * rating.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Validation Loss: {avg_val_loss:.6f}")
        
        if (epoch + 1) >= 20 and (epoch + 1) % 10 == 0:
            checkpoint_model_path = f"{model_base_path}_epoch{epoch+1}.pt"
            checkpoint_losses_path = f"{losses_base_path}_epoch{epoch+1}.pt"
            checkpoint_metrics_path = f"{metrics_base_path}_epoch{epoch+1}.pt"

            torch.save(model.state_dict(), checkpoint_model_path)
            torch.save({'train_loss': train_losses, 'val_loss': val_losses}, checkpoint_losses_path)

            # Evaluate and save metrics
            mse, mae, rmse = evaluate(model, test_loader)
            metrics = {
                'experiment': experiment_id,
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
        model (torch.nn.Module): The recommendation model to evaluate.
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
            gender = data['gender'].to(device)
            age = data['age'].to(device)
            occupation = data['occupation'].to(device)
            zip_code = data['zip_code'].to(device)

            outputs = model(user_id, movie_id, gender, age, occupation, zip_code)
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
    Main function to run non-private recommender system experiments.

    This function performs the following steps:
    1. Loads training, validation, and test datasets for a movie recommender system.
    2. Loads metadata to determine the number of users and movies.
    3. Defines a grid of experimental configurations with different learning rates and weight decays.
    4. For each configuration, initializes a matrix factorization model, optimizer, and loss function.
    5. Trains the model using the specified configuration, saving model checkpoints, losses, and metrics.
    6. Evaluates the model on the test dataset and saves evaluation metrics.

    Returns:
        int: Returns 1 upon successful completion of all experiments.
    """
    print("Non-private recommender system")

    print("Loading datasets...")
    train_dataset = MovieDataset('datasets/train_dataset.pt')
    val_dataset = MovieDataset('datasets/val_dataset.pt')
    test_dataset = MovieDataset('datasets/test_dataset.pt')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    metadata = torch.load('datasets/metadata.pt')
    n_users = metadata['n_users']
    n_movies = metadata['n_movies']

    # Define experimental configs
    learning_rates = [0.01, 0.005, 0.001]
    weight_decays = [1e-4, 1e-5, 1e-6]
    experiments = list(product(learning_rates, weight_decays))

    num_epochs = 300
    criterion = nn.MSELoss()

    for i, (learning_rate, weight_decay) in enumerate(experiments):
        print(f"\n=== Experiment {i+1} ===")
        print(f"Learning Rate: {learning_rate}, Weight Decay: {weight_decay}")

        model = MatrixFactorizationModel(n_users=n_users, n_movies=n_movies).to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        model_path = f'models/np/model_np_{i+1}'
        losses_path = f'losses/np/losses_np_{i+1}'
        metrics_path = f'metrics/np/metrics_np_{i+1}'

        print("Starting training...")
        train(model, train_loader, val_loader, test_loader, criterion, optimizer, i + 1, model_path, losses_path, metrics_path, num_epochs=num_epochs)


    print("\nAll experiments completed.")


    return 1


if __name__ == "__main__":
    main()

