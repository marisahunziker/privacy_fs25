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
        
        if (epoch + 1) >= 20 and (epoch + 1) % 5 == 0:
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
    print("Non-private recommender system")

    print("Loading datasets...")
    train_dataset = MovieDataset('datasets/train_dataset.pt')
    val_dataset = MovieDataset('datasets/val_dataset.pt')
    test_dataset = MovieDataset('datasets/test_dataset.pt')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    metadata = torch.load('datasets/metadata.pt')
    n_users = metadata['n_users']
    n_movies = metadata['n_movies']

    # Define experimental configs
    learning_rates = [0.01, 0.005, 0.001]
    weight_decays = [1e-4, 1e-5, 1e-6]
    experiments = list(product(learning_rates, weight_decays))

    num_epochs = 50
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

