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

    



def train(model, train_loader, val_loader, test_loader, criterion, optimizer,
          model_base_path, losses_base_path, metrics_base_path,
          privacy_engine, num_epochs=10, delta=1e-5):
    
    train_losses = []
    val_losses = []
    epsilon_targets = [round(x * 0.5, 2) for x in range(1, 21)]  # 0.5 to 10.0
    seen_epsilons = set()
    metrics_log = []

    iteration = 0
    epoch = 0

    reached_epsilon = False

    while not reached_epsilon:
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

        # Always check epsilon if privacy_engine is used
        epsilon = privacy_engine.get_epsilon(delta=delta)
        if epsilon >= 10.0:
            print(f"Reached ε = {epsilon:.2f}, stopping training.")
            reached_epsilon = True

        for eps_target in epsilon_targets:
            if eps_target not in seen_epsilons and epsilon >= eps_target:
                seen_epsilons.add(eps_target)
                print(f"[ε = {eps_target:.2f} | Iter = {iteration}] Evaluating model...")

                # Evaluate
                mse, mae, rmse = evaluate(model, test_loader)

                # Log results
                metrics_log.append({
                    'epsilon': eps_target,
                    'delta': delta,
                    'iterations': iteration,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse
                })

                print(f"  ➤ RMSE @ ε = {eps_target:.2f} → {rmse:.4f}")

        # Save checkpoint every 5 epochs after epoch 50, or if stopping
        if ((epoch + 1) >= 50 and (epoch + 1) % 5 == 0) or reached_epsilon:
            checkpoint_model_path = f"{model_base_path}_epoch{epoch+1}.pt"
            checkpoint_losses_path = f"{losses_base_path}_epoch{epoch+1}.pt"
            checkpoint_metrics_path = f"{metrics_base_path}_epoch{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_model_path)
            torch.save({'train_loss': train_losses, 'val_loss': val_losses}, checkpoint_losses_path)
            torch.save(metrics_log, checkpoint_metrics_path)

            print(f"Checkpoint saved at epoch {epoch+1}: {checkpoint_model_path}")

        epoch += 1

    # Save final model and logs after training ends
    final_model_path = f"{model_base_path}_final.pt"
    final_losses_path = f"{losses_base_path}_final.pt"
    final_metrics_path = f"{metrics_base_path}_final.pt"
    torch.save(model.state_dict(), final_model_path)
    torch.save({'train_loss': train_losses, 'val_loss': val_losses}, final_losses_path)
    torch.save(metrics_log, final_metrics_path)
    print(f"Final model and logs saved: {final_model_path}")





    
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
    print(f"Running in differentially private mode...")

    print("Loading datasets...")
    train_dataset = MovieDataset('datasets/train_dataset.pt')
    val_dataset = MovieDataset('datasets/val_dataset.pt')
    test_dataset = MovieDataset('datasets/test_dataset.pt')

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    metadata = torch.load('datasets/metadata.pt')
    n_users = metadata['n_users']
    n_movies = metadata['n_movies']

    # TODO: Hyperparameters (update based on prior tuning)
    learning_rate = 0.01
    weight_decay = 1e-4
    max_grad_norm = 1.0
    noise_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    for i, noise_multiplier in enumerate(noise_multipliers):
        print(f"\n=== Training with noise multiplier: {noise_multiplier} ===")

        # Recreate model and optimizer for each run
        model = MatrixFactorizationModel(n_users=n_users, n_movies=n_movies).to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Recreate train loader and attach PrivacyEngine
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            accountant='rdp'  # Use RDP accountant
        )

        # File paths
        model_path = f'models/dp/model_dp_{i+1}'
        losses_path = f'losses/dp/losses_dp_{i+1}'
        metrics_path = f'metrics/dp/metrics_dp_{i+1}'

        # Loss function
        criterion = nn.MSELoss()

        print("Starting training with differential privacy...")
        train(model=model,
              train_loader=train_loader,
              val_loader=val_loader,
              test_loader=test_loader,
              criterion=criterion,
              optimizer=optimizer,
              model_path=model_path,
              losses_path=losses_path,
              metrics_path=metrics_path,
              privacy_engine=privacy_engine)

    print("\nAll DP experiments completed.")


    return 1


if __name__ == "__main__":
    main()
