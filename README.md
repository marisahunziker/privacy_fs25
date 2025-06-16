## Privacy Preserving Movie Recommender System

This project was developed for the course *Privacy-Preserving Methods for Data Science and Distributed Systems* by Isabel Wagner at the University of Basel, spring semester 2025.

### Description

The goal of this project is to analyze the performance and utility of non-private and differentially private matrix factorization models using stochastic gradient descent for movie recommendations, based on the MovieLens 1M dataset.

The repository includes four main scripts for experimenting with matrix factorization-based recommender systems:

- **recommender_np.py**: Implements the non-private baseline model. Supports running multiple experiments with different learning rates and weight decay values.
- **recommender_np_no_metadata.py**: Similar to the baseline, but excludes user metadata from training, allowing comparison of the impact of metadata.
- **recommender_dp_clip.py**: Enables experimentation with differential privacy by tuning the gradient norm clipping parameter.
- **recommender_dp.py**: Implements a differentially private recommender system, allowing training with various noise multipliers.

### Set-Up

1. **Directory Structure**  
    Ensure the following directory structure is in place:
    ```
    data/
    datasets/
    losses/
         clip/
         dp/
         np/
    metrics/
         clip/
         dp/
         np/
    models/
         clip/
         dp/
         np/
    ```

2. **Dataset Download**  
    Download the MovieLens 1M dataset from [GroupLens](https://grouplens.org/datasets/movielens/1m/) and place it in the `data/` folder.

3. **Preprocessing**  
    Run `data_preprocessor.ipynb` to preprocess the dataset and save it as a `TensorDataset` in `datasets/`.

4. **Dataset Analysis**  
    Use `data_analyser.ipynb` to compute key metrics for the preprocessed and split datasets.

5. **Running Experiments**  
    Execute the scripts. Metrics and model checkpoints are saved in their respective directories.

### Evaluation

The `evaluation` folder contains a Jupyter notebook for generating plots and evaluating model performance.