# Movie Recommendation System

This project implements a movie recommendation system using various approaches, including traditional collaborative filtering, Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), and LightGCN. The system leverages the MovieLens 100k dataset to recommend movies to users based on their past interactions.

## Project Structure

- **Recommendation_System.ipynb**: The main Jupyter notebook containing the implementation of all recommendation models (Traditional, GCN, GAT, LightGCN).
- **u.data**: User-item interaction data (user_id, item_id, rating, timestamp).
- **u.item**: Movie metadata (movie_id, movie_title, etc.).
- **README.md**: Project documentation.
- **requirements.txt**: Python package dependencies.

## Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- NVIDIA GPU (optional, for faster training with PyTorch and torch_geometric)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the MovieLens 100k dataset:
   - Place `u.data` and `u.item` files in the project root directory.
   - These files can be obtained from the [MovieLens 100k dataset](https://grouplens.org/datasets/movielens/100k/).

## Usage

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook Recommendation_System.ipynb
   ```

2. Run the notebook cells sequentially to:
   - Install dependencies (e.g., torch_geometric).
   - Preprocess the MovieLens dataset.
   - Train and evaluate the recommendation models (Traditional, GCN, GAT, LightGCN).
   - Generate recommendations for a specific user (e.g., user_id=196).

3. Example output:
   - The notebook prints recommended movies for user 196 using each model.
   - For LightGCN, it also lists movies already watched by the user.

## Models Implemented

1. **Traditional Collaborative Filtering**:
   - Uses cosine similarity on a user-item rating matrix to recommend items.
2. **Graph Convolutional Network (GCN)**:
   - Models user-item interactions as a bipartite graph and uses GCN layers to learn embeddings.
3. **Graph Attention Network (GAT)**:
   - Employs attention mechanisms to weigh neighbor contributions in the graph.
4. **LightGCN**:
   - A simplified GCN variant optimized for recommendation tasks, using normalized adjacency matrices.

## Dataset

The project uses the MovieLens 100k dataset:
- **u.data**: Contains 100,000 ratings with columns: `user_id`, `item_id`, `rating`, `timestamp`.
- **u.item**: Contains movie metadata with columns: `item_id`, `movie_title`, etc.

Ensure these files are in the project directory before running the notebook.

## Requirements

See `requirements.txt` for a complete list of dependencies. Key packages include:
- pandas
- numpy
- scikit-learn
- torch
- torch_geometric
- scipy

## Notes

- The LightGCN model uses a subsampled dataset (500 users, 2000 interactions) for faster training. Modify the sampling parameters in the notebook for full dataset usage.
- GPU acceleration is recommended for training GCN, GAT, and LightGCN models.
- The notebook assumes `u.data` and `u.item` are in the correct format. Verify file encodings (latin-1 for u.item).



## Acknowledgments

- MovieLens dataset provided by [GroupLens](https://grouplens.org/datasets/movielens/).
- Implementation inspired by PyTorch Geometric and LightGCN research papers.
