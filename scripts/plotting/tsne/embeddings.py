import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def extract_embeddings(
    model,
    X,
    device,
    batch_size=256,
):
    """
    Extract latent representations from a model.

    The model must implement:
        model.get_embedding(x)

    Returns:
        numpy array with shape (N, latent_dim)
    """

    model.eval()

    X_tensor = torch.as_tensor(
        X,
        dtype=torch.float32,
    )

    loader = DataLoader(
        TensorDataset(X_tensor),
        batch_size=batch_size,
        shuffle=False,
    )

    embeddings = []

    with torch.no_grad():

        for (X_batch,) in loader:

            X_batch = X_batch.to(device)

            features = model.get_embedding(
                X_batch
            )

            embeddings.append(
                features.cpu().numpy()
            )

    return np.concatenate(
        embeddings,
        axis=0,
    )