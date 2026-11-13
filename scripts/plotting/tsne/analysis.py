import numpy as np

from sklearn.manifold import TSNE


def run_tsne(
    embeddings,
    perplexity=30,
    random_state=42,
):
    """
    Convert latent embeddings to 2-D using t-SNE.
    """

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Expected embeddings with shape "
            "(N, features). "
            f"Got {embeddings.shape}"
        )

    if embeddings.shape[0] <= perplexity:
        raise ValueError(
            f"Perplexity ({perplexity}) must be "
            f"smaller than number of samples "
            f"({embeddings.shape[0]})."
        )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )

    return tsne.fit_transform(
        embeddings
    )