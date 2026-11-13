from scripts.models.compact_mlp import CompactMLP_FD


MODEL_REGISTRY = {
    "compact_mlp": CompactMLP_FD,
}


def get_model(model_name, **kwargs):

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {list(MODEL_REGISTRY)}"
        )

    return MODEL_REGISTRY[model_name](**kwargs)