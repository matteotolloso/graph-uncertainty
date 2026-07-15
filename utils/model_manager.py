import os
import re

def find_best_checkpoints(
    dataset_name: str,
    num_models: int,
    metric_name: str = "val_auroc",
    mode: str = "max",
    fallback_metric_names: tuple[str, ...] = ("val_f1",),
) -> list[str]:
    """
    Finds the paths of the top-performing model checkpoints for a given dataset.
    Checkpoints are expected to include the monitored validation metric in the
    filename, e.g. {run_id}_{dataset_name}_val_auroc={score:.4f}.ckpt.

    Args:
        dataset_name (str): The name of the dataset (e.g., 'Cora').
        num_models (int): The number of top models to retrieve (M).

    Returns:
        A list of file paths to the best checkpoints.
    """

    checkpoint_dir = "checkpoints"
    print(f"Searching for '{dataset_name}' checkpoints in: {checkpoint_dir}")

    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Root checkpoint directory not found: '{checkpoint_dir}'. Please run training first.")

    def collect_checkpoints(metric: str):
        checkpoints = []
        score_pattern = re.compile(rf"{re.escape(metric)}=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\.ckpt")

        for filename in os.listdir(checkpoint_dir):
            # Check if the filename matches the dataset and is a checkpoint file
            if f"_{dataset_name}_" in filename and filename.endswith(".ckpt"):
                match = score_pattern.search(filename)
                if match:
                    score = float(match.group(1))
                    full_path = os.path.join(checkpoint_dir, filename)
                    checkpoints.append((full_path, score))
        return checkpoints

    selected_metric = metric_name
    checkpoints = collect_checkpoints(selected_metric)

    for fallback_metric in fallback_metric_names:
        if checkpoints:
            break
        checkpoints = collect_checkpoints(fallback_metric)
        if checkpoints:
            selected_metric = fallback_metric
            print(
                f"Warning: No '{metric_name}' checkpoints found for '{dataset_name}'. "
                f"Falling back to legacy '{fallback_metric}' checkpoints."
            )

    checkpoints.sort(key=lambda x: x[1], reverse=(mode == "max"))

    if not checkpoints:
        raise FileNotFoundError(
            f"No valid checkpoints found for dataset '{dataset_name}' in {checkpoint_dir} "
            f"with metric '{metric_name}' or fallbacks {fallback_metric_names}.")

    # Select the top M models
    top_checkpoints = checkpoints[:num_models]
    
    if len(top_checkpoints) < num_models:
        print(f"Warning: Found only {len(top_checkpoints)} checkpoints for '{dataset_name}', but {num_models} were requested.")

    print(f"Found {len(top_checkpoints)} top models for '{dataset_name}' using '{selected_metric}':")
    for path, score in top_checkpoints:
        print(f"  - Path: {os.path.basename(path)}, Score: {score:.4f}")
        
    return [path for path, score in top_checkpoints]

def search_best_model(
    save_path,
    dataset_name,
    metric_name: str = "val_auroc",
    mode: str = "max",
    fallback_metric_names: tuple[str, ...] = ("val_f1",),
):
    """
    Search for the best model in the given save path based on the dataset name
    and validation metric in the checkpoint filename.

    Args:
        save_path (str): The path where the models are saved.
        dataset_name (str): The name of the dataset to search for.

    Returns:
        str: The path to the best model file.
    """
    def search_metric(metric: str):
        best_score = float('-inf') if mode == "max" else float("inf")
        best_model_path = None
        pattern = re.compile(
            rf".+_{re.escape(dataset_name)}_{re.escape(metric)}=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\.ckpt$"
        )

        for filename in os.listdir(save_path):
            match = pattern.match(filename)
            if match:
                score_str = match.group(1)
                try:
                    score = float(score_str)
                    is_better = score > best_score if mode == "max" else score < best_score
                    if is_better:
                        best_score = score
                        best_model_path = os.path.join(save_path, filename)
                except ValueError:
                    continue  # In case of malformed float string
        return best_model_path

    selected_metric = metric_name
    best_model_path = search_metric(selected_metric)

    for fallback_metric in fallback_metric_names:
        if best_model_path is not None:
            break
        best_model_path = search_metric(fallback_metric)
        if best_model_path is not None:
            selected_metric = fallback_metric
            print(
                f"Warning: No '{metric_name}' checkpoint found for '{dataset_name}' in '{save_path}'. "
                f"Falling back to legacy '{fallback_metric}'."
            )

    if best_model_path is None:
        raise FileNotFoundError(
            f"No valid model files found for dataset '{dataset_name}' in '{save_path}' "
            f"with metric '{metric_name}' or fallbacks {fallback_metric_names}"
        )

    print(f"Selected checkpoint for '{dataset_name}' using '{selected_metric}': {os.path.basename(best_model_path)}")
    return best_model_path
