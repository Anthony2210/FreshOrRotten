"""Évaluation du baseline_model FreshOrRotten."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from train import create_image_dataset, get_image_size, load_config, scan_image_files, split_dataset


def load_test_set(config, reports_dir, dataset_path):
    """Charge le test_set depuis le split sauvegardé ou le recrée."""
    training_config = config["training"]
    split_filename = training_config.get("split_filename", "standard_split.csv")
    split_path = reports_dir / split_filename

    if split_path.exists():
        split_df = pd.read_csv(split_path)
        test_set = split_df[split_df["split"] == "test"].copy()

        if test_set.empty:
            raise ValueError("Le fichier de split ne contient aucun test_set.")

        test_set["image_path"] = test_set["image_path"].apply(lambda path: dataset_path / path)
        return test_set

    print("Aucun split sauvegardé trouvé. Le standard_split est recréé depuis config.yaml.")

    image_index_df = scan_image_files(dataset_path, config["data"]["labels"])
    _, _, test_set = split_dataset(
        image_index_df=image_index_df,
        validation_size=float(training_config.get("validation_size", 0.2)),
        test_size=float(training_config.get("test_size", 0.2)),
        random_seed=int(training_config.get("random_seed", 42)),
    )

    return test_set


def collect_predictions(model, test_dataset):
    """Récupère les labels réels et les prédictions du modèle."""
    true_labels = []

    for _, labels in test_dataset:
        true_labels.extend(labels.numpy())

    prediction_scores = model.predict(test_dataset).reshape(-1)
    predicted_labels = (prediction_scores >= 0.5).astype(int)

    return np.array(true_labels).astype(int), predicted_labels, prediction_scores


def evaluate_model(config, model_path=None):
    """Évalue le modèle sur le test_set."""
    paths_config = config["paths"]
    training_config = config["training"]

    dataset_path = PROJECT_ROOT / paths_config["raw_data_dir"]
    reports_dir = PROJECT_ROOT / Path(paths_config["results_report"]).parent
    model_dir = PROJECT_ROOT / paths_config["model_dir"]

    if model_path is None:
        model_path = model_dir / training_config.get("model_filename", "baseline_model.keras")

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    image_size = get_image_size(config)
    batch_size = int(training_config.get("batch_size", 32))
    random_seed = int(training_config.get("random_seed", 42))

    test_set = load_test_set(config, reports_dir, dataset_path)
    test_dataset = create_image_dataset(
        test_set,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        random_seed=random_seed,
    )

    model = tf.keras.models.load_model(model_path)

    evaluation_result = model.evaluate(test_dataset, return_dict=True)
    true_labels, predicted_labels, _ = collect_predictions(model, test_dataset)

    metrics = {
        "loss": evaluation_result.get("loss"),
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, zero_division=0),
        "f1_score": f1_score(true_labels, predicted_labels, zero_division=0),
    }

    confusion_matrix_df = pd.DataFrame(
        confusion_matrix(true_labels, predicted_labels, labels=[0, 1]),
        index=["actual_fresh", "actual_rotten"],
        columns=["predicted_fresh", "predicted_rotten"],
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(reports_dir / "evaluation_metrics.csv", index=False)
    confusion_matrix_df.to_csv(reports_dir / "confusion_matrix.csv")

    print("Résultats sur le test_set :")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("Matrice de confusion :")
    print(confusion_matrix_df)


def parse_args():
    parser = argparse.ArgumentParser(description="Évalue la baseline CNN FreshOrRotten.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Chemin vers config.yaml.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Chemin optionnel vers un modèle .keras.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    project_config = load_config(args.config)
    evaluate_model(project_config, model_path=args.model_path)
