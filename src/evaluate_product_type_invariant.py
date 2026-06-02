"""Évaluation du modèle qui réduit la dépendance au product_type."""

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

from model import GradientReversal
from split_strategy import load_split_file
from train import create_image_dataset, get_image_size, load_config


def get_invariant_reports_dir(config):
    """Retourne le dossier de résultats de la méthode adversariale."""
    invariant_config = config.get("product_type_invariant", {})
    return PROJECT_ROOT / invariant_config.get("reports_dir", "reports/product_type_invariant")


def get_output_settings(config, split_name):
    """Retourne les chemins utiles pour l'évaluation."""
    paths_config = config["paths"]
    invariant_config = config.get("product_type_invariant", {})
    model_dir = PROJECT_ROOT / paths_config["model_dir"]
    reports_dir = get_invariant_reports_dir(config)

    if split_name == "unseen":
        return {
            "model_path": model_dir
            / invariant_config.get("unseen_model_filename", "product_type_invariant_unseen.keras"),
            "split_path": reports_dir
            / invariant_config.get("unseen_split_filename", "product_type_invariant_unseen_split.csv"),
            "metrics_filename": invariant_config.get(
                "unseen_metrics_filename",
                "product_type_invariant_unseen_metrics.csv",
            ),
            "confusion_matrix_filename": invariant_config.get(
                "unseen_confusion_matrix_filename",
                "product_type_invariant_unseen_confusion_matrix.csv",
            ),
            "product_type_metrics_filename": invariant_config.get(
                "unseen_product_type_metrics_filename",
                "product_type_invariant_unseen_metrics_by_product_type.csv",
            ),
        }

    return {
        "model_path": model_dir / invariant_config.get("model_filename", "product_type_invariant_model.keras"),
        "split_path": reports_dir
        / invariant_config.get("split_filename", "product_type_invariant_standard_split.csv"),
        "metrics_filename": invariant_config.get(
            "metrics_filename",
            "product_type_invariant_standard_metrics.csv",
        ),
        "confusion_matrix_filename": invariant_config.get(
            "confusion_matrix_filename",
            "product_type_invariant_standard_confusion_matrix.csv",
        ),
        "product_type_metrics_filename": None,
    }


def get_freshness_scores(predictions):
    """Récupère la sortie freshness d'un modèle à plusieurs sorties."""
    if isinstance(predictions, dict):
        return np.asarray(predictions["freshness"]).reshape(-1)

    if isinstance(predictions, (list, tuple)):
        return np.asarray(predictions[0]).reshape(-1)

    return np.asarray(predictions).reshape(-1)


def collect_predictions(model, test_dataset):
    """Récupère les labels réels et les scores freshness."""
    true_labels = []

    for _, labels in test_dataset:
        true_labels.extend(labels.numpy())

    predictions = model.predict(test_dataset, verbose=0)
    prediction_scores = get_freshness_scores(predictions)
    predicted_labels = (prediction_scores >= 0.5).astype(int)

    return np.array(true_labels).astype(int), predicted_labels, prediction_scores


def calculate_binary_loss(true_labels, prediction_scores):
    """Calcule la binary_crossentropy sur la sortie freshness."""
    true_tensor = tf.convert_to_tensor(true_labels.reshape(-1, 1), dtype=tf.float32)
    score_tensor = tf.convert_to_tensor(prediction_scores.reshape(-1, 1), dtype=tf.float32)
    loss_values = tf.keras.losses.binary_crossentropy(true_tensor, score_tensor)
    return float(tf.reduce_mean(loss_values).numpy())


def calculate_metrics(true_labels, predicted_labels, prediction_scores):
    """Calcule les métriques principales de fraîcheur."""
    return {
        "loss": calculate_binary_loss(true_labels, prediction_scores),
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, zero_division=0),
        "f1_score": f1_score(true_labels, predicted_labels, zero_division=0),
    }


def calculate_metrics_by_product_type(test_set, true_labels, predicted_labels):
    """Calcule les métriques par product_type pour le protocole unseen."""
    prediction_df = test_set[["product_type"]].copy().reset_index(drop=True)
    prediction_df["true_label"] = true_labels
    prediction_df["predicted_label"] = predicted_labels

    rows = []
    for product_type, product_df in prediction_df.groupby("product_type"):
        rows.append(
            {
                "product_type": product_type,
                "image_count": len(product_df),
                "accuracy": accuracy_score(product_df["true_label"], product_df["predicted_label"]),
                "precision": precision_score(
                    product_df["true_label"],
                    product_df["predicted_label"],
                    zero_division=0,
                ),
                "recall": recall_score(
                    product_df["true_label"],
                    product_df["predicted_label"],
                    zero_division=0,
                ),
                "f1_score": f1_score(
                    product_df["true_label"],
                    product_df["predicted_label"],
                    zero_division=0,
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("product_type")


def add_metrics_row(rows, metrics_path, method, protocol):
    """Ajoute une ligne de comparaison si le fichier existe."""
    if not metrics_path.exists():
        return

    metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
    rows.append({"method": method, "protocol": protocol, **metrics})


def save_method_comparison(config):
    """Compare baseline et méthode adversariale quand les CSV existent."""
    main_reports_dir = PROJECT_ROOT / Path(config["paths"]["results_report"]).parent
    invariant_reports_dir = get_invariant_reports_dir(config)
    invariant_config = config.get("product_type_invariant", {})

    rows = []
    add_metrics_row(rows, main_reports_dir / "evaluation_metrics.csv", "baseline_cnn", "standard_split")
    add_metrics_row(
        rows,
        invariant_reports_dir
        / invariant_config.get("metrics_filename", "product_type_invariant_standard_metrics.csv"),
        "product_type_invariant_cnn",
        "standard_split",
    )
    add_metrics_row(
        rows,
        main_reports_dir / "unseen_category_evaluation_metrics.csv",
        "baseline_cnn",
        "unseen_category_split",
    )
    add_metrics_row(
        rows,
        invariant_reports_dir
        / invariant_config.get("unseen_metrics_filename", "product_type_invariant_unseen_metrics.csv"),
        "product_type_invariant_cnn",
        "unseen_category_split",
    )

    if len(rows) < 2:
        return

    comparison_df = pd.DataFrame(rows)
    comparison_path = invariant_reports_dir / invariant_config.get(
        "comparison_filename",
        "product_type_invariant_comparison.csv",
    )
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Comparaison sauvegardée : {comparison_path}")


def evaluate_product_type_invariant_model(config, split_name="standard"):
    """Évalue le modèle adversarial sur le split demandé."""
    paths_config = config["paths"]
    training_config = config["training"]
    output_settings = get_output_settings(config, split_name)

    dataset_path = PROJECT_ROOT / paths_config["raw_data_dir"]
    reports_dir = get_invariant_reports_dir(config)
    image_size = get_image_size(config)
    batch_size = int(training_config.get("batch_size", 32))
    random_seed = int(training_config.get("random_seed", 42))

    model_path = output_settings["model_path"]
    split_path = output_settings["split_path"]

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    if not split_path.exists():
        raise FileNotFoundError(f"Split introuvable : {split_path}. Lance d'abord l'entraînement.")

    test_set = load_split_file(split_path, dataset_path, split_name="test")
    test_dataset = create_image_dataset(
        test_set,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        random_seed=random_seed,
    )

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"GradientReversal": GradientReversal},
    )

    true_labels, predicted_labels, prediction_scores = collect_predictions(model, test_dataset)
    metrics = calculate_metrics(true_labels, predicted_labels, prediction_scores)

    confusion_matrix_df = pd.DataFrame(
        confusion_matrix(true_labels, predicted_labels, labels=[0, 1]),
        index=["actual_fresh", "actual_rotten"],
        columns=["predicted_fresh", "predicted_rotten"],
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(reports_dir / output_settings["metrics_filename"], index=False)
    confusion_matrix_df.to_csv(reports_dir / output_settings["confusion_matrix_filename"])

    if split_name == "unseen":
        product_type_metrics_df = calculate_metrics_by_product_type(test_set, true_labels, predicted_labels)
        product_type_metrics_df.to_csv(reports_dir / output_settings["product_type_metrics_filename"], index=False)

    print("Résultats sur le test_set :")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("Matrice de confusion :")
    print(confusion_matrix_df)

    save_method_comparison(config)


def parse_args():
    parser = argparse.ArgumentParser(description="Évalue le modèle product_type-invariant.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Chemin vers config.yaml.",
    )
    parser.add_argument(
        "--split",
        choices=["standard", "unseen"],
        default="standard",
        help="Protocole d'évaluation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    project_config = load_config(args.config)
    evaluate_product_type_invariant_model(project_config, split_name=args.split)
