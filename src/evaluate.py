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

from split_strategy import (
    create_standard_split,
    create_unseen_category_split,
    load_split_file,
    normalize_category_list,
    save_split_file,
)
from train import create_image_dataset, get_image_size, load_config, scan_image_files


def load_standard_test_set(config, reports_dir, dataset_path):
    """Charge le test_set classique depuis le split sauvegardé ou le recrée."""
    training_config = config["training"]
    split_filename = training_config.get("split_filename", "standard_split.csv")
    split_path = reports_dir / split_filename

    if split_path.exists():
        return load_split_file(split_path, dataset_path, split_name="test")

    print("Aucun split sauvegardé trouvé. Le standard_split est recréé depuis config.yaml.")

    image_index_df = scan_image_files(dataset_path, config["data"]["labels"])
    training_set, validation_set, test_set = create_standard_split(
        image_index_df=image_index_df,
        validation_size=float(training_config.get("validation_size", 0.2)),
        test_size=float(training_config.get("test_size", 0.2)),
        random_seed=int(training_config.get("random_seed", 42)),
    )

    save_split_file(
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        reports_dir=reports_dir,
        dataset_path=dataset_path,
        split_filename=split_filename,
    )

    return test_set


def get_unseen_categories(config, selected_categories):
    """Lit les catégories non vues depuis la CLI ou config.yaml."""
    if selected_categories:
        return normalize_category_list(selected_categories)

    generalization_config = config.get("generalization", {})
    return normalize_category_list(generalization_config.get("unseen_categories", []))


def load_unseen_category_test_set(config, reports_dir, dataset_path, selected_categories):
    """Crée le test_set avec uniquement les catégories non vues."""
    training_config = config["training"]
    generalization_config = config.get("generalization", {})
    split_filename = generalization_config.get("unseen_split_filename", "unseen_category_split.csv")
    unseen_categories = get_unseen_categories(config, selected_categories)

    if not unseen_categories:
        split_path = reports_dir / split_filename

        if split_path.exists():
            print("Aucune catégorie fournie. Le split unseen sauvegardé est utilisé.")
            return load_split_file(split_path, dataset_path, split_name="test")

        raise ValueError(
            "Ajoutez des catégories avec --unseen-categories ou dans generalization.unseen_categories."
        )

    image_index_df = scan_image_files(dataset_path, config["data"]["labels"])
    training_set, validation_set, test_set = create_unseen_category_split(
        image_index_df=image_index_df,
        unseen_categories=unseen_categories,
        validation_size=float(training_config.get("validation_size", 0.2)),
        random_seed=int(training_config.get("random_seed", 42)),
    )

    save_split_file(
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        reports_dir=reports_dir,
        dataset_path=dataset_path,
        split_filename=split_filename,
    )

    print(f"Catégories non vues : {', '.join(unseen_categories)}")
    print(f"Images train : {len(training_set)}")
    print(f"Images validation : {len(validation_set)}")
    print(f"Images test unseen : {len(test_set)}")

    return test_set


def collect_predictions(model, test_dataset):
    """Récupère les labels réels et les prédictions du modèle."""
    true_labels = []

    for _, labels in test_dataset:
        true_labels.extend(labels.numpy())

    prediction_scores = model.predict(test_dataset).reshape(-1)
    predicted_labels = (prediction_scores >= 0.5).astype(int)

    return np.array(true_labels).astype(int), predicted_labels, prediction_scores


def calculate_metrics(true_labels, predicted_labels, loss=None):
    """Calcule les métriques principales."""
    metrics = {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, zero_division=0),
        "f1_score": f1_score(true_labels, predicted_labels, zero_division=0),
    }

    if loss is not None:
        metrics = {"loss": loss, **metrics}

    return metrics


def calculate_metrics_by_product_type(test_set, true_labels, predicted_labels):
    """Calcule les métriques pour chaque product_type."""
    prediction_df = test_set[["product_type"]].copy().reset_index(drop=True)
    prediction_df["true_label"] = true_labels
    prediction_df["predicted_label"] = predicted_labels

    category_results = []

    for product_type, product_df in prediction_df.groupby("product_type"):
        true_category_labels = product_df["true_label"]
        predicted_category_labels = product_df["predicted_label"]

        category_results.append(
            {
                "product_type": product_type,
                "image_count": len(product_df),
                **calculate_metrics(true_category_labels, predicted_category_labels),
            }
        )

    return pd.DataFrame(category_results).sort_values("product_type")


def save_comparison_report(reports_dir, unseen_metrics, generalization_config):
    """Compare les résultats unseen avec le split classique si disponible."""
    standard_metrics_path = reports_dir / "evaluation_metrics.csv"
    comparison_filename = generalization_config.get("comparison_filename", "evaluation_comparison.csv")

    if not standard_metrics_path.exists():
        return None

    standard_metrics = pd.read_csv(standard_metrics_path).iloc[0].to_dict()
    comparison_df = pd.DataFrame(
        [
            {"protocol": "standard_split", **standard_metrics},
            {"protocol": "unseen_category_split", **unseen_metrics},
        ]
    )
    comparison_df.to_csv(reports_dir / comparison_filename, index=False)

    return comparison_df


def evaluate_model(config, model_path=None, split_name="standard", unseen_categories=None):
    """Évalue le modèle sur le test_set demandé."""
    paths_config = config["paths"]
    training_config = config["training"]
    generalization_config = config.get("generalization", {})

    dataset_path = PROJECT_ROOT / paths_config["raw_data_dir"]
    reports_dir = PROJECT_ROOT / Path(paths_config["results_report"]).parent
    model_dir = PROJECT_ROOT / paths_config["model_dir"]

    if model_path is None and split_name == "unseen":
        model_path = model_dir / generalization_config.get("unseen_model_filename", "baseline_model_unseen.keras")
    elif model_path is None:
        model_path = model_dir / training_config.get("model_filename", "baseline_model.keras")

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    image_size = get_image_size(config)
    batch_size = int(training_config.get("batch_size", 32))
    random_seed = int(training_config.get("random_seed", 42))

    if split_name == "unseen":
        test_set = load_unseen_category_test_set(config, reports_dir, dataset_path, unseen_categories)
        metrics_filename = generalization_config.get(
            "unseen_metrics_filename", "unseen_category_evaluation_metrics.csv"
        )
        confusion_matrix_filename = generalization_config.get(
            "unseen_confusion_matrix_filename", "unseen_category_confusion_matrix.csv"
        )
        category_metrics_filename = generalization_config.get(
            "unseen_category_metrics_filename", "unseen_category_metrics_by_product_type.csv"
        )
    else:
        test_set = load_standard_test_set(config, reports_dir, dataset_path)
        metrics_filename = "evaluation_metrics.csv"
        confusion_matrix_filename = "confusion_matrix.csv"
        category_metrics_filename = None

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

    metrics = calculate_metrics(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        loss=evaluation_result.get("loss"),
    )

    confusion_matrix_df = pd.DataFrame(
        confusion_matrix(true_labels, predicted_labels, labels=[0, 1]),
        index=["actual_fresh", "actual_rotten"],
        columns=["predicted_fresh", "predicted_rotten"],
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(reports_dir / metrics_filename, index=False)
    confusion_matrix_df.to_csv(reports_dir / confusion_matrix_filename)

    print("Résultats sur le test_set :")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("Matrice de confusion :")
    print(confusion_matrix_df)

    if split_name == "unseen":
        category_metrics_df = calculate_metrics_by_product_type(test_set, true_labels, predicted_labels)
        category_metrics_df.to_csv(reports_dir / category_metrics_filename, index=False)

        print("Résultats par product_type :")
        print(category_metrics_df)

        comparison_df = save_comparison_report(reports_dir, metrics, generalization_config)

        if comparison_df is not None:
            print("Comparaison avec le standard_split :")
            print(comparison_df)


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
    parser.add_argument(
        "--split",
        choices=["standard", "unseen"],
        default="standard",
        help="Protocole d'évaluation.",
    )
    parser.add_argument(
        "--unseen-categories",
        nargs="*",
        default=None,
        help="Catégories à retirer du train pour le test unseen.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    project_config = load_config(args.config)
    evaluate_model(
        project_config,
        model_path=args.model_path,
        split_name=args.split,
        unseen_categories=args.unseen_categories,
    )
