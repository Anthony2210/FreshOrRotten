"""Analyse du biais product_type dans les features du CNN."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from feature_distance_uncertainty import build_feature_extractor
from split_strategy import load_split_file
from train import create_image_dataset, get_image_size
from uncertainty import get_dataset_path, get_protocol_settings, get_reports_dir, load_config


def load_standard_sets(config):
    """Charge les splits standard déjà sauvegardés."""
    reports_dir = get_reports_dir(config)
    dataset_path = get_dataset_path(config)
    split_filename = config["training"].get("split_filename", "standard_split.csv")
    split_path = reports_dir / split_filename

    if not split_path.exists():
        raise FileNotFoundError(f"Split standard introuvable : {split_path}")

    training_set = load_split_file(split_path, dataset_path, split_name="train")
    validation_set = load_split_file(split_path, dataset_path, split_name="validation")
    test_set = load_split_file(split_path, dataset_path, split_name="test")

    return training_set, validation_set, test_set


def collect_features(feature_extractor, image_set, image_size, batch_size, random_seed):
    """Extrait les features internes sans charger toutes les images à la main."""
    dataset = create_image_dataset(
        image_set,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        random_seed=random_seed,
    )

    features = feature_extractor.predict(dataset, verbose=0)
    product_types = image_set["product_type"].astype(str).to_numpy()

    return features, product_types


def train_product_type_probe(training_features, training_product_types, random_seed):
    """Entraîne un classifieur simple pour prédire product_type depuis les features."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        ),
    ).fit(training_features, training_product_types)


def calculate_majority_baseline(training_product_types, target_product_types):
    """Mesure la baseline qui prédit toujours le type majoritaire du train."""
    majority_product_type = pd.Series(training_product_types).mode().iloc[0]
    majority_accuracy = (target_product_types == majority_product_type).mean()

    return majority_product_type, float(majority_accuracy)


def evaluate_probe(split_name, model, features, product_types, training_product_types):
    """Calcule les métriques du probe sur un split."""
    predictions = model.predict(features)
    majority_product_type, majority_accuracy = calculate_majority_baseline(
        training_product_types,
        product_types,
    )

    return {
        "split": split_name,
        "image_count": len(product_types),
        "product_type_count": len(pd.unique(product_types)),
        "majority_product_type": majority_product_type,
        "majority_baseline_accuracy": majority_accuracy,
        "accuracy": accuracy_score(product_types, predictions),
        "balanced_accuracy": balanced_accuracy_score(product_types, predictions),
        "macro_f1_score": f1_score(product_types, predictions, average="macro", zero_division=0),
        "weighted_f1_score": f1_score(product_types, predictions, average="weighted", zero_division=0),
    }


def build_product_type_report(product_types, predictions):
    """Construit les métriques détaillées par product_type."""
    report = classification_report(
        product_types,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for product_type, metrics in report.items():
        if product_type in {"accuracy", "macro avg", "weighted avg"}:
            continue

        rows.append(
            {
                "product_type": product_type,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1-score"],
                "support": int(metrics["support"]),
            }
        )

    return pd.DataFrame(rows).sort_values("product_type")


def build_confusion_matrix_df(product_types, predictions):
    """Construit une matrice de confusion lisible par product_type."""
    labels = sorted(pd.unique(product_types))
    matrix = confusion_matrix(product_types, predictions, labels=labels)
    confusion_df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )

    return confusion_df


def run_product_type_analysis(config):
    """Lance l'analyse de dépendance au product_type."""
    training_config = config["training"]
    analysis_config = config.get("product_type_bias_analysis", {})
    reports_dir = get_reports_dir(config)
    reports_dir.mkdir(parents=True, exist_ok=True)

    protocol_settings = get_protocol_settings(config, protocol="standard")
    model_path = protocol_settings["model_path"]

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    image_size = get_image_size(config)
    batch_size = int(training_config.get("batch_size", 32))
    random_seed = int(training_config.get("random_seed", 42))
    layer_index = int(analysis_config.get("feature_layer_index", -3))

    training_set, validation_set, test_set = load_standard_sets(config)
    model = tf.keras.models.load_model(model_path)
    feature_extractor = build_feature_extractor(model, layer_index)

    print("Extraction des features : train")
    training_features, training_product_types = collect_features(
        feature_extractor,
        training_set,
        image_size,
        batch_size,
        random_seed,
    )

    print("Extraction des features : validation")
    validation_features, validation_product_types = collect_features(
        feature_extractor,
        validation_set,
        image_size,
        batch_size,
        random_seed,
    )

    print("Extraction des features : test")
    test_features, test_product_types = collect_features(
        feature_extractor,
        test_set,
        image_size,
        batch_size,
        random_seed,
    )

    probe = train_product_type_probe(training_features, training_product_types, random_seed)

    validation_predictions = probe.predict(validation_features)
    test_predictions = probe.predict(test_features)

    metrics_df = pd.DataFrame(
        [
            {
                "feature_layer_index": layer_index,
                **evaluate_probe(
                    "validation",
                    probe,
                    validation_features,
                    validation_product_types,
                    training_product_types,
                ),
            },
            {
                "feature_layer_index": layer_index,
                **evaluate_probe(
                    "test",
                    probe,
                    test_features,
                    test_product_types,
                    training_product_types,
                ),
            },
        ]
    )

    product_type_df = build_product_type_report(test_product_types, test_predictions)
    confusion_df = build_confusion_matrix_df(test_product_types, test_predictions)

    metrics_filename = analysis_config.get("metrics_filename", "product_type_probe_metrics.csv")
    product_type_filename = analysis_config.get(
        "product_type_metrics_filename",
        "product_type_probe_by_product_type.csv",
    )
    confusion_filename = analysis_config.get(
        "confusion_matrix_filename",
        "product_type_probe_confusion_matrix.csv",
    )

    metrics_df.to_csv(reports_dir / metrics_filename, index=False)
    product_type_df.to_csv(reports_dir / product_type_filename, index=False)
    confusion_df.to_csv(reports_dir / confusion_filename)

    print(f"Métriques sauvegardées : {reports_dir / metrics_filename}")
    print(f"Métriques par product_type sauvegardées : {reports_dir / product_type_filename}")
    print(f"Matrice de confusion sauvegardée : {reports_dir / confusion_filename}")
    print(metrics_df)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse si les features du CNN encodent product_type.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Chemin vers config.yaml.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    project_config = load_config(args.config)
    run_product_type_analysis(project_config)
