"""Incertitude par distance dans l'espace de représentation du CNN."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from split_strategy import load_split_file
from train import create_image_dataset, get_image_size
from uncertainty import (
    calculate_binary_metrics,
    get_dataset_path,
    get_predicted_labels,
    get_protocol_settings,
    get_reports_dir,
    load_config,
)


def load_protocol_sets(config, protocol):
    """Charge train, validation et test pour un protocole donné."""
    reports_dir = get_reports_dir(config)
    dataset_path = get_dataset_path(config)
    protocol_settings = get_protocol_settings(config, protocol)
    split_path = reports_dir / protocol_settings["split_filename"]

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split introuvable : {split_path}. Lance d'abord l'entraînement du protocole {protocol}."
        )

    training_set = load_split_file(split_path, dataset_path, split_name="train")
    validation_set = load_split_file(split_path, dataset_path, split_name="validation")
    test_set = load_split_file(split_path, dataset_path, split_name="test")

    return training_set, validation_set, test_set


def build_feature_extractor(model, layer_index):
    """Utilise une couche interne du CNN comme représentation de l'image."""
    try:
        feature_layer = model.layers[layer_index]
    except IndexError as error:
        raise ValueError(f"Couche introuvable avec layer_index={layer_index}.") from error

    return tf.keras.Model(inputs=model.inputs, outputs=feature_layer.output)


def collect_scores_and_features(model, feature_extractor, image_set, image_size, batch_size, random_seed):
    """Calcule les prédictions sigmoid et les features internes."""
    dataset = create_image_dataset(
        image_set,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        random_seed=random_seed,
    )

    prediction_scores = model.predict(dataset, verbose=0).reshape(-1)
    features = feature_extractor.predict(dataset, verbose=0).astype("float32")
    true_labels = image_set["label"].astype(int).to_numpy()

    return true_labels, prediction_scores, features


def fit_feature_scaler(training_features):
    """Standardise les features pour éviter qu'une dimension domine la distance."""
    feature_mean = training_features.mean(axis=0)
    feature_std = training_features.std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0

    return feature_mean, feature_std


def transform_features(features, feature_mean, feature_std):
    """Applique la standardisation calculée sur le training_set."""
    return (features - feature_mean) / feature_std


def fit_class_prototypes(training_features, training_labels):
    """Calcule un prototype fresh et un prototype rotten dans l'espace de features."""
    prototypes = {}

    for label in [0, 1]:
        label_features = training_features[training_labels == label]

        if len(label_features) == 0:
            raise ValueError(f"Aucune feature trouvée pour le label {label}.")

        prototypes[label] = label_features.mean(axis=0)

    return prototypes


def calculate_predicted_class_distances(features, predicted_labels, prototypes):
    """Mesure la distance au prototype de la classe prédite."""
    prototype_matrix = np.vstack([prototypes[int(label)] for label in predicted_labels])
    distances = np.linalg.norm(features - prototype_matrix, axis=1)
    return distances


def evaluate_with_distance_threshold(true_labels, prediction_scores, distance_scores, threshold):
    """Évalue le rejet uncertain basé sur la distance aux prototypes."""
    predicted_labels = get_predicted_labels(prediction_scores)
    accepted_mask = distance_scores <= threshold
    uncertain_mask = ~accepted_mask

    baseline_metrics = calculate_binary_metrics(true_labels, predicted_labels)
    accepted_count = int(accepted_mask.sum())
    uncertain_count = int(uncertain_mask.sum())
    image_count = int(len(true_labels))

    result = {
        "image_count": image_count,
        "threshold": threshold,
        "baseline_accuracy": baseline_metrics["accuracy"],
        "baseline_precision": baseline_metrics["precision"],
        "baseline_recall": baseline_metrics["recall"],
        "baseline_f1_score": baseline_metrics["f1_score"],
        "accepted_image_count": accepted_count,
        "uncertain_image_count": uncertain_count,
        "coverage": accepted_count / image_count if image_count else 0,
        "uncertain_rate": uncertain_count / image_count if image_count else 0,
        "mean_distance": float(np.mean(distance_scores)),
        "median_distance": float(np.median(distance_scores)),
    }

    if accepted_count == 0:
        result.update(
            {
                "accepted_accuracy": np.nan,
                "accepted_precision": np.nan,
                "accepted_recall": np.nan,
                "accepted_f1_score": np.nan,
            }
        )
    else:
        accepted_metrics = calculate_binary_metrics(true_labels[accepted_mask], predicted_labels[accepted_mask])
        result.update(
            {
                "accepted_accuracy": accepted_metrics["accuracy"],
                "accepted_precision": accepted_metrics["precision"],
                "accepted_recall": accepted_metrics["recall"],
                "accepted_f1_score": accepted_metrics["f1_score"],
            }
        )

    if uncertain_count == 0:
        result["uncertain_error_rate"] = np.nan
    else:
        # Un bon rejet doit contenir plus d'erreurs que les images acceptées.
        uncertain_errors = predicted_labels[uncertain_mask] != true_labels[uncertain_mask]
        result["uncertain_error_rate"] = float(uncertain_errors.mean())

    return result


def build_distance_threshold_grid(distance_scores):
    """Construit des seuils à partir des quantiles observés sur validation."""
    quantiles = np.round(np.arange(0.05, 1.001, 0.01), 4)
    thresholds = np.quantile(distance_scores, quantiles)
    return np.unique(np.round(thresholds, 6))


def calibrate_distance_threshold(config, true_labels, prediction_scores, distance_scores):
    """Choisit un seuil de distance avec le validation_set."""
    feature_distance_config = config.get("feature_distance_uncertainty", {})
    target_accuracy = float(feature_distance_config.get("target_accuracy", 0.95))
    minimum_coverage = float(feature_distance_config.get("minimum_coverage", 0.50))

    grid_rows = []
    for threshold in build_distance_threshold_grid(distance_scores):
        row = evaluate_with_distance_threshold(true_labels, prediction_scores, distance_scores, threshold)
        grid_rows.append(row)

    grid_df = pd.DataFrame(grid_rows)
    coverage_candidates = grid_df[grid_df["coverage"] >= minimum_coverage].copy()

    if coverage_candidates.empty:
        coverage_candidates = grid_df.copy()

    target_candidates = coverage_candidates[
        coverage_candidates["accepted_accuracy"] >= target_accuracy
    ].copy()

    if not target_candidates.empty:
        # Pour la distance, on garde la plus grande couverture qui atteint l'objectif.
        selected_row = target_candidates.sort_values(["coverage", "threshold"], ascending=[False, False]).iloc[0]
        selection_reason = "target_accuracy_reached"
    else:
        selected_row = coverage_candidates.sort_values(
            ["accepted_accuracy", "coverage"], ascending=[False, False]
        ).iloc[0]
        selection_reason = "best_available_accuracy"

    threshold_info = {
        "threshold": float(selected_row["threshold"]),
        "target_accuracy": target_accuracy,
        "minimum_coverage": minimum_coverage,
        "validation_accepted_accuracy": float(selected_row["accepted_accuracy"]),
        "validation_coverage": float(selected_row["coverage"]),
        "selection_reason": selection_reason,
    }

    return threshold_info, grid_df


def evaluate_by_product_type(test_set, prediction_scores, distance_scores, threshold, protocol):
    """Calcule les métriques de distance pour chaque product_type."""
    rows = []
    test_set = test_set.reset_index(drop=True)

    for product_type, product_df in test_set.groupby("product_type"):
        row_indices = product_df.index.to_numpy()
        true_labels = product_df["label"].astype(int).to_numpy()
        product_prediction_scores = prediction_scores[row_indices]
        product_distance_scores = distance_scores[row_indices]
        metrics = evaluate_with_distance_threshold(
            true_labels=true_labels,
            prediction_scores=product_prediction_scores,
            distance_scores=product_distance_scores,
            threshold=threshold,
        )
        rows.append({"protocol": protocol, "product_type": product_type, **metrics})

    return pd.DataFrame(rows).sort_values(["protocol", "product_type"])


def evaluate_protocol(config, protocol):
    """Lance l'analyse par distance pour un protocole."""
    training_config = config["training"]
    feature_distance_config = config.get("feature_distance_uncertainty", {})
    protocol_settings = get_protocol_settings(config, protocol)
    model_path = protocol_settings["model_path"]

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    image_size = get_image_size(config)
    batch_size = int(training_config.get("batch_size", 32))
    random_seed = int(training_config.get("random_seed", 42))
    layer_index = int(feature_distance_config.get("feature_layer_index", -3))

    training_set, validation_set, test_set = load_protocol_sets(config, protocol)
    model = tf.keras.models.load_model(model_path)
    feature_extractor = build_feature_extractor(model, layer_index)

    training_labels, _, training_features = collect_scores_and_features(
        model, feature_extractor, training_set, image_size, batch_size, random_seed
    )
    validation_labels, validation_scores, validation_features = collect_scores_and_features(
        model, feature_extractor, validation_set, image_size, batch_size, random_seed
    )
    test_labels, test_scores, test_features = collect_scores_and_features(
        model, feature_extractor, test_set, image_size, batch_size, random_seed
    )

    feature_mean, feature_std = fit_feature_scaler(training_features)
    scaled_training_features = transform_features(training_features, feature_mean, feature_std)
    scaled_validation_features = transform_features(validation_features, feature_mean, feature_std)
    scaled_test_features = transform_features(test_features, feature_mean, feature_std)

    prototypes = fit_class_prototypes(scaled_training_features, training_labels)

    validation_predicted_labels = get_predicted_labels(validation_scores)
    validation_distances = calculate_predicted_class_distances(
        scaled_validation_features,
        validation_predicted_labels,
        prototypes,
    )
    threshold_info, calibration_grid_df = calibrate_distance_threshold(
        config,
        validation_labels,
        validation_scores,
        validation_distances,
    )

    test_predicted_labels = get_predicted_labels(test_scores)
    test_distances = calculate_predicted_class_distances(
        scaled_test_features,
        test_predicted_labels,
        prototypes,
    )
    test_metrics = evaluate_with_distance_threshold(
        true_labels=test_labels,
        prediction_scores=test_scores,
        distance_scores=test_distances,
        threshold=threshold_info["threshold"],
    )

    product_type_metrics_df = evaluate_by_product_type(
        test_set=test_set,
        prediction_scores=test_scores,
        distance_scores=test_distances,
        threshold=threshold_info["threshold"],
        protocol=protocol,
    )

    metrics_row = {
        "protocol": protocol,
        "feature_layer_index": layer_index,
        **threshold_info,
        **test_metrics,
    }

    calibration_grid_df.insert(0, "protocol", protocol)

    return metrics_row, calibration_grid_df, product_type_metrics_df


def run_feature_distance_analysis(config, protocol):
    """Lance l'analyse et sauvegarde les résultats."""
    feature_distance_config = config.get("feature_distance_uncertainty", {})
    reports_dir = get_reports_dir(config)
    reports_dir.mkdir(parents=True, exist_ok=True)

    protocols = ["standard", "unseen"] if protocol == "both" else [protocol]

    metrics_rows = []
    calibration_frames = []
    product_type_frames = []

    for current_protocol in protocols:
        print(f"Analyse feature-distance : {current_protocol}")
        metrics_row, calibration_grid_df, product_type_metrics_df = evaluate_protocol(config, current_protocol)
        metrics_rows.append(metrics_row)
        calibration_frames.append(calibration_grid_df)
        product_type_frames.append(product_type_metrics_df)

    metrics_df = pd.DataFrame(metrics_rows)
    calibration_df = pd.concat(calibration_frames, ignore_index=True)
    product_type_df = pd.concat(product_type_frames, ignore_index=True)

    metrics_filename = feature_distance_config.get("metrics_filename", "feature_distance_metrics.csv")
    calibration_filename = feature_distance_config.get(
        "calibration_grid_filename", "feature_distance_calibration_grid.csv"
    )
    product_type_filename = feature_distance_config.get(
        "product_type_metrics_filename", "feature_distance_by_product_type.csv"
    )

    metrics_df.to_csv(reports_dir / metrics_filename, index=False)
    calibration_df.to_csv(reports_dir / calibration_filename, index=False)
    product_type_df.to_csv(reports_dir / product_type_filename, index=False)

    print(f"Métriques sauvegardées : {reports_dir / metrics_filename}")
    print(f"Grille de calibration sauvegardée : {reports_dir / calibration_filename}")
    print(f"Métriques par product_type sauvegardées : {reports_dir / product_type_filename}")
    print(metrics_df)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse l'incertitude par distance de features.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Chemin vers config.yaml.",
    )
    parser.add_argument(
        "--protocol",
        choices=["standard", "unseen", "both"],
        default="both",
        help="Protocole à analyser.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    project_config = load_config(args.config)
    run_feature_distance_analysis(project_config, protocol=args.protocol)
