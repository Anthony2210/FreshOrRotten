"""Analyse d'incertitude hybride pour FreshOrRotten."""

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

from feature_distance_uncertainty import (
    build_feature_extractor,
    calculate_predicted_class_distances,
    collect_scores_and_features,
    fit_class_prototypes,
    fit_feature_scaler,
    load_protocol_sets,
    transform_features,
)
from train import get_image_size
from uncertainty import (
    calculate_binary_metrics,
    get_confidence_scores,
    get_predicted_labels,
    get_protocol_settings,
    get_reports_dir,
    load_config,
)


def build_confidence_threshold_grid(config):
    """Construit les seuils de confiance testés sur validation."""
    hybrid_config = config.get("hybrid_uncertainty", {})
    start = float(hybrid_config.get("confidence_threshold_start", 0.50))
    end = float(hybrid_config.get("confidence_threshold_end", 0.99))
    step = float(hybrid_config.get("confidence_threshold_step", 0.01))

    return np.round(np.arange(start, end + step / 2, step), 4)


def build_distance_threshold_grid(config, distance_scores):
    """Construit les seuils de distance à partir des quantiles de validation."""
    hybrid_config = config.get("hybrid_uncertainty", {})
    start = float(hybrid_config.get("distance_quantile_start", 0.10))
    end = float(hybrid_config.get("distance_quantile_end", 1.00))
    step = float(hybrid_config.get("distance_quantile_step", 0.02))

    quantiles = np.round(np.arange(start, end + step / 2, step), 4)
    thresholds = np.quantile(distance_scores, quantiles)
    return np.unique(np.round(thresholds, 6))


def evaluate_with_hybrid_threshold(
    true_labels,
    prediction_scores,
    distance_scores,
    confidence_threshold,
    distance_threshold,
):
    """Évalue le rejet uncertain avec confiance sigmoid et distance de features."""
    predicted_labels = get_predicted_labels(prediction_scores)
    confidence_scores = get_confidence_scores(prediction_scores)

    # Une image est acceptée seulement si les deux signaux sont favorables.
    accepted_mask = (confidence_scores >= confidence_threshold) & (distance_scores <= distance_threshold)
    uncertain_mask = ~accepted_mask

    baseline_metrics = calculate_binary_metrics(true_labels, predicted_labels)
    accepted_count = int(accepted_mask.sum())
    uncertain_count = int(uncertain_mask.sum())
    image_count = int(len(true_labels))

    result = {
        "image_count": image_count,
        "confidence_threshold": confidence_threshold,
        "distance_threshold": distance_threshold,
        "baseline_accuracy": baseline_metrics["accuracy"],
        "baseline_precision": baseline_metrics["precision"],
        "baseline_recall": baseline_metrics["recall"],
        "baseline_f1_score": baseline_metrics["f1_score"],
        "accepted_image_count": accepted_count,
        "uncertain_image_count": uncertain_count,
        "coverage": accepted_count / image_count if image_count else 0,
        "uncertain_rate": uncertain_count / image_count if image_count else 0,
        "mean_confidence": float(np.mean(confidence_scores)),
        "mean_distance": float(np.mean(distance_scores)),
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
        # Ce taux mesure si le rejet capture vraiment des erreurs.
        uncertain_errors = predicted_labels[uncertain_mask] != true_labels[uncertain_mask]
        result["uncertain_error_rate"] = float(uncertain_errors.mean())

    return result


def calibrate_hybrid_thresholds(config, true_labels, prediction_scores, distance_scores):
    """Choisit les deux seuils à partir du validation_set."""
    hybrid_config = config.get("hybrid_uncertainty", {})
    target_accuracy = float(hybrid_config.get("target_accuracy", 0.95))
    minimum_coverage = float(hybrid_config.get("minimum_coverage", 0.50))

    grid_rows = []
    confidence_thresholds = build_confidence_threshold_grid(config)
    distance_thresholds = build_distance_threshold_grid(config, distance_scores)

    for confidence_threshold in confidence_thresholds:
        for distance_threshold in distance_thresholds:
            row = evaluate_with_hybrid_threshold(
                true_labels=true_labels,
                prediction_scores=prediction_scores,
                distance_scores=distance_scores,
                confidence_threshold=confidence_threshold,
                distance_threshold=distance_threshold,
            )
            grid_rows.append(row)

    grid_df = pd.DataFrame(grid_rows)
    coverage_candidates = grid_df[grid_df["coverage"] >= minimum_coverage].copy()

    if coverage_candidates.empty:
        coverage_candidates = grid_df.copy()

    target_candidates = coverage_candidates[
        coverage_candidates["accepted_accuracy"] >= target_accuracy
    ].copy()

    if not target_candidates.empty:
        # On garde la plus grande couverture parmi les seuils qui atteignent l'objectif.
        selected_row = target_candidates.sort_values(
            ["coverage", "accepted_accuracy"],
            ascending=[False, False],
        ).iloc[0]
        selection_reason = "target_accuracy_reached"
    else:
        selected_row = coverage_candidates.sort_values(
            ["accepted_accuracy", "coverage"],
            ascending=[False, False],
        ).iloc[0]
        selection_reason = "best_available_accuracy"

    threshold_info = {
        "confidence_threshold": float(selected_row["confidence_threshold"]),
        "distance_threshold": float(selected_row["distance_threshold"]),
        "target_accuracy": target_accuracy,
        "minimum_coverage": minimum_coverage,
        "validation_accepted_accuracy": float(selected_row["accepted_accuracy"]),
        "validation_coverage": float(selected_row["coverage"]),
        "selection_reason": selection_reason,
    }

    return threshold_info, grid_df


def prepare_protocol_scores(config, protocol):
    """Calcule scores sigmoid et distances de features pour un protocole."""
    training_config = config["training"]
    hybrid_config = config.get("hybrid_uncertainty", {})
    protocol_settings = get_protocol_settings(config, protocol)
    model_path = protocol_settings["model_path"]

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    image_size = get_image_size(config)
    batch_size = int(training_config.get("batch_size", 32))
    random_seed = int(training_config.get("random_seed", 42))
    layer_index = int(hybrid_config.get("feature_layer_index", -3))

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

    test_predicted_labels = get_predicted_labels(test_scores)
    test_distances = calculate_predicted_class_distances(
        scaled_test_features,
        test_predicted_labels,
        prototypes,
    )

    return {
        "layer_index": layer_index,
        "validation_labels": validation_labels,
        "validation_scores": validation_scores,
        "validation_distances": validation_distances,
        "test_set": test_set,
        "test_labels": test_labels,
        "test_scores": test_scores,
        "test_distances": test_distances,
    }


def evaluate_by_product_type(test_set, prediction_scores, distance_scores, threshold_info, protocol):
    """Calcule les métriques hybrides pour chaque product_type."""
    rows = []
    test_set = test_set.reset_index(drop=True)

    for product_type, product_df in test_set.groupby("product_type"):
        row_indices = product_df.index.to_numpy()
        true_labels = product_df["label"].astype(int).to_numpy()
        product_scores = prediction_scores[row_indices]
        product_distances = distance_scores[row_indices]

        metrics = evaluate_with_hybrid_threshold(
            true_labels=true_labels,
            prediction_scores=product_scores,
            distance_scores=product_distances,
            confidence_threshold=threshold_info["confidence_threshold"],
            distance_threshold=threshold_info["distance_threshold"],
        )
        rows.append({"protocol": protocol, "product_type": product_type, **metrics})

    return pd.DataFrame(rows).sort_values(["protocol", "product_type"])


def evaluate_protocol(config, protocol):
    """Lance l'analyse hybride pour un protocole."""
    prepared_scores = prepare_protocol_scores(config, protocol)

    threshold_info, calibration_grid_df = calibrate_hybrid_thresholds(
        config=config,
        true_labels=prepared_scores["validation_labels"],
        prediction_scores=prepared_scores["validation_scores"],
        distance_scores=prepared_scores["validation_distances"],
    )

    test_metrics = evaluate_with_hybrid_threshold(
        true_labels=prepared_scores["test_labels"],
        prediction_scores=prepared_scores["test_scores"],
        distance_scores=prepared_scores["test_distances"],
        confidence_threshold=threshold_info["confidence_threshold"],
        distance_threshold=threshold_info["distance_threshold"],
    )

    product_type_metrics_df = evaluate_by_product_type(
        test_set=prepared_scores["test_set"],
        prediction_scores=prepared_scores["test_scores"],
        distance_scores=prepared_scores["test_distances"],
        threshold_info=threshold_info,
        protocol=protocol,
    )

    metrics_row = {
        "protocol": protocol,
        "feature_layer_index": prepared_scores["layer_index"],
        **threshold_info,
        **test_metrics,
    }

    calibration_grid_df.insert(0, "protocol", protocol)

    return metrics_row, calibration_grid_df, product_type_metrics_df


def run_hybrid_analysis(config, protocol):
    """Lance l'analyse hybride et sauvegarde les CSV."""
    hybrid_config = config.get("hybrid_uncertainty", {})
    reports_dir = get_reports_dir(config)
    reports_dir.mkdir(parents=True, exist_ok=True)

    protocols = ["standard", "unseen"] if protocol == "both" else [protocol]

    metrics_rows = []
    calibration_frames = []
    product_type_frames = []

    for current_protocol in protocols:
        print(f"Analyse hybride : {current_protocol}")
        metrics_row, calibration_grid_df, product_type_metrics_df = evaluate_protocol(config, current_protocol)
        metrics_rows.append(metrics_row)
        calibration_frames.append(calibration_grid_df)
        product_type_frames.append(product_type_metrics_df)

    metrics_df = pd.DataFrame(metrics_rows)
    calibration_df = pd.concat(calibration_frames, ignore_index=True)
    product_type_df = pd.concat(product_type_frames, ignore_index=True)

    metrics_filename = hybrid_config.get("metrics_filename", "hybrid_uncertainty_metrics.csv")
    calibration_filename = hybrid_config.get(
        "calibration_grid_filename", "hybrid_uncertainty_calibration_grid.csv"
    )
    product_type_filename = hybrid_config.get(
        "product_type_metrics_filename", "hybrid_uncertainty_by_product_type.csv"
    )

    metrics_df.to_csv(reports_dir / metrics_filename, index=False)
    calibration_df.to_csv(reports_dir / calibration_filename, index=False)
    product_type_df.to_csv(reports_dir / product_type_filename, index=False)

    print(f"Métriques sauvegardées : {reports_dir / metrics_filename}")
    print(f"Grille de calibration sauvegardée : {reports_dir / calibration_filename}")
    print(f"Métriques par product_type sauvegardées : {reports_dir / product_type_filename}")
    print(metrics_df)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse l'incertitude hybride de FreshOrRotten.")
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
    run_hybrid_analysis(project_config, protocol=args.protocol)
