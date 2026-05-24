"""Analyse d'incertitude calibrée pour FreshOrRotten."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from split_strategy import load_split_file
from train import create_image_dataset, get_image_size


def load_config(config_path):
    """Charge la configuration du projet."""
    if not config_path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def get_reports_dir(config):
    """Retourne le dossier où les résultats sont sauvegardés."""
    return PROJECT_ROOT / Path(config["paths"]["results_report"]).parent


def get_dataset_path(config):
    """Retourne le chemin du dataset local."""
    return PROJECT_ROOT / config["paths"]["raw_data_dir"]


def get_protocol_settings(config, protocol):
    """Retourne le modèle et le split associés au protocole demandé."""
    training_config = config["training"]
    generalization_config = config.get("generalization", {})
    model_dir = PROJECT_ROOT / config["paths"]["model_dir"]

    if protocol == "standard":
        return {
            "model_path": model_dir / training_config.get("model_filename", "baseline_model.keras"),
            "split_filename": training_config.get("split_filename", "standard_split.csv"),
        }

    return {
        "model_path": model_dir / generalization_config.get("unseen_model_filename", "baseline_model_unseen.keras"),
        "split_filename": generalization_config.get("unseen_split_filename", "unseen_category_split.csv"),
    }


def load_protocol_split(config, protocol):
    """Charge validation_set et test_set pour un protocole donné."""
    reports_dir = get_reports_dir(config)
    dataset_path = get_dataset_path(config)
    protocol_settings = get_protocol_settings(config, protocol)
    split_path = reports_dir / protocol_settings["split_filename"]

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split introuvable : {split_path}. Lance d'abord l'entraînement du protocole {protocol}."
        )

    validation_set = load_split_file(split_path, dataset_path, split_name="validation")
    test_set = load_split_file(split_path, dataset_path, split_name="test")

    return validation_set, test_set


def collect_prediction_scores(model, image_set, image_size, batch_size, random_seed):
    """Retourne les labels réels et les scores sigmoid du modèle."""
    dataset = create_image_dataset(
        image_set,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        random_seed=random_seed,
    )

    true_labels = image_set["label"].astype(int).to_numpy()
    prediction_scores = model.predict(dataset, verbose=0).reshape(-1)

    return true_labels, prediction_scores


def get_predicted_labels(prediction_scores):
    """Convertit les scores sigmoid en labels fresh ou rotten."""
    return (prediction_scores >= 0.5).astype(int)


def get_confidence_scores(prediction_scores):
    """Calcule la confiance associée à la classe prédite."""
    return np.maximum(prediction_scores, 1 - prediction_scores)


def calculate_binary_metrics(true_labels, predicted_labels):
    """Calcule les métriques classiques pour la classe positive rotten."""
    return {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, zero_division=0),
        "f1_score": f1_score(true_labels, predicted_labels, zero_division=0),
    }


def evaluate_with_threshold(true_labels, prediction_scores, threshold):
    """Évalue le modèle quand les prédictions peu confiantes deviennent uncertain."""
    predicted_labels = get_predicted_labels(prediction_scores)
    confidence_scores = get_confidence_scores(prediction_scores)
    accepted_mask = confidence_scores >= threshold
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
        # Ce taux indique si la règle uncertain capture vraiment des erreurs.
        uncertain_errors = predicted_labels[uncertain_mask] != true_labels[uncertain_mask]
        result["uncertain_error_rate"] = float(uncertain_errors.mean())

    return result


def build_threshold_grid(config):
    """Construit la grille de seuils testés sur validation."""
    uncertainty_config = config.get("uncertainty", {})
    start = float(uncertainty_config.get("threshold_grid_start", 0.50))
    end = float(uncertainty_config.get("threshold_grid_end", 0.99))
    step = float(uncertainty_config.get("threshold_step", 0.01))

    return np.round(np.arange(start, end + step / 2, step), 4)


def calibrate_threshold(config, true_labels, prediction_scores):
    """Choisit le seuil uncertain à partir du validation_set."""
    uncertainty_config = config.get("uncertainty", {})
    target_accuracy = float(uncertainty_config.get("target_accuracy", 0.90))
    minimum_coverage = float(uncertainty_config.get("minimum_coverage", 0.50))

    grid_rows = []
    for threshold in build_threshold_grid(config):
        row = evaluate_with_threshold(true_labels, prediction_scores, threshold)
        grid_rows.append(row)

    grid_df = pd.DataFrame(grid_rows)
    coverage_candidates = grid_df[grid_df["coverage"] >= minimum_coverage].copy()

    if coverage_candidates.empty:
        coverage_candidates = grid_df.copy()

    target_candidates = coverage_candidates[
        coverage_candidates["accepted_accuracy"] >= target_accuracy
    ].copy()

    if not target_candidates.empty:
        # On garde le seuil le plus bas qui atteint l'objectif pour éviter trop de rejets.
        selected_row = target_candidates.sort_values(["threshold"]).iloc[0]
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


def evaluate_by_product_type(test_set, prediction_scores, threshold, protocol):
    """Calcule les métriques uncertain pour chaque product_type."""
    rows = []
    test_set = test_set.reset_index(drop=True)

    for product_type, product_df in test_set.groupby("product_type"):
        row_indices = product_df.index.to_numpy()
        true_labels = product_df["label"].astype(int).to_numpy()
        product_scores = prediction_scores[row_indices]
        metrics = evaluate_with_threshold(true_labels, product_scores, threshold)
        rows.append({"protocol": protocol, "product_type": product_type, **metrics})

    return pd.DataFrame(rows).sort_values(["protocol", "product_type"])


def evaluate_protocol(config, protocol):
    """Lance l'analyse d'incertitude pour un protocole."""
    training_config = config["training"]
    reports_dir = get_reports_dir(config)
    protocol_settings = get_protocol_settings(config, protocol)
    model_path = protocol_settings["model_path"]

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    image_size = get_image_size(config)
    batch_size = int(training_config.get("batch_size", 32))
    random_seed = int(training_config.get("random_seed", 42))

    validation_set, test_set = load_protocol_split(config, protocol)
    model = tf.keras.models.load_model(model_path)

    validation_labels, validation_scores = collect_prediction_scores(
        model, validation_set, image_size, batch_size, random_seed
    )
    threshold_info, calibration_grid_df = calibrate_threshold(config, validation_labels, validation_scores)

    test_labels, test_scores = collect_prediction_scores(model, test_set, image_size, batch_size, random_seed)
    test_metrics = evaluate_with_threshold(test_labels, test_scores, threshold_info["threshold"])
    product_type_metrics_df = evaluate_by_product_type(
        test_set=test_set,
        prediction_scores=test_scores,
        threshold=threshold_info["threshold"],
        protocol=protocol,
    )

    metrics_row = {
        "protocol": protocol,
        **threshold_info,
        **test_metrics,
    }

    calibration_grid_df.insert(0, "protocol", protocol)

    return metrics_row, calibration_grid_df, product_type_metrics_df


def run_uncertainty_analysis(config, protocol):
    """Lance l'analyse et sauvegarde les CSV."""
    uncertainty_config = config.get("uncertainty", {})
    reports_dir = get_reports_dir(config)
    reports_dir.mkdir(parents=True, exist_ok=True)

    protocols = ["standard", "unseen"] if protocol == "both" else [protocol]

    metrics_rows = []
    calibration_frames = []
    product_type_frames = []

    for current_protocol in protocols:
        print(f"Analyse d'incertitude : {current_protocol}")
        metrics_row, calibration_grid_df, product_type_metrics_df = evaluate_protocol(config, current_protocol)
        metrics_rows.append(metrics_row)
        calibration_frames.append(calibration_grid_df)
        product_type_frames.append(product_type_metrics_df)

    metrics_df = pd.DataFrame(metrics_rows)
    calibration_df = pd.concat(calibration_frames, ignore_index=True)
    product_type_df = pd.concat(product_type_frames, ignore_index=True)

    metrics_filename = uncertainty_config.get("metrics_filename", "uncertainty_metrics.csv")
    calibration_filename = uncertainty_config.get(
        "calibration_grid_filename", "uncertainty_calibration_grid.csv"
    )
    product_type_filename = uncertainty_config.get(
        "product_type_metrics_filename", "uncertainty_by_product_type.csv"
    )

    metrics_df.to_csv(reports_dir / metrics_filename, index=False)
    calibration_df.to_csv(reports_dir / calibration_filename, index=False)
    product_type_df.to_csv(reports_dir / product_type_filename, index=False)

    print(f"Métriques sauvegardées : {reports_dir / metrics_filename}")
    print(f"Grille de calibration sauvegardée : {reports_dir / calibration_filename}")
    print(f"Métriques par product_type sauvegardées : {reports_dir / product_type_filename}")
    print(metrics_df)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse l'incertitude calibrée de FreshOrRotten.")
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
    run_uncertainty_analysis(project_config, protocol=args.protocol)
