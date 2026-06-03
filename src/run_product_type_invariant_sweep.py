"""Lance plusieurs expériences product_type-invariant sans écraser les rapports existants."""

import argparse
import copy
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from evaluate_product_type_invariant import evaluate_product_type_invariant_model
from feature_product_type_analysis import run_product_type_analysis
from train import load_config
from train_product_type_invariant import train_product_type_invariant_model


def format_weight(weight):
    """Transforme un poids en nom de dossier stable."""
    return f"{weight:g}".replace(".", "_")


def resolve_project_path(path):
    """Retourne un chemin absolu depuis la racine du projet."""
    path = Path(path)
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def apply_overrides(config, args):
    """Applique les paramètres Colab sans modifier config.yaml."""
    config = copy.deepcopy(config)

    if args.raw_data_dir is not None:
        config["paths"]["raw_data_dir"] = str(args.raw_data_dir)

    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)

    if args.batch_size is not None:
        config["training"]["batch_size"] = int(args.batch_size)

    if args.early_stopping_patience is not None:
        config["training"]["early_stopping_patience"] = int(args.early_stopping_patience)

    return config


def build_output_root(output_root):
    """Crée un dossier daté pour éviter d'écraser les expériences précédentes."""
    if output_root is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = PROJECT_ROOT / "reports" / "product_type_invariant_sweep" / run_name
    else:
        output_root = resolve_project_path(output_root)

    if output_root.exists():
        raise FileExistsError(
            f"Le dossier existe déjà : {output_root}. "
            "Choisissez un autre --output-root pour ne pas écraser les résultats."
        )

    output_root.mkdir(parents=True)
    return output_root


def build_experiment_config(base_config, weight, output_root):
    """Prépare une configuration isolée pour un poids adversarial."""
    config = copy.deepcopy(base_config)
    tag = format_weight(weight)
    reports_dir = output_root / f"weight_{tag}"

    invariant_config = config.setdefault("product_type_invariant", {})
    invariant_config["adversarial_weight"] = float(weight)
    invariant_config["reports_dir"] = str(reports_dir)

    # Chaque modèle a un nom distinct pour éviter d'écraser les modèles déjà entraînés.
    invariant_config["model_filename"] = f"product_type_invariant_weight_{tag}_standard.keras"
    invariant_config["unseen_model_filename"] = f"product_type_invariant_weight_{tag}_unseen.keras"

    return config, reports_dir


def get_splits(split_name):
    """Retourne les protocoles à lancer."""
    if split_name == "both":
        return ["standard", "unseen"]

    return [split_name]


def read_metrics_row(metrics_path):
    """Lit une ligne de métriques si le fichier existe."""
    if not metrics_path.exists():
        return None

    return pd.read_csv(metrics_path).iloc[0].to_dict()


def read_probe_metrics(probe_dir):
    """Lit les métriques de la sonde product_type si elles existent."""
    metrics_path = probe_dir / "product_type_probe_metrics.csv"
    if not metrics_path.exists():
        return {}

    metrics_df = pd.read_csv(metrics_path)
    test_rows = metrics_df[metrics_df["split"] == "test"]
    if test_rows.empty:
        return {}

    test_metrics = test_rows.iloc[0]
    return {
        "probe_accuracy": test_metrics["accuracy"],
        "probe_balanced_accuracy": test_metrics["balanced_accuracy"],
        "probe_macro_f1_score": test_metrics["macro_f1_score"],
        "probe_weighted_f1_score": test_metrics["weighted_f1_score"],
    }


def collect_summary_row(config, reports_dir, weight, protocol, probe_metrics):
    """Construit une ligne de synthèse pour le sweep."""
    invariant_config = config["product_type_invariant"]

    if protocol == "standard":
        metrics_filename = invariant_config.get("metrics_filename", "product_type_invariant_standard_metrics.csv")
    else:
        metrics_filename = invariant_config.get("unseen_metrics_filename", "product_type_invariant_unseen_metrics.csv")

    metrics = read_metrics_row(reports_dir / metrics_filename)
    if metrics is None:
        return None

    return {
        "adversarial_weight": weight,
        "protocol": protocol,
        **metrics,
        **probe_metrics,
    }


def run_weight_experiment(base_config, weight, splits, output_root, unseen_categories, run_probe):
    """Lance entraînement, évaluation et sonde pour un poids."""
    experiment_config, reports_dir = build_experiment_config(base_config, weight, output_root)
    summary_rows = []

    print(f"\n=== Poids adversarial {weight:g} ===")
    print(f"Rapports : {reports_dir}")

    for split_name in splits:
        print(f"\n--- Entraînement {split_name} ---")
        train_product_type_invariant_model(
            experiment_config,
            split_name=split_name,
            unseen_categories=unseen_categories,
        )

        print(f"\n--- Évaluation {split_name} ---")
        evaluate_product_type_invariant_model(experiment_config, split_name=split_name)

    probe_metrics = {}
    if run_probe and "standard" in splits:
        model_dir = PROJECT_ROOT / experiment_config["paths"]["model_dir"]
        model_path = model_dir / experiment_config["product_type_invariant"]["model_filename"]
        probe_dir = reports_dir / "product_type_probe"

        print("\n--- Sonde product_type ---")
        run_product_type_analysis(
            experiment_config,
            model_path=model_path,
            output_dir=probe_dir,
            feature_layer_name="freshness_embedding",
        )
        probe_metrics = read_probe_metrics(probe_dir)
    elif run_probe:
        print("Sonde ignorée : elle utilise le modèle standard, mais le split standard n'a pas été lancé.")

    for split_name in splits:
        row = collect_summary_row(experiment_config, reports_dir, weight, split_name, probe_metrics)
        if row is not None:
            summary_rows.append(row)

    return summary_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Teste plusieurs poids adversariaux pour la méthode product_type-invariant."
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--weights", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    parser.add_argument("--split", choices=["standard", "unseen", "both"], default="both")
    parser.add_argument("--unseen-categories", nargs="*", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--raw-data-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--run-probe", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    base_config = apply_overrides(load_config(args.config), args)
    output_root = build_output_root(args.output_root)
    splits = get_splits(args.split)

    all_rows = []
    for weight in args.weights:
        rows = run_weight_experiment(
            base_config=base_config,
            weight=weight,
            splits=splits,
            output_root=output_root,
            unseen_categories=args.unseen_categories,
            run_probe=args.run_probe,
        )
        all_rows.extend(rows)

        summary_df = pd.DataFrame(all_rows)
        summary_path = output_root / "product_type_invariant_sweep_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Synthèse mise à jour : {summary_path}")

    print("\nExpériences terminées.")
    print(f"Dossier de sortie : {output_root}")


if __name__ == "__main__":
    main()
