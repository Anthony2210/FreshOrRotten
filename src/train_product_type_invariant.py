"""Entraînement d'un CNN qui réduit la dépendance au product_type."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import tensorflow as tf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from model import build_product_type_invariant_model
from split_strategy import create_standard_split, create_unseen_category_split, normalize_category_list, save_split_file
from train import get_image_size, load_config, load_image, scan_image_files


def get_unseen_categories(config, selected_categories):
    """Lit les catégories non vues depuis la CLI ou config.yaml."""
    if selected_categories:
        return normalize_category_list(selected_categories)

    generalization_config = config.get("generalization", {})
    return normalize_category_list(generalization_config.get("unseen_categories", []))


def encode_product_types(training_set, validation_set):
    """Encode les product_type présents dans le training_set."""
    product_types = sorted(training_set["product_type"].astype(str).unique())
    product_type_to_label = {product_type: index for index, product_type in enumerate(product_types)}

    training_set = training_set.copy()
    validation_set = validation_set.copy()

    training_set["product_type_label"] = training_set["product_type"].map(product_type_to_label)
    validation_set["product_type_label"] = validation_set["product_type"].map(product_type_to_label)

    if validation_set["product_type_label"].isna().any():
        missing_product_types = sorted(
            validation_set.loc[validation_set["product_type_label"].isna(), "product_type"].unique()
        )
        raise ValueError(f"product_type absent du training_set : {missing_product_types}")

    training_set["product_type_label"] = training_set["product_type_label"].astype("int32")
    validation_set["product_type_label"] = validation_set["product_type_label"].astype("int32")

    mapping_df = pd.DataFrame(
        [
            {"product_type": product_type, "product_type_label": label}
            for product_type, label in product_type_to_label.items()
        ]
    )

    return training_set, validation_set, mapping_df


def load_invariant_image(image_path, freshness_label, product_type_label, image_size):
    """Charge une image avec les deux labels utilisés pendant l'entraînement."""
    image, freshness_label = load_image(image_path, freshness_label, image_size)

    return image, {
        "freshness": freshness_label,
        "product_type": product_type_label,
    }


def create_invariant_dataset(image_index_df, image_size, batch_size, shuffle=False, random_seed=42):
    """Crée un Dataset TensorFlow avec freshness et product_type."""
    image_paths = image_index_df["image_path"].astype(str).tolist()
    freshness_labels = image_index_df["label"].astype("float32").tolist()
    product_type_labels = image_index_df["product_type_label"].astype("int32").tolist()

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, freshness_labels, product_type_labels))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=random_seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda image_path, freshness_label, product_type_label: load_invariant_image(
            image_path,
            freshness_label,
            product_type_label,
            image_size,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def get_output_settings(config, split_name):
    """Retourne les noms de fichiers pour l'expérience adversariale."""
    invariant_config = config.get("product_type_invariant", {})

    if split_name == "unseen":
        return {
            "model_filename": invariant_config.get(
                "unseen_model_filename",
                "product_type_invariant_unseen.keras",
            ),
            "history_filename": invariant_config.get(
                "unseen_history_filename",
                "product_type_invariant_unseen_history.csv",
            ),
            "split_filename": invariant_config.get(
                "unseen_split_filename",
                "product_type_invariant_unseen_split.csv",
            ),
            "mapping_filename": invariant_config.get(
                "unseen_mapping_filename",
                "product_type_invariant_unseen_mapping.csv",
            ),
        }

    return {
        "model_filename": invariant_config.get("model_filename", "product_type_invariant_model.keras"),
        "history_filename": invariant_config.get("history_filename", "product_type_invariant_history.csv"),
        "split_filename": invariant_config.get("split_filename", "product_type_invariant_standard_split.csv"),
        "mapping_filename": invariant_config.get("mapping_filename", "product_type_invariant_mapping.csv"),
    }


def train_product_type_invariant_model(config, split_name="standard", unseen_categories=None):
    """Lance l'entraînement du modèle product_type-invariant."""
    paths_config = config["paths"]
    training_config = config["training"]
    invariant_config = config.get("product_type_invariant", {})

    dataset_path = PROJECT_ROOT / paths_config["raw_data_dir"]
    model_dir = PROJECT_ROOT / paths_config["model_dir"]
    reports_dir = PROJECT_ROOT / invariant_config.get("reports_dir", "reports/product_type_invariant")

    image_size = get_image_size(config)
    batch_size = int(training_config.get("batch_size", 32))
    epochs = int(training_config.get("epochs", 20))
    random_seed = int(training_config.get("random_seed", 42))
    validation_size = float(training_config.get("validation_size", 0.2))
    test_size = float(training_config.get("test_size", 0.2))
    learning_rate = float(training_config.get("learning_rate", 0.001))
    dropout_rate = float(training_config.get("dropout_rate", 0.3))
    early_stopping_patience = int(training_config.get("early_stopping_patience", 5))
    adversarial_weight = float(invariant_config.get("adversarial_weight", 0.1))

    output_settings = get_output_settings(config, split_name)

    tf.keras.utils.set_random_seed(random_seed)
    image_index_df = scan_image_files(dataset_path, config["data"]["labels"])

    if split_name == "unseen":
        selected_unseen_categories = get_unseen_categories(config, unseen_categories)

        if not selected_unseen_categories:
            raise ValueError(
                "Ajoutez des catégories avec --unseen-categories ou dans generalization.unseen_categories."
            )

        training_set, validation_set, test_set = create_unseen_category_split(
            image_index_df=image_index_df,
            unseen_categories=selected_unseen_categories,
            validation_size=validation_size,
            random_seed=random_seed,
        )
        print(f"Catégories non vues : {', '.join(selected_unseen_categories)}")
    else:
        training_set, validation_set, test_set = create_standard_split(
            image_index_df=image_index_df,
            validation_size=validation_size,
            test_size=test_size,
            random_seed=random_seed,
        )

    training_set, validation_set, mapping_df = encode_product_types(training_set, validation_set)

    print(f"Images train : {len(training_set)}")
    print(f"Images validation : {len(validation_set)}")
    print(f"Images test : {len(test_set)}")
    print(f"product_type vus pendant l'entraînement : {len(mapping_df)}")
    print(f"Poids adversarial : {adversarial_weight}")

    training_dataset = create_invariant_dataset(
        training_set,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        random_seed=random_seed,
    )
    validation_dataset = create_invariant_dataset(
        validation_set,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        random_seed=random_seed,
    )

    model = build_product_type_invariant_model(
        image_size=image_size,
        product_type_count=len(mapping_df),
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        adversarial_weight=adversarial_weight,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / output_settings["model_filename"]

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_freshness_loss",
            mode="min",
            patience=early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_freshness_loss",
            mode="min",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )

    history_df = pd.DataFrame(history.history)
    history_df.index = history_df.index + 1
    history_df.index.name = "epoch"
    history_df.to_csv(reports_dir / output_settings["history_filename"])

    mapping_df.to_csv(reports_dir / output_settings["mapping_filename"], index=False)

    save_split_file(
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        reports_dir=reports_dir,
        dataset_path=dataset_path,
        split_filename=output_settings["split_filename"],
    )

    print(f"Meilleur modèle sauvegardé : {model_path}")
    print(f"Historique sauvegardé : {reports_dir / output_settings['history_filename']}")
    print(f"Split sauvegardé : {reports_dir / output_settings['split_filename']}")
    print(f"Mapping product_type sauvegardé : {reports_dir / output_settings['mapping_filename']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Entraîne un CNN moins dépendant au product_type.")
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
        help="Protocole d'entraînement.",
    )
    parser.add_argument(
        "--unseen-categories",
        nargs="*",
        default=None,
        help="Catégories à retirer du train pour le protocole unseen.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    project_config = load_config(args.config)
    train_product_type_invariant_model(
        project_config,
        split_name=args.split,
        unseen_categories=args.unseen_categories,
    )
