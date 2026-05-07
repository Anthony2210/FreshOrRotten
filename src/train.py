"""Entraînement du baseline_model FreshOrRotten."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from model import build_baseline_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(config_path):
    """Charge la configuration du projet."""
    if not config_path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def get_image_size(config):
    """Retourne la taille attendue par TensorFlow : hauteur, largeur."""
    image_config = config["data"]["image_size"]
    return (int(image_config["height"]), int(image_config["width"]))


def normalize_name(name):
    """Normalise un nom de dossier pour détecter les labels."""
    return name.lower().strip().replace("-", "_").replace(" ", "_")


def detect_freshness(path_parts):
    """Déduit fresh ou rotten à partir du chemin de l'image."""
    normalized_parts = [normalize_name(part) for part in path_parts]

    for part in normalized_parts:
        if "rotten" in part:
            return "rotten"

    for part in normalized_parts:
        if "fresh" in part:
            return "fresh"

    return None


def detect_product_type(relative_path):
    """Déduit le type du produit à partir du premier dossier utile."""
    folder_name = relative_path.parts[0]
    product_type = normalize_name(folder_name)
    product_type = product_type.replace("_fresh", "").replace("_rotten", "")
    product_type = product_type.replace("fresh_", "").replace("rotten_", "")
    return product_type


def scan_image_files(dataset_path, labels):
    """Parcourt le dataset et retourne les chemins utiles à l'entraînement."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dossier du dataset introuvable : {dataset_path}")

    records = []
    skipped_images = 0

    for image_path in sorted(dataset_path.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_path = image_path.relative_to(dataset_path)
        freshness = detect_freshness(relative_path.parts)

        if freshness not in labels:
            skipped_images += 1
            continue

        records.append(
            {
                "image_path": image_path,
                "product_type": detect_product_type(relative_path),
                "freshness": freshness,
                "label": labels[freshness],
            }
        )

    image_index_df = pd.DataFrame(records)

    if image_index_df.empty:
        raise ValueError("Aucune image fresh ou rotten n'a été trouvée.")

    if image_index_df["label"].nunique() < 2:
        raise ValueError("Le dataset doit contenir au moins une classe fresh et une classe rotten.")

    if skipped_images > 0:
        print(f"{skipped_images} image(s) ignorée(s), car leur freshness est inconnue.")

    return image_index_df


def split_dataset(image_index_df, validation_size, test_size, random_seed):
    """Crée le split classique train / validation / test."""
    temporary_size = validation_size + test_size

    if temporary_size <= 0 or temporary_size >= 1:
        raise ValueError("validation_size + test_size doit être compris entre 0 et 1.")

    stratify_labels = image_index_df["label"]

    training_set, temporary_set = train_test_split(
        image_index_df,
        test_size=temporary_size,
        random_state=random_seed,
        stratify=stratify_labels,
    )

    test_ratio = test_size / temporary_size
    temporary_stratify = temporary_set["label"]

    validation_set, test_set = train_test_split(
        temporary_set,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=temporary_stratify,
    )

    return training_set, validation_set, test_set


def load_image(image_path, label, image_size):
    """Charge une image et applique le même prétraitement pour train et test."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size)

    # Les pixels sont normalisés comme attendu par le CNN.
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def create_image_dataset(image_index_df, image_size, batch_size, shuffle=False, random_seed=42):
    """Crée un tf.data.Dataset à partir des chemins d'images."""
    image_paths = image_index_df["image_path"].astype(str).tolist()
    labels = image_index_df["label"].astype("float32").tolist()

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=random_seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda image_path, label: load_image(image_path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def save_split_file(training_set, validation_set, test_set, reports_dir, dataset_path, split_filename):
    """Sauvegarde le split utilisé pour pouvoir refaire l'évaluation."""
    split_frames = []

    for split_name, split_df in [
        ("train", training_set),
        ("validation", validation_set),
        ("test", test_set),
    ]:
        current_df = split_df.copy()
        current_df["split"] = split_name
        split_frames.append(current_df)

    split_df = pd.concat(split_frames, ignore_index=True)

    # On sauvegarde des chemins relatifs pour éviter de figer un chemin local Windows.
    split_df["image_path"] = split_df["image_path"].apply(
        lambda path: Path(path).relative_to(dataset_path).as_posix()
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(reports_dir / split_filename, index=False)


def train_model(config):
    """Lance l'entraînement complet de la baseline CNN."""
    paths_config = config["paths"]
    training_config = config["training"]

    dataset_path = PROJECT_ROOT / paths_config["raw_data_dir"]
    model_dir = PROJECT_ROOT / paths_config["model_dir"]
    reports_dir = PROJECT_ROOT / Path(paths_config["results_report"]).parent

    image_size = get_image_size(config)
    batch_size = int(training_config.get("batch_size", 32))
    epochs = int(training_config.get("epochs", 20))
    random_seed = int(training_config.get("random_seed", 42))
    validation_size = float(training_config.get("validation_size", 0.2))
    test_size = float(training_config.get("test_size", 0.2))
    learning_rate = float(training_config.get("learning_rate", 0.001))
    dropout_rate = float(training_config.get("dropout_rate", 0.3))
    early_stopping_patience = int(training_config.get("early_stopping_patience", 5))
    model_filename = training_config.get("model_filename", "baseline_model.keras")
    history_filename = training_config.get("history_filename", "training_history.csv")
    split_filename = training_config.get("split_filename", "standard_split.csv")

    tf.keras.utils.set_random_seed(random_seed)

    image_index_df = scan_image_files(dataset_path, config["data"]["labels"])
    training_set, validation_set, test_set = split_dataset(
        image_index_df=image_index_df,
        validation_size=validation_size,
        test_size=test_size,
        random_seed=random_seed,
    )

    print(f"Images train : {len(training_set)}")
    print(f"Images validation : {len(validation_set)}")
    print(f"Images test : {len(test_set)}")

    training_dataset = create_image_dataset(
        training_set,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        random_seed=random_seed,
    )
    validation_dataset = create_image_dataset(
        validation_set,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        random_seed=random_seed,
    )

    model = build_baseline_model(
        image_size=image_size,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / model_filename

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
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
    history_df.to_csv(reports_dir / history_filename)

    save_split_file(
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        reports_dir=reports_dir,
        dataset_path=dataset_path,
        split_filename=split_filename,
    )

    print(f"Meilleur modèle sauvegardé : {model_path}")
    print(f"Historique sauvegardé : {reports_dir / history_filename}")
    print(f"Split sauvegardé : {reports_dir / split_filename}")


def parse_args():
    parser = argparse.ArgumentParser(description="Entraîne la baseline CNN FreshOrRotten.")
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
    train_model(project_config)
