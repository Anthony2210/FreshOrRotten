"""Stratégies de split pour FreshOrRotten."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def normalize_category_name(category_name):
    """Normalise un nom de catégorie pour le comparer au product_type."""
    return category_name.lower().strip().replace("-", "_").replace(" ", "_")


def normalize_category_list(unseen_categories):
    """Retourne une liste propre de catégories non vues."""
    return [normalize_category_name(category) for category in unseen_categories if category.strip()]


def create_standard_split(image_index_df, validation_size, test_size, random_seed):
    """Crée le split classique train / validation / test."""
    temporary_size = validation_size + test_size

    if temporary_size <= 0 or temporary_size >= 1:
        raise ValueError("validation_size + test_size doit être compris entre 0 et 1.")

    training_set, temporary_set = train_test_split(
        image_index_df,
        test_size=temporary_size,
        random_state=random_seed,
        stratify=image_index_df["label"],
    )

    test_ratio = test_size / temporary_size

    validation_set, test_set = train_test_split(
        temporary_set,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=temporary_set["label"],
    )

    return training_set, validation_set, test_set


def create_unseen_category_split(image_index_df, unseen_categories, validation_size, random_seed):
    """Crée un split où le test_set contient seulement des catégories non vues."""
    normalized_categories = normalize_category_list(unseen_categories)

    if not normalized_categories:
        raise ValueError("La liste unseen_categories ne doit pas être vide.")

    available_categories = set(image_index_df["product_type"].unique())
    missing_categories = sorted(set(normalized_categories) - available_categories)

    if missing_categories:
        raise ValueError(f"Catégories absentes du dataset : {missing_categories}")

    unseen_category_mask = image_index_df["product_type"].isin(normalized_categories)

    test_set = image_index_df[unseen_category_mask].copy()
    seen_categories_df = image_index_df[~unseen_category_mask].copy()

    if test_set.empty:
        raise ValueError("Le test_set non vu est vide.")

    if seen_categories_df.empty:
        raise ValueError("Le training_set serait vide avec ces catégories non vues.")

    if seen_categories_df["label"].nunique() < 2:
        raise ValueError("Les catégories vues doivent contenir fresh et rotten.")

    if test_set["label"].nunique() < 2:
        raise ValueError("Les catégories non vues doivent contenir fresh et rotten.")

    training_set, validation_set = train_test_split(
        seen_categories_df,
        test_size=validation_size,
        random_state=random_seed,
        stratify=seen_categories_df["label"],
    )

    return training_set, validation_set, test_set


def save_split_file(training_set, validation_set, test_set, reports_dir, dataset_path, split_filename):
    """Sauvegarde un split pour pouvoir reproduire l'évaluation."""
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

    # Les chemins relatifs évitent de figer le chemin local d'une machine.
    split_df["image_path"] = split_df["image_path"].apply(
        lambda path: Path(path).relative_to(dataset_path).as_posix()
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(reports_dir / split_filename, index=False)


def load_split_file(split_path, dataset_path, split_name="test"):
    """Charge un split sauvegardé et reconstruit les chemins des images."""
    split_df = pd.read_csv(split_path)
    selected_set = split_df[split_df["split"] == split_name].copy()

    if selected_set.empty:
        raise ValueError(f"Le fichier de split ne contient aucun ensemble {split_name}.")

    selected_set["image_path"] = selected_set["image_path"].apply(lambda path: dataset_path / path)
    return selected_set
