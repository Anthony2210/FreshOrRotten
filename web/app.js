const MODEL_URL = new URL("./model/model.json?v=tfkeras-compat-1", window.location.href).toString();
const IMAGE_SIZE = 128;
const CONFIDENT_THRESHOLD = 0.8;
const UNCERTAIN_THRESHOLD = 0.6;

const imageInput = document.querySelector("#image-input");
const imagePreview = document.querySelector("#image-preview");
const emptyPreview = document.querySelector("#empty-preview");
const analyzeButton = document.querySelector("#analyze-button");
const modelStatus = document.querySelector("#model-status");
const resultBox = document.querySelector("#result-box");
const predictionValue = document.querySelector("#prediction-value");
const confidenceValue = document.querySelector("#confidence-value");
const confidenceBar = document.querySelector("#confidence-bar");

let model = null;
let userImageIsReady = false;

function setModelStatus(message, statusClass = "", detail = "") {
  const shouldShowMessage = statusClass && statusClass !== "is-ready";
  modelStatus.className = "technical-note";
  modelStatus.textContent = shouldShowMessage ? `${message}. ${detail}` : "";

  if (statusClass) {
    modelStatus.classList.add(statusClass);
  }
}

function setAnalyzeButtonState() {
  analyzeButton.disabled = !model || !userImageIsReady;
}

function updateConfidenceBar(confidenceScore) {
  const safeScore = Number.isFinite(confidenceScore) ? confidenceScore : 0;
  confidenceBar.style.width = `${Math.round(safeScore * 100)}%`;
}

function clearResult() {
  resultBox.className = "result-box";
  resultBox.querySelector(".result-label").textContent = "En attente.";
  resultBox.querySelector(".result-message").textContent = "Image prête. Lance l'analyse quand tu veux.";
  predictionValue.textContent = "-";
  confidenceValue.textContent = "-";
  updateConfidenceBar(0);
}

function renderResult(label, message, prediction, confidenceScore, statusClass) {
  resultBox.className = `result-box ${statusClass}`;
  resultBox.querySelector(".result-label").textContent = label;
  resultBox.querySelector(".result-message").textContent = message;
  predictionValue.textContent = prediction;
  confidenceValue.textContent = Number.isFinite(confidenceScore)
    ? `${Math.round(confidenceScore * 100)} %`
    : "-";
  updateConfidenceBar(confidenceScore);
}

function getPredictionText(prediction) {
  return prediction === "rotten" ? "Pourri" : "Frais";
}

function interpretPrediction(predictionScore) {
  const prediction = predictionScore >= 0.5 ? "rotten" : "fresh";
  const confidenceScore = prediction === "rotten" ? predictionScore : 1 - predictionScore;
  const predictionText = getPredictionText(prediction);

  if (confidenceScore >= CONFIDENT_THRESHOLD) {
    return {
      label: predictionText,
      message: `Le modèle pense que le produit est ${predictionText.toLowerCase()}.`,
      prediction: predictionText,
      confidenceScore,
      statusClass: "is-success",
    };
  }

  if (confidenceScore >= UNCERTAIN_THRESHOLD) {
    return {
      label: "Incertain",
      message: `Le modèle hésite. Il penche vers : ${predictionText.toLowerCase()}.`,
      prediction: predictionText,
      confidenceScore,
      statusClass: "is-warning",
    };
  }

  return {
    label: "Photo à refaire",
    message: "Le score est trop faible. Essaie une photo plus nette, mieux éclairée et centrée.",
    prediction: predictionText,
    confidenceScore,
    statusClass: "is-error",
  };
}

function preprocessImage(imageElement) {
  // Les images sont normalisées comme pendant l'entraînement.
  return tf.tidy(() =>
    tf.browser
      .fromPixels(imageElement)
      .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE])
      .toFloat()
      .div(255)
      .expandDims(0)
  );
}

async function checkModelFile() {
  const response = await fetch(MODEL_URL, { cache: "no-store" });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
}

async function loadModel() {
  let modelFileWasFound = false;

  if (window.location.protocol === "file:") {
    model = null;
    setModelStatus(
      "Ouvre avec un serveur local",
      "is-warning",
      "Le navigateur bloque souvent le modèle en file://. Lance : python -m http.server 8000 dans le dossier web."
    );
    setAnalyzeButtonState();
    return;
  }

  if (!window.tf) {
    model = null;
    setModelStatus(
      "TensorFlow.js non chargé",
      "is-error",
      "Vérifie la connexion internet ou héberge tf.min.js localement."
    );
    setAnalyzeButtonState();
    return;
  }

  try {
    setModelStatus("Recherche du modèle...", "", `Chemin testé : ${MODEL_URL}`);
    await checkModelFile();
    modelFileWasFound = true;

    model = await tf.loadLayersModel(MODEL_URL);
    setModelStatus("", "is-ready", "");
  } catch (error) {
    model = null;

    if (modelFileWasFound) {
      setModelStatus(
        "Modèle trouvé, mais illisible",
        "is-error",
        "Le fichier model.json existe, mais TensorFlow.js n'arrive pas à le charger. Regarde la console du navigateur."
      );
    } else {
      setModelStatus(
        "Modèle introuvable",
        "is-warning",
        "Il faut avoir web/model/model.json et le fichier .bin à côté de cette page."
      );
    }

    console.warn("Chargement du modèle impossible :", error);
  }

  setAnalyzeButtonState();
}

function resetImagePreview() {
  userImageIsReady = false;
  imagePreview.hidden = true;
  emptyPreview.hidden = false;
  clearResult();
  setAnalyzeButtonState();
}

function handleImageSelection(event) {
  const file = event.target.files[0];

  if (!file) {
    resetImagePreview();
    return;
  }

  if (!file.type.startsWith("image/")) {
    userImageIsReady = false;
    renderResult("Image invalide", "Choisis un fichier image.", "-", NaN, "is-error");
    setAnalyzeButtonState();
    return;
  }

  const imageUrl = URL.createObjectURL(file);
  userImageIsReady = false;
  imagePreview.hidden = true;

  imagePreview.onload = () => {
    URL.revokeObjectURL(imageUrl);
    userImageIsReady = true;
    imagePreview.hidden = false;
    emptyPreview.hidden = true;
    clearResult();
    setAnalyzeButtonState();
  };

  imagePreview.onerror = () => {
    URL.revokeObjectURL(imageUrl);
    resetImagePreview();
    renderResult("Image invalide", "Impossible de lire cette image.", "-", NaN, "is-error");
  };

  imagePreview.src = imageUrl;
}

async function analyzeImage() {
  if (!model) {
    renderResult(
      "Modèle absent",
      "Le modèle TensorFlow.js doit être chargé avant l'analyse.",
      "-",
      NaN,
      "is-error"
    );
    return;
  }

  if (!userImageIsReady) {
    renderResult("Aucune photo", "Ajoute une image avant de lancer l'analyse.", "-", NaN, "is-error");
    return;
  }

  analyzeButton.disabled = true;
  analyzeButton.textContent = "Analyse...";

  try {
    const inputTensor = preprocessImage(imagePreview);
    const predictionTensor = model.predict(inputTensor);
    const predictionScore = (await predictionTensor.data())[0];

    inputTensor.dispose();
    predictionTensor.dispose();

    const result = interpretPrediction(predictionScore);
    renderResult(
      result.label,
      result.message,
      result.prediction,
      result.confidenceScore,
      result.statusClass
    );
  } catch (error) {
    renderResult("Analyse impossible", "Une erreur est survenue pendant la prédiction.", "-", NaN, "is-error");
    console.error("Erreur pendant la prédiction :", error);
  } finally {
    analyzeButton.textContent = "Analyser";
    setAnalyzeButtonState();
  }
}

imageInput.addEventListener("change", handleImageSelection);
analyzeButton.addEventListener("click", analyzeImage);

loadModel();
