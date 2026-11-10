const form = document.getElementById("inspection-form");
const fileInput = document.getElementById("image-input");
const dropzone = document.getElementById("dropzone");
const previewImage = document.getElementById("preview-image");
const placeholder = document.getElementById("preview-placeholder");
const fileCard = document.getElementById("selected-file");
const fileName = document.getElementById("file-name");
const fileInfo = document.getElementById("file-info");
const pickerLabel = document.getElementById("file-picker-label");
const previewCaption = document.getElementById("preview-caption");
const previewDetail = document.getElementById("preview-detail");
const resultPanel = document.getElementById("result-panel");
const resultTitle = document.getElementById("result-title");
const resultDescription = document.getElementById("result-description");
const resultIcon = document.getElementById("result-icon");
const iconUrl = resultIcon.getAttribute("href").split("#")[0];
const resultImage = document.getElementById("result-image");
const resultFileName = document.getElementById("result-file-name");
const confidenceValue = document.getElementById("confidence-value");
const confidenceFill = document.getElementById("confidence-fill");
const confidenceNote = document.getElementById("confidence-note");
const analyzeButton = document.getElementById("analyze-button");
const analyzeLabel = document.getElementById("analyze-label");
const clearButton = document.getElementById("clear-button");
const errorMessage = document.getElementById("upload-error");

const MAX_FILE_BYTES = 5 * 1024 * 1024;
const MAX_IMAGE_PIXELS = 10_000_000;
const IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/bmp", "image/x-ms-bmp", "image/x-bmp"]);
let selectedFile = null;
let imageReady = false;
let previewUrl = null;
let selectionVersion = 0;
let activeRequest = null;
let dragDepth = 0;

function updateControls() {
    analyzeButton.disabled = !selectedFile || !imageReady || activeRequest !== null;
    analyzeLabel.textContent = activeRequest ? "Analyzing…" : "Analyze Image";
    clearButton.disabled = !selectedFile && previewImage.hidden && errorMessage.hidden;
}

function setResult(state, title, description) {
    resultPanel.dataset.state = state;
    resultPanel.setAttribute("aria-busy", String(state === "analyzing"));
    resultTitle.textContent = title;
    resultDescription.textContent = description;
    const icon = state === "ok" ? "check" : (["defective", "error"].includes(state) ? "alert" : "scan");
    resultIcon.setAttribute("href", `${iconUrl}#${icon}`);
    confidenceValue.textContent = "—";
    confidenceFill.style.width = "0%";
    confidenceNote.textContent = "A score appears after analysis.";
    resultImage.hidden = true;
    resultFileName.textContent = "";
}

function hideError() {
    errorMessage.hidden = true;
    errorMessage.textContent = "";
}

function releasePreview() {
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }
}

function resetSelection(clearInput = true) {
    // Each selection gets a new version. Responses from older versions are ignored.
    selectionVersion += 1;
    activeRequest?.controller.abort();
    activeRequest = null;
    selectedFile = null;
    imageReady = false;
    releasePreview();
    previewImage.onload = null;
    previewImage.onerror = null;
    previewImage.hidden = true;
    previewImage.removeAttribute("src");
    placeholder.hidden = false;
    dropzone.classList.remove("has-image", "is-dragging");
    fileCard.classList.remove("has-file");
    fileName.textContent = "No image selected";
    fileInfo.textContent = "Choose a casting image to begin.";
    pickerLabel.textContent = "Choose image";
    previewCaption.textContent = "No image selected";
    previewDetail.textContent = "Your image will appear here";
    dragDepth = 0;
    if (clearInput) fileInput.value = "";
    hideError();
    setResult("idle", "Awaiting image", "Choose a casting image to begin your inspection.");
    updateControls();
}

function rejectImage(message) {
    resetSelection();
    errorMessage.textContent = message;
    errorMessage.hidden = false;
    setResult("error", "Unable to load image", "Choose a valid casting image and try again.");
    updateControls();
}

function formatBytes(bytes) {
    return bytes >= 1024 * 1024 ? `${(bytes / (1024 * 1024)).toFixed(1)} MB` : `${(bytes / 1024).toFixed(1)} KB`;
}

function selectFile(file) {
    resetSelection(false);
    if (!file) return;
    if (!file.size) return rejectImage("The selected image is empty.");
    if (file.size > MAX_FILE_BYTES) return rejectImage("Choose an image smaller than 5 MiB.");
    if ((file.type && !IMAGE_TYPES.has(file.type)) || (!file.type && !/\.(jpe?g|png|bmp)$/i.test(file.name))) {
        return rejectImage("Choose a JPEG, PNG, or BMP image.");
    }

    selectedFile = file;
    const version = selectionVersion;
    fileName.textContent = file.name;
    fileInfo.textContent = "Loading preview…";
    fileCard.classList.add("has-file");
    pickerLabel.textContent = "Replace image";
    previewCaption.textContent = file.name;
    previewDetail.textContent = "Original image";
    setResult("idle", "Preparing image", "Checking the selected image before analysis.");

    previewImage.onload = () => {
        if (version !== selectionVersion) return;
        const { naturalWidth: width, naturalHeight: height } = previewImage;
        if (width * height > MAX_IMAGE_PIXELS) return rejectImage("Choose an image with at most 10 million pixels.");
        imageReady = true;
        fileInfo.textContent = `${width} × ${height} px · ${formatBytes(file.size)}`;
        setResult("idle", "Ready to analyze", "The result for this image will appear here.");
        updateControls();
    };
    previewImage.onerror = () => {
        if (version === selectionVersion) rejectImage("This image could not be opened. Choose a readable JPEG, PNG, or BMP file.");
    };
    previewUrl = URL.createObjectURL(file);
    previewImage.src = previewUrl;
    previewImage.hidden = false;
    placeholder.hidden = true;
    dropzone.classList.add("has-image");
    updateControls();
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!selectedFile || !imageReady || activeRequest) return;

    const file = selectedFile;
    const version = selectionVersion;
    const controller = new AbortController();
    activeRequest = { controller, version };
    hideError();
    setResult("analyzing", "Analyzing image", "Checking the selected casting image…");
    updateControls();
    const body = new FormData();
    body.append("image", file, file.name);

    try {
        const response = await fetch(form.dataset.predictUrl, { method: "POST", body, signal: controller.signal });
        const result = await response.json();
        if (version !== selectionVersion) return;
        if (!response.ok) throw new Error(result.error || "The image could not be analyzed. Please try again.");
        if (!["ok", "defective"].includes(result.status) || !Number.isFinite(result.confidence) || result.confidence < 0 || result.confidence > 100) {
            throw new Error("The server returned an invalid result. Please try again.");
        }
        const title = result.status === "ok" ? "PASS - OK" : "DEFECTIVE";
        const description = result.status === "ok" ? "The model found no surface defect in this image." : "The model detected a potential surface defect.";
        setResult(result.status, title, description);
        confidenceValue.textContent = `${result.confidence.toFixed(2)}%`;
        confidenceFill.style.width = `${result.confidence}%`;
        confidenceNote.textContent = "Model score for this classification.";
        resultFileName.textContent = file.name;
        resultImage.hidden = false;
    } catch (error) {
        if (version !== selectionVersion || error.name === "AbortError") return;
        errorMessage.textContent = error instanceof TypeError || error instanceof SyntaxError ? "Could not reach the prediction service. Please try again." : error.message;
        errorMessage.hidden = false;
        setResult("error", "Analysis failed", "Try again or choose another casting image.");
    } finally {
        // An older request must never change controls for a newer selection.
        if (activeRequest?.controller === controller) {
            activeRequest = null;
            updateControls();
        }
    }
});

fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));
clearButton.addEventListener("click", () => resetSelection());
document.querySelectorAll("[data-pick-image]").forEach((button) => {
    button.hidden = false;
    button.addEventListener("click", () => fileInput.click());
});
clearButton.hidden = false;

dropzone.addEventListener("dragenter", (event) => {
    event.preventDefault();
    dragDepth += 1;
    dropzone.classList.add("is-dragging");
});
dropzone.addEventListener("dragover", (event) => event.preventDefault());
dropzone.addEventListener("dragleave", (event) => {
    event.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) dropzone.classList.remove("is-dragging");
});
dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    const [file] = event.dataTransfer.files;
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    selectFile(file);
});
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());
window.addEventListener("beforeunload", () => {
    activeRequest?.controller.abort();
    releasePreview();
});
updateControls();
