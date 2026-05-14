#!/bin/bash
# Download all ML/DL model weights using direct curl/wget downloads.
#
# This script downloads weights directly from official sources without Python dependencies.
#
# Usage:
#   ./scripts/download_model_weights.sh                    # Download all
#   ./scripts/download_model_weights.sh --deepface         # Download only DeepFace
#   ./scripts/download_model_weights.sh --dry-run         # Show what would be downloaded
#   ./scripts/download_model_weights.sh --force           # Re-download existing files
#
# Available options:
#   --all              Download all weights (default)
#   --deepface         DeepFace face recognition models
#   --doctr            DocTR document text recognition
#   --huggingface      FinBERT, GLiNER2
#   --mediapipe        MediaPipe hand gesture detector
#   --nltk             NLTK sentiment lexicon
#   --photoholmes      PhotoHolmes forgery detection
#   --spacy            spaCy language models

set -e

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
APP_DIR="${PROJECT_ROOT}/app"

# Model directories
DEEPFACE_TARGET="${APP_DIR}/deepface/.deepface/weights"
DOCTR_TARGET="${APP_DIR}/models/doctr/models"
FINBERT_TARGET="${APP_DIR}/models/finbert"
GLINER2_TARGET="${APP_DIR}/models/gliner2"
MEDIAPIPE_TARGET="${APP_DIR}/models/mediapipe"
NLTK_TARGET="${APP_DIR}/models/nltk"
PHOTOHOLMES_TARGET="${APP_DIR}/photoholmes/weights"
SPACY_TARGET="${APP_DIR}/models/spacy"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Flags
DRY_RUN=false
FORCE=false

# ============================================================================
# Utility Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}INFO:${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_section() {
    echo ""
    echo "============================================================"
    echo " $1"
    echo "============================================================"
}

# Get file size in bytes (cross-platform: stat -f%z on macOS, stat -c%s on Linux)
get_file_size() {
    local file="$1"
    stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0"
}

# Check if file exists and is non-empty (with optional minimum size)
check_file_exists_and_valid() {
    local file="$1"
    local min_size="${2:-1}"  # Default minimum 1 byte (just check non-zero)

    if [ ! -f "$file" ]; then
        return 1  # File doesn't exist
    fi

    local file_size=$(get_file_size "$file")

    if [ "$file_size" -lt "$min_size" ]; then
        # File exists but is too small (likely corrupted/empty)
        return 1
    fi

    return 0  # File exists and is valid
}

# Download a file with optional progress bar and validation
download_file() {
    local url="$1"
    local target="$2"
    local show_progress="${3:-true}"
    local min_size="${4:-1}"  # Default minimum 1 byte (just check non-zero)

    # Skip if exists, is valid, and not forcing
    if check_file_exists_and_valid "$target" "$min_size" && [ "$FORCE" != "true" ]; then
        return 0
    fi

    mkdir -p "$(dirname "$target")"

    if [ "$DRY_RUN" = "true" ]; then
        echo "Would download: $url -> $target"
        return 0
    fi

    # Download to temp file first to avoid leaving empty files on failure
    local tmp_file=$(mktemp)
    local download_success=false

    log_info "  Downloading $(basename "$target")..."

    if [ "$show_progress" = "true" ]; then
        if curl -fL --progress-bar "$url" -o "$tmp_file" 2>/dev/null; then
            download_success=true
        fi
    else
        if curl -fLs "$url" -o "$tmp_file" 2>/dev/null; then
            download_success=true
        fi
    fi

    if [ "$download_success" = true ]; then
        # Verify file size before moving to target
        local downloaded_size=$(get_file_size "$tmp_file")
        if [ "$downloaded_size" -ge "$min_size" ]; then
            mv "$tmp_file" "$target"
            log_info "    → $(basename "$target") ($(format_size "$downloaded_size"))"
            return 0
        else
            rm -f "$tmp_file"
            log_error "  Downloaded file is too small ($downloaded_size bytes < $min_size bytes)"
            return 1
        fi
    else
        rm -f "$tmp_file"
        log_error "  Failed to download $(basename "$target")"
        return 1
    fi
}

# Format file size for human-readable output
format_size() {
    local bytes=$1
    if [ "$bytes" -ge 1073741824 ]; then
        echo "$((bytes / 1073741824))GB"
    elif [ "$bytes" -ge 1048576 ]; then
        echo "$((bytes / 1048576))MB"
    elif [ "$bytes" -ge 1024 ]; then
        echo "$((bytes / 1024))KB"
    else
        echo "${bytes}B"
    fi
}

# Extract ZIP archive
extract_zip() {
    local zip_file="$1"
    local target_dir="$2"

    if [ "$DRY_RUN" = "true" ]; then
        echo "Would extract: $zip_file -> $target_dir"
        return 0
    fi

    mkdir -p "$target_dir"

    # Validate it's a ZIP file
    if ! file "$zip_file" | grep -qi "zip"; then
        log_error "  File is not a valid ZIP archive"
        file "$zip_file" | head -1
        return 1
    fi

    # Test ZIP integrity before extraction
    if ! unzip -t "$zip_file" >/dev/null 2>&1; then
        log_error "  ZIP file is corrupted or invalid"
        return 1
    fi

    # Extract (show errors, don't suppress with -q)
    if unzip -o "$zip_file" -d "$target_dir" 2>&1; then
        return 0
    else
        log_error "  Failed to extract $(basename "$zip_file")"
        return 1
    fi
}

# Extract BZ2 archive
extract_bz2() {
    local bz2_file="$1"
    local target_dir="$2"
    local output_name="$3"

    if [ "$DRY_RUN" = "true" ]; then
        echo "Would extract: $bz2_file -> $target_dir"
        return 0
    fi

    mkdir -p "$target_dir"
    if bunzip2 -k -c "$bz2_file" > "$target_dir/$output_name" 2>/dev/null; then
        return 0
    else
        log_error "  Failed to extract $(basename "$bz2_file")"
        return 1
    fi
}

# Check if directory exists and has files
check_dir_has_files() {
    local path="$1"
    if [ -d "$path" ] && [ -n "$(ls -A "$path" 2>/dev/null)" ]; then
        return 0
    fi
    return 1
}

# ============================================================================
# DeepFace Download
# ============================================================================

download_deepface() {
    log_section "DeepFace"

    if ! $FORCE && check_dir_has_files "$DEEPFACE_TARGET"; then
        # Verify all expected files exist and are valid (non-empty)
        local all_valid=true
        for f in vgg_face_weights.h5 facenet_weights.h5 facenet512_weights.h5 openface_weights.h5 \
                 deepid_keras_weights.h5 arcface_weights.h5 face_recognition_sface_2021dec.onnx \
                 GhostFaceNet_W1.3_S1_ArcFace.h5 VGGFace2_DeepFace_weights_val-0.9034.h5 \
                 dlib_face_recognition_resnet_model_v1.dat \
                 retinaface.h5 2.7_80x80_MiniFASNetV2.pth 4_0_0_80x80_MiniFASNetV1SE.pth; do
            if ! check_file_exists_and_valid "$DEEPFACE_TARGET/$f"; then
                all_valid=false
                break
            fi
        done
        if [ "$all_valid" = true ]; then
            log_info "Already exists at ${DEEPFACE_TARGET#$PROJECT_ROOT/} (skipped)"
            return 0
        fi
    fi

    if [ "$DRY_RUN" = "true" ]; then
        log_info "Would download DeepFace weights to ${DEEPFACE_TARGET#$PROJECT_ROOT/}"
        return 0
    fi

    mkdir -p "$DEEPFACE_TARGET"
    local tmp_dir=$(mktemp -d)
    local success_count=0
    local total_count=13

    log_info "Downloading DeepFace weights..."

    # VGG-Face
    if download_file "https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5" \
            "$DEEPFACE_TARGET/vgg_face_weights.h5" "true"; then
        success_count=$((success_count + 1))
    fi

    # Facenet
    if download_file "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5" \
            "$DEEPFACE_TARGET/facenet_weights.h5" "true"; then
        success_count=$((success_count + 1))
    fi

    # Facenet512
    if download_file "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5" \
            "$DEEPFACE_TARGET/facenet512_weights.h5" "true"; then
        success_count=$((success_count + 1))
    fi

    # OpenFace
    if download_file "https://github.com/serengil/deepface_models/releases/download/v1.0/openface_weights.h5" \
            "$DEEPFACE_TARGET/openface_weights.h5" "true"; then
        success_count=$((success_count + 1))
    fi

    # DeepFace (ZIP format)
    if download_file "https://github.com/swghosh/DeepFace/releases/download/weights-vggface2-2d-aligned/VGGFace2_DeepFace_weights_val-0.9034.h5.zip" \
            "$tmp_dir/deepface.zip" "true"; then
        if extract_zip "$tmp_dir/deepface.zip" "$tmp_dir"; then
            # Check for the extracted file (may be in subdirectory)
            local extracted_file=$(find "$tmp_dir" -name "VGGFace2_DeepFace_weights_val-0.9034.h5" -type f | head -1)
            if [ -n "$extracted_file" ]; then
                mv "$extracted_file" "$DEEPFACE_TARGET/"
                success_count=$((success_count + 1))
                log_success "  Extracted VGGFace2 weights"
            else
                log_error "  VGGFace2 weights not found in ZIP"
            fi
        fi
    fi

    # DeepID
    if download_file "https://github.com/serengil/deepface_models/releases/download/v1.0/deepid_keras_weights.h5" \
            "$DEEPFACE_TARGET/deepid_keras_weights.h5" "true"; then
        success_count=$((success_count + 1))
    fi

    # ArcFace
    if download_file "https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5" \
            "$DEEPFACE_TARGET/arcface_weights.h5" "true"; then
        success_count=$((success_count + 1))
    fi

    # SFace
    if download_file "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" \
            "$DEEPFACE_TARGET/face_recognition_sface_2021dec.onnx" "true"; then
        success_count=$((success_count + 1))
    fi

    # GhostFaceNet
    if download_file "https://github.com/HamadYA/GhostFaceNets/releases/download/v1.2/GhostFaceNet_W1.3_S1_ArcFace.h5" \
            "$DEEPFACE_TARGET/GhostFaceNet_W1.3_S1_ArcFace.h5" "true"; then
        success_count=$((success_count + 1))
    fi

    # Dlib (BZ2 format)
    if download_file "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2" \
            "$tmp_dir/dlib_face_recognition_resnet_model_v1.dat.bz2" "true"; then
        if extract_bz2 "$tmp_dir/dlib_face_recognition_resnet_model_v1.dat.bz2" "$DEEPFACE_TARGET" \
                "dlib_face_recognition_resnet_model_v1.dat"; then
            if [ -f "$DEEPFACE_TARGET/dlib_face_recognition_resnet_model_v1.dat" ]; then
                success_count=$((success_count + 1))
            fi
        fi
    fi

    # RetinaFace (face detector)
    if download_file "https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5" \
            "$DEEPFACE_TARGET/retinaface.h5" "true"; then
        success_count=$((success_count + 1))
    fi

    # Anti-spoofing models (FASNet)
    if download_file "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth" \
            "$DEEPFACE_TARGET/2.7_80x80_MiniFASNetV2.pth" "true"; then
        success_count=$((success_count + 1))
    fi

    if download_file "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth" \
            "$DEEPFACE_TARGET/4_0_0_80x80_MiniFASNetV1SE.pth" "true"; then
        success_count=$((success_count + 1))
    fi

    rm -rf "$tmp_dir"

    file_count=$(find "$DEEPFACE_TARGET" -type f 2>/dev/null | wc -l)
    total_size=$(du -sm "$DEEPFACE_TARGET" 2>/dev/null | cut -f1)

    if [ $success_count -eq $total_count ]; then
        log_success "Downloaded $file_count file(s) (${total_size} MB)"
        return 0
    elif [ $success_count -gt 0 ]; then
        log_warning "$success_count/$total_count downloads succeeded"
        return 1
    else
        log_error "Failed to download DeepFace weights"
        return 1
    fi
}

# ============================================================================
# DocTR Download
# ============================================================================

download_doctr() {
    log_section "DocTR"

    # Check if both files exist and are valid (non-empty)
    if ! $FORCE && check_dir_has_files "$DOCTR_TARGET"; then
        if check_file_exists_and_valid "$DOCTR_TARGET/db_resnet50-79bd7d70.pt" && \
           check_file_exists_and_valid "$DOCTR_TARGET/crnn_vgg16_bn-0417f351.pt"; then
            log_info "Already exists at ${DOCTR_TARGET#$PROJECT_ROOT/} (skipped)"
            return 0
        fi
    fi

    if [ "$DRY_RUN" = "true" ]; then
        log_info "Would download DocTR weights to ${DOCTR_TARGET#$PROJECT_ROOT/}"
        return 0
    fi

    mkdir -p "$DOCTR_TARGET"
    local success_count=0

    log_info "Downloading DocTR weights..."

    # Detection model (db_resnet50)
    if download_file "https://doctr-static.mindee.com/models?id=v0.7.0/db_resnet50-79bd7d70.pt&src=0" \
            "$DOCTR_TARGET/db_resnet50-79bd7d70.pt" "true"; then
        success_count=$((success_count + 1))
    fi

    # Recognition model (crnn_vgg16_bn)
    if download_file "https://doctr-static.mindee.com/models?id=v0.12.0/crnn_vgg16_bn-0417f351.pt&src=0" \
            "$DOCTR_TARGET/crnn_vgg16_bn-0417f351.pt" "true"; then
        success_count=$((success_count + 1))
    fi

    if [ $success_count -eq 2 ]; then
        file_count=$(find "$DOCTR_TARGET" -type f | wc -l)
        total_size=$(du -sm "$DOCTR_TARGET" | cut -f1)
        log_success "Downloaded $file_count file(s) (${total_size} MB)"
        return 0
    else
        log_error "Failed to download DocTR weights"
        return 1
    fi
}

# Ensure Poetry is using the correct Python version
ensure_poetry_python() {
    local target_python=""

    # Method 1: Read from .python-version file
    if [ -f "$PROJECT_ROOT/.python-version" ]; then
        local version=$(cat "$PROJECT_ROOT/.python-version" | tr -d '[:space:]')
        # Find Python with this version
        if command -v pyenv >/dev/null 2>&1; then
            target_python=$(pyenv which python3 "$version" 2>/dev/null)
        fi

        # Fallback to finding python3.<version> in PATH
        if [ -z "$target_python" ]; then
            target_python=$(command -v "python3.$version" 2>/dev/null)
        fi
    fi

    # Method 2: Use existing virtualenv's Python
    if [ -z "$target_python" ] && [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
        target_python="$PROJECT_ROOT/.venv/bin/python"
    fi

    # Method 3: Try common Python versions
    if [ -z "$target_python" ]; then
        for version in "3.11" "3.12" "3.10"; do
            if command -v "python3.$version" >/dev/null 2>&1; then
                target_python=$(command -v "python3.$version")
                break
            fi
        done
    fi

    # Set Poetry to use this Python
    if [ -n "$target_python" ]; then
        if [ "$VERBOSE" = "true" ]; then
            log_info "    Setting Poetry Python to: $target_python"
        fi
        poetry env use "$target_python" >/dev/null 2>&1
        return 0
    else
        log_error "    Could not find a compatible Python (3.10-3.12)"
        return 1
    fi
}

# ============================================================================
# HuggingFace Models (FinBERT, GLiNER2)
# ============================================================================

download_huggingface() {
    log_section "HuggingFace Models"

    # Check if all files exist and are valid (non-empty)
    local all_valid=true
    if ! $FORCE && check_dir_has_files "$FINBERT_TARGET" && check_dir_has_files "$GLINER2_TARGET"; then
        # Check FinBERT files
        if ! check_file_exists_and_valid "$FINBERT_TARGET/pytorch_model.bin"; then
            all_valid=false
        fi
        if ! check_file_exists_and_valid "$FINBERT_TARGET/config.json"; then
            all_valid=false
        fi
        if ! check_file_exists_and_valid "$FINBERT_TARGET/vocab.txt"; then
            all_valid=false
        fi
        if ! check_file_exists_and_valid "$FINBERT_TARGET/tokenizer_config.json"; then
            all_valid=false
        fi
        # Check GLiNER2 files
        if ! check_file_exists_and_valid "$GLINER2_TARGET/config.json"; then
            all_valid=false
        fi
        if ! check_file_exists_and_valid "$GLINER2_TARGET/model.safetensors"; then
            all_valid=false
        fi
        if [ "$all_valid" = true ]; then
            log_info "Already exists at ${FINBERT_TARGET#$PROJECT_ROOT/} and ${GLINER2_TARGET#$PROJECT_ROOT/} (skipped)"
            return 0
        fi
    fi

    if [ "$DRY_RUN" = "true" ]; then
        log_info "Would download HuggingFace models:"
        log_info "  - FinBERT (ProsusAI/finbert)"
        log_info "  - GLiNER2 (fastino/gliner2-large-v1)"
        return 0
    fi

    mkdir -p "$FINBERT_TARGET"
    mkdir -p "$GLINER2_TARGET"

    local finbert_success=0
    local gliner2_success=0

    log_info "Downloading HuggingFace models..."

    # FinBERT files
    log_info "  Downloading FinBERT (ProsusAI/finbert)..."

    if download_file "https://huggingface.co/ProsusAI/finbert/resolve/main/pytorch_model.bin" \
            "$FINBERT_TARGET/pytorch_model.bin" "true"; then
        finbert_success=$((finbert_success + 1))
    fi

    if download_file "https://huggingface.co/ProsusAI/finbert/resolve/main/config.json" \
            "$FINBERT_TARGET/config.json" "true"; then
        finbert_success=$((finbert_success + 1))
    fi

    if download_file "https://huggingface.co/ProsusAI/finbert/resolve/main/vocab.txt" \
            "$FINBERT_TARGET/vocab.txt" "true"; then
        finbert_success=$((finbert_success + 1))
    fi

    if download_file "https://huggingface.co/ProsusAI/finbert/resolve/main/tokenizer_config.json" \
            "$FINBERT_TARGET/tokenizer_config.json" "true"; then
        finbert_success=$((finbert_success + 1))
    fi

    # GLiNER2 files - fastino/gliner2-large-v1
    # Note: GLINER2 uses Xet storage which fails with curl (504 Gateway Timeout)
    # Use Python huggingface_hub instead
    log_info "  Downloading GLiNER2 (fastino/gliner2-large-v1) using Python..."

    if [ "$DRY_RUN" = "true" ]; then
        log_info "    Would download using: python scripts/download_gliner2.py $GLINER2_TARGET"
        gliner2_success=2
    else
        # Ensure Poetry is using the correct Python version
        if ! ensure_poetry_python; then
            gliner2_success=0
            log_error "    Failed to set Poetry Python version"
            log_error "    Please ensure Python 3.10-3.12 is available"
            log_error "    Run: pyenv local 3.11.7"
        else
            # Check if huggingface_hub is installed, install if missing
            local skip_download=false
            if ! poetry run python3 -c "import huggingface_hub" 2>/dev/null; then
                log_info "    huggingface_hub not found, installing..."
                if poetry run pip install huggingface_hub >/dev/null 2>&1; then
                    log_info "    → huggingface_hub installed successfully"
                else
                    gliner2_success=0
                    skip_download=true
                    log_error "    Failed to install huggingface_hub"
                    log_error "    Please run: poetry add huggingface_hub"
                fi
            fi

            if [ "$skip_download" = false ]; then
                set +e  # Don't exit on error for this command
                local stderr_file=$(mktemp)

                # Show diagnostic info if VERBOSE is set
                if [ "$VERBOSE" = "true" ]; then
                    log_info "    Running: poetry run python3 ${SCRIPT_DIR}/download_gliner2.py $GLINER2_TARGET"
                    log_info "    Current directory: $(pwd)"
                    log_info "    SCRIPT_DIR: $SCRIPT_DIR"
                fi

                # Run the download script via Poetry
                poetry run python3 "${SCRIPT_DIR}/download_gliner2.py" "$GLINER2_TARGET" 2>"$stderr_file"
                local exit_code=$?  # Capture exit code IMMEDIATELY

                if [ $exit_code -eq 0 ]; then
                    gliner2_success=2
                    log_info "    → GLiNER2 downloaded successfully"
                else
                    gliner2_success=0
                    log_error "    Failed to download GLiNER2"
                    log_error "    Exit code: $exit_code"

                    if [ -s "$stderr_file" ]; then
                        log_error "    Error output:"
                        while IFS= read -r line; do
                            log_error "      $line"
                        done < "$stderr_file"
                    fi
                fi

                rm -f "$stderr_file"
                set -e  # Re-enable exit on error
            fi
        fi
    fi

    if [ $finbert_success -eq 4 ] && [ $gliner2_success -eq 2 ]; then
        log_success "HuggingFace models downloaded"
        return 0
    else
        log_warning "FinBERT: $finbert_success/4 files, GLiNER2: $gliner2_success/2 files (config.json + model.safetensors)"
        return 1
    fi
}

# ============================================================================
# MediaPipe Download
# ============================================================================

download_mediapipe() {
    log_section "MediaPipe"

    if ! $FORCE && check_file_exists_and_valid "$MEDIAPIPE_TARGET/hand_landmarker.task"; then
        log_info "Already exists at ${MEDIAPIPE_TARGET#$PROJECT_ROOT/} (skipped)"
        return 0
    fi

    if [ "$DRY_RUN" = "true" ]; then
        log_info "Would download MediaPipe model to ${MEDIAPIPE_TARGET#$PROJECT_ROOT/}"
        return 0
    fi

    mkdir -p "$MEDIAPIPE_TARGET"

    log_info "Downloading MediaPipe hand landmark model..."

    if download_file "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" \
            "$MEDIAPIPE_TARGET/hand_landmarker.task" "true"; then
        file_size=$(get_file_size "$MEDIAPIPE_TARGET/hand_landmarker.task")
        log_success "Downloaded hand_landmarker.task ($(format_size "$file_size"))"
        return 0
    else
        log_error "Failed to download MediaPipe model"
        return 1
    fi
}

# ============================================================================
# NLTK Download
# ============================================================================

download_nltk() {
    log_section "NLTK"

    local nltk_sentiment_dir="${NLTK_TARGET}/sentiment"

    # Check if vader_lexicon.txt exists and is valid (non-empty)
    if ! $FORCE && check_file_exists_and_valid "$nltk_sentiment_dir/vader_lexicon.txt"; then
        log_info "Already exists at ${nltk_sentiment_dir#$PROJECT_ROOT/} (skipped)"
        return 0
    fi

    if [ "$DRY_RUN" = "true" ]; then
        log_info "Would download NLTK data to ${NLTK_TARGET#$PROJECT_ROOT/}"
        return 0
    fi

    mkdir -p "$NLTK_TARGET"
    local tmp_dir=$(mktemp -d)

    log_info "Downloading NLTK VADER lexicon..."

    if download_file "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/sentiment/vader_lexicon.zip" \
            "$tmp_dir/vader_lexicon.zip" "true"; then
        if extract_zip "$tmp_dir/vader_lexicon.zip" "$NLTK_TARGET"; then
            # CRITICAL FIX: Move vader_lexicon/ to sentiment/ for NLTK compatibility
            # ZIP extracts to vader_lexicon/ but code expects sentiment/vader_lexicon.txt
            if [ -d "${NLTK_TARGET}/vader_lexicon" ] && [ ! -d "$nltk_sentiment_dir" ]; then
                mv "${NLTK_TARGET}/vader_lexicon" "$nltk_sentiment_dir"
                log_info "  Moved vader_lexicon/ to sentiment/ for NLTK compatibility"
            fi
            # Verify final location
            if check_file_exists_and_valid "$nltk_sentiment_dir/vader_lexicon.txt"; then
                rm -rf "$tmp_dir"
                file_count=$(find "$NLTK_TARGET" -type f | wc -l)
                log_success "Downloaded $file_count file(s)"
                return 0
            else
                rm -rf "$tmp_dir"
                log_error "  vader_lexicon.txt not found at $nltk_sentiment_dir/"
                return 1
            fi
        else
            rm -rf "$tmp_dir"
            log_error "Failed to extract NLTK data"
            return 1
        fi
    else
        rm -rf "$tmp_dir"
        log_error "Failed to download NLTK data"
        return 1
    fi
}

# ============================================================================
# PhotoHolmes Download
# ============================================================================

download_photoholmes() {
    log_section "PhotoHolmes"

    if [ "$DRY_RUN" = "true" ]; then
        log_info "Would download PhotoHolmes weights to ${PHOTOHOLMES_TARGET#$PROJECT_ROOT/}:"
        echo "  - AdaptiveCFANet (raw GitHub)"
        echo "  - CAT-Net (Google Drive - 2 files)"
        echo "  - EXIF as Language (Google Drive - needs pruning)"
        echo "  - FOCAL (Google Drive - 2 files)"
        echo "  - TruFor (direct URL)"
        echo "  - PSCCNet (manual download only)"
        return 0
    fi

    mkdir -p "$PHOTOHOLMES_TARGET"
    local success_count=0
    local total_methods=6  # AdaptiveCFANet, CAT-Net, EXIF, FOCAL, PSCCNet, TruFor

    log_info "Downloading PhotoHolmes weights..."

    # ========================================================================
    # 1. AdaptiveCFANet - download from raw GitHub and rename to weights.pth
    # ========================================================================
    local adaptive_cfa_dir="${PHOTOHOLMES_TARGET}/adaptive_cfa_net"
    local adaptive_cfa_target="${adaptive_cfa_dir}/weights.pth"

    if check_file_exists_and_valid "$adaptive_cfa_target"; then
        log_info "  [1/6] AdaptiveCFANet already exists (skipped)"
        success_count=$((success_count + 1))
    else
        mkdir -p "$adaptive_cfa_dir"
        log_info "  [1/6] Downloading AdaptiveCFANet..."

        # Download to temp file first, then move if valid (non-empty)
        local tmp_file=$(mktemp).pt
        mkdir -p "$(dirname "$tmp_file")"
        if curl -fL --progress-bar "https://raw.githubusercontent.com/qbammey/adaptive_cfa_forensics/master/src/models/pretrained.pt" -o "$tmp_file" 2>/dev/null; then
            # Verify file is non-empty before moving
            local downloaded_size=$(get_file_size "$tmp_file")
            if [ "$downloaded_size" -gt 0 ]; then
                mv "$tmp_file" "$adaptive_cfa_target"
                success_count=$((success_count + 1))
                log_success "  [1/6] Downloaded AdaptiveCFANet ($(format_size "$downloaded_size"))"
            else
                rm -f "$tmp_file"
                log_error "  [1/6] Downloaded file is empty"
            fi
        else
            rm -f "$tmp_file"
            log_error "  [1/6] Failed to download AdaptiveCFANet"
        fi
    fi

    # ========================================================================
    # 2. CAT-Net - Google Drive (2 files: v1 and v2)
    # ========================================================================
    local catnet_dir="${PHOTOHOLMES_TARGET}/catnet"
    mkdir -p "$catnet_dir"

    # CAT-Net v1
    local catnet_v1_target="${catnet_dir}/CAT_full_v1.pth"
    local catnet_v1_valid=false
    if check_file_exists_and_valid "$catnet_v1_target"; then
        log_info "  [2a/6] CAT-Net v1 already exists (skipped)"
        catnet_v1_valid=true
    else
        log_info "  [2a/6] Downloading CAT-Net v1..."
        if command -v gdown >/dev/null 2>&1; then
            local tmp_file=$(mktemp).pth
            if gdown -q "https://drive.google.com/file/d/1NXLDCn0ABG7eWEXltGZ4SyIsREhOUhRM/view" -O "$tmp_file" 2>/dev/null; then
                local downloaded_size=$(get_file_size "$tmp_file")
                if [ "$downloaded_size" -gt 0 ]; then
                    mv "$tmp_file" "$catnet_v1_target"
                    log_success "  [2a/6] Downloaded CAT-Net v1 ($(format_size "$downloaded_size"))"
                    catnet_v1_valid=true
                else
                    rm -f "$tmp_file"
                    log_error "  [2a/6] Downloaded file is empty"
                fi
            else
                log_error "  [2a/6] Failed to download CAT-Net v1"
            fi
        else
            log_error "  [2a/6] gdown not found - install with: pip install gdown"
        fi
    fi

    # CAT-Net v2
    local catnet_v2_target="${catnet_dir}/CAT_full_v2.pth"
    local catnet_v2_valid=false
    if check_file_exists_and_valid "$catnet_v2_target"; then
        log_info "  [2b/6] CAT-Net v2 already exists (skipped)"
        catnet_v2_valid=true
    else
        log_info "  [2b/6] Downloading CAT-Net v2..."
        if command -v gdown >/dev/null 2>&1; then
            local tmp_file=$(mktemp).pth
            if gdown -q "https://drive.google.com/file/d/1tyOKVdx6UMys2OcNpUj9r6scxNIpcoLE/view" -O "$tmp_file" 2>/dev/null; then
                local downloaded_size=$(get_file_size "$tmp_file")
                if [ "$downloaded_size" -gt 0 ]; then
                    mv "$tmp_file" "$catnet_v2_target"
                    log_success "  [2b/6] Downloaded CAT-Net v2 ($(format_size "$downloaded_size"))"
                    catnet_v2_valid=true
                else
                    rm -f "$tmp_file"
                    log_error "  [2b/6] Downloaded file is empty"
                fi
            else
                log_error "  [2b/6] Failed to download CAT-Net v2"
            fi
        else
            log_error "  [2b/6] gdown not found - install with: pip install gdown"
        fi
    fi

    # Count CAT-Net as success if both files exist and are valid
    if [ "$catnet_v1_valid" = true ] && [ "$catnet_v2_valid" = true ]; then
        success_count=$((success_count + 1))
    fi

    # ========================================================================
    # 3. EXIF as Language - Google Drive (needs pruning)
    # ========================================================================
    local exif_dir="${PHOTOHOLMES_TARGET}/exif_as_language"
    mkdir -p "$exif_dir"

    local exif_original="${exif_dir}/wrapper_75_new.pth"
    local exif_pruned="${exif_dir}/weights.pth"

    if check_file_exists_and_valid "$exif_pruned"; then
        log_info "  [3/6] EXIF as Language already exists (skipped)"
        success_count=$((success_count + 1))
    elif check_file_exists_and_valid "$exif_original"; then
        log_info "  [3/6] EXIF as Language weights downloaded but need pruning..."
        if [ "$DRY_RUN" = "true" ]; then
            log_info "  [3/6] Would run: python -m photoholmes.methods.exif_as_language prune-weights"
            success_count=$((success_count + 1))
        else
            log_info "  [3/6] Pruning weights..."
            if (cd "$exif_dir" && python -m photoholmes.methods.exif_as_language prune-weights >/dev/null 2>&1); then
                if [ -f "$exif_pruned" ]; then
                    log_success "  [3/6] Pruned EXIF as Language weights"
                    success_count=$((success_count + 1))
                else
                    log_error "  [3/6] Pruning failed - weights.pth not created"
                fi
            else
                log_error "  [3/6] Pruning failed - Run manually: python -m photoholmes.methods.exif_as_language prune-weights"
            fi
        fi
    else
        log_info "  [3/6] Downloading EXIF as Language..."
        if command -v gdown >/dev/null 2>&1; then
            local tmp_file=$(mktemp).pth
            if gdown -q --id "17MW-fZRRQQ8dSRv52X_9DmcmdQD7TmHZ" -O "$tmp_file" 2>/dev/null; then
                local downloaded_size=$(get_file_size "$tmp_file")
                if [ "$downloaded_size" -gt 0 ]; then
                    mv "$tmp_file" "$exif_original"
                    log_success "  [3/6] Downloaded EXIF as Language ($(format_size "$downloaded_size"))"
                    # Prune the weights to create weights.pth
                    log_info "  [3/6] Pruning weights..."
                    if (cd "$exif_dir" && python -m photoholmes.methods.exif_as_language prune-weights >/dev/null 2>&1); then
                        if [ -f "$exif_pruned" ]; then
                            log_success "  [3/6] Pruned EXIF as Language weights"
                            success_count=$((success_count + 1))
                        else
                            log_error "  [3/6] Pruning failed - weights.pth not created"
                        fi
                    else
                        log_error "  [3/6] Pruning failed - Run manually: python -m photoholmes.methods.exif_as_language prune-weights"
                    fi
                else
                    rm -f "$tmp_file"
                    log_error "  [3/6] Downloaded file is empty"
                fi
            else
                log_error "  [3/6] Failed to download EXIF as Language"
            fi
        else
            log_error "  [3/6] gdown not found - install with: pip install gdown"
        fi
    fi

    # ========================================================================
    # 4. FOCAL - Google Drive (2 files: ViT and HRNet)
    # ========================================================================
    local focal_dir="${PHOTOHOLMES_TARGET}/focal"
    mkdir -p "$focal_dir"

    # FOCAL ViT
    local focal_vit_target="${focal_dir}/VIT_weights.pth"
    local focal_vit_valid=false
    if check_file_exists_and_valid "$focal_vit_target"; then
        log_info "  [4a/6] FOCAL ViT already exists (skipped)"
        focal_vit_valid=true
    else
        log_info "  [4a/6] Downloading FOCAL ViT..."
        if command -v gdown >/dev/null 2>&1; then
            local tmp_file=$(mktemp).pth
            if gdown -q --id "1GQMU8FHwi2K3XkkHhe71bt-RQvuA2VQ4" -O "$tmp_file" 2>/dev/null; then
                local downloaded_size=$(get_file_size "$tmp_file")
                if [ "$downloaded_size" -gt 0 ]; then
                    mv "$tmp_file" "$focal_vit_target"
                    log_success "  [4a/6] Downloaded FOCAL ViT ($(format_size "$downloaded_size"))"
                    focal_vit_valid=true
                else
                    rm -f "$tmp_file"
                    log_error "  [4a/6] Downloaded file is empty"
                fi
            else
                log_error "  [4a/6] Failed to download FOCAL ViT"
            fi
        else
            log_error "  [4a/6] gdown not found - install with: pip install gdown"
        fi
    fi

    # FOCAL HRNet
    local focal_hrnet_target="${focal_dir}/HRNet_weights.pth"
    local focal_hrnet_valid=false
    if check_file_exists_and_valid "$focal_hrnet_target"; then
        log_info "  [4b/6] FOCAL HRNet already exists (skipped)"
        focal_hrnet_valid=true
    else
        log_info "  [4b/6] Downloading FOCAL HRNet..."
        if command -v gdown >/dev/null 2>&1; then
            local tmp_file=$(mktemp).pth
            if gdown -q --id "1O_iyg5Tg_iZ5u_yGcU_MhKVH-c6MIpdR" -O "$tmp_file" 2>/dev/null; then
                local downloaded_size=$(get_file_size "$tmp_file")
                if [ "$downloaded_size" -gt 0 ]; then
                    mv "$tmp_file" "$focal_hrnet_target"
                    log_success "  [4b/6] Downloaded FOCAL HRNet ($(format_size "$downloaded_size"))"
                    focal_hrnet_valid=true
                else
                    rm -f "$tmp_file"
                    log_error "  [4b/6] Downloaded file is empty"
                fi
            else
                log_error "  [4b/6] Failed to download FOCAL HRNet"
            fi
        else
            log_error "  [4b/6] gdown not found - install with: pip install gdown"
        fi
    fi

    # Count FOCAL as success if both files exist and are valid
    if [ "$focal_vit_valid" = true ] && [ "$focal_hrnet_valid" = true ]; then
        success_count=$((success_count + 1))
    fi

    # ========================================================================
    # 5. TruFor - Direct URL (ZIP contains tar file)
    # ========================================================================
    local trufor_dir="${PHOTOHOLMES_TARGET}/trufor"

    # Check for the extracted weights file
    if check_file_exists_and_valid "$trufor_dir/trufor.pth"; then
        log_info "  [5/6] TruFor already exists (skipped)"
        success_count=$((success_count + 1))
    else
        log_info "  [5/6] Downloading TruFor..."
        local tmp_zip=$(mktemp).zip
        local tmp_extract=$(mktemp -d)
        local download_success=false

        # Download with curl directly
        mkdir -p "$(dirname "$tmp_zip")"
        if curl -fL --progress-bar "https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip" -o "$tmp_zip" 2>/dev/null; then
            # Check minimum size (1MB for TruFor weights)
            local zip_size=$(get_file_size "$tmp_zip")
            if [ "$zip_size" -lt 1000000 ]; then
                log_error "  [5/6] Downloaded file too small ($zip_size bytes)"
                rm -f "$tmp_zip"
            else
                # Validate it's actually a ZIP file
                local file_type=$(file "$tmp_zip" | head -1)
                if ! echo "$file_type" | grep -qi "zip"; then
                    log_error "  [5/6] Downloaded file is not a ZIP archive"
                    log_error "  [5/6] File type: $file_type"
                    rm -f "$tmp_zip"
                else
                    # Test ZIP integrity
                    if ! unzip -t "$tmp_zip" >/dev/null 2>&1; then
                        log_error "  [5/6] ZIP file is corrupted"
                        rm -f "$tmp_zip"
                    else
                        mkdir -p "$trufor_dir"
                        # Unzip to get the tar file
                        if unzip -q -o "$tmp_zip" -d "$tmp_extract" 2>/dev/null; then
                            # Move the .pth.tar file as trufor.pth (PyTorch can load it directly)
                            # The tar contains an internal archive/ structure - don't extract it
                            if [ -f "$tmp_extract/weights/trufor.pth.tar" ]; then
                                local tar_size=$(get_file_size "$tmp_extract/weights/trufor.pth.tar")
                                if [ "$tar_size" -gt 0 ]; then
                                    mv "$tmp_extract/weights/trufor.pth.tar" "$trufor_dir/trufor.pth"
                                    download_success=true
                                    log_success "  [5/6] Downloaded TruFor ($(format_size "$tar_size"))"
                                fi
                            fi
                        fi
                    fi
                fi
            fi
        fi

        # Cleanup
        rm -f "$tmp_zip"
        rm -rf "$tmp_extract"

        if [ "$download_success" = true ]; then
            success_count=$((success_count + 1))
        else
            rm -rf "$trufor_dir"  # Clean up incomplete download
            log_error "  [5/6] Failed to download TruFor"
        fi
    fi

    # ========================================================================
    # 6. PSCCNet - Direct URL (ZIP)
    # ========================================================================
    local psccnet_dir="${PHOTOHOLMES_TARGET}/psccnet"
    local psccnet_valid=false

    # Check for all three weight files
    if check_file_exists_and_valid "$psccnet_dir/FENet.pth" && \
       check_file_exists_and_valid "$psccnet_dir/SegNet.pth" && \
       check_file_exists_and_valid "$psccnet_dir/ClsNet.pth"; then
        log_info "  [6/6] PSCCNet already exists (skipped)"
        success_count=$((success_count + 1))
        psccnet_valid=true
    else
        log_info "  [6/6] Downloading PSCCNet..."
        local tmp_zip=$(mktemp).zip
        local tmp_extract=$(mktemp -d)
        local download_success=false

        # Download with curl directly
        mkdir -p "$(dirname "$tmp_zip")"
        if curl -fL --progress-bar "https://www.immin.io/public/assets/business/psccnet.zip" -o "$tmp_zip" 2>/dev/null; then
            # Check minimum size (1MB for PSCCNet weights)
            local zip_size=$(get_file_size "$tmp_zip")
            if [ "$zip_size" -lt 1000000 ]; then
                log_error "  [6/6] Downloaded file too small ($zip_size bytes)"
                rm -f "$tmp_zip"
            else
                # Validate it's actually a ZIP file
                local file_type=$(file "$tmp_zip" | head -1)
                if ! echo "$file_type" | grep -qi "zip"; then
                    log_error "  [6/6] Downloaded file is not a ZIP archive"
                    log_error "  [6/6] File type: $file_type"
                    rm -f "$tmp_zip"
                else
                    # Test ZIP integrity
                    if ! unzip -t "$tmp_zip" >/dev/null 2>&1; then
                        log_error "  [6/6] ZIP file is corrupted"
                        rm -f "$tmp_zip"
                    else
                        mkdir -p "$psccnet_dir"
                        # Extract the zip
                        if unzip -q -o "$tmp_zip" -d "$tmp_extract" 2>/dev/null; then
                            # Move the three weight files
                            local fenet_valid=false
                            local segnet_valid=false
                            local clsnet_valid=false

                            if [ -f "$tmp_extract/psccnet/FENet.pth" ]; then
                                local fenet_size=$(get_file_size "$tmp_extract/psccnet/FENet.pth")
                                if [ "$fenet_size" -gt 0 ]; then
                                    mv "$tmp_extract/psccnet/FENet.pth" "$psccnet_dir/FENet.pth"
                                    fenet_valid=true
                                fi
                            fi

                            if [ -f "$tmp_extract/psccnet/SegNet.pth" ]; then
                                local segnet_size=$(get_file_size "$tmp_extract/psccnet/SegNet.pth")
                                if [ "$segnet_size" -gt 0 ]; then
                                    mv "$tmp_extract/psccnet/SegNet.pth" "$psccnet_dir/SegNet.pth"
                                    segnet_valid=true
                                fi
                            fi

                            if [ -f "$tmp_extract/psccnet/ClsNet.pth" ]; then
                                local clsnet_size=$(get_file_size "$tmp_extract/psccnet/ClsNet.pth")
                                if [ "$clsnet_size" -gt 0 ]; then
                                    mv "$tmp_extract/psccnet/ClsNet.pth" "$psccnet_dir/ClsNet.pth"
                                    clsnet_valid=true
                                fi
                            fi

                            # Count as success only if all three files were extracted
                            if [ "$fenet_valid" = true ] && [ "$segnet_valid" = true ] && [ "$clsnet_valid" = true ]; then
                                download_success=true
                                psccnet_valid=true
                                local total_size=$(($(get_file_size "$psccnet_dir/FENet.pth") + $(get_file_size "$psccnet_dir/SegNet.pth") + $(get_file_size "$psccnet_dir/ClsNet.pth")))
                                log_success "  [6/6] Downloaded PSCCNet ($(format_size "$total_size"))"
                            fi
                        fi
                    fi
                fi
            fi
        else
            log_error "  [6/6] Failed to download PSCCNet"
        fi

        # Cleanup
        rm -f "$tmp_zip"
        rm -rf "$tmp_extract"

        if [ "$download_success" = true ]; then
            success_count=$((success_count + 1))
        else
            rm -rf "$psccnet_dir"  # Clean up incomplete download
            log_error "  [6/6] Failed to download PSCCNet"
        fi
    fi

    # Summary
    echo ""
    log_info "  PhotoHolmes methods downloaded: $success_count/$total_methods"
    if [ $success_count -eq $total_methods ]; then
        log_success "All PhotoHolmes weights downloaded successfully"
        return 0
    elif [ $success_count -gt 0 ]; then
        log_warning "Some PhotoHolmes weights downloaded ($success_count/$total_methods)"
        return 0
    else
        log_error "Failed to download PhotoHolmes weights"
        return 1
    fi
}

# ============================================================================
# spaCy Download
# ============================================================================

download_spacy() {
    log_section "spaCy"

    local spacy_model_path="${SPACY_TARGET}/en_core_web_sm"

    # Check if key model files exist and are valid (non-empty)
    if ! $FORCE && check_dir_has_files "$spacy_model_path"; then
        if check_file_exists_and_valid "$spacy_model_path/en_core_web_sm-3.7.1-py3-none-any.whl" || \
           (check_file_exists_and_valid "$spacy_model_path/model.bin" && \
            check_file_exists_and_valid "$spacy_model_path/vocab"); then
            log_info "Already exists at ${spacy_model_path#$PROJECT_ROOT/} (skipped)"
            return 0
        fi
    fi

    if [ "$DRY_RUN" = "true" ]; then
        log_info "Would download spaCy model to ${SPACY_TARGET#$PROJECT_ROOT/}"
        return 0
    fi

    mkdir -p "$SPACY_TARGET"
    local tmp_dir=$(mktemp -d)

    log_info "Downloading spaCy language model..."

    if download_file "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl" \
            "$tmp_dir/spacy.whl" "true"; then
        if extract_zip "$tmp_dir/spacy.whl" "$tmp_dir"; then
            # Find the extracted en_core_web_sm directory and move it
            if [ -d "$tmp_dir/en_core_web_sm" ]; then
                rm -rf "$spacy_model_path"  # Remove old version if exists
                mv "$tmp_dir/en_core_web_sm" "$SPACY_TARGET/"
                rm -rf "$tmp_dir"
                file_count=$(find "$spacy_model_path" -type f | wc -l)
                total_size=$(du -sm "$spacy_model_path" | cut -f1)
                log_success "Downloaded $file_count file(s) (${total_size} MB)"
                return 0
            else
                rm -rf "$tmp_dir"
                log_error "Failed to extract spaCy model"
                return 1
            fi
        else
            rm -rf "$tmp_dir"
            log_error "Failed to extract spaCy model"
            return 1
        fi
    else
        rm -rf "$tmp_dir"
        log_error "Failed to download spaCy model"
        return 1
    fi
}

# ============================================================================
# Parse Arguments
# ============================================================================

DOWNLOAD_ALL=false
DOWNLOAD_DEEPFACE=false
DOWNLOAD_DOCTR=false
DOWNLOAD_HUGGINGFACE=false
DOWNLOAD_MEDIAPIPE=false
DOWNLOAD_NLTK=false
DOWNLOAD_PHOTOHOLMES=false
DOWNLOAD_SPACY=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --all)
            DOWNLOAD_ALL=true
            shift
            ;;
        --deepface)
            DOWNLOAD_DEEPFACE=true
            shift
            ;;
        --doctr)
            DOWNLOAD_DOCTR=true
            shift
            ;;
        --huggingface)
            DOWNLOAD_HUGGINGFACE=true
            shift
            ;;
        --mediapipe)
            DOWNLOAD_MEDIAPIPE=true
            shift
            ;;
        --nltk)
            DOWNLOAD_NLTK=true
            shift
            ;;
        --photoholmes)
            DOWNLOAD_PHOTOHOLMES=true
            shift
            ;;
        --spacy)
            DOWNLOAD_SPACY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Download ML/DL model weights from official sources."
            echo ""
            echo "Options:"
            echo "  --all              Download all weights (default)"
            echo "  --deepface         Download DeepFace weights"
            echo "  --doctr            Download DocTR weights"
            echo "  --huggingface      Download HuggingFace models"
            echo "  --mediapipe        Download MediaPipe model"
            echo "  --nltk             Download NLTK data"
            echo "  --photoholmes      Download PhotoHolmes weights"
            echo "  --spacy            Download spaCy model"
            echo "  --dry-run          Show what would be downloaded"
            echo "  --force            Re-download existing files"
            echo "  --verbose          Show detailed diagnostic information"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# If no specific downloader selected, run all
if [ "$DOWNLOAD_DEEPFACE" = false ] && [ "$DOWNLOAD_DOCTR" = false ] && \
   [ "$DOWNLOAD_HUGGINGFACE" = false ] && [ "$DOWNLOAD_MEDIAPIPE" = false ] && \
   [ "$DOWNLOAD_NLTK" = false ] && [ "$DOWNLOAD_PHOTOHOLMES" = false ] && \
   [ "$DOWNLOAD_SPACY" = false ]; then
    DOWNLOAD_ALL=true
fi

# ============================================================================
# Main
# ============================================================================

echo ""
echo "============================================================"
echo " Model Weights Download Script"
echo " Using direct curl/wget downloads"
echo "============================================================"
echo ""

# Track results (using simple variables for bash 3.x compatibility)
RESULT_DEEPFACE=1
RESULT_DOCTR=1
RESULT_HUGGINGFACE=1
RESULT_MEDIAPIPE=1
RESULT_NLTK=1
RESULT_PHOTOHOLMES=1
RESULT_SPACY=1

if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_DEEPFACE" = true ]; then
    if download_deepface; then
        RESULT_DEEPFACE=0
    fi
fi

if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_DOCTR" = true ]; then
    if download_doctr; then
        RESULT_DOCTR=0
    fi
fi

if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_HUGGINGFACE" = true ]; then
    if download_huggingface; then
        RESULT_HUGGINGFACE=0
    fi
fi

if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_MEDIAPIPE" = true ]; then
    if download_mediapipe; then
        RESULT_MEDIAPIPE=0
    fi
fi

if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_NLTK" = true ]; then
    if download_nltk; then
        RESULT_NLTK=0
    fi
fi

if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_PHOTOHOLMES" = true ]; then
    if download_photoholmes; then
        RESULT_PHOTOHOLMES=0
    fi
fi

if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_SPACY" = true ]; then
    if download_spacy; then
        RESULT_SPACY=0
    fi
fi

# ============================================================================
# Summary
# ============================================================================

if [ "$DRY_RUN" = false ]; then
    echo ""
    echo "============================================================"
    echo " Summary"
    echo "============================================================"

    # Display results
    if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_DEEPFACE" = true ]; then
        if [ $RESULT_DEEPFACE -eq 0 ]; then
            echo -e " ${GREEN}✓${NC} deepface"
        else
            echo -e " ${RED}✗${NC} deepface"
        fi
    fi

    if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_DOCTR" = true ]; then
        if [ $RESULT_DOCTR -eq 0 ]; then
            echo -e " ${GREEN}✓${NC} doctr"
        else
            echo -e " ${RED}✗${NC} doctr"
        fi
    fi

    if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_HUGGINGFACE" = true ]; then
        if [ $RESULT_HUGGINGFACE -eq 0 ]; then
            echo -e " ${GREEN}✓${NC} huggingface"
        else
            echo -e " ${RED}✗${NC} huggingface"
        fi
    fi

    if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_MEDIAPIPE" = true ]; then
        if [ $RESULT_MEDIAPIPE -eq 0 ]; then
            echo -e " ${GREEN}✓${NC} mediapipe"
        else
            echo -e " ${RED}✗${NC} mediapipe"
        fi
    fi

    if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_NLTK" = true ]; then
        if [ $RESULT_NLTK -eq 0 ]; then
            echo -e " ${GREEN}✓${NC} nltk"
        else
            echo -e " ${RED}✗${NC} nltk"
        fi
    fi

    if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_PHOTOHOLMES" = true ]; then
        if [ $RESULT_PHOTOHOLMES -eq 0 ]; then
            echo -e " ${GREEN}✓${NC} photoholmes"
        else
            echo -e " ${RED}✗${NC} photoholmes"
        fi
    fi

    if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_SPACY" = true ]; then
        if [ $RESULT_SPACY -eq 0 ]; then
            echo -e " ${GREEN}✓${NC} spacy"
        else
            echo -e " ${RED}✗${NC} spacy"
        fi
    fi

    echo ""

    # Count successes
    SUCCESS_COUNT=0
    TOTAL_COUNT=0

    if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_DEEPFACE" = true ]; then
        TOTAL_COUNT=$((TOTAL_COUNT + 1))
        if [ $RESULT_DEEPFACE -eq 0 ]; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        fi
    fi

    if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_DOCTR" = true ]; then
        TOTAL_COUNT=$((TOTAL_COUNT + 1))
        if [ $RESULT_DOCTR -eq 0 ]; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        fi
    fi

    if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_HUGGINGFACE" = true ]; then
        TOTAL_COUNT=$((TOTAL_COUNT + 1))
        if [ $RESULT_HUGGINGFACE -eq 0 ]; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        fi
    fi

    if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_MEDIAPIPE" = true ]; then
        TOTAL_COUNT=$((TOTAL_COUNT + 1))
        if [ $RESULT_MEDIAPIPE -eq 0 ]; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        fi
    fi

    if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_NLTK" = true ]; then
        TOTAL_COUNT=$((TOTAL_COUNT + 1))
        if [ $RESULT_NLTK -eq 0 ]; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        fi
    fi

    if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_PHOTOHOLMES" = true ]; then
        TOTAL_COUNT=$((TOTAL_COUNT + 1))
        if [ $RESULT_PHOTOHOLMES -eq 0 ]; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        fi
    fi

    if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_SPACY" = true ]; then
        TOTAL_COUNT=$((TOTAL_COUNT + 1))
        if [ $RESULT_SPACY -eq 0 ]; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        fi
    fi

    if [ $SUCCESS_COUNT -eq $TOTAL_COUNT ]; then
        log_success "All $TOTAL_COUNT downloaders succeeded!"
    else
        log_warning "$SUCCESS_COUNT/$TOTAL_COUNT succeeded"
    fi

    echo "============================================================"
    echo ""
fi

# Exit with error code if any failed
# Only check the downloaders that were actually run
if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_DEEPFACE" = true ]; then
    if [ $RESULT_DEEPFACE -ne 0 ]; then
        exit 1
    fi
fi

if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_DOCTR" = true ]; then
    if [ $RESULT_DOCTR -ne 0 ]; then
        exit 1
    fi
fi

if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_HUGGINGFACE" = true ]; then
    if [ $RESULT_HUGGINGFACE -ne 0 ]; then
        exit 1
    fi
fi

if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_MEDIAPIPE" = true ]; then
    if [ $RESULT_MEDIAPIPE -ne 0 ]; then
        exit 1
    fi
fi

if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_NLTK" = true ]; then
    if [ $RESULT_NLTK -ne 0 ]; then
        exit 1
    fi
fi

if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_PHOTOHOLMES" = true ]; then
    if [ $RESULT_PHOTOHOLMES -ne 0 ]; then
        exit 1
    fi
fi

if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_SPACY" = true ]; then
    if [ $RESULT_SPACY -ne 0 ]; then
        exit 1
    fi
fi

exit 0
