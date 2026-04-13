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
#   --insightface      InsightFace face recognition

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
INSIGHTFACE_TARGET="${APP_DIR}/models/insightface"

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

# Download a file with optional progress bar
download_file() {
    local url="$1"
    local target="$2"
    local show_progress="${3:-true}"

    # Skip if exists and not forcing
    if [ -f "$target" ] && [ "$FORCE" != "true" ]; then
        return 0
    fi

    mkdir -p "$(dirname "$target")"

    if [ "$DRY_RUN" = "true" ]; then
        echo "Would download: $url -> $target"
        return 0
    fi

    log_info "  Downloading $(basename "$target")..."

    if [ "$show_progress" = "true" ]; then
        if curl -fL --progress-bar "$url" -o "$target"; then
            return 0
        else
            log_error "  Failed to download $(basename "$target")"
            rm -f "$target"
            return 1
        fi
    else
        if curl -fLs "$url" -o "$target"; then
            return 0
        else
            log_error "  Failed to download $(basename "$target")"
            rm -f "$target"
            return 1
        fi
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
    if unzip -q -o "$zip_file" -d "$target_dir" 2>/dev/null; then
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
        log_info "Already exists at ${DEEPFACE_TARGET#$PROJECT_ROOT/} (skipped)"
        return 0
    fi

    if [ "$DRY_RUN" = "true" ]; then
        log_info "Would download DeepFace weights to ${DEEPFACE_TARGET#$PROJECT_ROOT/}"
        return 0
    fi

    mkdir -p "$DEEPFACE_TARGET"
    local tmp_dir=$(mktemp -d)
    local success_count=0
    local total_count=10

    log_info "Downloading DeepFace weights..."

    # VGG-Face
    if download_file "https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5" \
            "$DEEPFACE_TARGET/vgg_face_weights.h5"; then
        success_count=$((success_count + 1))
    fi

    # Facenet
    if download_file "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5" \
            "$DEEPFACE_TARGET/facenet_weights.h5"; then
        success_count=$((success_count + 1))
    fi

    # Facenet512
    if download_file "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5" \
            "$DEEPFACE_TARGET/facenet512_weights.h5"; then
        success_count=$((success_count + 1))
    fi

    # OpenFace
    if download_file "https://github.com/serengil/deepface_models/releases/download/v1.0/openface_weights.h5" \
            "$DEEPFACE_TARGET/openface_weights.h5"; then
        success_count=$((success_count + 1))
    fi

    # DeepFace (ZIP format)
    if download_file "https://github.com/swghosh/DeepFace/releases/download/weights-vggface2-2d-aligned/VGGFace2_DeepFace_weights_val-0.9034.h5.zip" \
            "$tmp_dir/deepface.zip"; then
        if extract_zip "$tmp_dir/deepface.zip" "$tmp_dir"; then
            mv "$tmp_dir"/*.h5 "$DEEPFACE_TARGET/" 2>/dev/null || true
            success_count=$((success_count + 1))
        fi
    fi

    # DeepID
    if download_file "https://github.com/serengil/deepface_models/releases/download/v1.0/deepid_keras_weights.h5" \
            "$DEEPFACE_TARGET/deepid_keras_weights.h5"; then
        success_count=$((success_count + 1))
    fi

    # ArcFace
    if download_file "https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5" \
            "$DEEPFACE_TARGET/arcface_weights.h5"; then
        success_count=$((success_count + 1))
    fi

    # SFace
    if download_file "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" \
            "$DEEPFACE_TARGET/face_recognition_sface_2021dec.onnx"; then
        success_count=$((success_count + 1))
    fi

    # GhostFaceNet
    if download_file "https://github.com/HamadYA/GhostFaceNets/releases/download/v1.2/GhostFaceNet_W1.3_S1_ArcFace.h5" \
            "$DEEPFACE_TARGET/GhostFaceNet_W1.3_S1_ArcFace.h5"; then
        success_count=$((success_count + 1))
    fi

    # Dlib (BZ2 format)
    if download_file "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2" \
            "$tmp_dir/dlib_face_recognition_resnet_model_v1.dat.bz2"; then
        if extract_bz2 "$tmp_dir/dlib_face_recognition_resnet_model_v1.dat.bz2" "$DEEPFACE_TARGET" \
                "dlib_face_recognition_resnet_model_v1.dat"; then
            success_count=$((success_count + 1))
        fi
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

    if ! $FORCE && check_dir_has_files "$DOCTR_TARGET"; then
        log_info "Already exists at ${DOCTR_TARGET#$PROJECT_ROOT/} (skipped)"
        return 0
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
            "$DOCTR_TARGET/db_resnet50-79bd7d70.pt"; then
        success_count=$((success_count + 1))
    fi

    # Recognition model (crnn_vgg16_bn)
    if download_file "https://doctr-static.mindee.com/models?id=v0.12.0/crnn_vgg16_bn-0417f351.pt&src=0" \
            "$DOCTR_TARGET/crnn_vgg16_bn-0417f351.pt"; then
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

# ============================================================================
# HuggingFace Models (FinBERT, GLiNER2)
# ============================================================================

download_huggingface() {
    log_section "HuggingFace Models"

    if [ "$DRY_RUN" = "true" ]; then
        log_info "Would download HuggingFace models:"
        log_info "  - FinBERT (ProsusAI/finbert)"
        log_info "  - GLiNER2 (urchade/gliner_large-v2.1)"
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
            "$FINBERT_TARGET/pytorch_model.bin"; then
        finbert_success=$((finbert_success + 1))
    fi

    if download_file "https://huggingface.co/ProsusAI/finbert/resolve/main/config.json" \
            "$FINBERT_TARGET/config.json"; then
        finbert_success=$((finbert_success + 1))
    fi

    if download_file "https://huggingface.co/ProsusAI/finbert/resolve/main/vocab.txt" \
            "$FINBERT_TARGET/vocab.txt"; then
        finbert_success=$((finbert_success + 1))
    fi

    if download_file "https://huggingface.co/ProsusAI/finbert/resolve/main/tokenizer_config.json" \
            "$FINBERT_TARGET/tokenizer_config.json"; then
        finbert_success=$((finbert_success + 1))
    fi

    # GLiNER2 files - urchade/gliner_large-v2.1 only has 2 files
    log_info "  Downloading GLiNER2 (urchade/gliner_large-v2.1)..."

    if download_file "https://huggingface.co/urchade/gliner_large-v2.1/resolve/main/gliner_config.json" \
            "$GLINER2_TARGET/gliner_config.json"; then
        gliner2_success=$((gliner2_success + 1))
    fi

    if download_file "https://huggingface.co/urchade/gliner_large-v2.1/resolve/main/pytorch_model.bin" \
            "$GLINER2_TARGET/pytorch_model.bin"; then
        gliner2_success=$((gliner2_success + 1))
    fi

    if [ $finbert_success -eq 4 ] && [ $gliner2_success -eq 2 ]; then
        log_success "HuggingFace models downloaded"
        return 0
    else
        log_warning "FinBERT: $finbert_success/4 files, GLiNER2: $gliner2_success/2 files"
        return 1
    fi
}

# ============================================================================
# MediaPipe Download
# ============================================================================

download_mediapipe() {
    log_section "MediaPipe"

    if ! $FORCE && [ -f "$MEDIAPIPE_TARGET/hand_landmarker.task" ]; then
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
            "$MEDIAPIPE_TARGET/hand_landmarker.task"; then
        file_size=$(du -sm "$MEDIAPIPE_TARGET" | cut -f1)
        log_success "Downloaded hand_landmarker.task (${file_size} MB)"
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

    if ! $FORCE && check_dir_has_files "$nltk_sentiment_dir"; then
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
            "$tmp_dir/vader_lexicon.zip"; then
        if extract_zip "$tmp_dir/vader_lexicon.zip" "$NLTK_TARGET"; then
            rm -rf "$tmp_dir"
            file_count=$(find "$NLTK_TARGET" -type f | wc -l)
            log_success "Downloaded $file_count file(s)"
            return 0
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
        log_info "Would download PhotoHolmes weights to ${PHOTOHOLMES_TARGET#$PROJECT_ROOT/}"
        return 0
    fi

    mkdir -p "$PHOTOHOLMES_TARGET"
    local success_count=0

    log_info "Downloading PhotoHolmes weights..."

    # AdaptiveCFANet - download from raw GitHub and rename to weights.pth
    local adaptive_cfa_dir="${PHOTOHOLMES_TARGET}/adaptive_cfa_net"
    local adaptive_cfa_target="${adaptive_cfa_dir}/weights.pth"

    if [ -f "$adaptive_cfa_target" ] && [ "$FORCE" != "true" ]; then
        log_info "  AdaptiveCFANet already exists (skipped)"
        success_count=$((success_count + 1))
    else
        mkdir -p "$adaptive_cfa_dir"
        log_info "  Downloading AdaptiveCFANet..."

        # Download as pretrained.pt and rename to weights.pth
        local tmp_file=$(mktemp)
        if download_file "https://raw.githubusercontent.com/qbammey/adaptive_cfa_forensics/master/src/models/pretrained.pt" \
                "$tmp_file" "false"; then
            mv "$tmp_file" "$adaptive_cfa_target"
            success_count=$((success_count + 1))
            log_success "  Downloaded AdaptiveCFANet"
        else
            rm -f "$tmp_file"
            log_error "  Failed to download AdaptiveCFANet"
        fi
    fi

    # PSCCNet - CANNOT be downloaded via direct URL
    # The checkpoint files require manual download from the repository
    log_warning "  PSCCNet weights cannot be downloaded automatically"
    log_info "  Manual download required for PSCCNet:"
    echo ""
    echo "    Option 1: Clone the PSCC-Net repository"
    echo "      git clone https://github.com/proteus1991/PSCC-Net.git /tmp/psccnet"
    echo "      mkdir -p ${PHOTOHOLMES_TARGET#$PROJECT_ROOT/}/psccnet"
    echo "      cp /tmp/psccnet/checkpoint/HRNet_checkpoint/HRNet.pth \\"
    echo "         ${PHOTOHOLMES_TARGET#$PROJECT_ROOT/}/psccnet/FENet.pth"
    echo "      cp /tmp/psccnet/checkpoint/NLCDetection_checkpoint/NLCDetection.pth \\"
    echo "         ${PHOTOHOLMES_TARGET#$PROJECT_ROOT/}/psccnet/SegNet.pth"
    echo "      cp /tmp/psccnet/checkpoint/DetectionHead_checkpoint/DetectionHead.pth \\"
    echo "         ${PHOTOHOLMES_TARGET#$PROJECT_ROOT/}/psccnet/ClsNet.pth"
    echo ""
    echo "    Option 2: Download from Baidu Cloud (password: js74)"
    echo "      See: https://github.com/proteus1991/PSCC-Net"
    echo ""

    # Summary
    if [ $success_count -eq 1 ]; then
        log_success "AdaptiveCFANet downloaded (PSCCNet requires manual download)"
        return 0
    else
        log_error "Failed to download AdaptiveCFANet"
        return 1
    fi
}

# ============================================================================
# spaCy Download
# ============================================================================

download_spacy() {
    log_section "spaCy"

    local spacy_model_path="${SPACY_TARGET}/en_core_web_sm"

    if ! $FORCE && check_dir_has_files "$spacy_model_path"; then
        log_info "Already exists at ${spacy_model_path#$PROJECT_ROOT/} (skipped)"
        return 0
    fi

    if [ "$DRY_RUN" = "true" ]; then
        log_info "Would download spaCy model to ${SPACY_TARGET#$PROJECT_ROOT/}"
        return 0
    fi

    mkdir -p "$SPACY_TARGET"
    local tmp_dir=$(mktemp -d)

    log_info "Downloading spaCy language model..."

    if download_file "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl" \
            "$tmp_dir/spacy.whl"; then
        if extract_zip "$tmp_dir/spacy.whl" "$tmp_dir"; then
            # Find the extracted en_core_web_sm directory and move it
            if [ -d "$tmp_dir/en_core_web_sm" ]; then
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
# InsightFace Download
# ============================================================================

download_insightface() {
    log_section "InsightFace"

    local insightface_model_path="${INSIGHTFACE_TARGET}/buffalo_l"

    if ! $FORCE && check_dir_has_files "$insightface_model_path"; then
        log_info "Already exists at ${insightface_model_path#$PROJECT_ROOT/} (skipped)"
        return 0
    fi

    if [ "$DRY_RUN" = "true" ]; then
        log_info "Would download InsightFace model to ${insightface_model_path#$PROJECT_ROOT/}"
        return 0
    fi

    mkdir -p "$INSIGHTFACE_TARGET"
    local tmp_dir=$(mktemp -d)

    log_info "Downloading InsightFace buffalo_l model weights (288MB)..."

    if download_file "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" \
            "$tmp_dir/buffalo_l.zip"; then
        if extract_zip "$tmp_dir/buffalo_l.zip" "$insightface_model_path"; then
            rm -rf "$tmp_dir"
            file_count=$(find "$insightface_model_path" -name "*.onnx" | wc -l)
            total_size=$(du -sm "$insightface_model_path" | cut -f1)
            log_success "Downloaded $file_count ONNX file(s) (${total_size} MB)"

            # List downloaded files
            for file in "$insightface_model_path"/*.onnx; do
                if [ -f "$file" ]; then
                    size=$(du -sm "$file" | cut -f1)
                    basename "$file" | sed "s/^/    - /"
                    echo "      (${size} MB)"
                fi
            done
            return 0
        else
            rm -rf "$tmp_dir"
            log_error "Failed to extract InsightFace model"
            return 1
        fi
    else
        rm -rf "$tmp_dir"
        log_error "Failed to download InsightFace model"
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
DOWNLOAD_INSIGHTFACE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
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
        --insightface)
            DOWNLOAD_INSIGHTFACE=true
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
            echo "  --insightface      Download InsightFace model"
            echo "  --dry-run          Show what would be downloaded"
            echo "  --force            Re-download existing files"
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
   [ "$DOWNLOAD_SPACY" = false ] && [ "$DOWNLOAD_INSIGHTFACE" = false ]; then
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
RESULT_INSIGHTFACE=1

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

if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_INSIGHTFACE" = true ]; then
    if download_insightface; then
        RESULT_INSIGHTFACE=0
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

    if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_INSIGHTFACE" = true ]; then
        if [ $RESULT_INSIGHTFACE -eq 0 ]; then
            echo -e " ${GREEN}✓${NC} insightface"
        else
            echo -e " ${RED}✗${NC} insightface"
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

    if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_INSIGHTFACE" = true ]; then
        TOTAL_COUNT=$((TOTAL_COUNT + 1))
        if [ $RESULT_INSIGHTFACE -eq 0 ]; then
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
if [ $RESULT_DEEPFACE -ne 0 ] || [ $RESULT_DOCTR -ne 0 ] || \
   [ $RESULT_HUGGINGFACE -ne 0 ] || [ $RESULT_MEDIAPIPE -ne 0 ] || \
   [ $RESULT_NLTK -ne 0 ] || [ $RESULT_PHOTOHOLMES -ne 0 ] || \
   [ $RESULT_SPACY -ne 0 ] || [ $RESULT_INSIGHTFACE -ne 0 ]; then
    exit 1
fi

exit 0
