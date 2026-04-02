#!/bin/bash

# Client Testing Script for Shamir Secret Sharing
# This script tests sending secret shares to all 3 OSINT instances
#
# NOTE: This script uses the deprecated /api/jobs/analyze-async endpoint.
# The secure replacement is /api/jobs/analyze-async-signed which requires
# ECDSA signature verification. This test script needs to be updated to:
# 1. Generate a SECP256k1 keypair
# 2. Sign the request with the private key
# 3. Include signature components (r, s) in the request body
#
# For now, this script is kept for basic connectivity testing only.

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
INSTANCE_URLS=(
    "http://localhost:12410"
    "http://localhost:12411"
    "http://localhost:12412"
)

# Test data
TEST_PUBLIC_KEY="test_public_key_$(date +%s)"
TEST_DEVICE_ID="test_device_$(date +%s)"
TEST_MOBILE_NUMBER="+1234567890"

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_instance() {
    local url=$1
    local instance_num=$2

    if curl -s -f "$url/health" > /dev/null 2>&1; then
        print_success "Instance $instance_num is healthy"
        return 0
    else
        print_error "Instance $instance_num is not responding at $url"
        return 1
    fi
}

generate_otp() {
    local url=$1
    local instance_num=$2

    print_info "Generating OTP from Instance $instance_num..."

    local response=$(curl -s -X POST \
        "$url/api/otp/random-number/6?mobile_number=$TEST_MOBILE_NUMBER" \
        -H "Content-Type: application/json" 2>/dev/null)

    if [ $? -eq 0 ]; then
        local otp=$(echo "$response" | grep -o '"random_number":"[0-9]*"' | cut -d'"' -f4)
        if [ -n "$otp" ]; then
            print_success "OTP generated: $otp"
            echo "$otp"
            return 0
        fi
    fi

    print_warning "Failed to generate OTP from Instance $instance_num"
    echo "123456"  # Fallback OTP
    return 1
}

create_test_selfie() {
    local otp=$1
    local output_file=$2

    # Create a simple test image with OTP in filename
    # Using base64 encoded 1x1 pixel PNG as placeholder
    local base64_image="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    echo "$base64_image" | base64 -d > "$output_file"
    print_success "Created test selfie: $output_file"
}

submit_secret_share() {
    local instance_num=$1
    local url=$2
    local secret_share=$3
    local otp=$4

    print_info "Submitting Share $instance_num to Instance $instance_num..."

    # Create test selfie with OTP in filename
    local selfie_filename="selfie_OTP${otp}_test.png"
    local selfie_path="/tmp/$selfie_filename"
    create_test_selfie "$otp" "$selfie_path"

    # Convert image to base64
    local selfie_base64=$(base64 -i "$selfie_path")

    # Create payload
    local payload=$(cat <<EOF
{
    "client_public_key": "$TEST_PUBLIC_KEY",
    "secret_share": "$secret_share",
    "documents": [
        {
            "type": "selfie",
            "data": "$selfie_base64",
            "filename": "$selfie_filename"
        }
    ]
}
EOF
)

    # Submit to instance
    local response=$(curl -s -X POST \
        "$url/api/jobs/analyze-async" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>/dev/null)

    if echo "$response" | grep -q "job_id\|session_id"; then
        print_success "Share $instance_num submitted successfully"
        echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
        rm -f "$selfie_path"
        return 0
    else
        print_error "Failed to submit Share $instance_num"
        echo "$response"
        rm -f "$selfie_path"
        return 1
    fi
}

test_basic_connectivity() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════╗"
    echo "║   Step 1: Testing Basic Connectivity          ║"
    echo "╚════════════════════════════════════════════════╝"
    echo -e "${NC}"

    local all_healthy=true
    for i in "${!INSTANCE_URLS[@]}"; do
        check_instance "${INSTANCE_URLS[$i]}" $((i+1)) || all_healthy=false
    done

    if [ "$all_healthy" = true ]; then
        print_success "All instances are healthy!"
    else
        print_error "Some instances are not healthy. Please check with: ./deploy/status-instances.sh"
        exit 1
    fi
    echo ""
}

test_shamir_submission() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════╗"
    echo "║   Step 2: Testing Shamir Secret Sharing       ║"
    echo "╚════════════════════════════════════════════════╝"
    echo -e "${NC}"

    # Simulate 3 secret shares (in real implementation, client creates these)
    local shares=(
        "share_1_base64_encoded_data_here"
        "share_2_base64_encoded_data_here"
        "share_3_base64_encoded_data_here"
    )

    print_info "Test Scenario:"
    echo "  - Client public key: $TEST_PUBLIC_KEY"
    echo "  - Device identifier: $TEST_DEVICE_ID"
    echo "  - Mobile number: $TEST_MOBILE_NUMBER"
    echo ""

    # Generate OTPs from each instance
    local otps=()
    for i in "${!INSTANCE_URLS[@]}"; do
        local otp=$(generate_otp "${INSTANCE_URLS[$i]}" $((i+1)))
        otps+=("$otp")
        sleep 1
    done

    echo ""

    # Submit shares to each instance
    local success_count=0
    for i in "${!INSTANCE_URLS[@]}"; do
        echo ""
        if submit_secret_share $((i+1)) "${INSTANCE_URLS[$i]}" "${shares[$i]}" "${otps[$i]}"; then
            success_count=$((success_count + 1))
        fi
        sleep 2
    done

    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    if [ $success_count -eq 3 ]; then
        print_success "All 3 shares submitted successfully!"
        print_info "Shamir Secret Sharing test completed"
    else
        print_warning "Only $success_count/3 shares were submitted successfully"
        print_info "This may be expected if full validation pipeline is active"
    fi
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

show_next_steps() {
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "  1. Implement actual Shamir Secret Sharing client-side"
    echo "  2. Split secret into 3 shares using cryptographic library"
    echo "  3. Send each share to corresponding instance"
    echo "  4. Verify shares are stored encrypted in database"
    echo "  5. Test secret recovery with temp_public_key"
    echo ""
    echo -e "${YELLOW}Useful Commands:${NC}"
    echo "  Check instance status:  ./deploy/status-instances.sh"
    echo "  View logs:              tail -f deploy/logs/instance_*/server.log"
    echo "  Stop instances:         ./deploy/stop-instances.sh"
    echo ""
}

# Main execution
main() {
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║       OSINT Client Testing - Shamir Secret Sharing             ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""

    # Test connectivity
    test_basic_connectivity

    # Test Shamir submission
    test_shamir_submission

    # Show next steps
    show_next_steps
}

main
