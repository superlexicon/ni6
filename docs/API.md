# IM-OSINT API Documentation

## Overview

The IM-OSINT API provides document verification, identity management, OSINT background screening, and cryptographic key management services.

**Base URL:** `http://your-server:port`

**Content-Type:** `application/json`

---

## Table of Contents

1. [Health Endpoints](#1-health-endpoints)
2. [LLM Forgery Detection](#2-llm-forgery-detection)
3. [Key Management](#3-key-management)
4. [Key Management (Add/Remove)](#4-key-management-addremove)
5. [OTP Service](#5-otp-service)
6. [Verification Endpoints](#6-verification-endpoints)
7. [Secret Share Recovery](#7-secret-share-recovery)
8. [User Data Deletion](#8-user-data-deletion)
9. [Security Features](#9-security-features)

---

## 1. Health Endpoints

### 1.1 Health Check

**GET** `/api/health`

Returns service health status and list of available API endpoints.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-15T10:30:00Z",
  "service": "im-osint-api",
  "version": "1.0.0",
  "apis": [
    "/api/health",
    "/api/jobs/analyze-async",
    "/api/jobs/verification",
    ...
  ]
}
```

### 1.2 GPU Health Check

**GET** `/health/gpu`

Returns GPU availability and resource usage.

**Response (200 OK):**
```json
{
  "gpu_available": true,
  "device": "cuda:0",
  "memory_used": 2048,
  "memory_total": 8192,
  "memory_free": 6144,
  "active_models": ["DeepFace", "PhotoHolmes", "DocTR"],
  "warnings": [],
  "recommendations": []
}
```

---

## 2. LLM Forgery Detection

### 2.1 Detect Forgery

**POST** `/api/forgery/detect`

Detects document forgery using PhotoHolmes AI analysis.

**Query Parameters:**
- `text_require` (boolean, required) - Whether text extraction is required

**Request Body:**
```json
{
  "image_data": "base64_encoded_image",
  "image_format": "png"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "is_forgery": false,
    "is_photoshopped": false,
    "forgery_confidence": 0.15,
    "photoshopped_confidence": 0.08,
    "analysis_details": {
      "methods_used": ["ELA", "Shadow Analysis", "Noise Analysis"],
      "forgery_indicators": []
    }
  }
}
```

### 2.2 Detailed Forgery Detection

**POST** `/detect-forgery-detailed`

Provides research-backed detailed forgery analysis.

**Query Parameters:**
- `text_require` (boolean, required) - Whether text extraction is required

**Request Body:**
```json
{
  "image_data": "base64_encoded_image",
  "image_format": "png"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "is_forgery": false,
    "confidence": 0.12,
    "detailed_analysis": {
      "error_level_analysis": {...},
      "latent_digits": [...],
      "noise_inconsistency": {...},
      "printing_artifacts": {...},
      "geometric_distortions": {...}
    }
  }
}
```

---

## 3. Key Management

### 3.1 Get Server Public Key

**GET** `/api/key/public-key`

Returns the server's public key for encrypting client data.

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "public_key": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----",
    "key_id": "server_key_v1",
    "algorithm": "RSA-4096"
  }
}
```

### 3.2 Create Share Key

**POST** `/api/key/create`

Creates a user share key for multi-device support.

**Rate Limiting:** 10 requests per hour per IP

**Request Body:**
```json
{
  "client_public_key": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----"
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "data": {
    "share_key_id": "sk_abc123",
    "created_at": "2026-01-15T10:30:00Z",
    "device_count": 1
  }
}
```

**Rate Limit Exceeded (429 Too Many Requests):**
```json
{
  "detail": "Rate limit exceeded: 10 requests per hour"
}
```

### 3.3 Add Device Key

**POST** `/device-key/add`

Adds an additional device key for multi-device support.

**Rate Limiting:** 20 requests per hour per IP

**Request Body:**
```json
{
  "new_public_key": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----",
  "secret_share": "encrypted_secret_share",
  "signature": "signature_for_verification"
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "data": {
    "device_added": true,
    "device_count": 2
  }
}
```

---

## 4. Key Management (Add/Remove)

### 4.1 Add New Public Key

**POST** `/keys/add`

Adds a new public key to user identity with face verification.

**Request Body (Encrypted Envelope):**
```json
{
  "client_public_key": "existing_registered_key",
  "encrypted_key": "encrypted_session_key",
  "key_iv": "initialization_vector",
  "encrypted_payload": "encrypted_request_payload",
  "payload_iv": "payload_initialization_vector",
  "otp_code": "123456"
}
```

**Decrypted Payload Contents:**
```json
{
  "new_public_key": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC PUBLIC KEY-----",
  "secret_share": "encrypted_secret_share",
  "selfie_data": "base64_selfie_image",
  "filename": "selfie.jpg",
  "otp_code": "123456"
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "job_abc123",
  "status": "processing",
  "message": "Key addition job submitted for processing"
}
```

### 4.2 Remove Public Key

**DELETE** `/keys/remove`

Removes a public key from user identity with face verification.

**Request Body (Encrypted Envelope):**
```json
{
  "client_public_key": "existing_registered_key",
  "encrypted_key": "encrypted_session_key",
  "key_iv": "initialization_vector",
  "encrypted_payload": "encrypted_request_payload",
  "payload_iv": "payload_initialization_vector",
  "otp_code": "123456"
}
```

**Decrypted Payload Contents:**
```json
{
  "public_key_to_remove": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC PUBLIC KEY-----",
  "selfie_data": "base64_selfie_image",
  "filename": "selfie.jpg",
  "otp_code": "123456"
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "job_def456",
  "status": "processing",
  "message": "Key removal job submitted for processing"
}
```

### 4.3 Get Key Management Job Status (NEW - Recommended)

**POST** `/keys/status`

Retrieves the status of a key management job with signature verification.

**Request Body:**
```json
{
  "job_id": "job_abc123",
  "public_key": "client_public_key_hex",
  "timestamp": 1642250000,
  "signature": {
    "r": "signature_r_value",
    "s": "signature_s_value"
  }
}
```

**Authentication:**
- Sign the message `"request:{timestamp}"` with your private key
- Timestamp must be within 5 minutes (replay protection)
- Public key must be registered in the system

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "job_id": "job_abc123",
    "status": "completed",
    "result": {
      "key_added": true,
      "face_verified": true,
      "similarity": 0.92
    }
  }
}
```

### 4.4 Get Key Management Job Status (DEPRECATED)

**GET** `/keys/status/{job_id}`

**DEPRECATED:** Use `POST /keys/status` with signature verification instead.

Retrieves the status of a key management job.

**Path Parameters:**
- `job_id` (string) - Job identifier

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "job_id": "job_abc123",
    "status": "completed",
    "result": {
      "key_added": true,
      "face_verified": true,
      "similarity": 0.92
    }
  }
}
```

---

## 5. OTP Service

### 5.1 Request OTP (Signed, Encrypted Response)

**POST** `/api/otp/`

**SECURE ENDPOINT**

Requests an OTP with signature verification and encrypted response.
Creates user identity and user keys during request.

**Security Features:**
- ECDSA signature verification (proves key ownership)
- Creates user_identity_index and user_keys records
- Returns encrypted OTP (hybrid encryption)

**Rate Limiting:** 3 requests per 10 minutes per mobile number

**Request Body:**
```json
{
  "client_public_key": "client_public_key_hex",
  "mobile_number": "+1234567890",
  "country_code": "+1",
  "secret_share": "0:SGVsbG8gV29ybGQ=",
  "otp_length": 6,
  "timestamp": 1642250000,
  "signature": {
    "r": "signature_r_hex",
    "s": "signature_s_hex"
  },
  "target_server_public_key": "server_public_key_hex"
}
```

**Authentication:**
- Sign the message "otp:{timestamp}" with your private key
- Timestamp must be within 5 minutes (replay protection)

**Response (201 Created):**
```json
{
  "client_public_key": "ephemeral_server_key_hex",
  "encrypted_key": "encrypted_aes_key_base64",
  "key_iv": "iv_base64",
  "encrypted_payload": "encrypted_response_base64",
  "payload_iv": "payload_iv_base64"
}
```

**Decrypted Payload Contents:**
```json
{
  "otp": "123456",
  "otp_id": "otp_abc123",
  "expires_at": "2026-01-15T11:00:00Z",
  "sent_at": "2026-01-15T10:30:00Z",
  "user_identity_id": "uuid-here"
}
```

**Client-Side Decryption:**
```python
from app.core.key.hybrid_crypto import HybridCrypto

hybrid = HybridCrypto()
decrypted = hybrid.decrypt_envelope(response_dict)
import json
payload = json.loads(decrypted.payload)
otp = payload['otp']
```

**Selfie Submission Changes:**

Selfie submissions no longer require secret_share, mobile_number, or country_code.
These are now provided during the OTP request.

**New Selfie Request:**
```json
{
  "client_public_key": "registered_key_hex",
  "timestamp": 1642250000,
  "signature": {"r": "...", "s": "..."},
  "encrypted_key": "...",
  "encrypted_payload": "...",
  ...
}
```

User data is looked up via client_public_key from the user_keys table.

**Error Responses:**

**Invalid Signature (400 Bad Request):**
```json
{
  "detail": "Invalid signature"
}
```

**Expired Timestamp (400 Bad Request):**
```json
{
  "detail": "Timestamp expired. Maximum age is 300 seconds"
}
```

**Rate Limit Exceeded (429 Too Many Requests):**
```json
{
  "detail": "Rate limit exceeded: 3 requests per 10 minutes per mobile number"
}
```

### 5.2 Generate and Send OTP (DEPRECATED)

**POST** `/api/otp/random-number/{length}`

**DEPRECATED:** Use `POST /api/otp/` with signed request instead.

Generates a random OTP and sends it via AWS SMS.
This endpoint does NOT create user identity or encrypt the response.

**Rate Limiting:** 3 requests per 10 minutes per mobile number

**Path Parameters:**
- `length` (integer) - Length of OTP code (typically 4-6 digits)

**Request Body:**
```json
{
  "mobile_number": "+1234567890",
  "public_key": "client_public_key_hex"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "message": "OTP successfully sent to +1234567890",
    "mobile_number": "+1234567890",
    "otp_length": 6,
    "delivery_method": "sms",
    "sent_at": "2026-01-15T10:30:00Z",
    "otp_id": "otp_abc123"
  }
}
```

### 5.3 Legacy OTP Generation (DEPRECATED)

**GET** `/api/otp/legacy/{length}`

**DEPRECATED:** OTP returned in response (should only be sent via SMS).

Legacy OTP generation for backward compatibility.

**Path Parameters:**
- `length` (integer) - Length of OTP code

**Query Parameters:**
- `email` (string) - Email address

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "otp": "654321",
    "expires_at": "2026-01-15T11:00:00Z"
  }
}
```

---

## 6. Verification Endpoints

### 6.1 Submit Document Analysis Job (SECURE - Recommended)

**POST** `/api/jobs/analyze-async-signed`

Submits a document analysis job with signature verification and state validation.

**SECURITY FEATURES:**
- ECDSA signature verification (proves key ownership)
- State validation BEFORE queuing (prevents garbage submissions)
- Encrypted envelope (payload security)
- Selfie filename validation (requires otpXXXXXX pattern)

**Supported Document Types:**
- `selfie` - Selfie image for face verification (filename must contain otpXXXXXX)
- `passport` - Passport or national ID (requires state == 1)
- `bank_statement` - Bank statement document (requires state == 1)
- `tax_statement` - Tax statement document (independent)
- `tax_return` - Tax return document (requires state == 1)
- `driving_license` - Driving license document (requires state == 1)
- `national_id` - National ID card (requires state == 1)
- `resume` - Resume/CV document (requires state == 1)
- `auto` - Auto-detect document type using GLiNER2 (requires state == 1)

**State Validation Rules:**
- Selfie: No previous state required, but filename must contain "otpXXXXXX" pattern
- All other documents: Require state == 1 (selfie completed)

**Request Body:**
```json
{
  "client_public_key": "registered_public_key_hex",
  "timestamp": 1642250000,
  "signature": {
    "r": "signature_r_hex",
    "s": "signature_s_hex"
  },
  "encrypted_key": "encrypted_aes_key_base64",
  "key_iv": "iv_base64",
  "encrypted_payload": "encrypted_payload_base64",
  "payload_iv": "payload_iv_base64",
  "target_server_public_key": "server_public_key_hex",
  "callback_url": "https://callback-url/webhook"
}
```

**Authentication:**
- Sign the message `"request:{timestamp}"` with your private key
- Timestamp must be within 5 minutes (replay protection)
- Public key must be registered in the system

**Selfie Filename Requirement:**
- Must contain pattern: `otpXXXXXX` (otp + 6 digits, case-insensitive)
- Examples: `selfie_otp123456.jpg`, `OTP999999_selfie.png`

**Auto-Detection Hints:**
When using `document_type: "auto"`, you can optionally provide hints to guide detection:
- `document_type`: Hint for document type (e.g., "tax_statement", "bank_statement")
- `country`: Hint for country code (ISO 2-letter, e.g., "AE", "US")
- `entity`: Hint for entity identifier (e.g., "trc", "emirates_id")

**Response (202 Accepted):**
```json
{
  "success": true,
  "job_id": "job_xyz789",
  "status": "pending",
  "message": "Job queued successfully"
}
```

**Error Responses:**

```json
{
  "success": false,
  "job_id": "",
  "status": "failed",
  "message": "State validation failed: Complete selfie step first"
}
```

```json
{
  "success": false,
  "job_id": "",
  "status": "failed",
  "message": "Selfie filename must contain otpXXXXXX pattern (e.g., selfie_otp123456.jpg)"
}
```

```json
{
  "detail": "Invalid signature"
}
```

### 6.2 Get Verification State (Recommended)

**POST** `/api/jobs/verification`

Retrieves the verification state and results with signature verification.

**Request Body:**
```json
{
  "public_key": "client_public_key_hex",
  "timestamp": 1642250000,
  "signature": {
    "r": "signature_r_value",
    "s": "signature_s_value"
  }
}
```

**Authentication:**
- Sign the message `"request:{timestamp}"` with your private key
- Timestamp must be within 5 minutes (replay protection)
- Public key must be registered in the system

**Response (200 OK):**
```json
{
  "client_public_key": "abc123...",
  "state": "passport_pending",
  "verification_state": 1,
  "user_identity_id": "uuid-here",
  "sequence_no": 1,
  "docs_auth_score": 85.5,
  "id_veri_score": 92.3,
  "selfie_result": {
    "forgery_checks": {
      "ela": {"score": 0.1, "threshold": 0.5, "passed": true},
      "shadow_analysis": {"score": 0.15, "threshold": 0.5, "passed": true}
    },
    "other_checks": {
      "anti_spoofing_score": 0.85,
      "face_detected": true,
      "otp_verified": true
    },
    "extracted_data_encrypted": {
      "version": "ecies_v1",
      "ephemeral_public_key": "04abc123...",
      "encrypted_data": "base64_encoded_ciphertext",
      "iv": "base64_encoded_iv"
    }
  },
  "passport_result": {
    "forgery_checks": {...},
    "other_checks": {
      "face_match_confidence": 92.5,
      "document_expiry_valid": true
    },
    "extracted_data_encrypted": {
      "version": "ecies_v1",
      "ephemeral_public_key": "04def456...",
      "encrypted_data": "base64_encoded_ciphertext",
      "iv": "base64_encoded_iv"
    }
  },
  "bank_statement_result": null,
  "jobid_inprogress": "job_xyz789"
}
```

### 6.3 Get Verification State (REMOVED)

**GET** `/api/jobs/verification/{client_public_key}`

**REMOVED:** This endpoint has been deleted. Use `POST /api/jobs/verification` with signature verification instead.

The GET endpoint was removed because it did not require signature verification, allowing:
- User enumeration attacks (public key in URL)
- Unauthorized access to verification results

**Migration:** All clients must use `POST /api/jobs/verification` with ECDSA signature verification.

**IMPORTANT: ECIES Encryption Note**

All PII (Personally Identifiable Information) is encrypted using **ECIES (Elliptic Curve Integrated Encryption Scheme)** with ephemeral keys. This ensures **user-only decryption** - the server stores encrypted data but **cannot decrypt it**.

**PII Encryption Details:**
- Encryption: ECIES with ephemeral SECP256k1 keypair
- Decryption: Only possible with user's private key
- Ephemeral key: New key generated for each encryption (discarded after)
- Envelope format: `{"version": "ecies_v1", "ephemeral_public_key": "...", "encrypted_data": "...", "iv": "..."}`

**Client-Side Decryption Required:**
The `extracted_data_encrypted` field contains the ECIES envelope. Clients must decrypt this using their private key to access PII fields.

**State Values:**
- `selfie_pending` - No user_identity_id in user_keys (state: 0)
- `passport_pending` - Row exists, passport_expiry_date IS NULL (state: 1)
- `bank_pending` - passport_expiry_date IS NOT NULL, bank_statement_date IS NULL (state: 2)
- `completed` - Both passport_expiry_date AND bank_statement_date are NOT NULL (state: 3)

**Note:** Identity uniqueness is enforced by face biometrics trigger (trg_face_biometrics_cross_identity_check),
not by passport or document hashes. Same document can be submitted with different encryption for different keys.

**Field Descriptions:**
- `client_public_key`: Client's public key
- `state`: Current verification state string
- `verification_state`: Current verification state int (0-3)
- `user_identity_id`: User identity ID
- `sequence_no`: Current sequence number (0-3)
- `docs_auth_score`: Document authentication score % (0-100) based on PhotoHolmes checks
- `id_veri_score`: Identity verification score % (0-100) based on validation checks
- `selfie_result`: Analysis result if submitted (null if not submitted)
  - `extracted_data_encrypted`: ECIES envelope with encrypted PII (client must decrypt with private key)
- `passport_result`: Analysis result if submitted (null if not submitted)
  - `extracted_data_encrypted`: ECIES envelope with encrypted PII (client must decrypt with private key)
- `idcard_result`: Analysis result if submitted (null if not submitted)
  - `extracted_data_encrypted`: ECIES envelope with encrypted PII (client must decrypt with private key)
- `bank_statement_result`: Analysis result if submitted (null if not submitted)
  - `extracted_data_encrypted`: ECIES envelope with encrypted PII (client must decrypt with private key)
- `other_results`: List of other document results that don't have dedicated fields
  - Includes document types like `tax_return`, `tax_statement`, `national_id`, `driving_license`, `resume`
  - Each result contains:
    - `filename`: Original filename of the submitted document
    - `document_type`: Type of document processed
    - `job_id`: Job identifier for the document processing
    - `result`: Boolean indicating if processing was successful
    - `verification_state`: Verification state at time of processing
    - `processing_time_seconds`: Time taken to process the document
    - `docs_auth_score`: Document authentication score
    - `id_veri_score`: Identity verification score
    - `forgery_checks`: PhotoHolmes forgery detection results (if available)
    - `other_checks`: Additional validation checks (if available)
    - `extracted_data_encrypted`: ECIES envelope with encrypted PII (client must decrypt with private key)

### 6.4 Get Verification Progress

**GET** `/api/jobs/verification/{client_public_key}/progress`

Gets real-time verification progress.

**Path Parameters:**
- `client_public_key` (string) - Client public key (hex format)

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "progress_percentage": 75,
    "current_step": "OSINT background check",
    "steps_completed": [
      "Document upload",
      "Passport verification",
      "Selfie verification",
      "Bank statement analysis"
    ],
    "steps_remaining": [
      "OSINT screening",
      "Final risk assessment"
    ]
  }
}
```

---

## 7. Secret Share Recovery

### 7.1 Request Secret Share (SECURE)

**POST** `/api/share/request`

Requests secret share recovery with ECDSA signature verification and face verification.

**SECURITY FEATURES:**
- ECDSA signature verification (proves key ownership)
- Public key must be registered in the system
- Timestamp validation (5-minute window, replay protection)
- Face verification before releasing shares
- Re-encrypted shares using temporary public key

**TEMPORARY KEYPAIR FLOW:**

For secure share recovery, the client generates a **temporary keypair**:
1. Client generates ephemeral SECP256k1 keypair
2. `temp_public_key` is sent in the share request
3. Server re-encrypts shares with `temp_public_key` using ECIES
4. Only the client with matching `temp_private_key` can decrypt

**Request Body:**
```json
{
  "public_key": "original_registered_public_key_hex",
  "timestamp": 1642250000,
  "signature": {
    "r": "signature_r_hex",
    "s": "signature_s_hex"
  },
  "temp_public_key": "ephemeral_public_key_hex",
  "selfie_data": "base64_encoded_selfie",
  "otp_code": "123456",
  "filename": "recovery_selfie.jpg",
  "callback_url": "https://callback-url/webhook",
  "target_server_public_key": "server_public_key_hex"
}
```

**Authentication:**
- Sign the message `"request:{timestamp}"` with your **original** private key
- Timestamp must be within 5 minutes (replay protection)
- Public key must be registered in the system
- The `temp_public_key` is an ephemeral key for this recovery session only

**Response (202 Accepted):**
```json
{
  "job_id": "job_recovery_123",
  "status": "processing",
  "message": "Secret share recovery job submitted"
}
```

**Error Responses:**

**Invalid Signature (401 Unauthorized):**
```json
{
  "detail": "Invalid signature"
}
```

**Invalid Public Key (401 Unauthorized):**
```json
{
  "detail": "Invalid public key or user not found"
}
```

**Expired Timestamp (400 Bad Request):**
```json
{
  "detail": "Timestamp expired. Maximum age is 300 seconds"
}
```

### 7.2 Get Recovery Status (SECURE - with Temp Key Support)

**POST** `/api/share/status`

Gets the status of a secret share recovery request with signature verification.

**IMPORTANT - Temp Public Key Support:**
- Use the **same `temp_public_key`** that was sent in the share request
- Sign the request with the **matching `temp_private_key`**
- Server verifies signature against `temp_public_key`
- Server looks up submission by `temp_public_key` in `document_submissions`
- Response is encrypted with `temp_public_key` (ECIES)

**Request Body:**
```json
{
  "job_id": "job_recovery_123",
  "public_key": "temp_public_key_hex",
  "timestamp": 1642250000,
  "signature": {
    "r": "signature_r_value",
    "s": "signature_s_value"
  }
}
```

**Authentication:**
- Sign the message `"request:{timestamp}"` with your **temp_private_key**
- Timestamp must be within 5 minutes (replay protection)
- The `temp_public_key` does NOT need to be in `user_keys` table
- The signature verification alone proves ownership

**Response (200 OK) - Job Pending:**
```json
{
  "success": false,
  "message": "Recovery in progress (status: processing)"
}
```

**Response (200 OK) - Job Completed with ECIES Encrypted Shares:**
```json
{
  "success": true,
  "message": "Recovery completed",
  "extracted_data_encrypted": {
    "version": "ecies_v1",
    "ephemeral_public_key": "04abc123...",
    "encrypted_data": "base64_encoded_ciphertext",
    "iv": "base64_encoded_iv"
  }
}
```

**Decrypted Data Contents (after ECIES decryption):**
```json
{
  "success": true,
  "shares": [
    {
      "public_key": "original_public_key_hex",
      "share": "0:SGVsbG8gV29ybGQ=",
      "server_url": "http://server1:12410"
    },
    {
      "public_key": "original_public_key_hex",
      "share": "1:V29ybGQgSGVsbG8=",
      "server_url": "http://server2:12411"
    }
  ],
  "total_shares": 2,
  "face_match_confidence": 92.5,
  "faces_checked": 1
}
```

**ECIES Decryption (Client-Side):**
The `extracted_data_encrypted` field contains an ECIES envelope that only the client with the matching `temp_private_key` can decrypt.

**Error Responses:**

**Invalid Signature (401 Unauthorized):**
```json
{
  "detail": "Invalid signature"
}
```

**Job Not Found (404 Not Found):**
```json
{
  "detail": "No recovery job found for this public key"
}
```

### 7.3 Get Recovery Status (REMOVED)

**GET** `/api/share/status/{job_id}`

**REMOVED:** This endpoint has been deleted. Use `POST /api/share/status` with signature verification.

The GET endpoint was removed because it did not require signature verification, allowing:
- Job ID enumeration attacks
- Unauthorized access to recovery results

**Migration:** All clients must use `POST /api/share/status` with ECDSA signature verification.

**Important Note - Temp Key Support:**
The POST endpoint supports temporary public keys for share recovery. The temp public key:
- Does NOT need to exist in the `user_keys` table
- Is generated by the client for each recovery session
- Is used to encrypt the response (ECIES) so only the client can decrypt
- Must match the private key used to sign the status request

### 7.4 Temporary Keypair Flow (Share Recovery)

**Overview:**

The secret share recovery flow uses a temporary keypair to ensure that re-encrypted shares can only be decrypted by the client that initiated the recovery.

**Step-by-Step Flow:**

**1. Client Generates Temporary Keypair**
```dart
// Generate ephemeral SECP256k1 keypair
final keypair = generateSecp256k1KeyPair();
final tempPublicKey = keypair.publicKey.toHex();
final tempPrivateKey = keypair.privateKey;
```

**2. Client Requests Share Recovery**
```json
POST /api/share/request
{
  "public_key": "original_registered_key",
  "signature": {"r": "...", "s": "..."},  // Signed with original private key
  "temp_public_key": "temp_key_hex",      // Ephemeral key for encryption
  "selfie_data": "...",
  "otp_code": "123456"
}
```

**3. Server Processes Request**
- Verifies signature with original public key
- Validates selfie (face match, OTP, forgery detection)
- Re-encrypts shares with `temp_public_key` using ECIES
- Stores submission with `client_public_key = temp_public_key`
- Returns `job_id`

**4. Client Checks Status**
```json
POST /api/share/status
{
  "job_id": "job_recovery_123",
  "public_key": "temp_key_hex",           // Same as share request
  "signature": {"r": "...", "s": "..."}   // Signed with temp_private_key
}
```

**5. Server Returns Status**
- Verifies signature with `temp_public_key` ✅
- Does NOT check `user_keys` (temp key not stored there)
- Looks up submission by `temp_public_key` in `document_submissions`
- Returns ECIES-encrypted response

**6. Client Decrypts Response**
```dart
// Decrypt with temp_private_key
final decrypted = decryptEciesEnvelope(
  response['extracted_data_encrypted'],
  privateKey: tempPrivateKey
);
final shares = decrypted['shares'];
```

**Security Benefits:**

1. **Forward Secrecy:** Temporary key is discarded after recovery
2. **Single-Use:** Each recovery session uses a new temp key
3. **Exclusive Access:** Only the client with `temp_private_key` can decrypt
4. **No Server Storage:** Temp private key never leaves the client
5. **Replay Protection:** Timestamp + signature prevent replay attacks

### 7.5 Submit Voluntary Selfie

**POST** `/api/face/submit`

Submits a voluntary selfie to improve face matching accuracy.

**Request Body:**
```json
{
  "public_key": "client_public_key_hex",
  "selfie_data": "base64_selfie_image"
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "data": {
    "selfie_id": "selfie_abc123",
    "stored_at": "2026-01-15T10:30:00Z",
    "message": "Selfie stored successfully for improved face matching"
  }
}
```

---

## 8. User Data Deletion

### 8.1 Delete User Presence

**POST** `/api/user/delete-presence`

Permanently deletes all user data with signature verification.

**Request Body:**
```json
{
  "message": "delete:1642250000",
  "signature": {
    "r": "signature_r_value",
    "s": "signature_s_value"
  },
  "public_key": "client_public_key_hex"
}
```

**Message Format:**
- Must be: `delete:{unix_timestamp}`
- Timestamp must be within 5 minutes (replay protection)

**Signature Requirements:**
- ECDSA signature with SHA-256
- Public key recovered from signature components
- Signature verified before deletion

**Data Deleted:**
- Document analysis jobs
- Document submissions
- Face biometrics
- OTP records
- User keys
- User identity index

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "deletion_status": "completed",
    "records_deleted": {
      "document_analysis_jobs": 5,
      "document_submissions": 10,
      "face_biometrics": 3,
      "otp_records": 15,
      "user_keys": 2,
      "user_identity_index": 1
    },
    "deleted_at": "2026-01-15T10:30:00Z"
  }
}
```

---

## 9. Security Features

### 9.1 State Validation

The job submission system validates verification state BEFORE queuing to prevent garbage data submissions.

**State Validation Rules:**

| Document Type | Required State | Description |
|---------------|----------------|-------------|
| `selfie` | State 0 | New user, ready for selfie. Filename must contain `otpXXXXXX` pattern |
| `passport` | State 1 | Selfie must be completed |
| `bank_statement` | State 1 | Selfie must be completed |
| `national_id` | State 1 | Selfie must be completed |
| `driving_license` | State 1 | Selfie must be completed |
| `resume` | State 1 | Selfie must be completed |
| `tax_statement` | Independent | Can be submitted anytime |
| `id_card` | Independent | Can be submitted anytime |

**Selfie Filename Validation:**
- Selfie filenames must contain `otpXXXXXX` pattern (OTP + 6 digits)
- Case-insensitive matching
- Prevents garbage selfie submissions without knowing expected format
- Examples: `selfie_otp123456.jpg`, `OTP999999_selfie.png`

**Benefits:**
- Prevents queue flooding with invalid submissions
- Validates verification state BEFORE database insertion
- Reduces resource consumption on invalid requests

### 9.2 Rate Limiting

The API implements rate limiting to prevent abuse and protect against DoS attacks.

**Rate Limiting Rules:**

| Endpoint | Limit | Scope |
|----------|-------|-------|
| `POST /api/otp/random-number/{length}` | 3 per 10 minutes | Per mobile number |
| `POST /api/key/create` | 10 per hour | Per IP |
| `POST /device-key/add` | 20 per hour | Per IP |
| `POST /api/jobs/analyze-async` | 30 per hour | Per IP |
| Default | 100 per hour | Per IP |

**Rate Limit Headers:**
When rate limits are enforced, the response includes:
```
X-RateLimit-Limit: 3
X-RateLimit-Remaining: 2
X-RateLimit-Reset: 1642250123
```

**Rate Limit Exceeded Response:**
```json
{
  "detail": "Rate limit exceeded"
}
```

### 9.3 Signature Verification

Sensitive endpoints require ECDSA signature verification to prevent unauthorized access.

**Signing Process:**
1. Create a message: `"request:{unix_timestamp}"` or `"delete:{unix_timestamp}"`
2. Sign the message with your private key using ECDSA (SHA-256)
3. Include the signature components (r, s) in the request

**Example (Python):**
```python
import time
from ecdsa import SigningKey, SECP256k1, util

# Get current timestamp
timestamp = int(time.time())
message = f"request:{timestamp}"

# Sign with private key
private_key = SigningKey.from_string(bytes.fromhex(your_private_key_hex), curve=SECP256k1)
signature = private_key.sign(message.encode(), hashfunc=hashlib.sha256, sigencode=util.sigencode_string)

# Split into r and s components
r = signature[:32].hex()
s = signature[32:].hex()

# Make request
request_body = {
    "public_key": your_public_key_hex,
    "timestamp": timestamp,
    "signature": {"r": r, "s": s}
}
```

### 9.4 ECIES PII Encryption

All PII (Personally Identifiable Information) is encrypted using ECIES for user-only decryption.

**Encrypted PII Fields:**
- Selfie: `otp_number`
- Passport: `full_name`, `date_of_birth`, `passport_number`, `passport_country`, `nationality`, `place_of_birth`, `sex`
- Bank Statement: `account_holder_name`, `account_number`, `address`
- Resume: `full_name`, `email`, `phone_number`, `address`

**Encryption Details:**
- Method: ECIES with ephemeral SECP256k1 keypair
- New ephemeral key generated for each encryption
- Ephemeral private key discarded after encryption (server cannot decrypt)
- Only user with their private key can derive shared secret and decrypt

**Envelope Format:**
```json
{
  "version": "ecies_v1",
  "ephemeral_public_key": "04abc123...",
  "encrypted_data": "base64_encoded_ciphertext",
  "iv": "base64_encoded_iv"
}
```

### 9.5 Request Signing Pattern

All new secure endpoints follow the same request signing pattern:

**Request Structure:**
```json
{
  "public_key": "client_public_key_hex",
  "timestamp": 1642250000,
  "signature": {
    "r": "signature_r_hex",
    "s": "signature_s_hex"
  }
}
```

**Benefits:**
- No sensitive data in URL paths (prevents logging exposure)
- Proof of key ownership via signature
- Replay protection via timestamp validation
- Unified security pattern across all endpoints

### 9.6 Removed Endpoints

The following endpoints have been **removed** from the API. Use the secure replacements shown below:

| Removed Endpoint | Secure Replacement | Reason |
|------------------|-------------------|--------|
| `POST /api/jobs/analyze-async` | `POST /api/jobs/analyze-async-signed` | No state validation before queuing |
| `GET /api/jobs/verification/{public_key}` | `POST /api/jobs/verification` | Public key in URL logged, no signature verification |
| `GET /keys/status/{job_id}` | `POST /keys/status` | Job ID enumeration |
| `GET /share/status/{job_id}` | `POST /share/status` | Job ID enumeration |
| `POST /api/resume/extract` | `POST /api/jobs/analyze-async-signed` | Consolidate to job system |

**Migration Guide:**
- All removed endpoints lacked signature verification
- Use POST endpoints with ECDSA signature verification
- Sign `"request:{timestamp}"` message with your private key
- Include signature components (r, s) in request body

### 9.7 Supported Document Types

The job-based system supports the following document types:

| Document Type | Description | Required Fields |
|---------------|-------------|-----------------|
| `selfie` | Selfie image for face verification | None (user data looked up via client_public_key) |
| `passport` | Passport or national ID | None |
| `bank_statement` | Bank statement document | None |
| `tax_statement` | Tax statement document | None |
| `tax_return` | Tax return document | None |
| `driving_license` | Driving license document | None |
| `national_id` | National ID card | None |
| `resume` | Resume/CV document | None |
| `auto` | Auto-detect document type with GLiNER2 | None (hints optional) |

**Submitting a Resume:**
```json
{
  "client_public_key": "client_public_key_hex",
  "iv": "initialization_vector",
  "target_server_public_key": "server_public_key_hex",
  "files": [{
    "filename": "resume.pdf",
    "file_data": "base64_encoded_resume",
    "file_type": "resume"
  }]
}
```

**Submitting with Auto-Detection:**
```json
{
  "client_public_key": "client_public_key_hex",
  "iv": "initialization_vector",
  "target_server_public_key": "server_public_key_hex",
  "files": [{
    "filename": "document.pdf",
    "file_data": "base64_encoded_document",
    "file_type": "auto",
    "hints": {
      "country": "AE",
      "document_type": "tax_statement"
    }
  }]
}
```

---

## Common Response Format

All responses follow this standard format:

**Success Response:**
```json
{
  "success": true,
  "data": { ... }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": { ... }
  }
}
```

### Sequential Job Response Fields

For document verification endpoints, responses use the `SequentialJobResponse` format:

**Success Response:**
```json
{
  "result": true,
  "status": "completed",
  "job_id": "...",
  "verification_state": 1,
  "sequence_no": 1,
  "processing_time_seconds": 2.45,
  "user_identity_id": "uuid-here",
  "extracted_data": {...},
  "forgery_checks": {...},
  "other_checks": {...}
}
```

**Error Response (with error field):**
```json
{
  "result": false,
  "status": "failed",
  "error": "OTP validation failed - incorrect OTP",
  "job_id": "...",
  "verification_state": 0,
  "sequence_no": 0,
  "processing_time_seconds": 1.23
}
```

| Field | Type | Description |
|-------|------|-------------|
| `result` | boolean | `true` if verification passed, `false` otherwise |
| `status` | string | `"completed"` if `result=true`, `"failed"` if `result=false` (computed field) |
| `error` | string (optional) | Descriptive error message when `result=false` |
| `job_id` | string | Unique job identifier |
| `verification_state` | integer | Current verification state (0-3) |
| `sequence_no` | integer | Current sequence number (0-3) |
| `processing_time_seconds` | float | Processing time in seconds |
| `user_identity_id` | string (optional) | User identity ID if created |
| `extracted_data` | object (optional) | Extracted document data |
| `forgery_checks` | object (optional) | PhotoHolmes forgery detection results |
| `other_checks` | object (optional) | Additional validation checks |

### Common Error Messages

| Error Message | Description |
|---------------|-------------|
| `OTP validation failed - incorrect OTP` | OTP code does not match |
| `OTP validation failed - OTP expired` | OTP has exceeded validity period |
| `OTP validation failed - no OTP found for this public key` | No OTP record exists |
| `OTP validation failed - OTP already verified` | OTP was already used |
| `Duplicate face detected - this face is already registered to another user` | Face matches another user |
| `Failed to create user identity` | Database error during identity creation |
| `Duplicate document - this document was previously submitted` | Document hash already exists |
| `Rate limit exceeded` | Too many requests |

**Common HTTP Status Codes:**
- `200 OK` - Request succeeded
- `201 Created` - Resource created successfully
- `202 Accepted` - Request accepted for async processing
- `400 Bad Request` - Invalid request parameters
| `401 Unauthorized` - Authentication failed
- `403 Forbidden` - Authorization failed
- `404 Not Found` - Resource not found
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

---

## Environment Configuration

Required environment variables:

```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=3306
DB_USER=admin
DB_PASSWORD=admin
DB_NAME=im_osint

# OSSPEP Database (for sanctions/PEP checks)
OSSPEP_DB_HOST=localhost
OSSPEP_DB_PORT=3306
OSSPEP_DB_USER=admin
OSSPEP_DB_PASSWORD=admin
OSSPEP_DB_NAME=osspep

# Encryption Keys
SEED=<seed_for_otp_generation>
# Note: SEED is used for all encryption purposes:
# - Generates SECP256k1 keypair (server identity/signing)
# - Derives NaCl SecretBox key for secret share encryption (via SHA-256)
# - Derives Fernet key for database field encryption (via HKDF with context)

# AWS Configuration (for SMS OTP)
AWS_ACCESS_KEY_ID=<your_key>
AWS_SECRET_ACCESS_KEY=<your_secret>
AWS_REGION=us-east-1
AWS_SNS_TOPIC_ARN=<your_topic_arn>

# Verification Thresholds
VERIFICATION_FACE_MATCH_THRESHOLD=70.0
VERIFICATION_NAME_MATCH_THRESHOLD=70.0

# OSINT Configuration
OSINT_ENABLE_FACE_VERIFIED_NEGATIVE_NEWS=true
OSINT_FACE_VERIFIED_SIMILARITY_THRESHOLD=0.85
OSINT_ENABLE_NLP_ENHANCED_ANALYSIS=true
OSINT_NLP_SENTIMENT_THRESHOLD=-0.3
```

---

## WebSocket Support (Optional)

For real-time updates on job status and verification progress, WebSocket connections may be supported.

**WebSocket Endpoint:** `ws://your-server/ws/jobs/{job_id}`

---

## Versioning

API version: `1.1.0`

All endpoints are subject to change. Check the `/api/health` endpoint for the latest available endpoints.
