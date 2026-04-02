# NI6 (Neutral Intelligence Service 6)

[Interactive Documentation →](https://superlexicon.github.io/ni6/)

**Algorithmic Consistency. Verifiable Neutrality.**

---

## Executive Summary

NI6 eliminates the need for trusted third-party custodians by combining cryptographic secret sharing with biometric verification, enabling true self-custody with secure, decentralized recovery. More importantly, NI6 replaces subjective human risk assessment with algorithmic consistency - delivering verifiable neutrality in every verification decision.

Traditional identity verification and KYC processes suffer from inherent human bias: different analysts make different judgments on identical cases, cultural background influences scrutiny, fatigue degrades quality, and business pressure can compromise risk assessments.

**The NI6 Difference**: Every verification decision is produced by deterministic algorithms that apply identical criteria to every applicant regardless of geography, score risk based on quantifiable signals, maintain consistent thresholds across millions of verifications, and document every factor in the decision chain.

### Value Proposition at a Glance

| What We Solve | The NI6 Solution |
|--------------|------------------|
| Centralized custody risk | Distributed secret sharing (2-of-3 threshold) |
| Expensive background checks | Self-contained OSINT with zero API costs |
| SMS OTP vulnerabilities | Video selfie with hand gesture authentication |
| Subjective risk assessment | Algorithmic consistency with full audit trails |
| False positives in screening | Face-verified negative news search |

### Target Markets

- **Cryptocurrency Exchanges**: Self-custodial user onboarding with decentralized recovery
- **Fintech Platforms**: KYC/AML verification with algorithmic neutrality
- **Corporate Background Screening**: Consistent, bias-free employee verification
- **Regulated Industries**: Audit-proof compliance with reproducible decisions

---

## The NI6 Philosophy: Why Neutrality?

### The Problem with Human Risk Assessment

| Human Bias Type | Impact on KYC/Screening | NI6 Algorithmic Solution |
|-----------------|------------------------|--------------------------|
| **Cultural Bias** | Names from certain regions trigger extra scrutiny | Face biometrics + fingerprint matching eliminate name-based pre-judgment |
| **Inconsistency** | Different analysts judge identical cases differently | Deterministic algorithms produce identical outputs for identical inputs |
| **Fatigue** | Quality degrades after hours of manual review | Algorithms don't tire - millionth verification = first verification |
| **Business Pressure** | Sales team influence on "risky but valuable" applicants | No human override path - audit trail shows if any thresholds were modified |
| **Subjectivity** | "Political exposure" varies by analyst background | Binary confidence model (33%, 66%, 99%) with clear evidence trails |
| **Recency Bias** | Recent fraud stories heighten scrutiny of similar profiles | Risk scores based on actual evidence, not news cycle timing |

### Algorithmic Consistency = Verifiable Neutrality

Every NI6 verification produces:

1. **Quantified Scores**: Face match (92.3%), document authenticity (85.7%), risk (15/100)
2. **Evidence Chain**: Exact forgery detection results, sentiment sources, biometric comparisons
3. **Reproducible Results**: Same inputs always produce same outputs
4. **Full Audit Trail**: Every factor, threshold, and decision point is logged

**Compliance Benefit**: Regulators increasingly require algorithmic transparency. NI6 provides this by design - not as an afterthought.

---

## Problem Statement

### Custody Risks

Centralized exchanges act as single points of failure. When exchange custodial wallets are compromised, users lose everything. The alternative - self-custody - introduces the critical problem of key recovery: lose your private key, lose your assets.

### KYC Fragmentation

Businesses must cobble together multiple expensive service providers for:
- Document verification (Jumio, Onfido)
- Background screening (Worldcheck, Refinitiv)
- Biometric authentication (identity.com)
- Sanctions/PEP checks (manual processes)

Each integration adds cost, complexity, and security surface area.

### Recovery Vulnerability

Traditional 2FA and SMS OTP systems are vulnerable to:
- SIM swapping attacks
- SS7 protocol vulnerabilities
- Social engineering
- SMS interception

### Identity Fraud

Sophisticated attackers now use:
- AI-generated deepfakes
- Photoshopped documents
- Stolen identity credentials
- Synthetic identities

### Compliance Costs

External services like Worldcheck charge expensive per-query fees while providing no self-service capability. Manual review costs escalate quickly with volume.

---

## Existing Solutions & Competition

| Solution Type | Examples | Limitations |
|--------------|----------|-------------|
| **Centralized Hot/Cold Wallets** | Binance, Coinbase | Single point of failure, custodial risk, "not your keys" |
| **MPC Custody** | Fireblocks, Copper | Still centralized, requires trusted third party, expensive |
| **Traditional KYC** | Jumio, Onfido | Document-only verification, no key recovery, expensive |
| **Background Screening** | Worldcheck, Refinitiv | Expensive API costs, no self-service, black-box algorithms |
| **Identity Verification** | Stripe Identity, Identity.com | No decentralized recovery, still custodial |

**The NI6 Advantage**: None of these solutions address both custody AND verification in a decentralized, algorithmically neutral system.

---

## Our Solution - Distributed Architecture

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   DISTRIBUTED ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────┘

                              ┌─────────────┐
                              │   CLIENT    │
                              │   DEVICE    │
                              │             │
                              │ Private Key │
                              └──────┬──────┘
                                     │
                                     │ Shamir Secret Sharing
                                     │ (splits secret into 3 parts)
                                     │
            ┌────────────────────────┼────────────────────────┐
            │                        │                        │
            ▼                        ▼                        ▼
     ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
     │  SERVER 1   │          │  SERVER 2   │          │  SERVER 3   │
     │  Share 0    │          │  Share 1    │          │  Share 2    │
     └─────────────┘          └─────────────┘          └─────────────┘
            │                        │                        │
            └────────────────────────┼────────────────────────┘
                                     │
                                     │ Any 2 shares reconstruct
                                     │ the original secret
                                     │
                              ┌──────┴──────┐
                              │   CLIENT    │
                              │   DEVICE    │
                              │ Private Key │
                              │   Recovered │
                              └─────────────┘

Shamir Secret Sharing (2-of-3 threshold):
- Client splits secret into 3 shares
- Each share sent to independent server
- No single server has complete secret
- Any 2 shares can reconstruct the original
- Loss of 1 server = no data loss
```

### Biometric Registration Flow

1. Client generates SECP256k1 keypair
2. Client captures selfie/photo/video
3. Face embedding extracted (Facenet512)
4. Secret split into 3 Shamir shares
5. Each share encrypted and sent to different server
6. Face biometric stored alongside each share

### Dual Proof-of-Liveliness System

#### Active Video Selfie (Passwordless Recovery)

- Hand gesture OTP extraction (0-5 finger counts)
- MediaPipe-based finger detection (21 landmarks)
- Frame-accurate sampling at 4.5s, 8.5s, 12.5s, 16.5s
- Anti-spoofing via gesture dynamics
- No SMS required - truly passwordless

#### Passive Selfie Image (Traditional Verification)

- PhotoHolmes 8-method forgery detection
- EXIF metadata analysis
- Error Level Analysis (ELA)
- Noise inconsistency detection
- DeepFace anti-spoofing

---

## Document Analysis & OCR

### Supported Documents

| Document | OCR Engine | Forgery Detection | Field Extraction |
|----------|------------|-------------------|------------------|
| Passport | DoTR | 8-method PhotoHolmes | Logic-based |
| ID Card | DoTR | 8-method PhotoHolmes | Unified extractor |
| Bank Statement | DoTR | PDF metadata + PhotoHolmes | Spatial clustering |
| Tax Statement | DoTR | PhotoHolmes | Hybrid LLM + regex |
| Resume | DoTR | PhotoHolmes | LLM-assisted |

### Logic-Based Extraction (Country-Agnostic)

- Single O(N) pass algorithm
- No template dependencies
- Multi-format date parsing
- Geometry-based name extraction
- Works with damaged documents

---

## OSINT Background Screening (Self-Contained)

**Key Differentiator**: No external API costs - uses locally maintained databases

### Anonymized Dossier Components

| Check Type | Source | Cost | Update Frequency |
|------------|--------|------|------------------|
| Sanctions | OFAC, EU, UN lists | FREE | Real-time sync |
| PEP Screening | Wikipedia scraping | FREE | Weekly sync |
| Criminal Records | Local database | FREE | Configurable |
| Negative News | DuckDuckGo search + NLP | FREE | Real-time |
| Face-Verified News | Image search + DeepFace | FREE | Real-time |

### Risk Scoring

- Binary confidence model (33%, 66%, 99%)
- VADER sentiment + FinBERT classification
- Categories: Conviction, Investigation, Allegation
- Threshold-based decision logic

---

## Face-Verified Negative News Search

**Problem**: Traditional name search produces false positives (same name, different person)

**Solution**:
1. DuckDuckGo image search for target name
2. Batch face detection (DeepFace + Facenet512)
3. Compare embeddings to stored selfie
4. Only analyze face-verified articles
5. Apply NLP sentiment analysis

**Benefits**:
- Eliminates false positives
- Reduces manual review
- More accurate risk assessment

---

## Digital Footprint Tracker (In Development)

**Planned Features**:
- Continuous monitoring of registered entities
- Social media activity tracking
- Corporate registry changes
- News alert notifications
- Risk score updates

---

## Technology Overview

### Cryptography

| Component | Implementation |
|-----------|----------------|
| Key Generation | SECP256k1 (Bitcoin curve) |
| Secret Sharing | Shamir's Scheme (2-of-3) |
| Encryption | ECIES + AES-256-GCM hybrid |
| Signatures | ECDSA with SHA-256 |
| At-Rest | NaCl SecretBox (XSalsa20-Poly1305) |

### Computer Vision

| Component | Model/Technology |
|-----------|------------------|
| Face Embeddings | Facenet512 (512-dim vectors) |
| Face Detection | RetinaFace |
| Hand Gesture | MediaPipe (21 landmarks) |
| Forgery Detection | PhotoHolmes (8 methods) |
| Anti-Spoofing | DeepFace classifiers |

### OCR & NLP

| Component | Technology |
|-----------|------------|
| OCR Engine | DoTR (Document Text Recognition) |
| Sentiment Analysis | VADER + FinBERT |
| Entity Recognition | spaCy NER |
| LLM Assistance | Configurable backend |

### Infrastructure

| Layer | Technology |
|-------|------------|
| Database | MariaDB with VECTOR type |
| Queue | RQ (Redis Queue) |
| Async | Python asyncio |
| GPU Support | CUDA, ROCm, Metal (MPS) |

---

## API Specifications

For complete API documentation, see [`docs/API.md`](docs/API.md)

### Core Endpoints

| Endpoint | Purpose | Auth |
|----------|---------|------|
| `POST /api/otp/` | Request OTP (encrypted response) | ECDSA signature |
| `POST /api/jobs/analyze-async-signed` | Submit document | ECDSA + state validation |
| `POST /api/jobs/verification` | Get verification state | ECDSA signature |
| `POST /api/share/request` | Recover secret | Face + ECDSA |
| `POST /keys/add` | Add public key | Face + ECDSA |
| `DELETE /keys/remove` | Remove public key | Face + ECDSA |

### Security Features

- State validation before queuing
- Rate limiting (3 OTP/10min, etc.)
- Replay protection (5-minute timestamp window)
- ECIES PII encryption (user-only decryption)
- Temporary keypairs for recovery
- Full audit trail on all decisions

---

## References

### Critical Dependencies

- **PhotoHolmes**: Forgery detection framework
- **DeepFace**: Face recognition and anti-spoofing
- **MediaPipe**: Hand gesture detection
- **DoTR**: Document OCR engine
- **Shamir**: Secret sharing implementation

### Research & Standards

- ICAO 9303 (Machine Readable Travel Documents)
- NIST Digital Identity Guidelines
- GDPR Compliance Considerations
- Shamir's Secret Sharing (1979)

### Open Source Projects

- Facenet512: Face embedding model
- VADER: Sentiment analysis
- FinBERT: Financial text classification
- spaCy: NLP and NER

---

## License

SPDX-License-Identifier: GPL-3.0

---

## Quick Start

### Prerequisites

NI6 requires the following infrastructure components:

- **Poetry** (Python dependency management)
- **MariaDB** (MySQL-compatible database for each instance)
- **RethinkDB** (Real-time database for cross-instance OTP synchronization)
- **Python 3.10+** with GPU support (CUDA/ROCm/Metal optional but recommended)

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/ni6.git
cd ni6

# Install dependencies with Poetry
poetry install

# Download model weights (required for forgery detection & face recognition)
python scripts/download_model_weights.py

# Copy environment configuration
cp deploy/.env.example deploy/.env

# Edit deploy/.env with your database credentials
# Required: MariaDB host, port, user, password
# Required: RethinkDB host, port, database
```

### Database Setup

```bash
# Create databases and load schemas
./deploy/setup-database.sh
```

This script creates 3 separate databases (`osint_db1`, `osint_db2`, `osint_db3`) and sets up RethinkDB for OTP synchronization.

### Starting the Services

```bash
# Start all 3 NI6 instances (ports 12410, 12411, 12412)
./deploy/start-instances.sh
```

### Deploy Scripts Reference

| Script | Purpose |
|--------|---------|
| `./deploy/start-instances.sh` | Starts 3 NI6 instances on ports 12410-12412. Each instance gets its own database and unique SEED for cryptographic operations. |
| `./deploy/stop-instances.sh` | Gracefully stops all running instances and frees up ports. |
| `./deploy/status-instances.sh` | Shows status of all instances (running/healthy), PIDs, log locations, and recent log entries. |
| `./deploy/setup-database.sh` | Initializes MariaDB databases (3 separate DBs) and RethinkDB tables. Run this before first startup. |
| `./deploy/test-client.sh` | Basic connectivity testing script (note: uses deprecated endpoints, needs update for ECDSA signature support). |

### Viewing Logs

```bash
# Tail all instance logs
tail -f deploy/logs/instance_*/server.log

# View specific instance log
tail -f deploy/logs/instance_1/server.log
```

### Instance Configuration

Each instance is configured as follows:

| Instance | Port | Database | Purpose |
|----------|------|----------|---------|
| Instance 1 | 12410 | osint_db1 | Receives Share 0 |
| Instance 2 | 12411 | osint_db2 | Receives Share 1 |
| Instance 3 | 12412 | osint_db3 | Receives Share 2 |

**Note**: Client applications should split secrets using Shamir's Secret Sharing (2-of-3 threshold) and send one share to each instance. Any 2 shares can reconstruct the original secret.

---

## Contact

For inquiries about NI6 integration or partnership opportunities, contact: [mano@superlexicon.com]
