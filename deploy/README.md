# OSINT Localhost Deployment

This directory contains scripts for deploying and testing 3 OSINT instances on localhost for Shamir Secret Sharing testing.

## Overview

The deployment runs 3 separate OSINT server instances on different ports:
- **Instance 1**: `http://localhost:12410`
- **Instance 2**: `http://localhost:12411`
- **Instance 3**: `http://localhost:12412`

Each instance has its own separate MySQL database (`osint_db1`, `osint_db2`, `osint_db3`) for complete data isolation.

## Architecture

```
Client Application
       |
       | (Splits secret into 3 Shamir shares)
       |
       +--- Share 1 ---> Instance 1 (Port 12410) ---> MySQL (osint_db1)
       |
       +--- Share 2 ---> Instance 2 (Port 12411) ---> MySQL (osint_db2)
       |
       +--- Share 3 ---> Instance 3 (Port 12412) ---> MySQL (osint_db3)
```

## Prerequisites

1. **Python 3.12+** with Poetry installed
2. **MySQL Server** running on `localhost:3306`
3. **Ports available**: 12410, 12411, 12412

## Quick Start

### 1. Setup Database

First, initialize the MySQL database:

```bash
# Option 1: With root password
export DB_ROOT_PASSWORD='your_mysql_root_password'
./deploy/setup-database.sh

# Option 2: Without password (if MySQL allows root login without password)
./deploy/setup-database.sh
```

This creates:
- 3 Databases: `osint_db1`, `osint_db2`, `osint_db3`
- User: `osint_user` / `osint_password`
- Tables: All schema from `schema/schema.sql` (loaded into each database)

### 2. Start Instances

Start all 3 OSINT instances:

```bash
./deploy/start-instances.sh
```

This will:
- Kill any existing instances on ports 12410-12412
- Create instance-specific directories
- Start 3 server processes with environment variables passed via command line
- Each instance connects to its own database (osint_db1, osint_db2, osint_db3)
- Wait for health checks to pass

### 3. Check Status

View the status of all instances:

```bash
./deploy/status-instances.sh
```

Output includes:
- Running status (PID, port, health)
- Recent log entries
- Connection URLs
- Management commands

### 4. Test Client

Run the test client to verify Shamir secret sharing:

```bash
./deploy/test-client.sh
```

This simulates:
- Generating OTPs from each instance
- Splitting a secret into 3 shares
- Submitting each share to the corresponding instance

### 5. Stop Instances

Stop all running instances:

```bash
./deploy/stop-instances.sh
```

## Scripts Reference

### `setup-database.sh`

Initializes MySQL database for all instances.

**Environment Variables:**
- `DB_HOST` (default: `localhost`)
- `DB_PORT` (default: `3306`)
- `DB_NAME` (default: `osint_db` - will create `osint_db1`, `osint_db2`, `osint_db3`)
- `DB_USER` (default: `osint_user`)
- `DB_PASSWORD` (default: `osint_password`)
- `DB_ROOT_PASSWORD` (required for database creation)

### `start-instances.sh`

Starts 3 OSINT instances on ports 12410-12412.

**Features:**
- Automatically kills existing instances
- Creates instance-specific directories
- Starts processes with environment variables passed via command line (no .env files)
- Each instance connects to its own database
- Runs health checks
- Displays status summary

**Generated Files:**
- `deploy/instances/instance_*/` - Instance directories
- `deploy/logs/instance_*/server.log` - Server logs
- `deploy/pids.txt` - Process IDs

### `stop-instances.sh`

Stops all running instances.

**Features:**
- Kills processes from PID file
- Kills processes using ports 12410-12412
- Verifies all ports are free

### `status-instances.sh`

Displays detailed status of all instances.

**Shows:**
- Running status (PID, port)
- Health check status
- Recent log entries
- Connection URLs
- Management commands

### `test-client.sh`

Tests Shamir secret sharing with all 3 instances.

**Test Flow:**
1. Verify all instances are healthy
2. Generate OTP from each instance
3. Create test selfie with OTP
4. Submit secret share to each instance
5. Display results

## Directory Structure

```
deploy/
├── README.md                    # This file
├── setup-database.sh            # Database initialization (creates 3 DBs)
├── start-instances.sh           # Start all instances (env vars in CLI)
├── stop-instances.sh            # Stop all instances
├── status-instances.sh          # Check instance status
├── test-client.sh               # Test client for Shamir sharing
├── instances/                   # Instance directories (generated)
│   ├── instance_1/
│   ├── instance_2/
│   └── instance_3/
├── logs/                        # Server logs (generated)
│   ├── instance_1/server.log
│   ├── instance_2/server.log
│   └── instance_3/server.log
└── pids.txt                     # Process IDs (generated)
```

## Shamir Secret Sharing Workflow

### Initial Verification (Client splits secret)

1. **Client**: Generate OTP for mobile number from any instance
2. **Client**: Split secret seed into 3 Shamir shares
3. **Client**: For each instance (1, 2, 3):
   ```bash
   POST http://localhost:1241{0,1,2}/api/jobs/analyze-async
   {
     "client_public_key": "...",
     "secret_share": "share_{1,2,3}_plaintext",
     "documents": [
       {
         "type": "selfie",
         "data": "base64_image_with_otp",
         "filename": "selfie_OTP123456.jpg"
       }
     ]
   }
   ```
4. **Server**: Validates selfie (OTP, PhotoHolmes, anti-spoofing)
5. **Server**: Encrypts share with `key_injection_manager.encrypt_data()`
6. **Server**: Stores encrypted share in `user_keys.encrypted_secret_share`

### Secret Share Recovery (User requests share back)

1. **Client**: For each instance:
   ```bash
   POST http://localhost:1241{0,1,2}/api/share/request
   {
     "public_key": "...",
     "temp_public_key": "...",
     "selfie_data": "base64_verification_selfie",
     "filename": "selfie_OTP789012.jpg"
   }
   ```
2. **Server**: Validates verification selfie
3. **Server**: Matches face against last 3 stored faces (70% threshold)
4. **Server**: Decrypts share from storage
5. **Server**: Re-encrypts with `temp_public_key`
6. **Server**: Returns re-encrypted share
7. **Client**: Combines 3 shares to reconstruct secret seed

## Troubleshooting

### MySQL Connection Errors

```bash
# Check MySQL is running
mysqladmin ping -h localhost -P 3306

# Test connection
mysql -h localhost -P 3306 -u osint_user -posint_password osint_db

# Reset database
./deploy/setup-database.sh
```

### Port Already in Use

```bash
# Check what's using the port
lsof -i :12410

# Kill process manually
kill -9 <PID>

# Or use stop script
./deploy/stop-instances.sh
```

### Instance Won't Start

```bash
# Check logs
tail -f deploy/logs/instance_1/server.log

# Verify Python environment
poetry env info
poetry install

# Check port availability
lsof -i :12410
```

### Health Check Timeout

The instance may still be starting. Health checks can take up to 60 seconds. Check logs:

```bash
tail -f deploy/logs/instance_*/server.log
```

## Viewing Logs

### Real-time logs for all instances
```bash
tail -f deploy/logs/instance_*/server.log
```

### Logs for specific instance
```bash
tail -f deploy/logs/instance_1/server.log
tail -f deploy/logs/instance_2/server.log
tail -f deploy/logs/instance_3/server.log
```

### Search logs for errors
```bash
grep -i error deploy/logs/instance_*/server.log
```

## Database Access

### Connect to database
```bash
mysql -h localhost -P 3306 -u osint_user -posint_password osint_db
```

### View stored shares
```sql
SELECT
    user_public_key,
    mobile_number,
    created_at
FROM user_keys
ORDER BY created_at DESC;
```

### View face biometrics
```sql
SELECT
    public_key,
    document_type,
    submission_type,
    created_at
FROM face_biometrics
ORDER BY created_at DESC;
```

## Environment Variables

Each instance receives environment variables via command line (no .env files):

**Instance 1:**
```bash
INSTANCE_ID=1
INSTANCE_NAME=OSINT_Instance_1
PORT=12410
DB_NAME=osint_db1
# ... other vars
```

**Instance 2:**
```bash
INSTANCE_ID=2
INSTANCE_NAME=OSINT_Instance_2
PORT=12411
DB_NAME=osint_db2
# ... other vars
```

**Instance 3:**
```bash
INSTANCE_ID=3
INSTANCE_NAME=OSINT_Instance_3
PORT=12412
DB_NAME=osint_db3
# ... other vars
```

**Common Variables:**
- `DB_HOST=localhost`
- `DB_PORT=3306`
- `DB_USER=osint_user`
- `DB_PASSWORD=osint_password`
- `PYTORCH_ENABLE_MPS_FALLBACK=1`
- `HOST=0.0.0.0`
- `RELOAD=False`

## Production Considerations

This deployment is for **localhost testing only**. For production:

1. **Security**:
   - Use HTTPS with TLS certificates
   - Implement proper authentication
   - Use strong database passwords
   - Enable firewall rules

2. **High Availability**:
   - Deploy instances on separate servers
   - Use load balancer
   - Implement health monitoring
   - Setup automatic failover

3. **Database**:
   - Use managed MySQL service (AWS RDS, Google Cloud SQL)
   - Enable replication for backup
   - Configure connection pooling
   - Optimize indexes

4. **Secrets Management**:
   - Use environment-specific secrets
   - Rotate encryption keys
   - Implement key management service (AWS KMS, HashiCorp Vault)

5. **Logging & Monitoring**:
   - Centralized logging (ELK stack, CloudWatch)
   - Performance monitoring (Prometheus, Grafana)
   - Alert on errors and anomalies

## Next Steps

After successful localhost testing:

1. ✅ Verify all 3 instances can start and run simultaneously
2. ✅ Test database connectivity from all instances
3. ✅ Test OTP generation and validation
4. ✅ Implement client-side Shamir secret splitting
5. ✅ Test secret share submission to all instances
6. ✅ Verify encrypted shares are stored correctly
7. ✅ Test face matching and secret recovery
8. ✅ Implement proper error handling
9. ✅ Add comprehensive logging
10. ✅ Plan production deployment architecture

## Support

For issues or questions:
- Check logs: `tail -f deploy/logs/instance_*/server.log`
- Check status: `./deploy/status-instances.sh`
- Review documentation: `schema/schema.sql`, `app/main.py`
