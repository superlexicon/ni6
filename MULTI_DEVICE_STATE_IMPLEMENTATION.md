# Multi-Device Verification State Implementation Summary

## Overview

This implementation adds per-device verification state tracking to support multi-device scenarios where each device (identified by `client_public_key`) has its own verification progress.

## Problem Solved

Previously, `verification_state` was stored only in `user_identity_index` table, which shared a single state across all devices. In multi-device scenarios where different devices use different `client_public_key` values, all devices received the same verification state because they shared the same `user_identity_id`.

**Example Scenario:**
- Device A (`public_key_A`): Completed selfie, passport, and bank statement
- Device B (`public_key_B`): Only completed selfie
- **Before:** Both devices returned state 3 (overall identity state)
- **After:** Device A returns state 3, Device B returns state 1

## Architecture

### Table Relationships

```
user_keys (device-specific)
    ├── user_public_key (PK, unique per device)
    ├── user_identity_id (FK to user_identity_index)
    ├── verification_state (NEW) - Per-device state: 0-3
    ├── sequence_no (NEW) - Per-device progress: 0-3
    └── mobile_number

user_identity_index (shared across devices)
    ├── id (PK)
    ├── verification_state - Best unexpired state across devices
    ├── sequence_no - Best progress across devices
    └── full_name

document_submissions (per-submission record)
    ├── client_public_key (links to user_keys)
    ├── user_identity_id (links to user_identity_index)
    ├── verification_state - From user_keys (not user_identity_index)
    └── sequence_no - From user_keys (not user_identity_index)
```

### State Values

- **State 0:** Initial (ready for selfie)
- **State 1:** Selfie done (ready for passport)
- **State 2:** Passport done (ready for bank statement)
- **State 3:** Complete (bank statement submitted)

## Changes Made

### 1. Database Schema (`schema/migrations/add_user_keys_verification_state.sql`)

Added two new columns to `user_keys` table:
- `verification_state TINYINT NOT NULL DEFAULT 0`
- `sequence_no INT NOT NULL DEFAULT 0`
- Index `idx_verification_state` for efficient queries

Migration script initializes existing records with current `user_identity_index` state.

### 2. Repository Layer (`app/repositories/user_key_repository.py`)

Added new methods for per-device state management:
- `get_verification_state(user_public_key)` - Get device's verification state
- `get_sequence_no(user_public_key)` - Get device's sequence number
- `update_verification_state(user_public_key, state)` - Update device's verification state
- `update_sequence_no(user_public_key, seq)` - Update device's sequence number
- `update_state_and_sequence(user_public_key, state, seq)` - Update both at once

### 3. Service Layer (`app/services/verification_state_service.py`)

Updated `get_verification_state()` and `get_sequence_no()` to:
- Read from `user_keys` table for per-device state (when `client_public_key` is provided)
- Maintain backward compatibility for `user_identity_id` parameters (UUID format detection)
- Remove dependency on `document_submissions` table for state lookups

### 4. Selfie Service (`app/services/sequential_selfie_service.py`)

Updated to set state in `user_keys` on successful selfie verification:
- Update `user_keys` with `verification_state=1, sequence_no=1` for first submission
- Also update `user_identity_index` for overall identity state
- Skip state increment for multi-device links and resubmissions
- Revert state in `user_keys` on verification failures

### 5. Document Processor Base (`app/services/sequential_document_processor_base.py`)

Updated state increment logic:
- Passport/National ID/Driving License: Set `verification_state=2, sequence_no=2`
- Bank Statement: Set `verification_state=3, sequence_no=3`
- Update `user_keys` for per-device state
- Also update `user_identity_index` for overall identity state
- Revert state in `user_keys` on processing failures

### 6. Document Submission Repository (`app/repositories/document_submission_repository.py`)

Updated `_extract_submission_fields()` to:
- Get `verification_state` and `sequence_no` from `user_keys` instead of `response_data`
- This ensures submissions store the device-specific state, not overall identity state

## Migration Steps

### For Existing Databases

1. **Run migration script:**
   ```bash
   mysql -u app_user -p im_osint < schema/migrations/add_user_keys_verification_state.sql
   ```

2. **Verify migration:**
   ```sql
   SELECT user_public_key, verification_state, sequence_no
   FROM user_keys
   LIMIT 10;
   ```

3. **Deploy application changes** - No downtime required:
   - Schema changes are backward compatible (new columns have defaults)
   - Code changes can be deployed incrementally

### For New Installations

Simply run the updated `schema/schema.sql` file (v1.8.0+).

## Verification

### Test Scenario

```sql
-- Create test user with two devices
INSERT INTO user_keys (mobile_number, user_public_key, user_identity_id, verification_state, sequence_no)
VALUES
  ('+1234567890', 'key_device_A', 'identity_123', 3, 3),
  ('+1234567890', 'key_device_B', 'identity_123', 1, 1);
```

```python
# Device A should get state 3
state_A = verification_state_service.get_verification_state('key_device_A')
assert state_A == 3

# Device B should get state 1
state_B = verification_state_service.get_verification_state('key_device_B')
assert state_B == 1
```

## Edge Cases Handled

| Edge Case | Handling |
|-----------|----------|
| New device with no state | Default to 0 in user_keys |
| Device submits first document | Update user_keys with new state |
| UUID passed as parameter | Detected as user_identity_id, use user_identity_index |
| Device deleted | Its state is deleted with user_keys record |
| State update fails | Logs error, continues with current state |
| Multi-device link | Don't increment state (identity already verified) |
| Resubmission | Keep current state (don't increment) |

## Rollback

If needed, rollback the migration:

```sql
ALTER TABLE user_keys DROP COLUMN verification_state;
ALTER TABLE user_keys DROP COLUMN sequence_no;
ALTER TABLE user_keys DROP INDEX idx_verification_state;
```

## Files Modified

- `schema/schema.sql` - Updated user_keys table definition and version
- `schema/migrations/add_user_keys_verification_state.sql` - New migration script
- `app/repositories/user_key_repository.py` - Added state management methods
- `app/services/verification_state_service.py` - Updated to use user_keys
- `app/services/sequential_selfie_service.py` - Updated state management
- `app/services/sequential_document_processor_base.py` - Updated state management
- `app/repositories/document_submission_repository.py` - Updated state source

## Benefits

1. **Clean separation:** `user_keys` = per-device, `user_identity_index` = overall best state
2. **Direct lookup:** No need to query MAX from `document_submissions`
3. **Single source of truth:** Device state stored in one place
4. **Easy to query:** Simple SELECT on `user_keys`
5. **Clear semantics:** Each device has its own verification progress
6. **Backward compatible:** UUID detection maintains compatibility with existing call sites
