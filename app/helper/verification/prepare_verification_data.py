
class PrepareVerificationData:
    def prepare_verification_data(self, email: str, doc_verify_result, deepface_verify_result: bool, otp_verify_result: bool) -> list[dict]:
        return [
            # Bank Account
            {
                "email": email,
                "check_item": "Bank Account Match",
                "service_type": "Doctr",
                "verification_percentage": doc_verify_result.bank_statement_match_pct,
                "check_requirement": "User's bank account matches across documents and onboarding data.",
                "check_result": doc_verify_result.is_bank_statement_verified(),
                "category": "identity_verification",
                "manual_check": False,
            },
            # Full Name
            {
                "email": email,
                "check_item": "Full Name Match",
                "service_type": "Doctr",
                "verification_percentage": doc_verify_result.username_match_pct,
                "check_requirement": "User's full name matches across documents and onboarding data.",
                "check_result": doc_verify_result.is_username_verified(),
                "category": "identity_verification",
                "manual_check": False,
            },
            # Nationality
            {
                "email": email,
                "check_item": "Nationality Match",
                "service_type": "Doctr",
                "verification_percentage": doc_verify_result.nationality_match_pct,
                "check_requirement": "User's nationality matches across documents and onboarding data.",
                "check_result": doc_verify_result.is_nationality_verified(),
                "category": "identity_verification",
                "manual_check": False,
            },
            # Age
            {
                "email": email,
                "check_item": "DOB Check/ Age verification",
                "service_type": "Doctr",
                "verification_percentage": doc_verify_result.dob_match_pct,
                "check_requirement": "User's date of birth matches across documents and onboarding data.",
                "check_result": doc_verify_result.is_valid_age,
                "category": "identity_verification",
                "manual_check": False,
            },
            # Phone Number
            {
                "email": email,
                "check_item": "Phone Number Consistency",
                "service_type": "Doctr",
                "verification_percentage": doc_verify_result.phone_number_match_pct,
                "check_requirement": "User's phone number matches resume/onboarding data.",
                "check_result": doc_verify_result.is_phone_number_verified(),
                "category": "contact_and_address_information",
                "manual_check": False,
            },
            # Face Match Score
            {
                "email": email,
                "check_item": "Face Match Score",
                "service_type": "DeepFace",
                "verification_percentage": 100,
                "check_requirement": "Similarity score confirms user's ID photo and selfie match.",
                "check_result": deepface_verify_result,
                "category": "identity_verification",
                "manual_check": False,
            },
            # Deepfake Detection
            {
                "email": email,
                "check_item": "Deepfake Detection",
                "service_type": "DeepFace",
                "verification_percentage": 100,
                "check_requirement": "User's selfie/video shows no deepfake signs.",
                "check_result": deepface_verify_result,
                "category": "identity_verification",
                "manual_check": False,
            },
            # OTP Verification
            {
                "email": email,
                "check_item": "Selfie + Liveness Check",
                "service_type": "Doctr",
                "verification_percentage": 100,
                "check_requirement": "Matches with ID photo and passed liveness detection",
                "check_result": otp_verify_result,
                "category": "identity_verification",
                "manual_check": False,
            },
        ]
