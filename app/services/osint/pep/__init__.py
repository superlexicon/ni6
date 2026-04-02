"""
PEP (Politically Exposed Persons) Screening Package.

This package provides functionality for screening individuals against
PEP databases sourced from OpenSanctions (via OSSPEP service).

The main app only reads from the database - NO scraping is done here.
Scraping is handled by the separate OSSPEP Quarkus application.
"""

from app.services.osint.pep.pep_checker import PEPChecker

__all__ = [
    'PEPChecker',
]
