"""
Hardened Legal Events Extractor - ALWAYS produces five-column output
Uses centralized client and guarantees table creation even on failures
"""

import logging
from typing import Dict, List, Any

from ..core.langextract_client import LangExtractClient
from ..core.constants import (
    INTERNAL_FIELDS,
    DEFAULT_NO_CITATION,
    DEFAULT_NO_REFERENCE,
    DEFAULT_NO_PARTICULARS,
    DEFAULT_NO_DATE
)

logger = logging.getLogger(__name__)


class LegalEventsExtractor:
    """
    Hardened legal events extractor that ALWAYS produces five-column output
    Uses centralized client and validates all records before returning
    """

    def __init__(self):
        try:
            self.client = LangExtractClient()
            logger.info("✅ Legal events extractor initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LangExtract client: {e}")
            self.client = None

    def extract_legal_events(self, text: str, document_name: str) -> Dict[str, Any]:
        """
        Extract legal events with GUARANTEED five-column output

        Args:
            text: Document text
            document_name: Source document filename

        Returns:
            Dict with legal_events list - ALWAYS contains at least one record
        """
        # GUARANTEE: Always return a valid structure
        base_result = {
            "success": False,
            "legal_events": [],
            "total_events": 0,
            "document_reference": document_name,
            "error": None
        }

        # If no client available, create fallback record
        if not self.client:
            logger.warning("⚠️ LangExtract client not available - creating fallback record")
            fallback_record = self._create_fallback_record(document_name, "LangExtract client not available")
            base_result["legal_events"] = [fallback_record]
            base_result["total_events"] = 1
            return base_result

        # If no text, create fallback record
        if not text or not text.strip():
            logger.warning("⚠️ No text provided - creating fallback record")
            fallback_record = self._create_fallback_record(document_name, "No text content available")
            base_result["legal_events"] = [fallback_record]
            base_result["total_events"] = 1
            return base_result

        # Try LangExtract extraction
        try:
            extraction_result = self.client.extract_legal_events(text, document_name)

            if not extraction_result.get("success", False):
                error_msg = extraction_result.get("error", "Unknown extraction error")
                logger.warning(f"⚠️ LangExtract failed: {error_msg} - creating fallback record")
                fallback_record = self._create_fallback_record(document_name, f"LangExtract failed: {error_msg}")
                base_result["legal_events"] = [fallback_record]
                base_result["total_events"] = 1
                base_result["error"] = error_msg
                return base_result

            # Process successful extractions
            extractions = extraction_result.get("extractions", [])
            if not extractions:
                logger.warning("⚠️ No extractions returned - creating fallback record")
                fallback_record = self._create_fallback_record(document_name, "No legal events found in document")
                base_result["legal_events"] = [fallback_record]
                base_result["total_events"] = 1
                return base_result

            # Transform extractions to five-column format
            legal_events = []
            event_counter = 1

            for extraction in extractions:
                try:
                    # Extract data from LangExtract response
                    event_record = self._transform_extraction_to_record(
                        extraction,
                        event_counter,
                        document_name
                    )

                    # Validate record has all required fields
                    if self._validate_record(event_record):
                        legal_events.append(event_record)
                        event_counter += 1
                    else:
                        logger.warning(f"⚠️ Invalid record structure, skipping: {event_record}")

                except Exception as e:
                    logger.error(f"❌ Failed to process extraction: {e}")
                    continue

            # FINAL GUARANTEE: If no valid events, create fallback
            if not legal_events:
                logger.warning("⚠️ No valid events extracted - creating fallback record")
                fallback_record = self._create_fallback_record(document_name, "No valid legal events could be extracted")
                legal_events = [fallback_record]

            # Success result
            base_result.update({
                "success": True,
                "legal_events": legal_events,
                "total_events": len(legal_events)
            })

            logger.info(f"✅ Successfully extracted {len(legal_events)} legal events")
            return base_result

        except Exception as e:
            logger.error(f"❌ Critical error in legal events extraction: {e}")
            fallback_record = self._create_fallback_record(document_name, f"Critical extraction error: {str(e)}")
            base_result["legal_events"] = [fallback_record]
            base_result["total_events"] = 1
            base_result["error"] = str(e)
            return base_result

    def _transform_extraction_to_record(self, extraction: Dict, number: int, document_name: str) -> Dict[str, Any]:
        """
        Transform LangExtract extraction to five-column record

        Args:
            extraction: Raw extraction from LangExtract
            number: Sequential number for this event
            document_name: Source document filename

        Returns:
            Five-column record dictionary
        """
        # Extract basic information
        extraction_text = extraction.get("extraction_text", "")
        attributes = extraction.get("attributes") or {}

        # Build event particulars
        event_particulars = extraction_text or attributes.get("event_particulars", DEFAULT_NO_PARTICULARS)

        # Extract citation
        citation = attributes.get("citation", "")
        if not citation:
            # Try alternative fields
            citation = attributes.get("legal_reference", "")
        if not citation:
            citation = DEFAULT_NO_CITATION

        # Extract date
        date = attributes.get("date", "")
        if not date:
            # Try alternative date fields
            date = attributes.get("event_date", "") or attributes.get("normalized_date", "")
        if not date or not self._is_valid_date_format(date):
            date = DEFAULT_NO_DATE

        # Create the five-column record
        record = {
            "number": number,
            "date": date,
            "event_particulars": event_particulars,
            "citation": citation,
            "document_reference": document_name
        }

        return record

    def _is_valid_date_format(self, date_str: str) -> bool:
        """
        Validate that date string is in a reasonable format

        Args:
            date_str: Date string to validate

        Returns:
            True if date appears valid, False otherwise
        """
        if not date_str or not isinstance(date_str, str):
            return False

        # Check for common date patterns (YYYY-MM-DD, MM/DD/YYYY, etc.)
        import re

        # ISO format: YYYY-MM-DD
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            return True

        # US format: MM/DD/YYYY
        if re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', date_str):
            return True

        # Various other common formats
        if re.match(r'^\d{1,2}-\d{1,2}-\d{4}$', date_str) or re.match(r'^\d{4}/\d{2}/\d{2}$', date_str):
            return True

        return False

    def _create_fallback_record(self, document_name: str, reason: str) -> Dict[str, Any]:
        """
        Create a fallback record when extraction fails
        Ensures five-column format is always maintained

        Args:
            document_name: Source document filename
            reason: Reason for fallback

        Returns:
            Valid five-column record
        """
        return {
            "number": 1,
            "date": DEFAULT_NO_DATE,
            "event_particulars": f"Processing failed: {reason}",
            "citation": "No citation available (processing failed)",
            "document_reference": document_name
        }

    def _validate_record(self, record: Dict[str, Any]) -> bool:
        """
        Validate that record has all required five-column fields

        Args:
            record: Record to validate

        Returns:
            True if valid, False otherwise
        """
        required_fields = INTERNAL_FIELDS  # ["number", "event_particulars", "citation", "document_reference"]

        # Check all fields exist
        for field in required_fields:
            if field not in record:
                logger.error(f"❌ Missing required field: {field}")
                return False

            # Check field is not None or empty string
            value = record[field]
            if value is None or (isinstance(value, str) and not value.strip()):
                logger.error(f"❌ Empty value for required field: {field}")
                return False

        # Validate number field is positive integer
        if not isinstance(record["number"], int) or record["number"] <= 0:
            logger.error(f"❌ Invalid number field: {record['number']}")
            return False

        return True

    def is_available(self) -> bool:
        """Check if LangExtract is available"""
        return self.client and self.client.is_available()

    def get_required_env_vars(self) -> List[str]:
        """Return required environment variables"""
        return self.client.get_required_env_vars() if self.client else ["GEMINI_API_KEY"]