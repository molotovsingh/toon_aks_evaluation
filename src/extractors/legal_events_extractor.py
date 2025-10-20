"""
Legal Events Extractor - STRICT LangExtract Implementation per Assistant Guardrails
Maps directly to five-column output: No, Date, Event Particulars, Citation, Document Reference
"""

import logging
import os
from typing import List, Dict, Optional, Any
from datetime import datetime

try:
    import langextract as lx
    LANGEXTRACT_AVAILABLE = True
except ImportError:
    LANGEXTRACT_AVAILABLE = False

logger = logging.getLogger(__name__)


class LegalEventsExtractor:
    """
    STRICT LangExtract implementation for legal events extraction
    MANDATORY: Must use real langextract.extract API calls - no mocking allowed per guardrails
    """

    def __init__(self):
        self.available = LANGEXTRACT_AVAILABLE
        self.extractor = None

        if not self.available:
            logger.error("🚨 CRITICAL: langextract module not available")
            raise ImportError("langextract module required for legal events extraction")

        # Use centralized client for all LangExtract operations
        try:
            from ..core.langextract_client import LangExtractClient
            self.client = LangExtractClient()
            logger.info("✅ Using centralized LangExtract client")
        except Exception as e:
            logger.error(f"🚨 CRITICAL: Failed to initialize LangExtract client: {e}")
            raise

        self._setup_extractor()

    def _setup_extractor(self):
        """Setup langextract with legal events schema mapping to five mandatory columns"""
        try:
            # MANDATORY SCHEMA per guardrails - maps directly to five columns
            self.legal_events_schema = {
                "extraction_class": "LegalEventsExtraction",
                "description": "Extract legal events and structure them for five-column table output",
                "fields": {
                    "number": {
                        "type": "integer",
                        "description": "Sequential number for the legal event (starting from 1)"
                    },
                    "event_particulars": {
                        "type": "string",
                        "description": "Detailed description of the legal event, including dates, parties, and circumstances"
                    },
                    "citation": {
                        "type": "string",
                        "description": "Legal citation or reference (statute, case law, regulation) if applicable"
                    },
                    "document_reference": {
                        "type": "string",
                        "description": "Reference to the source document (filename, page, section)"
                    }
                }
            }

            # Few-shot examples using proper langextract.data.ExampleData with Extraction objects
            self.extraction_examples = [
                lx.data.ExampleData(
                    text="On January 15, 2024, the plaintiff filed a motion to dismiss pursuant to Rule 12(b)(6) of the Federal Rules of Civil Procedure.",
                    extractions=[
                        lx.data.Extraction(
                            extraction_text="Plaintiff filed motion to dismiss on January 15, 2024",
                            extraction_class="LegalEvent",
                            attributes={
                                "event_particulars": "Plaintiff filed motion to dismiss on January 15, 2024",
                                "citation": "Fed. R. Civ. P. 12(b)(6)",
                                "document_reference": "Motion Filing Document"
                            }
                        )
                    ]
                ),
                lx.data.ExampleData(
                    text="The contract executed on March 3, 2023, between ABC Corp and XYZ LLC, with an effective date of April 1, 2023, terminates on March 31, 2025.",
                    extractions=[
                        lx.data.Extraction(
                            extraction_text="Contract execution between ABC Corp and XYZ LLC on March 3, 2023, effective April 1, 2023, terminating March 31, 2025",
                            extraction_class="LegalEvent",
                            attributes={
                                "event_particulars": "Contract execution between ABC Corp and XYZ LLC on March 3, 2023, effective April 1, 2023, terminating March 31, 2025",
                                "citation": "Contract Agreement",
                                "document_reference": "Executed Contract Document"
                            }
                        )
                    ]
                )
            ]

            logger.info("✅ Legal events extractor configured with five-column schema")

        except Exception as e:
            logger.error(f"❌ Failed to setup legal events extractor: {e}")
            raise

    def extract_legal_events(self, text: str, document_name: str) -> Dict[str, Any]:
        """
        Extract legal events using centralized LangExtract client

        Args:
            text: Document text from Docling
            document_name: Source document filename

        Returns:
            Dict with structured legal events for five-column table
        """
        if not text or not text.strip():
            logger.error("❌ No text provided for legal events extraction")
            return {
                "success": False,
                "error": "No text content available",
                "legal_events": []
            }

        try:
            logger.info(f"🔍 Starting LangExtract API call via centralized client for: {document_name}")

            # Use centralized client for extraction
            extraction_result = self.client.extract_legal_events(text, document_name)

            if not extraction_result.get("success", False):
                error_msg = extraction_result.get("error", "Unknown extraction error")
                logger.error(f"❌ LangExtract extraction failed: {error_msg}")
                return {
                    "success": False,
                    "error": f"LangExtract API failure: {error_msg}",
                    "legal_events": []
                }

            # Process extractions into five-column format
            extractions = extraction_result.get("extractions", [])
            legal_events = []
            event_counter = 1

            for extraction in extractions:
                try:
                    # Extract data from centralized client response
                    attributes = extraction.get('attributes') or {}
                    extraction_text = extraction.get('extraction_text', '')

                    event_data = {
                        "number": event_counter,
                        "event_particulars": (attributes.get('event_particulars', '') or
                                             extraction_text or
                                             "Event details not available"),
                        "citation": (attributes.get('citation', '') or
                                   attributes.get('legal_reference', '') or
                                   "No citation available"),
                        "document_reference": (attributes.get('document_reference', '') or document_name)
                    }

                    legal_events.append(event_data)
                    event_counter += 1

                    logger.info(f"📋 Extracted legal event {event_counter-1}: {event_data['event_particulars'][:50]}...")

                except Exception as e:
                    logger.error(f"❌ Failed to process extraction: {e}")
                    continue

            if not legal_events:
                logger.error("❌ No legal events extracted - failing per guardrails")
                return {
                    "success": False,
                    "error": "No legal events could be extracted from document",
                    "legal_events": []
                }

            logger.info(f"✅ Successfully extracted {len(legal_events)} legal events")
            return {
                "success": True,
                "legal_events": legal_events,
                "total_events": len(legal_events),
                "document_reference": document_name
            }

        except Exception as e:
            logger.error(f"❌ CRITICAL: LangExtract API call failed: {e}")
            return {
                "success": False,
                "error": f"LangExtract API failure: {str(e)}",
                "legal_events": []
            }

    def is_available(self) -> bool:
        """Check if LangExtract is properly configured"""
        return self.available and self.client and self.client.is_available()

    def get_required_env_vars(self) -> List[str]:
        """Return list of required environment variables"""
        return self.client.get_required_env_vars() if self.client else ['GEMINI_API_KEY']