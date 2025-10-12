"""
Core Constants - Shared values across the legal events pipeline
"""

import os

# Five-column headers for legal events table - Date added as column 2
FIVE_COLUMN_HEADERS = ["No", "Date", "Event Particulars", "Citation", "Document Reference"]

# Internal field names used in processing (matching column order)
INTERNAL_FIELDS = ["number", "date", "event_particulars", "citation", "document_reference"]

# Default values for failed extractions
DEFAULT_NO_CITATION = "No citation available"
DEFAULT_NO_REFERENCE = "Unknown document"
DEFAULT_NO_PARTICULARS = "Event details not available"
DEFAULT_NO_DATE = "Date not available"

# ============================================================================
# LEGAL EVENTS PROMPT VERSIONS
# ============================================================================

# V1: Current production prompt (baseline)
LEGAL_EVENTS_PROMPT_V1 = """Extract legal events from this document. For each event, you must return exactly four JSON keys:

1. "event_particulars" - REQUIRED: Provide a complete description (2-8 sentences as appropriate) of what happened, including relevant context, parties involved, procedural background, implications, and any important details. Use verbatim or paraphrased text from the document. NEVER leave this field empty.
2. "citation" - Exact legal authority cited in the event (statute, rule, case, docket, etc.). Copy the verbatim reference from the document. Use empty string "" when no explicit legal citation appears.
3. "document_reference" - Leave as empty string "" (will be filled automatically with source filename)
4. "date" - Specific date mentioned (use empty string "" if no date is found)

CRITICAL REQUIREMENTS:
- "event_particulars" must ALWAYS contain meaningful, comprehensive text - provide a full description with sufficient context for legal analysis
- Use 2-8 sentences as appropriate to capture the complete legal significance and context of each event
- Include relevant background, procedural details, party information, and implications when available in the document
- Use empty strings ("") for missing values EXCEPT for "event_particulars" which must never be empty
- The "citation" field should only contain actual legal references mentioned in the text
- Always use empty string "" for "document_reference" - the system will populate it with the correct filename
- Return all four keys for every extraction

PROHIBITION: Never return an extraction with blank or empty "event_particulars" - every event must have a comprehensive, contextual description.

Extract all legally significant events, proceedings, filings, agreements, and deadlines."""

# V2: Enhanced prompt with improved granularity and recall (2025-10-11)
LEGAL_EVENTS_PROMPT_V2 = """Extract legal events from this document. For each event, you must return exactly four JSON keys:

1. "event_particulars" – REQUIRED: Provide a complete description (1–8 sentences as appropriate) of what happened, including context, parties, procedural posture, implications, and any relevant follow-up. Use verbatim or paraphrased text from the document. NEVER leave this field empty.
2. "citation" – Exact legal authority cited in the event (statute, rule, case, docket, etc.). Copy the verbatim reference from the document. Use the empty string "" when no explicit citation appears.
3. "document_reference" – Leave as the empty string "" (the pipeline fills the filename automatically).
4. "date" – Specific date or date range mentioned. Use the empty string "" only if no date is provided.

CRITICAL REQUIREMENTS:
- Create a SEPARATE event for every distinct dated action or milestone. Do not merge multiple dated actions even if they share a paragraph or storyline (e.g., meetings on July 5, July 10, and July 20 must produce three events if they represent distinct actions).
- Include interim milestones such as inspections, negotiation rounds, warnings, counter-offers, site visits, and status emails—not just formal filings or court orders.
- When the source states that only limited details are available (e.g., "response details not provided"), still create an event summarizing what is known and explicitly note the missing pieces.
- "event_particulars" must ALWAYS contain meaningful, contextual text; describe the action, participants, legal significance, and consequences.
- Use empty strings ("") for missing values EXCEPT for "event_particulars".
- The "citation" field should contain only actual legal references; use "" when none are present.
- Return all four keys for every event, in valid JSON form.

PROHIBITION: Never return an event with blank or superficial "event_particulars". Capture every legally significant development chronologically."""

# Active prompt selection (feature flag)
# Set USE_ENHANCED_PROMPT=true in .env to use V2, otherwise defaults to V1
LEGAL_EVENTS_PROMPT = (
    LEGAL_EVENTS_PROMPT_V2 if os.getenv("USE_ENHANCED_PROMPT", "false").lower() == "true"
    else LEGAL_EVENTS_PROMPT_V1
)

# API configuration
REQUIRED_ENV_VARS = ["GEMINI_API_KEY"]
DEFAULT_MODEL = "gemini-2.0-flash"

# ============================================================================
# MODEL IDENTIFIERS - Premium Models for Ground Truth Creation
# ============================================================================

# OpenAI Models
GPT_4O_MINI = "gpt-4o-mini"
GPT_4O = "gpt-4o"
GPT_5 = "gpt-5"
GPT_5_MINI = "gpt-5-mini"
GPT_5_NANO = "gpt-5-nano"

# Google Gemini Models
GEMINI_2_0_FLASH = "gemini-2.0-flash"
GEMINI_2_5_PRO = "gemini-2.5-pro"

# Anthropic Claude Models
CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
CLAUDE_SONNET_4_5 = "claude-sonnet-4-5"
CLAUDE_OPUS_4 = "claude-opus-4"
CLAUDE_OPUS_4_1 = "claude-opus-4-1"

# DeepSeek Models
DEEPSEEK_CHAT = "deepseek-chat"
DEEPSEEK_R1_DISTILL = "deepseek/deepseek-r1-distill-llama-70b"
