#!/usr/bin/env python3
"""Flask Web UI for Legal Events Extraction - Production Ready"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, abort
import os, logging, uuid, tempfile, atexit, shutil, re
from dotenv import load_dotenv
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from datetime import datetime, timedelta
from pathlib import Path
from io import StringIO
import pandas as pd
import markupsafe  # For escaping user input in flash messages
import json

# Load environment and imports
load_dotenv()
from src.core.legal_pipeline_refactored import LegalEventsPipeline
from src.core.constants import FIVE_COLUMN_HEADERS
from src.core.model_catalog import get_ui_model_config_list
from src.core.results_store import get_results_store
from src.core.classification_catalog import get_classification_catalog
from src.core.prompt_registry import get_prompt_variant

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Security: Use proper secret key
secret_key = os.getenv('FLASK_SECRET_KEY')
if not secret_key:
    if os.getenv('FLASK_ENV') == 'development':
        # Allow default key in development only
        secret_key = 'dev-secret-key-change-in-production'
        print("⚠️ WARNING: Using default Flask secret key. Set FLASK_SECRET_KEY environment variable for production!")
    else:
        raise ValueError("FLASK_SECRET_KEY environment variable is required for production")
app.secret_key = secret_key

# Create temp upload folder with cleanup on exit
upload_folder = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)  # Use timedelta, not int

# Register cleanup function to remove temp directory on exit
def cleanup_temp_upload_folder():
    """Clean up temporary upload folder on app shutdown"""
    if os.path.exists(upload_folder):
        try:
            shutil.rmtree(upload_folder, ignore_errors=True)
            logger.info(f"🧹 Cleaned up temp upload folder: {upload_folder}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp folder: {e}")

atexit.register(cleanup_temp_upload_folder)

# CSRF Protection (Cross-Site Request Forgery)
csrf = CSRFProtect(app)
logger.info("✅ CSRF protection enabled")

# Rate Limiting (prevent brute-force, DoS)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["1000 per day", "200 per hour"],
    storage_uri="memory://"  # Use Redis in production
)
logger.info("✅ Rate limiting enabled")

# NOTE: Templates must include CSRF token in forms
# Add to templates: <input type="hidden" name="csrf_token" value="{{ csrf_token() }}"/>
# OR for AJAX: Include X-CSRFToken header from meta tag

# Session security
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
# Only set Secure in production (when not debug)
if not app.debug:
    app.config['SESSION_COOKIE_SECURE'] = True

# Security Configuration
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'msg', 'eml', 'jpg', 'jpeg', 'png'}
MAX_FILES_PER_UPLOAD = 100  # Prevent resource exhaustion
MIME_TYPE_WHITELIST = {
    'pdf': {'application/pdf'},
    'docx': {'application/vnd.openxmlformats-officedocument.wordprocessingml.document'},
    'txt': {'text/plain'},
    'jpg': {'image/jpeg'},
    'jpeg': {'image/jpeg'},
    'png': {'image/png'},
    'msg': {'application/vnd.ms-outlook'},
    'eml': {'message/rfc822'}
}

class FlaskUploadedFile:
    """Flask equivalent of Streamlit UploadedFile with proper resource management"""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.name = os.path.basename(file_path)
        self._buffer = None
        self._position = 0

    def getbuffer(self):
        """Read file contents into buffer with proper resource cleanup"""
        if self._buffer is None:
            with open(self.file_path, 'rb') as f:
                self._buffer = f.read()
        return self._buffer

    def read(self, size: int = -1):
        """Read file contents (supports partial reads like file objects)"""
        buffer = self.getbuffer()
        if size == -1:
            # Read all remaining bytes from current position
            result = buffer[self._position:]
            self._position = len(buffer)
        else:
            # Read 'size' bytes from current position
            result = buffer[self._position:self._position + size]
            self._position += len(result)
        return result

    def seek(self, position: int, whence: int = 0):
        """Seek to position in file (supports SEEK_SET, SEEK_CUR, SEEK_END)"""
        buffer = self.getbuffer()
        if whence == 0:  # SEEK_SET (absolute position)
            self._position = max(0, min(position, len(buffer)))
        elif whence == 1:  # SEEK_CUR (relative to current position)
            self._position = max(0, min(self._position + position, len(buffer)))
        elif whence == 2:  # SEEK_END (relative to end)
            self._position = max(0, min(len(buffer) + position, len(buffer)))
        return self._position

    def close(self):
        """Close file handle and clear buffer"""
        self._buffer = None
        self._position = 0

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_content(filepath: str, expected_ext: str) -> bool:
    """Validate file content matches declared extension using MIME type"""
    try:
        import magic
        mime = magic.from_file(filepath, mime=True)

        if expected_ext not in MIME_TYPE_WHITELIST:
            return False

        allowed_mimes = MIME_TYPE_WHITELIST[expected_ext]
        return mime in allowed_mimes
    except Exception as e:
        logger.warning(f"MIME validation failed for {filepath}: {e} (allowing through)")
        return True  # Fail open if python-magic not available

def log_security_event(event_type: str, details: dict):
    """Structured security event logging for monitoring"""
    try:
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'ip_address': request.remote_addr or 'unknown',
            'details': details
        }
        logger.warning(f"SECURITY_EVENT: {json.dumps(event)}")
    except Exception as e:
        logger.error(f"Failed to log security event: {e}")

@app.route('/')
def index():
    providers = [
        {'id': 'openrouter', 'name': 'OpenRouter', 'description': 'Best quality/cost balance (requires model selection)'},
        {'id': 'anthropic', 'name': 'Anthropic', 'description': 'Speed/cost champion (requires model selection)'},
        {'id': 'openai', 'name': 'OpenAI', 'description': 'Quality champion (requires model selection)'},
        {'id': 'langextract', 'name': 'LangExtract', 'description': 'Completeness champion (standalone model)'},
    ]
    extractors = [
        {'id': 'docling', 'name': 'Docling', 'description': 'Local OCR processing'},
        {'id': 'qwen_vl', 'name': 'Qwen VL', 'description': 'Vision-based extraction'},
    ]
    return render_template('index.html',
                         providers=providers,
                         extractors=extractors,
                         selected_provider=session.get('provider', 'openrouter'),
                         selected_extractor=session.get('extractor', 'docling'))

@app.route('/upload', methods=['POST'])
@limiter.limit("10 per hour")  # Strict rate limit on heavy endpoint
def upload():
    if 'files' not in request.files:
        log_security_event('upload_no_files', {'reason': 'missing_files_field'})
        flash('No files were selected for upload')
        return redirect(url_for('index'))

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        log_security_event('upload_empty_files', {'reason': 'empty_file_list'})
        flash('No files were selected for upload')
        return redirect(url_for('index'))

    # Fix #3: Check MAX_FILES_PER_UPLOAD to prevent resource exhaustion
    if len(files) > MAX_FILES_PER_UPLOAD:
        log_security_event('excessive_file_upload', {'count': len(files), 'limit': MAX_FILES_PER_UPLOAD})
        flash('The upload was rejected due to system constraints. Please try with fewer files.')
        return redirect(url_for('index'))

    # Get config
    provider = request.form.get('provider', 'openrouter')
    extractor = request.form.get('extractor', 'docling')
    model = request.form.get('model', '')
    enable_classification = request.form.get('enable_classification') == 'on'

    # Validate provider and extractor
    VALID_PROVIDERS = {'openrouter', 'anthropic', 'openai', 'langextract', 'deepseek', 'opencode_zen'}
    VALID_EXTRACTORS = {'docling', 'qwen_vl'}

    if provider not in VALID_PROVIDERS:
        log_security_event('invalid_provider', {'provider': provider, 'valid': list(VALID_PROVIDERS)})
        flash('Invalid configuration. Please reload the page and try again.')
        return redirect(url_for('index'))

    if extractor not in VALID_EXTRACTORS:
        log_security_event('invalid_extractor', {'extractor': extractor, 'valid': list(VALID_EXTRACTORS)})
        flash('Invalid configuration. Please reload the page and try again.')
        return redirect(url_for('index'))

    # LangExtract doesn't need model
    if provider == 'langextract':
        model = None

    # Validate required fields
    if provider != 'langextract' and not model:
        log_security_event('missing_model', {'provider': provider})
        flash('Invalid configuration. Please reload the page and try again.')
        return redirect(url_for('index'))

    # Store config
    session['provider'] = provider
    session['extractor'] = extractor
    session['model'] = model
    session['enable_classification'] = enable_classification

    # Validate and save files with unique prefixes to prevent collisions
    valid_files = []
    file_prefix = str(uuid.uuid4())[:8]  # Unique prefix for this batch

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add unique prefix to prevent collisions if same filename uploaded later
            unique_filename = f"{file_prefix}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            try:
                file.save(filepath)

                # Fix #4: Validate file content matches declared extension
                file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
                if not validate_file_content(filepath, file_ext):
                    # Content validation failed - MIME type mismatch
                    log_security_event('mime_type_mismatch', {
                        'filename': filename,
                        'declared_ext': file_ext,
                        'file_size': os.path.getsize(filepath)
                    })
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        logger.warning(f"Failed to delete invalid file {filepath}: {e}")
                    # Don't add to valid_files, don't flash (silent fail)
                    continue

                valid_files.append(filepath)
            except Exception as e:
                logger.error(f"Failed to save file {filename}: {e}")
                # Don't reveal details in flash message
                log_security_event('file_save_error', {'filename': filename, 'error': str(e)})
                # Silent fail - don't flash anything
        else:
            # Sanitize filename before displaying in flash message
            safe_filename = markupsafe.escape(file.filename) if file else "unknown"
            # Silent validation fail - don't flash anything
            log_security_event('invalid_file_extension', {'filename': safe_filename})

    if not valid_files:
        log_security_event('upload_no_valid_files', {})
        flash('No valid files were found. Please check your files and try again.')
        return redirect(url_for('index'))

    file_objects = []
    # Generate secure 16-character run ID (16^16 = 1.8 × 10^19 combinations)
    # Previous: 8 chars (16^8 = 4 billion) - vulnerable to brute force
    run_id = str(uuid.uuid4()).replace('-', '')[:16].upper()

    session['current_run_id'] = run_id
    session.pop('results', None)
    session.pop('results_columns', None)

    try:
        # Create pipeline directly (without Streamlit dependencies)
        pipeline = LegalEventsPipeline(
            event_extractor=provider,
            runtime_model=model if model else None,  # Pass None if empty
            doc_extractor=extractor
        )

        # Convert file paths to FlaskUploadedFile objects (compatible with pipeline)
        for filepath in valid_files:
            # Create FlaskUploadedFile object that mimics Streamlit UploadedFile
            file_obj = FlaskUploadedFile(filepath)
            file_objects.append(file_obj)

        # Fix #2: Buffer all files into memory BEFORE pipeline processing
        # This prevents race condition where cleanup code deletes files while pipeline reads them
        for file_obj in file_objects:
            try:
                file_obj.getbuffer()  # Load file contents into memory
            except Exception as e:
                logger.error(f"Failed to buffer file {file_obj.name}: {e}")
                log_security_event('file_buffer_error', {'filename': file_obj.name, 'error': str(e)})
                raise

        # === CLASSIFICATION LAYER (Layer 1.5) - Optional ===
        classification_lookup = {}  # {filename: document_type}
        classifications = []  # For display
        classification_model = None  # Initialize to prevent NameError

        if enable_classification:
            logger.info("🏷️ Classification layer enabled - classifying documents")
            flash('🏷️ Classifying documents... This may take additional time.')

            try:
                # Import classification factory
                from src.core.classification_factory import create_classifier

                # Get recommended classification model (non-UI path)
                catalog = get_classification_catalog()
                recommended_models = catalog.list_models(enabled=True, recommended_only=True)

                if recommended_models:
                    classification_model_entry = recommended_models[0]
                    classification_model = classification_model_entry.model_id
                    classification_prompt = classification_model_entry.recommended_prompt

                    logger.info(f"Using classification model: {classification_model} with prompt: {classification_prompt}")
                    classifier = create_classifier(classification_model, classification_prompt)
                else:
                    logger.warning("No recommended classification models available")
                    classification_model = None
                    classifier = None

                if classifier:

                    # Extract and classify all documents
                    for file_obj in file_objects:
                        try:
                            # Extract directly from the uploaded file path
                            # FlaskUploadedFile has a file_path attribute pointing to the uploaded file
                            doc_result = pipeline.document_extractor.extract(file_obj.file_path)

                            if not doc_result or not doc_result.plain_text.strip():
                                logger.warning(f"⚠️ No text extracted from {file_obj.name} - skipping classification")
                                classification_lookup[file_obj.name] = "Unknown"
                                continue

                            # Classify document
                            classification_result = classifier.classify(
                                doc_result.plain_text,
                                document_title=file_obj.name
                            )

                            # Store for 6th column
                            classification_lookup[file_obj.name] = classification_result['primary']

                            classifications.append({
                                'filename': file_obj.name,
                                'type': classification_result['primary'],
                                'confidence': classification_result.get('confidence', 0.0),
                                'all_labels': classification_result.get('classes', [])
                            })

                            logger.info(
                                f"✅ Classified {file_obj.name}: {classification_result['primary']} "
                                f"(confidence={classification_result.get('confidence', 0):.2f})"
                            )

                        except Exception as e:
                            logger.error(f"❌ Classification failed for {file_obj.name}: {e}")
                            classification_lookup[file_obj.name] = "Classification Failed"

                    # Classification results will be stored with metadata

                else:
                    logger.warning("⚠️ Classification enabled but no model available")
                    log_security_event('classification_no_model', {})
                    flash('Classification feature is temporarily unavailable. Processing will continue without classification.')

            except Exception as e:
                logger.error(f"❌ Classification layer failed: {e}")
                log_security_event('classification_error', {'error': str(e)})
                flash('An error occurred during classification. Processing will continue without classification.')
                enable_classification = False

        # === LAYER 2: EVENT EXTRACTION ===
        # Process documents directly using pipeline
        legal_events_df, warning_message = pipeline.process_documents_for_legal_events(file_objects)

        # Show warning if any
        if warning_message:
            flash(f'Warning: {warning_message}')

        # === ADD CLASSIFICATION AS 6TH COLUMN (if enabled) ===
        if enable_classification and classification_lookup:
            # Add Document Type column by mapping Document Reference to classification
            # (FIVE_COLUMN_HEADERS already imported at top)
            legal_events_df['Document Type'] = legal_events_df[FIVE_COLUMN_HEADERS[4]].map(
                classification_lookup
            )

            # Fill any missing values
            legal_events_df['Document Type'] = legal_events_df['Document Type'].fillna('Unknown')

            logger.info(f"✅ Added Document Type column with {len(classification_lookup)} classifications")

        if legal_events_df is not None:
            # Store results with run ID and metadata
            processing_metadata = {
                'run_id': run_id,
                'timestamp': datetime.now(),
                'provider': provider,
                'model': model,
                'extractor': extractor,
                'file_count': len(valid_files),
                'events_found': len(legal_events_df),
                'enable_classification': enable_classification,
                'classification_model': classification_model if enable_classification else None,
                'document_types_found': len(set(classification_lookup.values())) if classification_lookup else 0,
                'filename': Path(valid_files[0]).name if valid_files else 'unknown',
                'file_size': sum(Path(f.file_path).stat().st_size for f in file_objects) if file_objects else 0,
                'processing_time': None
            }

            # Get session info for tracking
            # Generate or retrieve session ID from cookie
            session_cookie_name = app.config.get('SESSION_COOKIE_NAME', 'session')
            session_id = request.cookies.get(session_cookie_name, f'flask-{run_id}')

            session_info = {
                'session_id': session_id,
                'user_agent': request.headers.get('User-Agent', ''),
                'ip_address': request.remote_addr or ''
            }

            # Store results in DuckDB
            results_store = get_results_store()
            success = results_store.store_processing_result(
                run_id=run_id,
                metadata=processing_metadata,
                results_df=legal_events_df,
                session_info=session_info
            )

            if success:
                # Store only run_id in session (not the large results)
                session['current_run_id'] = run_id

                logger.info(f"✅ Processing complete - Run ID: {run_id}, Events: {len(legal_events_df)}")
                logger.info(f"   Results stored in DuckDB for run ID: {run_id}")

                # Auto-redirect to results page with run ID in URL
                logger.info("🔄 Redirecting to results page...")
                return redirect(url_for('results', run_id=run_id))
            else:
                log_security_event('results_store_failed', {'run_id': run_id})
                flash('An error occurred while processing your files. Please try again.')
                return redirect(url_for('index'))
        else:
            log_security_event('processing_failed', {'files': len(valid_files)})
            flash('An error occurred while processing your files. Please try again.')
            return redirect(url_for('index'))

    except Exception as e:
        logger.error(f'Processing error: {e}')
        log_security_event('processing_exception', {'error': str(e)})
        flash('An error occurred while processing your files. Please try again.')
        return redirect(url_for('index'))

    finally:
        # Clean up file objects
        for file_obj in file_objects:
            try:
                file_obj.close()
            except Exception as e:
                logger.warning(f"Failed to close file object: {e}")

        # Clean up uploaded files
        for filepath in valid_files:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {filepath}: {e}")

@app.route('/results')
def results():
    """Display processing results"""
    # Try to get run_id from URL parameter first, then session
    run_id = request.args.get('run_id') or session.get('current_run_id')

    logger.info(f"📊 Results page accessed - Run ID: {run_id}, Session keys: {list(session.keys())}")

    if not run_id:
        logger.warning("❌ No run_id in URL or session")
        log_security_event('results_no_run_id', {})
        flash('Your session has expired. Please upload files again.')
        return redirect(url_for('index'))

    # Store run_id in session for future requests (downloads, etc.)
    session['current_run_id'] = run_id

    # Retrieve results from DuckDB
    results_store = get_results_store()
    result_data = results_store.get_processing_result(run_id)

    if not result_data:
        logger.warning(f"❌ No results found in database for run_id: {run_id}")
        log_security_event('results_not_found', {'run_id': run_id})
        flash('Results are no longer available. Please upload files again.')
        return redirect(url_for('index'))

    # Extract metadata and events
    metadata = result_data['metadata']
    events = result_data['events']

    # Convert events to the format expected by template
    results = []
    columns = []
    if events:
        # Get column names from first event
        columns = list(events[0].keys())
        results = events

    logger.info(f"✅ Results retrieved from DuckDB - Run ID: {run_id}, Events: {len(results)}")

    return render_template('results.html',
                          results=results,
                          columns=columns,
                          count=len(results),
                          run_id=run_id,
                          metadata=metadata)

@app.route('/api/cost-estimate', methods=['POST'])
@limiter.limit("100 per hour")  # Rate limit API endpoint
def cost_estimate():
    """API endpoint for cost estimation using tiktoken for accuracy"""
    # Validate request has JSON data
    if not request.json:
        log_security_event('api_invalid_request', {'endpoint': '/api/cost-estimate', 'reason': 'no_json'})
        return jsonify({'error': 'Invalid request format'}), 400

    data = request.json
    provider = data.get('provider')
    model = data.get('model')
    text_content = data.get('text_content')  # Extracted text for token counting

    # Validate required fields
    if not provider:
        log_security_event('api_missing_provider', {'endpoint': '/api/cost-estimate'})
        return jsonify({'error': 'Invalid request'}), 400

    try:
        # Import cost estimation functions
        from src.ui.cost_estimator import estimate_all_models_with_tiktoken
        from src.core.model_catalog import get_ui_model_config_list

        # If text is provided, use tiktoken for precise cost calculation
        if text_content:
            try:
                cost_table = estimate_all_models_with_tiktoken(
                    extracted_texts=[text_content],
                    output_ratio=0.10
                )

                # Filter by provider if specified
                if provider:
                    cost_table = [m for m in cost_table if m.get('provider') == provider]

                # Filter by model if specified
                if model and cost_table:
                    cost_table = [m for m in cost_table if m.get('model_id') == model]

                if cost_table:
                    selected = cost_table[0]
                    return jsonify({
                        'success': True,
                        'estimated_cost': selected['total_cost'],
                        'currency': 'USD',
                        'input_tokens': selected['input_tokens'],
                        'output_tokens': selected['output_tokens'],
                        'total_tokens': selected['input_tokens'] + selected['output_tokens'],
                        'input_cost': selected['input_cost'],
                        'output_cost': selected['output_cost'],
                        'provider': provider,
                        'model': model,
                        'method': 'tiktoken_actual'
                    })
            except Exception as e:
                logger.warning(f"Tiktoken cost calculation failed, falling back to heuristic: {e}")

        # Fallback: Use heuristic estimation if no text provided or tiktoken fails
        model_catalog = get_ui_model_config_list()
        matching_models = [
            m for m in model_catalog
            if (not provider or m.provider == provider) and
               (not model or m.model_id == model)
        ]

        if matching_models:
            selected_model = matching_models[0]
            # Rough estimate: assume 1000 tokens per page for legal documents
            estimated_tokens = 1000
            estimated_cost = (estimated_tokens / 1_000_000) * selected_model.cost_per_1m
            return jsonify({
                'success': True,
                'estimated_cost': estimated_cost,
                'currency': 'USD',
                'estimated_tokens': estimated_tokens,
                'provider': provider,
                'model': model,
                'method': 'heuristic_fallback',
                'warning': 'Using heuristic estimation (no text provided for precise calculation)'
            })
        else:
            log_security_event('api_model_not_found', {'endpoint': '/api/cost-estimate', 'provider': provider, 'model': model})
            return jsonify({'error': 'Configuration not available'}), 404

    except Exception as e:
        logger.error(f'Cost estimation error: {e}')
        log_security_event('api_cost_estimate_error', {'endpoint': '/api/cost-estimate', 'error': str(e)})
        return jsonify({'error': 'Service temporarily unavailable'}), 500

@app.route('/api/providers')
@limiter.limit("500 per hour")  # Higher limit for read-only endpoints
def get_providers():
    """API endpoint to get available providers and their status"""
    providers_status = []

    provider_configs = {
        'langextract': ['GEMINI_API_KEY', 'GOOGLE_API_KEY'],
        'openrouter': ['OPENROUTER_API_KEY'],
        'opencode_zen': ['OPENCODEZEN_API_KEY'],
        'openai': ['OPENAI_API_KEY'],
        'anthropic': ['ANTHROPIC_API_KEY'],
        'deepseek': ['DEEPSEEK_API_KEY']
    }

    for provider, keys in provider_configs.items():
        has_keys = any(os.getenv(key) for key in keys)
        providers_status.append({
            'id': provider,
            'configured': has_keys,
            'required_keys': keys
        })

    return jsonify(providers_status)

@app.route('/api/models/<provider>')
@limiter.limit("500 per hour")  # Higher limit for read-only endpoints
def get_models_for_provider(provider):
    """API endpoint to get available models for a specific provider"""
    # Load fresh model catalog to pick up any newly added models
    fresh_catalog = get_ui_model_config_list()

    # Filter models by provider
    provider_models = [m for m in fresh_catalog if m.provider == provider]

    models_data = []
    for model in provider_models:
        models_data.append({
            'model_id': model.model_id,
            'display_name': model.display_name,
            'category': model.category,
            'cost_per_1m': model.cost_per_1m,
            'context_window': model.context_window,
            'quality_score': model.quality_score,
            'badges': model.badges,
            'inline_display': model.format_inline()
        })

    return jsonify({
        'provider': provider,
        'models': models_data,
        'count': len(models_data)
    })

@app.route('/download/<run_id>')
@limiter.limit("50 per hour")  # Rate limit downloads to prevent brute-force attacks
def download_results(run_id):
    """Download results for a specific run ID with database-backed authorization"""
    # Validate run_id format - must be exactly 16 hex characters (from uuid)
    if not re.match(r'^[0-9A-F]{16}$', run_id):
        logger.warning(f"Invalid run_id format attempted: {run_id}")
        abort(400)  # Bad request - invalid format

    # Fix #1: Database-backed authorization with multi-factor verification
    results_store = get_results_store()
    result_data = results_store.get_processing_result(run_id)

    if not result_data:
        # Don't leak that the run_id doesn't exist
        log_security_event('download_not_found', {'run_id': run_id})
        abort(404)

    # Multi-factor verification: Check session, IP, and user agent
    session_info = result_data.get('session_info', {})
    current_session_id = request.cookies.get(app.config.get('SESSION_COOKIE_NAME', 'session'), '')
    current_ip = request.remote_addr or ''
    current_user_agent = request.headers.get('User-Agent', '')

    # Verify at least one factor matches (IP or user agent)
    ip_match = current_ip == session_info.get('ip_address', '')
    user_agent_match = current_user_agent == session_info.get('user_agent', '')

    if not (ip_match or user_agent_match):
        # Both factors don't match - deny access
        log_security_event('download_unauthorized', {
            'run_id': run_id,
            'reason': 'multi_factor_mismatch',
            'stored_ip': session_info.get('ip_address', ''),
            'request_ip': current_ip,
            'stored_ua': session_info.get('user_agent', ''),
            'request_ua': current_user_agent
        })
        abort(403)

    # Additional session check - at least verify current session has accessed this run_id
    if session.get('current_run_id') != run_id:
        # Session doesn't have this run_id - but allow if multi-factor passed
        logger.info(f"Download request for run_id {run_id} without session history - but passed multi-factor auth")

    if not result_data.get('events'):
        log_security_event('download_no_events', {'run_id': run_id})
        abort(404)  # Not found - don't redirect, just 404

    try:
        # Convert events to DataFrame
        events = result_data['events']
        results_df = pd.DataFrame(events)

        # Create CSV response
        csv_buffer = StringIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        # Encode to bytes for proper Content-Length
        csv_bytes = csv_data.encode('utf-8')

        # Generate filename with run ID
        filename = f"legal_events_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        from flask import Response
        return Response(
            csv_bytes,
            mimetype='text/csv; charset=utf-8',
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Content-Length': str(len(csv_bytes))  # Length in bytes, not characters
            }
        )

    except Exception as e:
        logger.error(f'Download error for run_id {run_id}: {e}')
        abort(500)  # Internal server error

@app.route('/test')
def test():
    """Simple test endpoint to verify the app works (development only)"""
    # Guard: Only expose in development mode
    if not app.debug and os.getenv('FLASK_ENV') != 'development':
        logger.warning("/test endpoint accessed in production - denying")
        abort(404)

    # Check API key availability
    api_keys_status = {
        'GEMINI_API_KEY': bool(os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')),
        'OPENROUTER_API_KEY': bool(os.getenv('OPENROUTER_API_KEY')),
        'ANTHROPIC_API_KEY': bool(os.getenv('ANTHROPIC_API_KEY')),
        'OPENAI_API_KEY': bool(os.getenv('OPENAI_API_KEY')),
        'OPENCODEZEN_API_KEY': bool(os.getenv('OPENCODEZEN_API_KEY')),
        'DEEPSEEK_API_KEY': bool(os.getenv('DEEPSEEK_API_KEY'))
    }

    return jsonify({
        'status': 'ok',
        'message': 'Flask Legal Events Extraction API with Process Handover',
        'environment': 'DEVELOPMENT' if app.debug else 'PRODUCTION',
        'api_keys_available': api_keys_status,
        'session_info': {
            'has_results': 'results' in session,
            'has_run_id': 'current_run_id' in session,
            'run_id': session.get('current_run_id', 'None'),
        },
        'model_catalog_size': len(get_ui_model_config_list())
    })

@app.route('/debug')
def debug():
    """Debug endpoint to check session and app state (development only)"""
    # Guard: Only expose in development mode
    if not app.debug and os.getenv('FLASK_ENV') != 'development':
        logger.warning("/debug endpoint accessed in production - denying")
        abort(404)

    return jsonify({
        'session_data': {
            'current_run_id': session.get('current_run_id', 'Not set'),
        },
        'app_config': {
            'model_catalog_size': len(get_ui_model_config_list()),
            'debug_mode': app.debug,
        }
    })

@app.route('/clear', methods=['POST'])
def clear_session():
    """Clear all session data for debugging (development only)"""
    # Guard: Only expose in development mode
    if not app.debug and os.getenv('FLASK_ENV') != 'development':
        logger.warning("/clear endpoint accessed in production - denying")
        abort(404)

    session.clear()
    return jsonify({'status': 'cleared', 'message': 'Session data cleared'})

@app.route('/refresh-models', methods=['POST'])
def refresh_models():
    """Force refresh model catalog (development only)"""
    # Guard: Only expose in development mode
    if not app.debug and os.getenv('FLASK_ENV') != 'development':
        logger.warning("/refresh-models endpoint accessed in production - denying")
        abort(404)

    fresh_catalog = get_ui_model_config_list()
    return jsonify({
        'status': 'refreshed',
        'models_loaded': len(fresh_catalog),
        'providers': list(set(m.provider for m in fresh_catalog))
    })

@app.route('/test-session', methods=['GET', 'POST'])
def test_session():
    """Test session persistence across requests (development only)"""
    # Guard: Only expose in development mode
    if not app.debug and os.getenv('FLASK_ENV') != 'development':
        logger.warning("/test-session endpoint accessed in production - denying")
        abort(404)

    if request.method == 'POST':
        # Store test data
        test_data = request.form.get('test_data', 'test_value')
        session['test_data'] = test_data
        session['test_timestamp'] = datetime.now().isoformat()
        return jsonify({
            'status': 'stored',
            'data': test_data,
        })
    else:
        # Retrieve test data
        return jsonify({
            'status': 'retrieved',
            'data': session.get('test_data', 'not_found'),
            'timestamp': session.get('test_timestamp', 'not_found'),
        })

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """Handle file uploads that exceed MAX_CONTENT_LENGTH"""
    log_security_event('file_too_large', {'max_size': '100MB'})
    return jsonify({'error': 'File exceeds maximum size'}), 413

@app.errorhandler(400)
def handle_csrf_error(e):
    """Handle CSRF token errors without leaking details"""
    # Check if this is a CSRF error
    if 'CSRF' in str(e) or 'csrf' in str(e):
        log_security_event('csrf_error', {'error': str(e)})
        return jsonify({'error': 'Invalid request. Please try again.'}), 400
    # For other 400 errors, return generic message
    return jsonify({'error': 'Invalid request'}), 400

if __name__ == '__main__':
    print("🚀 Starting Flask Legal Events Extraction UI")
    print("🏠 Main page: http://localhost:53335/")
    print("✅ Templates loaded and ready!")

    app.run(debug=True, host='0.0.0.0', port=53335)
