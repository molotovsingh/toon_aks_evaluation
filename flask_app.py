#!/usr/bin/env python3
"""Flask Web UI for Legal Events Extraction - Production Ready"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import os, logging, uuid, tempfile
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from datetime import datetime
from pathlib import Path
from io import StringIO
import pandas as pd

# Load environment and imports
load_dotenv()
from src.core.legal_pipeline_refactored import LegalEventsPipeline
from src.utils.file_handler import FileHandler
from src.core.constants import FIVE_COLUMN_HEADERS
from src.core.model_catalog import get_ui_model_config_list
from src.ui.classification_ui import create_classification_config

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
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'msg', 'eml', 'jpg', 'jpeg', 'png'}

class FlaskUploadedFile:
    """Flask equivalent of Streamlit UploadedFile"""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.name = os.path.basename(file_path)
        self._file_obj = None

    def getbuffer(self):
        if self._file_obj is None:
            self._file_obj = open(self.file_path, 'rb')
        self._file_obj.seek(0)
        return self._file_obj.read()

    def read(self): return self.getbuffer()
    def seek(self, position): self._file_obj and self._file_obj.seek(position)
    def close(self): self._file_obj and self._file_obj.close(); self._file_obj = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
def upload():
    if 'files' not in request.files:
        flash('No files selected')
        return redirect(url_for('index'))

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        flash('No files selected')
        return redirect(url_for('index'))

    # Get config
    provider = request.form.get('provider', 'openrouter')
    extractor = request.form.get('extractor', 'docling')
    model = request.form.get('model', '')
    enable_classification = request.form.get('enable_classification') == 'on'

    # LangExtract doesn't need model
    if provider == 'langextract':
        model = None

    # Validate required fields
    if provider != 'langextract' and not model:
        flash('Model selection is required for this provider')
        return redirect(url_for('index'))

    # Store config
    session['provider'] = provider
    session['extractor'] = extractor
    session['model'] = model
    session['enable_classification'] = enable_classification

    # Validate and save files
    valid_files = []
    file_handler = FileHandler()

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            valid_files.append(filepath)
        else:
            flash(f'Invalid file type: {file.filename}')

    if not valid_files:
        flash('No valid files uploaded')
        return redirect(url_for('index'))

    file_objects = []
    run_id = str(uuid.uuid4())[:8].upper()

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

        # === CLASSIFICATION LAYER (Layer 1.5) - Optional ===
        classification_lookup = {}  # {filename: document_type}
        classifications = []  # For display

        if enable_classification:
            logger.info("🏷️ Classification layer enabled - classifying documents")
            flash('🏷️ Classifying documents... This may take additional time.')

            try:
                # Import classification factory
                from src.core.classification_factory import create_classifier

                # Create classifier (uses OpenRouter by default)
                classification_model, classification_prompt = create_classification_config()
                if classification_model:
                    classifier = create_classifier(classification_model, classification_prompt)

                    # Extract and classify all documents
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = Path(temp_dir)

                        for file_obj in file_objects:
                            try:
                                # Save and extract (uses cache)
                                file_path = file_handler.save_uploaded_file(file_obj, temp_path)
                                doc_result = pipeline.document_extractor.extract(file_path)

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
                    flash('⚠️ Classification requested but no classification model available')

            except Exception as e:
                logger.error(f"❌ Classification layer failed: {e}")
                flash(f'🚨 Classification failed: {str(e)}\n\nProceeding without classification...')
                enable_classification = False

        # === LAYER 2: EVENT EXTRACTION ===
        # Process documents directly using pipeline
        legal_events_df, warning_message = pipeline.process_documents_for_legal_events(file_objects)

        # Show warning if any
        if warning_message:
            flash(f'Warning: {warning_message}')

        # === ADD CLASSIFICATION AS 6TH COLUMN (if enabled) ===
        if enable_classification and classification_lookup:
            from src.core.constants import FIVE_COLUMN_HEADERS

            # Add Document Type column by mapping Document Reference to classification
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
                'timestamp': datetime.now().isoformat(),
                'provider': provider,
                'model': model,
                'extractor': extractor,
                'file_count': len(valid_files),
                'events_found': len(legal_events_df)
            }

            # Store results in session with enhanced metadata
            enhanced_metadata = processing_metadata.copy()
            enhanced_metadata.update({
                'enable_classification': enable_classification,
                'classification_model': 'openrouter' if enable_classification else None,
                'document_types_found': len(set(classification_lookup.values())) if classification_lookup else 0,
                'filename': getattr(valid_files[0], 'name', 'unknown') if valid_files else 'unknown',
                'file_size': sum(getattr(f, 'getbuffer', lambda: b'').__call__().__len__() for f in file_objects) if file_objects else 0,
                'processing_time': None
            })

            # Store results in session
            session['results'] = legal_events_df.to_json(orient='records')
            session['results_columns'] = list(legal_events_df.columns)
            session['processing_metadata'] = enhanced_metadata
            session['current_run_id'] = run_id

            logger.info(f"✅ Processing complete - Run ID: {run_id}, Events: {len(legal_events_df)}")
            logger.info(f"   Results stored in session for run ID: {run_id}")

            # Auto-redirect to results page with run ID in URL
            logger.info("🔄 Redirecting to results page...")
            return redirect(url_for('results', run_id=run_id))
        else:
            flash('Processing failed')
            return redirect(url_for('index'))

    except Exception as e:
        logger.error(f'Processing error: {e}')
        flash(f'Processing error: {str(e)}')
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
        flash('Invalid session - please upload files again')
        return redirect(url_for('index'))

    # Store run_id in session for future requests (downloads, etc.)
    session['current_run_id'] = run_id

    # Retrieve results from session
    results_json = session.get('results')
    if not results_json:
        logger.warning("❌ No results in session")
        flash('No results available - please upload files first')
        return redirect(url_for('index'))

    # Deserialize results
    results_columns = session.get('results_columns', [])
    metadata = session.get('processing_metadata', {})

    try:
        results_df = pd.read_json(StringIO(results_json), orient='records')
        results = results_df.to_dict('records')
        columns = results_columns or list(results_df.columns)
        logger.info(f"✅ DataFrame deserialized - {len(results)} rows")
    except Exception as e:
        logger.error(f"❌ Failed to deserialize results: {e}")
        results = []
        columns = []

    logger.info(f"✅ Results retrieved from session - Run ID: {run_id}, Events: {len(results)}")

    return render_template('results.html',
                         results=results,
                         columns=columns,
                         count=len(results),
                         run_id=run_id,
                         metadata=metadata)

@app.route('/api/cost-estimate', methods=['POST'])
def cost_estimate():
    """API endpoint for cost estimation"""
    data = request.json
    provider = data.get('provider')
    model = data.get('model')
    file_sizes = data.get('file_sizes', [])

    # Use existing cost estimation logic
    from src.ui.streamlit_common import display_cost_estimates

    # Mock cost calculation (would need to adapt display_cost_estimates)
    estimated_cost = len(file_sizes) * 0.01  # Placeholder

    return jsonify({
        'estimated_cost': estimated_cost,
        'currency': 'USD',
        'provider': provider,
        'model': model
    })

@app.route('/api/providers')
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
def download_results(run_id):
    """Download results for a specific run ID"""
    # Check if this run ID matches the current session
    current_run_id = session.get('current_run_id')
    if current_run_id != run_id:
        flash('Invalid or expired run ID')
        return redirect(url_for('index'))

    # Get results from session
    results_json = session.get('results')
    if not results_json:
        flash('No results available for download')
        return redirect(url_for('index'))

    try:
        results_df = pd.read_json(StringIO(results_json), orient='records')

        # Create CSV response
        csv_buffer = StringIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        # Generate filename with run ID
        filename = f"legal_events_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        from flask import Response
        return Response(
            csv_data,
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename={filename}',
                'Content-Length': len(csv_data)
            }
        )

    except Exception as e:
        logger.error(f'Download error: {e}')
        flash('Error generating download file')
        return redirect(url_for('results'))

@app.route('/test')
def test():
    """Simple test endpoint to verify the app works"""
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
        'api_keys_available': api_keys_status,
        'session_info': {
            'has_results': 'results' in session,
            'has_run_id': 'current_run_id' in session,
            'run_id': session.get('current_run_id', 'None'),
            'session_keys': list(session.keys())
        },
        'endpoints': [
            'GET / - Main page with status indicators',
            'POST /upload - File processing with unique run IDs',
            'GET /results - View results for current session only',
            'GET /download/<run_id> - Download CSV with run ID validation',
            'GET /debug - Session debugging info',
            'POST /clear - Clear session data',
            'POST /api/cost-estimate - Cost estimation',
            'GET /api/providers - Provider status',
            'GET /api/models/<provider> - Models for provider'
        ],
        'features': [
            '🔄 Process handover - each upload gets unique run ID',
            '🧹 Fresh results - no mixing with previous runs',
            '📥 Automatic download - CSV with run ID in filename',
            '🔒 Session validation - results tied to current upload',
            '📊 Processing indicators - real-time status updates',
            '🏷️ Document classification - optional AI categorization',
            '🔍 Debug endpoints for troubleshooting',
            '🔄 Dynamic model loading - automatically picks up new backend models'
        ],
        'model_catalog_size': len(get_ui_model_config_list())
    })

@app.route('/debug')
def debug():
    """Debug endpoint to check session and app state"""
    return jsonify({
        'session_keys': list(session.keys()),
        'session_data': {
            'current_run_id': session.get('current_run_id', 'Not set'),
        },
        'database_info': {
            'results_store_available': True,
            'current_run_id': session.get('current_run_id', 'None'),
        },
        'app_config': {
            'model_catalog_size': len(get_ui_model_config_list()),
            'debug_mode': app.debug,
            'secret_key_set': bool(app.secret_key)
        }
    })

@app.route('/clear', methods=['POST'])
def clear_session():
    """Clear all session data for debugging"""
    session.clear()
    return jsonify({'status': 'cleared', 'message': 'Session data cleared'})

@app.route('/refresh-models', methods=['POST'])
def refresh_models():
    """Force refresh model catalog (useful for development)"""
    fresh_catalog = get_ui_model_config_list()
    return jsonify({
        'status': 'refreshed',
        'models_loaded': len(fresh_catalog),
        'providers': list(set(m.provider for m in fresh_catalog))
    })

@app.route('/test-session', methods=['GET', 'POST'])
def test_session():
    """Test session persistence across requests"""
    if request.method == 'POST':
        # Store test data
        test_data = request.form.get('test_data', 'test_value')
        session['test_data'] = test_data
        session['test_timestamp'] = datetime.now().isoformat()
        return jsonify({
            'status': 'stored',
            'data': test_data,
            'session_keys': list(session.keys())
        })
    else:
        # Retrieve test data
        return jsonify({
            'status': 'retrieved',
            'data': session.get('test_data', 'not_found'),
            'timestamp': session.get('test_timestamp', 'not_found'),
            'session_keys': list(session.keys())
        })

if __name__ == '__main__':
    print("🚀 Starting Flask Legal Events Extraction UI")
    print("🏠 Main page: http://localhost:5001/")
    print("✅ Templates loaded and ready!")

    app.run(debug=True, host='0.0.0.0', port=5001)
