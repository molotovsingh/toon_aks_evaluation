# 🔒 Security Guidelines

## API Key Management

### ⚠️ Critical Security Rules

1. **NEVER commit API keys to version control**
2. **NEVER share API keys in chat, email, or documentation**
3. **ALWAYS use environment variables for API keys**
4. **ROTATE API keys if accidentally exposed**

### API Key Exposure Response

If an API key is accidentally exposed:

1. **Immediately revoke the exposed key**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
   - Find the exposed key and delete it

2. **Generate a new API key**:
   - Create replacement key with same permissions
   - Update your local `.env` file

3. **Check git history**:
   ```bash
   git log --all --full-history -- .env
   git log --all --full-history -S "AIzaSy" --source --all
   ```

4. **If committed to git**:
   - Contact repository administrator
   - Consider repository cleanup/rewrite if necessary

### Secure Development Practices

#### Environment Setup
```bash
# Copy template
cp .env.example .env

# Edit with actual keys (NEVER commit this file)
nano .env

# Verify .env is ignored
git status  # Should not show .env as tracked
```

#### Testing with API Keys
- Use mock keys for unit tests: `"mock_key_for_testing"`
- Only use real keys for integration tests
- Never hardcode keys in test files

#### Production Deployment
- Use cloud secret managers (AWS Secrets Manager, Google Secret Manager)
- Set environment variables at runtime
- Never include `.env` files in containers

### Code Security Patterns

#### ✅ Secure (Good)
```python
# Load from environment
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY required")
```

#### ❌ Insecure (Bad)
```python
# NEVER do this
api_key = "AIzaSyBut0pBetvmrWLkw69MBLy1zsdo5VK4C48"

# Or this
with open('secrets.txt') as f:
    api_key = f.read().strip()
```

### Security Monitoring

#### Regular Audits
- Run security scans on codebase
- Check for accidentally committed secrets
- Monitor API key usage in cloud console

#### Automated Checks
```bash
# Search for potential API key patterns
grep -r "AIzaSy" . --exclude-dir=.git --exclude-dir=.venv
grep -r "api_key.*=" . --include="*.py" | grep -v "os.getenv"
```

### Incident Response

If you discover a security issue:

1. **Stop using the compromised credential immediately**
2. **Revoke/rotate the credential**
3. **Document the incident**
4. **Review how the exposure occurred**
5. **Implement preventive measures**

## Dependency Security

### Recent Changes

#### .EML Email Parsing (2025-10-10)
**Security Assessment**: ✅ NO NEW DEPENDENCIES

The email parsing functionality (`src/core/email_parser.py`) was implemented using **Python standard library only**:
- `email.parser.BytesParser` - RFC-compliant email parsing
- `email.policy.default` - Secure email handling policies
- `html.parser.HTMLParser` - Safe HTML tag stripping

**Security Benefits**:
- ✅ No third-party dependency vulnerabilities
- ✅ No supply chain risks
- ✅ Standard library maintained by Python core team
- ✅ No additional security audits required

**Implementation Files**:
- `src/core/email_parser.py` (259 lines, stdlib only)
- No changes to `pyproject.toml` or `requirements.txt`

#### Image Processing (JPEG/PNG) (2025-10-10)
**Security Assessment**: ✅ NO NEW DEPENDENCIES

Native image support implemented using **Docling's built-in image backend**:
- Uses existing Tesseract/EasyOCR/OCRmac (already configured for PDFs)
- Docling's `InputFormat.IMAGE` with `PdfPipelineOptions`
- No new image processing libraries (Pillow, img2pdf, etc.)
- Docling handles image parsing and OCR securely

**Security Benefits**:
- ✅ No Pillow dependency (avoids frequent CVEs in image libraries)
- ✅ Docling image backend maintained by IBM Research
- ✅ Same OCR engine as PDF processing (security surface unchanged)
- ✅ No temporary file creation (in-memory processing)
- ✅ No image conversion dependencies (img2pdf, etc.)

**Implementation Files**:
- `src/core/document_processor.py` (image format configuration, lines 171-224)
- `src/utils/file_handler.py` (added jpg/jpeg/png to supported types)
- `app.py` (updated file uploader)
- No changes to `pyproject.toml` or `requirements.txt`

### Dependency Audit Process

When adding new dependencies:

1. **Check for known vulnerabilities**:
   ```bash
   uv pip list --format=json | uv run safety check --stdin
   ```

2. **Review dependency tree**:
   ```bash
   uv pip tree
   ```

3. **Verify package authenticity**:
   - Check PyPI package signatures
   - Review package maintainer history
   - Scan for typosquatting risks

4. **Document security review**:
   - Update this section with dependency name and purpose
   - Note any security considerations
   - Link to security audit if conducted

### Security Resources

- [Google API Security Best Practices](https://cloud.google.com/docs/authentication/best-practices)
- [OWASP Secrets Management](https://owasp.org/www-community/vulnerabilities/Use_of_hard-coded_cryptographic_key)
- [Git Secrets Prevention](https://git-secret.io/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)

---

**Remember: Security is everyone's responsibility. When in doubt, err on the side of caution.**