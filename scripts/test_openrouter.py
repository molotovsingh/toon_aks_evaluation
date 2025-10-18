#!/usr/bin/env python3
"""
OpenRouter Configuration Diagnostic Script
Validates API key, configuration, and connectivity with detailed error reporting
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from dotenv import load_dotenv
from src.core.config import OpenRouterConfig, env_str

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


class OpenRouterDiagnostic:
    """Comprehensive OpenRouter configuration validation"""

    def __init__(self, test_model: Optional[str] = None, verbose: bool = False):
        self.test_model = test_model
        self.verbose = verbose
        self.checks_passed = 0
        self.total_checks = 10
        self.config = None
        self.api_key_safe = None
        self.log_file = Path(__file__).parent / "openrouter_diagnostic.log"

        # Setup file logging
        file_handler = logging.FileHandler(self.log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    def log(self, message: str, level: str = "INFO"):
        """Log message to console and file"""
        if level == "INFO":
            logger.info(message)
        elif level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        elif level == "DEBUG" and self.verbose:
            logger.debug(message)

    def mask_api_key(self, api_key: str) -> str:
        """Safely mask API key for display"""
        if not api_key or len(api_key) < 16:
            return "***INVALID***"
        return f"{api_key[:8]}...{api_key[-8:]}"

    def print_header(self):
        """Print diagnostic header"""
        self.log("=" * 70)
        self.log("🔍 OpenRouter Configuration Diagnostic")
        self.log("=" * 70)
        self.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Log file: {self.log_file}")
        self.log("=" * 70)
        self.log("")

    def print_step(self, step: int, name: str):
        """Print step header"""
        self.log(f"\n[Step {step}/{self.total_checks}] {name}")
        self.log("-" * 50)

    def print_result(self, passed: bool, message: str):
        """Print check result"""
        icon = "✅" if passed else "❌"
        self.log(f"{icon} {message}")
        if passed:
            self.checks_passed += 1

    def check_environment_file(self) -> bool:
        """Step 1: Check if .env file exists"""
        self.print_step(1, "Environment File Check")

        project_root = Path(__file__).parent.parent
        env_file = project_root / ".env"

        if not env_file.exists():
            self.print_result(False, f".env file not found at {env_file}")
            self.log("💡 Create .env from .env.example: cp .env.example .env")
            return False

        self.print_result(True, f".env file found at {env_file}")

        # Load environment variables
        load_dotenv(env_file)
        self.log(f"   Loaded environment variables from {env_file}")
        return True

    def check_api_key_format(self) -> bool:
        """Step 2: Validate API key format"""
        self.print_step(2, "API Key Format Validation")

        api_key = os.getenv("OPENROUTER_API_KEY", "")

        if not api_key:
            self.print_result(False, "OPENROUTER_API_KEY not set in .env")
            self.log("💡 Get API key from: https://openrouter.ai/keys")
            self.log("💡 Add to .env: OPENROUTER_API_KEY=sk-or-v1-...")
            return False

        # Check for whitespace
        if api_key != api_key.strip():
            self.print_result(False, "API key contains leading/trailing whitespace")
            return False

        self.api_key_safe = self.mask_api_key(api_key)
        self.print_result(True, f"API key found: {self.api_key_safe}")

        # Check expected format (OpenRouter keys typically start with sk-or-v1-)
        if api_key.startswith("sk-or-v1-"):
            self.log(f"   Key format: Valid OpenRouter format (sk-or-v1-)")
        elif api_key.startswith("sk-"):
            self.log(f"   ⚠️  Key format: Starts with 'sk-' but not standard OpenRouter prefix")
        else:
            self.log(f"   ⚠️  Key format: Unexpected format (expected sk-or-v1-)")

        # Check length
        if len(api_key) < 20:
            self.log(f"   ⚠️  Key length seems short: {len(api_key)} characters")
        else:
            self.log(f"   Key length: {len(api_key)} characters")

        return True

    def check_configuration_loading(self) -> bool:
        """Step 3: Load and validate configuration"""
        self.print_step(3, "Configuration Loading")

        try:
            self.config = OpenRouterConfig()
            self.print_result(True, "Configuration loaded successfully")

            # Display configuration
            self.log(f"   Base URL: {self.config.base_url}")
            self.log(f"   Model: {self.config.model}")
            self.log(f"   Timeout: {self.config.timeout}s")
            self.log(f"   API Key: {self.mask_api_key(self.config.api_key)}")

            # Override model if test model specified
            if self.test_model:
                self.config.model = self.test_model
                self.log(f"   ⚠️  Test model override: {self.test_model}")

            return True

        except Exception as e:
            self.print_result(False, f"Configuration loading failed: {e}")
            return False

    def check_network_connectivity(self) -> bool:
        """Step 4: Test network connectivity"""
        self.print_step(4, "Network Connectivity")

        if not REQUESTS_AVAILABLE:
            self.print_result(False, "requests library not available")
            self.log("💡 Install: pip install requests")
            return False

        try:
            response = requests.get("https://openrouter.ai", timeout=10)
            self.print_result(True, f"Connected to openrouter.ai (HTTP {response.status_code})")
            return True

        except requests.exceptions.ConnectionError:
            self.print_result(False, "Connection failed - check internet/firewall")
            return False
        except requests.exceptions.Timeout:
            self.print_result(False, "Connection timeout")
            return False
        except Exception as e:
            self.print_result(False, f"Network error: {e}")
            return False

    def check_api_authentication(self) -> Tuple[bool, Optional[Dict]]:
        """Step 5: Test API authentication"""
        self.print_step(5, "API Authentication Test")

        url = f"{self.config.base_url}/models"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                self.print_result(True, "API key authenticated successfully")
                data = response.json()
                models_count = len(data.get("data", []))
                self.log(f"   Available models: {models_count}")
                return True, data

            elif response.status_code == 401:
                self.print_result(False, "API key invalid (401 Unauthorized)")
                self.log("💡 Verify key at: https://openrouter.ai/keys")
                return False, None

            elif response.status_code == 403:
                self.print_result(False, "API key forbidden (403)")
                self.log("💡 Check if key has required permissions")
                return False, None

            else:
                self.print_result(False, f"Unexpected status: {response.status_code}")
                self.log(f"   Response: {response.text[:200]}")
                return False, None

        except Exception as e:
            self.print_result(False, f"Authentication test failed: {e}")
            return False, None

    def check_model_availability(self, models_data: Optional[Dict]) -> bool:
        """Step 6: Check if configured model is available"""
        self.print_step(6, "Model Availability Check")

        if not models_data:
            self.log("⚠️  Skipping (no models data from previous step)")
            return False

        models = models_data.get("data", [])
        model_ids = [m.get("id") for m in models if m.get("id")]

        if self.config.model in model_ids:
            self.print_result(True, f"Model '{self.config.model}' is available")

            # Find model details
            model_info = next((m for m in models if m.get("id") == self.config.model), None)
            if model_info:
                pricing = model_info.get("pricing", {})
                self.log(f"   Prompt: ${pricing.get('prompt', 'N/A')} per token")
                self.log(f"   Completion: ${pricing.get('completion', 'N/A')} per token")

            return True
        else:
            self.print_result(False, f"Model '{self.config.model}' not found")

            # Suggest free alternatives
            free_models = [m for m in model_ids if ":free" in m]
            if free_models:
                self.log("\n   💡 Available free models:")
                for model in free_models[:5]:
                    self.log(f"      - {model}")
                self.log(f"\n   Set in .env: OPENROUTER_MODEL={free_models[0]}")

            return False

    def check_minimal_chat_completion(self) -> bool:
        """Step 7: Test minimal chat completion"""
        self.print_step(7, "Minimal Chat Completion Test")

        url = f"{self.config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": "Say 'test successful' in 3 words."}
            ],
            "max_tokens": 20
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                self.print_result(True, "Chat completion successful")
                self.log(f"   Response: {content[:100]}")

                # Check usage
                usage = data.get("usage", {})
                if usage:
                    self.log(f"   Tokens used: {usage.get('total_tokens', 'N/A')}")

                return True
            else:
                self.print_result(False, f"Chat completion failed: HTTP {response.status_code}")
                self.log(f"   Error: {response.text[:200]}")
                return False

        except Exception as e:
            self.print_result(False, f"Chat completion error: {e}")
            return False

    def check_json_response_format(self) -> bool:
        """Step 8: Test JSON response format (required for legal extraction)"""
        self.print_step(8, "JSON Response Format Test")

        url = f"{self.config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": "Return JSON: {\"status\": \"ok\", \"message\": \"test\"}"}
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 50
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Try to parse as JSON
                try:
                    json_content = json.loads(content)
                    self.print_result(True, "JSON response format supported")
                    self.log(f"   Parsed JSON: {json_content}")
                    return True
                except json.JSONDecodeError:
                    self.print_result(False, "Response is not valid JSON")
                    self.log(f"   Response: {content[:100]}")
                    return False
            else:
                self.print_result(False, f"JSON format test failed: HTTP {response.status_code}")
                error_text = response.text[:200]
                self.log(f"   Error: {error_text}")

                # Check if model doesn't support JSON mode
                if "json" in error_text.lower() and "not supported" in error_text.lower():
                    self.log("   ⚠️  Model may not support response_format: json_object")
                    self.log("   💡 Try a different model that supports structured output")

                return False

        except Exception as e:
            self.print_result(False, f"JSON format test error: {e}")
            return False

    def check_rate_limits(self) -> bool:
        """Step 9: Check rate limit information"""
        self.print_step(9, "Rate Limit Check")

        url = f"{self.config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )

            # Check rate limit headers
            rate_limit_headers = {
                "x-ratelimit-limit-requests": response.headers.get("x-ratelimit-limit-requests"),
                "x-ratelimit-remaining-requests": response.headers.get("x-ratelimit-remaining-requests"),
                "x-ratelimit-reset-requests": response.headers.get("x-ratelimit-reset-requests"),
            }

            has_rate_info = any(v is not None for v in rate_limit_headers.values())

            if has_rate_info:
                self.print_result(True, "Rate limit information available")
                for header, value in rate_limit_headers.items():
                    if value:
                        self.log(f"   {header}: {value}")
            else:
                self.print_result(True, "No rate limit headers (may be unlimited)")
                self.log("   Note: Rate limits may still apply server-side")

            return True

        except Exception as e:
            self.print_result(False, f"Rate limit check error: {e}")
            return False

    def check_full_integration(self) -> bool:
        """Step 10: Test full legal events extraction integration"""
        self.print_step(10, "Full Integration Test")

        from src.core.constants import LEGAL_EVENTS_PROMPT

        url = f"{self.config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        test_text = "On January 15, 2024, the plaintiff filed a motion to dismiss."

        messages = [
            {
                "role": "system",
                "content": LEGAL_EVENTS_PROMPT + "\n\nReturn your response as valid JSON array containing the extracted events."
            },
            {
                "role": "user",
                "content": f"Extract legal events from this document:\n\n{test_text}"
            }
        ]

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "max_tokens": 500
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                try:
                    events_data = json.loads(content)

                    # Check if response has expected structure
                    if isinstance(events_data, dict):
                        if "events" in events_data or "extractions" in events_data:
                            self.print_result(True, "Full integration test passed")
                            self.log(f"   Extracted structure: {list(events_data.keys())}")
                            return True
                        else:
                            # Check if it's a single event object
                            if "event_particulars" in events_data:
                                self.print_result(True, "Full integration test passed (single event)")
                                return True
                    elif isinstance(events_data, list):
                        self.print_result(True, "Full integration test passed (array)")
                        self.log(f"   Extracted {len(events_data)} events")
                        return True

                    self.print_result(False, "Response structure unexpected")
                    self.log(f"   Response: {json.dumps(events_data, indent=2)[:200]}")
                    return False

                except json.JSONDecodeError as e:
                    self.print_result(False, f"Failed to parse response as JSON: {e}")
                    self.log(f"   Content: {content[:200]}")
                    return False
            else:
                self.print_result(False, f"Integration test failed: HTTP {response.status_code}")
                self.log(f"   Error: {response.text[:200]}")
                return False

        except Exception as e:
            self.print_result(False, f"Integration test error: {e}")
            return False

    def print_summary(self):
        """Print final summary"""
        self.log("\n")
        self.log("=" * 70)
        self.log("📊 DIAGNOSTIC SUMMARY")
        self.log("=" * 70)

        percentage = (self.checks_passed / self.total_checks) * 100
        self.log(f"Checks Passed: {self.checks_passed}/{self.total_checks} ({percentage:.1f}%)")

        if self.checks_passed == self.total_checks:
            self.log("\n✅ ALL CHECKS PASSED - OpenRouter is properly configured!")
            self.log("\n💡 Next steps:")
            self.log("   1. Start Streamlit app: uv run streamlit run app.py")
            self.log("   2. Select 'OpenRouter' provider in the UI")
            self.log("   3. Upload a document and process")
        elif self.checks_passed >= 7:
            self.log("\n⚠️  MOSTLY WORKING - Some issues detected")
            self.log("\n💡 Review failed checks above and address issues")
        elif self.checks_passed >= 4:
            self.log("\n❌ PARTIAL CONFIGURATION - Significant issues")
            self.log("\n💡 Key issues to fix:")
            if not self.config or not self.config.api_key:
                self.log("   - Set OPENROUTER_API_KEY in .env file")
            self.log("   - Review error messages above")
        else:
            self.log("\n❌ CONFIGURATION FAILED - Critical issues")
            self.log("\n💡 Start here:")
            self.log("   1. Create .env from .env.example")
            self.log("   2. Get API key: https://openrouter.ai/keys")
            self.log("   3. Add to .env: OPENROUTER_API_KEY=sk-or-v1-...")

        self.log(f"\n📄 Detailed log saved to: {self.log_file}")
        self.log("=" * 70)

    def run(self) -> int:
        """Run all diagnostic checks"""
        self.print_header()

        # Run checks in sequence
        step1 = self.check_environment_file()
        if not step1:
            self.print_summary()
            return 2  # Critical failure

        step2 = self.check_api_key_format()
        if not step2:
            self.print_summary()
            return 2  # Critical failure

        step3 = self.check_configuration_loading()
        if not step3:
            self.print_summary()
            return 2  # Critical failure

        if not REQUESTS_AVAILABLE:
            self.log("\n❌ requests library not available - cannot continue")
            self.print_summary()
            return 2

        step4 = self.check_network_connectivity()
        step5_result, models_data = self.check_api_authentication()
        step6 = self.check_model_availability(models_data)

        # Continue with remaining checks even if some fail
        step7 = self.check_minimal_chat_completion()
        step8 = self.check_json_response_format()
        step9 = self.check_rate_limits()
        step10 = self.check_full_integration()

        self.print_summary()

        # Return exit code
        if self.checks_passed == self.total_checks:
            return 0  # All passed
        elif self.checks_passed >= 7:
            return 0  # Mostly working
        else:
            return 1  # Some failures


def main():
    parser = argparse.ArgumentParser(description="OpenRouter Configuration Diagnostic")
    parser.add_argument("--test-model", help="Override model for testing (e.g., google/gemini-2.0-flash-exp:free)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--list-providers", action="store_true", help="List all available event extraction providers from catalog")

    args = parser.parse_args()

    # Handle --list-providers flag
    if args.list_providers:
        from src.core.event_extractor_catalog import get_event_extractor_catalog

        catalog = get_event_extractor_catalog()
        providers = catalog.list_extractors(enabled=True)

        print("=" * 70)
        print("AVAILABLE EVENT EXTRACTION PROVIDERS")
        print("=" * 70)
        print()

        for provider in providers:
            status = "✓ Recommended" if provider.recommended else ""
            runtime = " [Runtime Model Support]" if provider.supports_runtime_model else ""
            print(f"• {provider.display_name} ({provider.provider_id}){runtime}")
            print(f"  {provider.notes}")
            if status:
                print(f"  {status}")
            print()

        print(f"Total: {len(providers)} enabled provider(s)")
        print()
        print("💡 Run diagnostic for specific provider:")
        print("   uv run python scripts/test_openrouter.py")
        print("=" * 70)
        sys.exit(0)

    diagnostic = OpenRouterDiagnostic(
        test_model=args.test_model,
        verbose=args.verbose
    )

    exit_code = diagnostic.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
