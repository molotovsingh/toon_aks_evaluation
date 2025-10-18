#!/usr/bin/env python3
"""
Utility to classify one or more legal documents using a small hosted model.

The script assembles a consistent prompt (see src/core/classification_prompt.py),
extracts representative text from each file (via Docling when necessary), and
optionally calls an API such as OpenRouter with the chosen model.

Usage examples:

Dry run with default demo files (no API call; prints prompt diagnostics):
    uv run python scripts/classify_documents.py

Execute real classification (requires OPENROUTER_API_KEY in environment):
    uv run python scripts/classify_documents.py --execute --model anthropic/claude-3-haiku
    uv run python scripts/classify_documents.py --execute --model google/gemini-flash-1.5 --max-chars 1800

You can target custom files with --files or --glob. Results are written to
output/classification/<filename>__<model>.json for auditing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

# Ensure repository root is available on sys.path for src imports.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.core.classification_prompt import build_classification_prompt  # noqa: E402
from src.core.config import load_config  # noqa: E402
from src.core.prompt_registry import (  # noqa: E402
    get_prompt_text,
    get_output_directory,
    get_prompt_version,
    get_prompt_for_v1_v2_flag,
    list_prompt_variants,
    print_all_variants,
    print_variant_info,
)
from src.ui.cost_estimator import estimate_cost, estimate_tokens  # noqa: E402

try:
    from src.core.document_processor import DocumentProcessor  # noqa: E402
except Exception as exc:  # pragma: no cover - defensive import guard
    raise RuntimeError("DocumentProcessor import failed. Ensure Docling dependencies are installed.") from exc


DEFAULT_DEMO_FILES: List[Path] = [
    Path("test_documents/abc_xyz_contract_dispute.txt"),
    Path("tests/test_documents/multiple_events_document.html"),
    Path("sample_pdf/famas_dispute/Answer to Request for Arbitration.pdf"),
]

DEFAULT_MODEL = "anthropic/claude-3-haiku"
DEFAULT_PROVIDER = "openrouter"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class ClassificationResult:
    document_path: Path
    model: str
    prompt_version: str
    response: dict
    raw_response: str
    api_latency_seconds: Optional[float] = None


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify legal documents with a hosted small model.")
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        help="Specific files to classify.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        help="Glob pattern to select files (evaluated relative to project root).",
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=["openrouter", "openai"],
        help=f"API provider to use (default: {DEFAULT_PROVIDER}). 'openrouter' uses OpenRouter API, 'openai' uses direct OpenAI API.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model identifier (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--base-url",
        default=OPENROUTER_BASE_URL,
        help="Chat completion endpoint (default points to OpenRouter).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the API call.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1600,
        help="Trim document excerpts to this many characters.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/classification"),
        help="Directory to store JSON responses.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Make actual API calls (requires relevant API key).",
    )
    parser.add_argument(
        "--prompt-version",
        default="v1",
        help="Semantic version tag for the prompt; stored alongside results.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout for API calls.",
    )
    parser.add_argument(
        "--multi-label",
        action="store_true",
        help="Use multi-label classification prompt (encourages multiple classes per document). Automatically routes output to classification_multilabel/ directory.",
    )
    parser.add_argument(
        "--prompt-variant",
        default=None,
        help=f"Multi-label prompt variant. Options: {', '.join(list_prompt_variants())}, v1, v2 (backward compat). Default: 'comprehensive' (your preference). Only applies when --multi-label is used.",
    )
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List all available prompt variants with descriptions and exit.",
    )
    parser.add_argument(
        "--show-prompt-info",
        type=str,
        metavar="VARIANT",
        help="Show detailed information about a specific prompt variant and exit.",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Show cost estimates for processing documents without making API calls.",
    )
    # Get available extractors from catalog (dynamically loaded)
    try:
        from src.core.document_extractor_catalog import get_doc_extractor_catalog
        catalog = get_doc_extractor_catalog()
        available_extractors = [e.extractor_id for e in catalog.list_extractors(enabled=True)]
    except Exception:
        # Fallback if catalog import fails
        available_extractors = ["docling", "qwen_vl"]

    parser.add_argument(
        "--doc-extractor",
        default="docling",
        choices=available_extractors if available_extractors else ["docling"],
        help=f"Document extractor to use for cost estimation (default: docling). Available: {', '.join(available_extractors)}. Affects Layer 1 costs in two-layer estimate.",
    )
    return parser.parse_args(argv)


def resolve_files(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    if args.files:
        paths.extend(args.files)
    if args.glob:
        paths.extend(Path(".").glob(args.glob))
    if not paths:
        paths.extend(DEFAULT_DEMO_FILES)
    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(resolved)
    return unique_paths


def extract_excerpt(processor: DocumentProcessor, path: Path, max_chars: int) -> str:
    extension = path.suffix.lower().lstrip(".")
    if extension in {"txt", "md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text.strip()[:max_chars]
    text, method = processor.extract_text(path, extension)
    if not text:
        raise ValueError(f"Unable to extract text from {path} using method={method}")
    return text[:max_chars].strip()


def build_multilabel_prompt(document_title: str, document_excerpt: str, variant: Optional[str] = None) -> tuple[str, str]:
    """
    Build multi-label classification prompt using prompt registry.

    Args:
        document_title: Title of the document (currently unused in prompt)
        document_excerpt: Text excerpt from document to classify
        variant: Prompt variant name ("comprehensive", "decisive", "v1", "v2", etc.)
                 If None, uses default from registry ("comprehensive")

    Returns:
        tuple: (system_message, user_message)
    """
    # Map old v1/v2 flags to new registry names for backward compatibility
    registry_variant = get_prompt_for_v1_v2_flag(variant)

    # Get prompt text from registry
    prompt_text = get_prompt_text(registry_variant)

    # Split the prompt into system and user parts
    # The prompt format expects the document excerpt at the end after "Document Excerpt:"
    system_message = prompt_text.split("Review the document below")[0].strip()
    user_template = "Review the document below and return valid JSON only.\n\nDocument Excerpt:\n{excerpt}"

    user_message = user_template.format(excerpt=document_excerpt.strip())

    return system_message, user_message


def parse_model_json(content: str) -> dict:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Drop leading ```json or ``` fence
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # Drop trailing ``` fence
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    if cleaned.lower().startswith("json:"):
        cleaned = cleaned[5:].strip()

    # Attempt direct parse; fallback to substring between first { and last }
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            return json.loads(snippet)
        raise


def call_model(
    system_message: str,
    user_message: str,
    *,
    model: str,
    base_url: str,
    temperature: float,
    timeout: float,
) -> tuple[dict, str, float]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Export it to make real API calls.")

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": [{"type": "text", "text": user_message}]},
        ],
    }

    response = requests.post(
        base_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    choice = data["choices"][0]["message"]["content"]
    parsed = parse_model_json(choice)
    latency = data.get("usage", {}).get("completion_time", 0.0)
    return parsed, choice, float(latency or 0.0)


def call_openai_direct(
    system_message: str,
    user_message: str,
    *,
    model: str,
    temperature: float,
    timeout: float,
) -> tuple[dict, str, float]:
    """
    Call OpenAI API directly using openai SDK

    This function provides direct OpenAI API access for ground truth model classification
    (e.g., GPT-5). Uses the same interface as call_model() for compatibility.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it to make real API calls with provider=openai.")

    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai library not installed. Run: uv pip install openai")

    client = OpenAI(api_key=api_key, timeout=timeout)

    # Check if model is GPT-5 (requires different API parameters)
    is_gpt5 = "gpt-5" in model.lower()
    uses_responses_api = is_gpt5  # GPT-5 uses Responses API for reasoning

    try:
        if uses_responses_api:
            # GPT-5: Use Responses API for advanced reasoning
            input_text = f"{system_message}\n\n{user_message}"
            response = client.responses.create(
                model=model,
                input=input_text,
                reasoning={"effort": "medium"},
                text={"verbosity": "medium"},
            )
            content = response.output_text
            latency = 0.0  # Responses API doesn't provide latency
        else:
            # GPT-4: Use Chat Completions API
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 1.0 if is_gpt5 else temperature,
            }

            # Add max tokens parameter
            if is_gpt5:
                kwargs["max_completion_tokens"] = 4096
            else:
                kwargs["max_tokens"] = 4096

            # Check if model supports JSON mode
            supports_json_mode = any(
                compatible in model.lower()
                for compatible in ["gpt-5", "gpt-4o", "gpt-4-turbo"]
            )
            if supports_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            latency = 0.0  # Chat Completions API doesn't provide latency

        # Parse the response
        parsed = parse_model_json(content)
        return parsed, content, latency

    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}") from e


def print_cost_estimates(files: List[Path], excerpts: List[str], model: str, doc_extractor: str = "docling") -> None:
    """
    Print two-layer cost estimates for processing documents.

    Args:
        files: List of file paths being processed
        excerpts: List of text excerpts extracted from files (can be None for pre-extraction estimate)
        model: Model identifier to estimate costs for
        doc_extractor: Document extractor ID ('docling', 'qwen_vl', 'gemini')
    """
    from src.ui.cost_estimator import estimate_cost_two_layer  # noqa: E402

    print("\n" + "=" * 80)
    print("TWO-LAYER COST ESTIMATES (Document + Event Extraction)")
    print("=" * 80)
    print(f"\nEstimation method:")
    print(f"  Layer 1 (Doc Extraction): Page count metadata (no paid extraction run)")
    print(f"  Layer 2 (Event Extraction): 4 chars = 1 token (±20% accuracy)")
    print(f"  Token split assumption: 90% input, 10% output\n")

    # Get two-layer cost estimate
    two_layer_result = estimate_cost_two_layer(
        uploaded_files=files,
        doc_extractor=doc_extractor,
        event_model=model,
        extracted_texts=excerpts  # Can be None - will use page count heuristic
    )

    print(f"Total documents: {len(files)}")
    print(f"Total pages: {two_layer_result['page_count']} (confidence: {two_layer_result['page_confidence']})")
    print(f"Document extractor: {doc_extractor}")
    print(f"Event extraction model: {model}")

    # Print results in table format
    print("\n" + "-" * 80)
    print(f"{'Layer / Metric':<40} {'Value':>20}")
    print("-" * 80)

    # Layer 1: Document Extraction
    print(f"{'LAYER 1: Document Extraction':<40} {''}")
    print(f"{'  Pages processed':<40} {two_layer_result['page_count']:>20,}")
    print(f"{'  Extractor':<40} {two_layer_result['document_extractor']:>20}")
    print(f"{'  Cost':<40} {two_layer_result['document_cost_display']:>20}")

    # Layer 2: Event Extraction
    print(f"{'LAYER 2: Event Extraction':<40} {''}")
    print(f"{'  Total tokens':<40} {two_layer_result['tokens_total']:>20,}")
    print(f"{'  Input tokens (90%)':<40} {two_layer_result['tokens_input']:>20,}")
    print(f"{'  Output tokens (10%)':<40} {two_layer_result['tokens_output']:>20,}")
    print(f"{'  Model':<40} {two_layer_result['event_model']:>20}")
    print(f"{'  Cost':<40} {two_layer_result['event_cost_display']:>20}")

    # Total
    print("-" * 80)
    if two_layer_result['pricing_available']:
        print(f"{'TOTAL ESTIMATED COST':<40} {two_layer_result['total_cost_display']:>20}")
    else:
        print(f"{'TOTAL ESTIMATED COST':<40} {'Pricing unavailable':>20}")
    print("-" * 80)

    if two_layer_result['pricing_available']:
        print(f"\nNote: {two_layer_result['note']}")
    else:
        print(f"\nNote: Pricing unavailable for one or more layers")

    print("=" * 80 + "\n")


def run_classification(args: argparse.Namespace) -> List[ClassificationResult]:
    load_dotenv()
    docling_config, _, _ = load_config()
    processor = DocumentProcessor(config=docling_config)
    files = resolve_files(args)
    if not files:
        raise RuntimeError("No files resolved for classification.")

    # Auto-route to appropriate directory when --multi-label is used
    if args.multi_label and args.output_dir == Path("output/classification"):
        # Use registry to get output directory
        registry_variant = get_prompt_for_v1_v2_flag(args.prompt_variant)
        args.output_dir = Path(get_output_directory(registry_variant))

    # Extract all excerpts first (needed for both estimation and processing)
    excerpts = []
    for path in files:
        excerpt = extract_excerpt(processor, path, args.max_chars)
        excerpts.append(excerpt)

    # If --estimate-only is set, show cost estimates and exit
    if args.estimate_only:
        print_cost_estimates(files, excerpts, args.model, args.doc_extractor)
        return []  # Return empty list since no classification was performed

    results: List[ClassificationResult] = []

    for idx, path in enumerate(files):
        excerpt = excerpts[idx]  # Use pre-extracted excerpt

        # Choose prompt based on --multi-label flag
        if args.multi_label:
            system_message, user_message = build_multilabel_prompt(path.name, excerpt, variant=args.prompt_variant)
            # Get version from registry
            registry_variant = get_prompt_for_v1_v2_flag(args.prompt_variant)
            prompt_version = get_prompt_version(registry_variant)
        else:
            prompt = build_classification_prompt(path.name, excerpt, prompt_version=args.prompt_version)
            system_message = prompt.system_message
            user_message = prompt.user_message
            prompt_version = prompt.prompt_version

        if args.execute:
            # Route to appropriate API based on provider
            if args.provider == "openai":
                response, raw_response, latency = call_openai_direct(
                    system_message,
                    user_message,
                    model=args.model,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
            else:  # openrouter
                response, raw_response, latency = call_model(
                    system_message,
                    user_message,
                    model=args.model,
                    base_url=args.base_url,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
        else:
            response = {
                "classes": ["DRY_RUN"],
                "primary": "DRY_RUN",
                "confidence": 0.0,
                "rationale": "Execution disabled; run with --execute to call the API.",
            }
            raw_response = json.dumps(response)
            latency = None

        results.append(
            ClassificationResult(
                document_path=path,
                model=args.model,
                prompt_version=prompt_version,
                response=response,
                raw_response=raw_response,
                api_latency_seconds=latency,
            )
        )
    return results


def save_results(results: Iterable[ClassificationResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        safe_name = result.document_path.stem.replace(" ", "_")
        output_path = output_dir / f"{safe_name}__{result.model.replace('/', '_')}.json"
        payload = {
            "document": str(result.document_path),
            "model": result.model,
            "prompt_version": result.prompt_version,
            "response": result.response,
            "raw_response": result.raw_response,
            "api_latency_seconds": result.api_latency_seconds,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    # Handle --list-prompts flag
    if args.list_prompts:
        print_all_variants()
        return 0

    # Handle --show-prompt-info flag
    if args.show_prompt_info:
        try:
            print_variant_info(args.show_prompt_info)
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return 0

    try:
        results = run_classification(args)
    except Exception as exc:  # pragma: no cover - CLI exception handling
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # If --estimate-only was used, results will be empty
    if args.estimate_only:
        print("Cost estimation completed. Use without --estimate-only to run classification.")
        return 0

    save_results(results, args.output_dir)

    for res in results:
        print(f"{res.document_path.name} → {res.response['primary']} (classes={res.response['classes']})")
    if not args.execute:
        print("Dry run completed. Use --execute with OPENROUTER_API_KEY set to make real API calls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
