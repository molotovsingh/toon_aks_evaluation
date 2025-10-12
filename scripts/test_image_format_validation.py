#!/usr/bin/env python3
"""
Quick validation: Test InputFormat.IMAGE configuration
Determines which pipeline options work with IMAGE format
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from docling.document_converter import DocumentConverter, FormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    ConvertPipelineOptions,
    AcceleratorOptions,
    TableStructureOptions,
    TableFormerMode
)
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.pipeline.simple_pipeline import SimplePipeline

print("="*80)
print("IMAGE FORMAT VALIDATION TEST")
print("="*80)

# Try Approach A: PdfPipelineOptions with StandardPdfPipeline
print("\n🧪 Test 1: IMAGE with PdfPipelineOptions + StandardPdfPipeline")
try:
    accelerator_options = AcceleratorOptions(device='cpu', num_threads=4)
    table_options = TableStructureOptions(
        mode=TableFormerMode.FAST,
        do_cell_matching=True
    )

    pdf_pipeline_options = PdfPipelineOptions(
        accelerator_options=accelerator_options,
        do_ocr=True,
        do_table_structure=True,
        table_structure_options=table_options,
        generate_page_images=False,
        images_scale=1.0,
        generate_picture_images=False,
        generate_table_images=False,
        generate_parsed_pages=False,
    )

    format_options = {
        InputFormat.IMAGE: FormatOption(
            pipeline_options=pdf_pipeline_options,
            backend=DoclingParseV4DocumentBackend,
            pipeline_cls=StandardPdfPipeline
        )
    }

    converter = DocumentConverter(format_options=format_options)
    print("✅ SUCCESS: IMAGE configured with PdfPipelineOptions + StandardPdfPipeline")
    test1_success = True
except Exception as e:
    print(f"❌ FAILED: {e}")
    test1_success = False

# Try Approach B: ConvertPipelineOptions with SimplePipeline
print("\n🧪 Test 2: IMAGE with ConvertPipelineOptions + SimplePipeline")
try:
    convert_pipeline_options = ConvertPipelineOptions(
        accelerator_options=accelerator_options,
    )

    format_options = {
        InputFormat.IMAGE: FormatOption(
            pipeline_options=convert_pipeline_options,
            backend=DoclingParseV4DocumentBackend,
            pipeline_cls=SimplePipeline
        )
    }

    converter = DocumentConverter(format_options=format_options)
    print("✅ SUCCESS: IMAGE configured with ConvertPipelineOptions + SimplePipeline")
    test2_success = True
except Exception as e:
    print(f"❌ FAILED: {e}")
    test2_success = False

# Summary
print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)
if test1_success:
    print("✅ RECOMMENDED: Use PdfPipelineOptions + StandardPdfPipeline")
    print("   → Same configuration as PDFs (OCR settings, table extraction)")
    print("   → Best for consistency with existing pipeline")
elif test2_success:
    print("✅ ALTERNATIVE: Use ConvertPipelineOptions + SimplePipeline")
    print("   → Simpler configuration")
    print("   → May need separate OCR configuration")
else:
    print("❌ BOTH APPROACHES FAILED")
    print("   → Need to investigate Docling IMAGE format requirements")

print()
