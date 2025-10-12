# Document Extraction Cache Testing Guide

## Overview
The document extraction cache prevents redundant Docling processing when switching between different LLM providers. This guide explains how to validate the caching implementation.

## What Was Implemented

### 1. Cache Architecture
- **Cache Key Format**: `{filename}:{file_size}:{doc_extractor_type}`
- **Cache Storage**: Streamlit session state (`st.session_state['doc_extraction_cache']`)
- **Cache Limit**: 10 files (LRU eviction)
- **Cache Scope**: Session-level (persists across provider changes, cleared on browser refresh)

### 2. Modified Files
- `src/core/legal_pipeline_refactored.py` - Core caching logic
- `src/ui/streamlit_common.py` - Cache stats and clear utilities
- `app.py` - Cache indicator UI

### 3. Expected Behavior
**Before caching**:
- Upload file + Process with OpenAI = 10s (4s Docling + 6s LLM)
- Switch to Anthropic + Process same file = 10s (4s Docling AGAIN + 6s LLM)
- **Total**: 20 seconds with duplicate Docling work

**After caching**:
- Upload file + Process with OpenAI = 10s (4s Docling + 6s LLM)
- Switch to Anthropic + Process same file = 6s (0s cached + 6s LLM)
- **Total**: 16 seconds, ~40% faster for provider comparison

## Manual Testing Instructions

### Test 1: Basic Cache Hit/Miss

1. **Start the app**:
   ```bash
   uv run python -m streamlit run app.py
   ```

2. **First extraction (cache miss)**:
   - Upload sample file: `sample_pdf/famas_dispute/Answer to Request for Arbitration.pdf`
   - Select provider: **LangExtract (Gemini)**
   - Click "Process Files"
   - **Expected logs**:
     ```
     💾 Cached: Answer to Request for Arbitration.pdf (3.45s extraction time)
     ```
   - Note the Docling extraction time (should be 2-10 seconds)

3. **Second extraction with different provider (cache hit)**:
   - Keep the same file uploaded
   - Change provider to: **OpenAI** or **Anthropic**
   - Click "Process Files" again
   - **Expected logs**:
     ```
     💾 Cache HIT: Answer to Request for Arbitration.pdf (skipping Docling extraction)
     ```
   - Check performance metrics - Docling time should be **0.00s**

4. **Verify in UI**:
   - Cache indicator should show: "💾 **Docling Cache**: 1 file"
   - Expand "ℹ️ Cache Info" to see cached filename

### Test 2: Document Extractor Type Separation

The cache key includes `doc_extractor_type` to keep Docling and Gemini extractions separate.

1. **Upload file with Docling extractor**:
   - Upload: `sample_pdf/famas_dispute/Answer to Request for Arbitration.pdf`
   - Document Processing: **🔧 Docling (Local Processing)**
   - Provider: **LangExtract**
   - Process

2. **Switch to Gemini extractor (cache miss expected)**:
   - Keep same file
   - Document Processing: **🌟 Gemini 2.5 (Cloud Vision)**
   - Provider: **LangExtract**
   - Process
   - **Expected**: Cache MISS (different extractor type creates new cache entry)

3. **Verify cache size**:
   - Cache should show: "💾 **Docling Cache**: 2 files"
   - Both extractions are cached separately

### Test 3: Cache Eviction (10 file limit)

1. **Upload 11 different files** (use files from `sample_pdf/amrapali_case/` + `sample_pdf/famas_dispute/`)
2. **Process each file** (can use same provider)
3. **Verify eviction**:
   - Cache should never exceed 10 files
   - Check logs for: `💾 Evicted cache entry: ...`
   - Oldest file should be removed first (LRU)

### Test 4: Cache Clear Button

1. **Build up cache** by processing 2-3 files
2. **Verify cache indicator** shows file count
3. **Click "Clear Cache"** button in ℹ️ Cache Info expander
4. **Expected**:
   - Cache indicator disappears
   - Logs show: `💾 Cleared N cache entries`
   - Next upload will be cache miss

### Test 5: Performance Timing Validation

1. **Upload file and process with Provider A**
2. **Check Performance Metrics section**:
   - "Avg Docling Time" should show actual extraction time (2-10s)
   - "Avg Extractor Time" should show LLM time
   - "Avg Total Time" = Docling + Extractor

3. **Switch to Provider B and process same file**
4. **Check Performance Metrics again**:
   - "Avg Docling Time" should be **0.00s** (cache hit)
   - "Avg Extractor Time" should show new LLM time
   - "Avg Total Time" should be lower than before

## Log Messages Reference

### Success Messages
```
💾 Initialized document extraction cache
💾 Cache HIT: filename.pdf (skipping Docling extraction)
💾 Cached: filename.pdf (3.45s extraction time)
💾 Evicted cache entry: old_filename.pdf:12345:docling
💾 Cleared 3 cache entries
```

### Debug Messages
```
Cache key: filename.pdf:1234567:docling
```

## Common Issues

### Issue: Cache not working (always cache miss)
**Symptoms**: Every extraction shows "💾 Cached" with extraction time, never shows "💾 Cache HIT"

**Causes**:
1. Browser refresh cleared session state
2. Different file size (modified file)
3. Different document extractor selected

**Solution**: Ensure same file and same document extractor type across provider changes

### Issue: Cache indicator not showing
**Symptoms**: No "💾 **Docling Cache**" in configuration section

**Cause**: Cache is empty (no files processed yet)

**Solution**: Process at least one file to populate cache

### Issue: Cache cleared unexpectedly
**Cause**: Browser refresh or session restart

**Solution**: This is expected - cache is session-scoped for testing purposes

## Validation Checklist

- [ ] Cache HIT logs appear when switching providers with same file
- [ ] Docling time shows 0.00s on cache hit
- [ ] Cache indicator shows correct file count
- [ ] Cache Info expander lists cached filenames
- [ ] Cache eviction works (max 10 files)
- [ ] Clear Cache button empties cache
- [ ] Performance metrics show correct timing
- [ ] Different document extractors create separate cache entries

## Performance Expectations

| Scenario | Expected Time | Cache Status |
|----------|--------------|--------------|
| First file upload | 2-10s Docling + 2-30s LLM | Miss |
| Same file, different provider | 0s Docling + 2-30s LLM | Hit |
| Different file | 2-10s Docling + 2-30s LLM | Miss |
| Same file, different extractor | 2-10s Docling + 2-30s LLM | Miss |

## Notes

- **Cache persistence**: Session-scoped only (cleared on browser refresh)
- **Cache size**: Limited to 10 files to prevent memory issues
- **Cache key**: Includes filename, file size, AND document extractor type
- **Graceful degradation**: If caching fails, falls back to fresh extraction
- **No cross-session caching**: Each browser tab has independent cache
