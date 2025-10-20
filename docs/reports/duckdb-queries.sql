-- ===========================================================================
-- DuckDB Pipeline Metadata Queries
-- ===========================================================================
--
-- SQL query templates for analyzing pipeline runs stored in DuckDB.
-- Use these queries for custom analytics and reporting.
--
-- How to run these queries:
--   1. Via Python REPL:
--      python -c "import duckdb; conn = duckdb.connect('runs.duckdb'); print(conn.execute('''...query...''').df())"
--
--   2. Via query_duckdb.py script:
--      python scripts/query_duckdb.py --db runs.duckdb --query avg-time
--
--   3. Via DuckDB CLI (if installed):
--      duckdb runs.duckdb "SELECT * FROM pipeline_runs LIMIT 5;"
--
-- For more information, see: docs/reports/duckdb-ingestion-plan.md
-- Order: duckdb-ingestion-001
-- ===========================================================================


-- ===========================================================================
-- 1. DATABASE STATISTICS
-- ===========================================================================
-- Get overall database statistics (runs, providers, models, environments)

SELECT
    COUNT(*) as total_runs,
    COUNT(DISTINCT provider_name) as unique_providers,
    COUNT(DISTINCT provider_model) as unique_models,
    COUNT(DISTINCT environment) as unique_environments,
    MIN(timestamp) as first_run,
    MAX(timestamp) as last_run
FROM pipeline_runs;


-- ===========================================================================
-- 2. AVERAGE EXTRACTOR TIME BY MODEL
-- ===========================================================================
-- Performance comparison across different models
-- Useful for identifying fastest models for production use

SELECT
    provider_model,
    COUNT(*) as total_runs,
    ROUND(AVG(extractor_seconds), 3) as avg_extractor_time,
    ROUND(MIN(extractor_seconds), 3) as min_time,
    ROUND(MAX(extractor_seconds), 3) as max_time,
    ROUND(AVG(total_seconds), 3) as avg_total_time
FROM pipeline_runs
WHERE status = 'success'
    AND extractor_seconds IS NOT NULL
GROUP BY provider_model
ORDER BY avg_extractor_time ASC;


-- ===========================================================================
-- 3. TOTAL COST BY PROVIDER
-- ===========================================================================
-- Track spending across different providers (when cost_usd is available)
-- Useful for budget analysis and cost optimization

SELECT
    provider_name,
    COUNT(*) as total_runs,
    ROUND(SUM(cost_usd), 4) as total_cost_usd,
    ROUND(AVG(cost_usd), 4) as avg_cost_per_run,
    ROUND(MIN(cost_usd), 4) as min_cost,
    ROUND(MAX(cost_usd), 4) as max_cost
FROM pipeline_runs
WHERE cost_usd IS NOT NULL
GROUP BY provider_name
ORDER BY total_cost_usd DESC;


-- ===========================================================================
-- 4. SUCCESS RATE BY PROVIDER
-- ===========================================================================
-- Monitor reliability of each provider
-- Useful for identifying providers with high failure rates

SELECT
    provider_name,
    COUNT(*) as total_runs,
    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_runs,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_runs,
    ROUND(100.0 * SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) / COUNT(*), 2) as success_rate_pct
FROM pipeline_runs
GROUP BY provider_name
ORDER BY success_rate_pct DESC;


-- ===========================================================================
-- 5. RECENT RUNS (LAST 7 DAYS)
-- ===========================================================================
-- Show recent activity and outcomes
-- Useful for monitoring current work and debugging recent failures

SELECT
    run_id,
    timestamp,
    provider_model,
    events_extracted,
    ROUND(total_seconds, 2) as total_seconds,
    status,
    input_filename
FROM pipeline_runs
WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '7 days'
ORDER BY timestamp DESC;


-- ===========================================================================
-- 6. RECENT RUNS (LAST 24 HOURS)
-- ===========================================================================
-- Similar to above, but for last 24 hours only

SELECT
    run_id,
    timestamp,
    provider_model,
    events_extracted,
    ROUND(total_seconds, 2) as total_seconds,
    status,
    input_filename
FROM pipeline_runs
WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '1 day'
ORDER BY timestamp DESC;


-- ===========================================================================
-- 7. SLOWEST DOCUMENTS
-- ===========================================================================
-- Identify performance bottlenecks and large files
-- Useful for optimization and capacity planning

SELECT
    input_filename,
    provider_model,
    ROUND(total_seconds, 2) as total_seconds,
    ROUND(input_size_bytes / 1024.0 / 1024.0, 2) as size_mb,
    events_extracted,
    status
FROM pipeline_runs
WHERE total_seconds IS NOT NULL
ORDER BY total_seconds DESC
LIMIT 20;


-- ===========================================================================
-- 8. FASTEST DOCUMENTS (SUCCESSFUL RUNS ONLY)
-- ===========================================================================
-- Identify efficient runs and optimal configurations

SELECT
    input_filename,
    provider_model,
    ROUND(total_seconds, 2) as total_seconds,
    ROUND(input_size_bytes / 1024.0 / 1024.0, 2) as size_mb,
    events_extracted
FROM pipeline_runs
WHERE status = 'success'
    AND total_seconds IS NOT NULL
    AND total_seconds > 0
ORDER BY total_seconds ASC
LIMIT 20;


-- ===========================================================================
-- 9. AVERAGE EVENTS EXTRACTED BY PROVIDER
-- ===========================================================================
-- Compare extraction thoroughness across providers
-- Useful for quality assessment and provider selection

SELECT
    provider_name,
    provider_model,
    COUNT(*) as total_runs,
    ROUND(AVG(events_extracted), 1) as avg_events,
    MIN(events_extracted) as min_events,
    MAX(events_extracted) as max_events
FROM pipeline_runs
WHERE status = 'success'
    AND events_extracted IS NOT NULL
GROUP BY provider_name, provider_model
ORDER BY avg_events DESC;


-- ===========================================================================
-- 10. DOCUMENT SIZE DISTRIBUTION
-- ===========================================================================
-- Analyze input file sizes to understand workload characteristics

SELECT
    CASE
        WHEN input_size_bytes < 100000 THEN '< 100 KB'
        WHEN input_size_bytes < 1000000 THEN '100 KB - 1 MB'
        WHEN input_size_bytes < 10000000 THEN '1 MB - 10 MB'
        WHEN input_size_bytes < 50000000 THEN '10 MB - 50 MB'
        ELSE '> 50 MB'
    END as size_category,
    COUNT(*) as total_files,
    ROUND(AVG(total_seconds), 2) as avg_processing_time
FROM pipeline_runs
WHERE input_size_bytes IS NOT NULL
GROUP BY size_category
ORDER BY MIN(input_size_bytes);


-- ===========================================================================
-- 11. FAILED RUNS ANALYSIS
-- ===========================================================================
-- Investigate failed runs to identify patterns and issues

SELECT
    provider_name,
    provider_model,
    error_message,
    COUNT(*) as failure_count,
    MAX(timestamp) as last_failure
FROM pipeline_runs
WHERE status = 'failed'
GROUP BY provider_name, provider_model, error_message
ORDER BY failure_count DESC;


-- ===========================================================================
-- 12. OCR ENGINE COMPARISON
-- ===========================================================================
-- Compare performance across different OCR engines

SELECT
    ocr_engine,
    COUNT(*) as total_runs,
    ROUND(AVG(docling_seconds), 3) as avg_docling_time,
    ROUND(AVG(total_seconds), 3) as avg_total_time,
    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_runs
FROM pipeline_runs
WHERE ocr_engine IS NOT NULL
GROUP BY ocr_engine
ORDER BY avg_docling_time ASC;


-- ===========================================================================
-- 13. TABLE MODE COMPARISON
-- ===========================================================================
-- Compare FAST vs ACCURATE table extraction modes

SELECT
    table_mode,
    COUNT(*) as total_runs,
    ROUND(AVG(docling_seconds), 3) as avg_docling_time,
    ROUND(AVG(events_extracted), 1) as avg_events,
    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_runs
FROM pipeline_runs
WHERE table_mode IS NOT NULL
GROUP BY table_mode;


-- ===========================================================================
-- 14. ENVIRONMENT TRACKING
-- ===========================================================================
-- Track runs by execution environment (machine/host)

SELECT
    environment,
    COUNT(*) as total_runs,
    COUNT(DISTINCT provider_name) as providers_used,
    MIN(timestamp) as first_run,
    MAX(timestamp) as last_run
FROM pipeline_runs
WHERE environment IS NOT NULL
GROUP BY environment
ORDER BY total_runs DESC;


-- ===========================================================================
-- 15. SESSION ANALYSIS (BATCH PROCESSING)
-- ===========================================================================
-- Analyze batch processing via session_label
-- Useful for tracking experiments and benchmark runs

SELECT
    session_label,
    COUNT(*) as total_runs,
    ROUND(AVG(total_seconds), 2) as avg_time,
    ROUND(SUM(total_seconds), 2) as total_time,
    MIN(timestamp) as session_start,
    MAX(timestamp) as session_end
FROM pipeline_runs
WHERE session_label IS NOT NULL
GROUP BY session_label
ORDER BY session_start DESC;


-- ===========================================================================
-- 16. TIME SERIES: RUNS PER DAY
-- ===========================================================================
-- Track activity over time

SELECT
    DATE_TRUNC('day', timestamp) as run_date,
    COUNT(*) as total_runs,
    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_runs
FROM pipeline_runs
GROUP BY run_date
ORDER BY run_date DESC
LIMIT 30;


-- ===========================================================================
-- 17. CONFIG VALIDATION (JSON COLUMN QUERYING)
-- ===========================================================================
-- Demonstrate JSON column queries and validate config consistency
-- Useful for debugging configuration mismatches

SELECT
    run_id,
    provider_name,
    provider_model,
    config_snapshot->>'provider' as config_provider,
    config_snapshot->>'provider_model' as config_model,
    CASE
        WHEN provider_name != config_snapshot->>'provider' THEN 'MISMATCH'
        WHEN provider_model != config_snapshot->>'provider_model' THEN 'MISMATCH'
        ELSE 'OK'
    END as validation_status
FROM pipeline_runs
WHERE config_snapshot IS NOT NULL
LIMIT 10;


-- ===========================================================================
-- 18. QUALITY METRICS: CITATIONS
-- ===========================================================================
-- Analyze citation extraction quality across providers

SELECT
    provider_name,
    provider_model,
    COUNT(*) as total_runs,
    ROUND(AVG(citations_found), 1) as avg_citations,
    ROUND(100.0 * AVG(citations_found) / NULLIF(AVG(events_extracted), 0), 1) as citation_rate_pct
FROM pipeline_runs
WHERE status = 'success'
    AND citations_found IS NOT NULL
    AND events_extracted IS NOT NULL
GROUP BY provider_name, provider_model
ORDER BY citation_rate_pct DESC;


-- ===========================================================================
-- 19. DETAIL LENGTH ANALYSIS
-- ===========================================================================
-- Analyze event description verbosity across providers

SELECT
    provider_name,
    provider_model,
    COUNT(*) as total_runs,
    ROUND(AVG(avg_detail_length), 0) as avg_chars,
    MIN(avg_detail_length) as min_chars,
    MAX(avg_detail_length) as max_chars
FROM pipeline_runs
WHERE avg_detail_length IS NOT NULL
GROUP BY provider_name, provider_model
ORDER BY avg_chars DESC;


-- ===========================================================================
-- 20. TOKEN USAGE ANALYSIS (IF AVAILABLE)
-- ===========================================================================
-- Track token consumption for cost optimization

SELECT
    provider_model,
    COUNT(*) as total_runs,
    SUM(tokens_input) as total_input_tokens,
    SUM(tokens_output) as total_output_tokens,
    SUM(tokens_input + tokens_output) as total_tokens,
    ROUND(AVG(tokens_input), 0) as avg_input_tokens,
    ROUND(AVG(tokens_output), 0) as avg_output_tokens
FROM pipeline_runs
WHERE tokens_input IS NOT NULL
    AND tokens_output IS NOT NULL
GROUP BY provider_model
ORDER BY total_tokens DESC;


-- ===========================================================================
-- 21. COMPREHENSIVE RUN DETAILS
-- ===========================================================================
-- View all details for a specific run (replace run_id)

SELECT *
FROM pipeline_runs
WHERE run_id = 'DL2-OA2-TS1-F-20251017231723';


-- ===========================================================================
-- 22. LATEST 10 RUNS
-- ===========================================================================
-- Quick view of most recent activity

SELECT
    timestamp,
    run_id,
    provider_model,
    input_filename,
    events_extracted,
    ROUND(total_seconds, 2) as seconds,
    status
FROM pipeline_runs
ORDER BY timestamp DESC
LIMIT 10;


-- ===========================================================================
-- END OF QUERY TEMPLATES
-- ===========================================================================
--
-- For Python integration examples, see: scripts/query_duckdb.py
-- For schema documentation, see: docs/reports/duckdb-ingestion-plan.md
--
