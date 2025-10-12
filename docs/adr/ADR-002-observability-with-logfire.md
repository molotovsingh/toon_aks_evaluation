# ADR-002: Observability with Pydantic Logfire

- **Status:** Deferred
- **Date:** 2025-10-11
- **Decision Date:** 2025-10-11
- **Review Date:** TBD (when moving to production)
- **Owner:** Development team
- **Related Docs:** `tests/run_all_tests.py`, `scripts/test_langextract_v1_v2.py`

## Context

During Phase 1 testing of the legal events extraction pipeline, we evaluated **Pydantic Logfire** as an observability solution for testing and monitoring. Logfire is an OpenTelemetry-based observability platform built by the Pydantic team, offering:

- **Pytest integration** via `capfire` fixture for capturing logs, spans, and metrics
- **Zero-config test mode** (automatic `send_to_logfire=False` during tests)
- **Snapshot testing** with inline-snapshot integration
- **Performance tracking** across pipeline stages
- **Cost monitoring** for API usage and token consumption

### Current Testing Approach

The project currently uses:
1. **Acceptance tests** (`tests/test_acceptance_criteria.py`) - Core functionality validation
2. **Performance tests** (`tests/test_performance_integration.py`) - Benchmark tracking
3. **Comparison scripts** (`scripts/test_langextract_v1_v2.py`) - Provider/prompt A/B testing with JSON+CSV exports
4. **Manual testing** via Streamlit UI with 6 provider options

### Project Phase

- **Current**: POC testing environment ("documents in → legal events out")
- **Scope**: Evaluating combinations of Docling + pluggable event extractors
- **Providers**: 6 extractors integrated (LangExtract, OpenRouter, OpenAI, Anthropic, DeepSeek, OpenCode Zen)
- **Goal**: Determine which parser+extractor combination works best for legal document processing

### Research Findings

**Logfire Capabilities:**
```python
# Example pytest integration
def test_extraction(capfire: CaptureLogfire):
    with logfire.span('langextract_extraction'):
        events = extractor.extract_events(text, metadata)

    # Assertions on captured spans
    spans = capfire.exporter.exported_spans
    assert len(spans) > 0
    assert spans[0].attributes['event_count'] == 21
```

**Potential Use Cases:**
- Track extraction timing across providers (OpenRouter vs LangExtract)
- Monitor prompt A/B test performance (V1 vs V2)
- Log API token usage and costs per provider
- Debug multi-stage pipeline failures (Docling → extraction)
- Generate performance dashboards

## Decision

**Defer Logfire integration** until the project moves from POC to production deployment.

### Rationale

**1. Current Approach is Sufficient**
- Comparison scripts like `test_langextract_v1_v2.py` already capture critical metrics:
  ```python
  results = {
      "v1": {"events": v1_events, "extraction_time": v1_time},
      "v2": {"events": v2_events, "extraction_time": v2_time},
      "comparison": {"event_count_diff": 7, "time_diff": 6.6}
  }
  ```
- JSON + CSV exports provide sufficient analysis capability
- Simple `time.perf_counter()` measurements meet current needs

**2. POC Phase Priorities**
- Focus: "Prove value fast" (project mantra)
- Goal: Test provider combinations, not build monitoring infrastructure
- Keep it boring: "Favor clear, well-understood patterns"

**3. Overhead vs. Benefit**
- **Overhead**: Additional dependency, configuration, learning curve
- **Benefit**: Minimal during POC (no production traffic, no multi-team usage)
- **Trade-off**: Not favorable at this stage

**4. No Clear Observability Gaps**
- No evidence of debugging difficulties requiring Logfire
- Current test suite provides adequate visibility
- Manual Streamlit testing catches integration issues

## Alternatives Considered

### 1. Integrate Logfire Now
*Rejected* for reasons above. Would add complexity without clear ROI during POC phase.

### 2. Enhanced Standard Logging
*Alternative approach* if more visibility needed:
```python
import logging
logger = logging.getLogger(__name__)

def extract_events(self, text, metadata):
    start = time.perf_counter()
    logger.info(f"Starting: provider={self.provider}, doc={metadata['document_name']}")

    events = self._extract(text)
    elapsed = time.perf_counter() - start

    logger.info(f"Completed: events={len(events)}, time={elapsed:.2f}s")
    return events
```

Benefits:
- ✅ No new dependencies
- ✅ Built into Python stdlib
- ✅ Familiar to all developers
- ✅ Easy to add structured fields

### 3. Custom Metrics Collection
*Alternative approach* for provider comparison:
```python
@dataclass
class ExtractionMetrics:
    provider: str
    event_count: int
    extraction_time: float
    token_count: int
    cost: float

    def save_to_csv(self, path: Path): ...
```

Benefits:
- ✅ Tailored to exact needs
- ✅ Simple, no dependencies
- ✅ Easy CSV/JSON export
- ✅ Aligns with current comparison script approach

### 4. Continue Current Approach (Selected)
*Chosen* - Keep using comparison scripts with JSON/CSV exports:
- ✅ Already working well (see `test_langextract_v1_v2.py` success)
- ✅ No additional complexity
- ✅ Clear, maintainable output
- ✅ Sufficient for POC evaluation

## Consequences

### Positive
- **Reduced complexity**: No new dependencies or configuration overhead
- **Faster iteration**: Focus remains on testing extractors, not building monitoring
- **Clear path forward**: Decision can be revisited with clear trigger criteria
- **Current approach validated**: Comparison scripts proven effective

### Negative / Mitigations
- **Limited real-time visibility**: Mitigated by manual Streamlit testing and test suite
- **No centralized dashboard**: Mitigated by CSV exports and manual analysis (sufficient for POC)
- **Manual correlation**: Must manually compare JSON files across runs (acceptable for current scale)

### Deferred Benefits (Available When Revisited)
When moving to production, Logfire will provide:
- Real-time monitoring dashboard
- Automatic performance regression detection
- Cost tracking across teams/projects
- Distributed tracing for multi-stage pipelines
- Alerting on quality degradation

## When to Revisit

Reconsider Logfire integration when **ANY** of these conditions are met:

### Production Triggers
1. ✅ **Production deployment**: Moving from POC to production service
2. ✅ **Multi-team usage**: More than one team using the system
3. ✅ **SLA requirements**: Need to monitor uptime/latency guarantees
4. ✅ **Cost tracking critical**: Budget monitoring becomes essential (API costs)
5. ✅ **Long-term quality tracking**: Need to track extraction quality over weeks/months

### Scale Triggers
6. ✅ **High volume**: Processing >100 documents/day
7. ✅ **Multiple environments**: Need to compare dev/staging/prod performance
8. ✅ **Customer-facing**: External users depend on extraction quality

### Debugging Triggers
9. ✅ **Complex failures**: Multi-stage pipeline failures difficult to debug
10. ✅ **Performance issues**: Need to identify bottlenecks in extraction pipeline
11. ✅ **Provider comparison**: Need automated A/B testing across providers

### Team Triggers
12. ✅ **Observability culture**: Team adopts OpenTelemetry/observability practices
13. ✅ **Shared dashboards**: Need to communicate metrics to stakeholders

## Implementation Guidance (When Revisited)

When the decision is revisited, follow this implementation plan:

### Phase 1: Basic Integration (1-2 days)
```python
# 1. Install Logfire
pip install logfire

# 2. Configure for tests
# conftest.py
import logfire
logfire.configure(send_to_logfire=False)  # Test mode

# 3. Add capfire to key tests
def test_provider_comparison(capfire: CaptureLogfire):
    with logfire.span('openrouter_extraction'):
        openrouter_events = extract_with_openrouter(text)

    with logfire.span('langextract_extraction'):
        langextract_events = extract_with_langextract(text)

    # Assert on timing, event counts
    spans = capfire.exporter.exported_spans
    assert len(spans) == 2
```

### Phase 2: Production Configuration (2-3 days)
```python
# 4. Production setup
logfire.configure(
    send_to_logfire=True,
    service_name='legal-events-extractor',
    environment='production'
)

# 5. Add pipeline instrumentation
with logfire.span('document_extraction', document_name=doc.name):
    markdown = docling.extract(pdf)

with logfire.span('event_extraction', provider=config.provider):
    events = extractor.extract_events(markdown, metadata)
    logfire.info('Extracted {count} events', count=len(events))
```

### Phase 3: Metrics & Dashboards (3-5 days)
- Set up Logfire project
- Create dashboards for provider comparison
- Configure alerting on performance degradation
- Add cost tracking metrics

### Estimated Effort
- **Basic integration**: 1-2 days
- **Production deployment**: 3-5 days total
- **Full observability setup**: 1-2 weeks

## References

- **Logfire Documentation**: https://logfire.pydantic.dev/docs/
- **Testing Guide**: https://logfire.pydantic.dev/docs/reference/advanced/testing/
- **GitHub**: https://github.com/pydantic/logfire
- **Current Testing**: `tests/run_all_tests.py`, `scripts/test_langextract_v1_v2.py`

## Decision Log

| Date | Action | Rationale |
|------|--------|-----------|
| 2025-10-11 | Deferred | POC phase - current testing approach sufficient |
| TBD | Review | When moving to production or hitting scale triggers |

## Notes

- This decision was made after successful validation of comparison script approach (see `test_langextract_v1_v2.py` - proved V1 vs V2 prompt comparison works well with simple JSON/CSV exports)
- Decision aligns with project mantras: "Keep it boring", "Prove value fast", "Start small, scale smart"
- Logfire remains a strong candidate for production observability - this is a timing decision, not a rejection of the technology
