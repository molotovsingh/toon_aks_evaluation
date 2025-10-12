# 📚 Documentation Index

This directory contains design documents, architecture decisions, and reference materials for the paralegal document processing project.

## 🎯 Purpose

The `docs/` directory serves as the central repository for:
- **Product Requirements** - Business logic and feature specifications
- **Architecture Decisions** - Technical design choices and rationale
- **Reference Materials** - Guides, troubleshooting, and operational documentation

## 📋 Current Documents

### Product Requirements Documents (PRDs)
- **[pluggable_extractors_prd.md](pluggable_extractors_prd.md)** - Product requirements and specifications for the modular extractor architecture

### Architecture Decision Records (ADRs)
- **[ADR-001: Pluggable Extractors](adr/ADR-001-pluggable-extractors.md)** - Decision record for implementing pluggable document and event extraction interfaces
- **[ADR-002: Observability with Logfire](adr/ADR-002-observability-with-logfire.md)** - Deferred decision on Pydantic Logfire integration for testing and monitoring (to revisit in production)

### Orders & Planning
- **[orders/](orders/)** - Development orders and structured task specifications
- **[orders/example-order-template.json](orders/example-order-template.json)** - Reference order showing required structure, context, and guardrails.
- **[orders/event-extractor-001.json](orders/event-extractor-001.json)** - Phase 1: registry bootstrap groundwork for multi-provider extractors.
- **[orders/event-extractor-002.json](orders/event-extractor-002.json)** - Phase 2: wire OpenRouter/OpenCode Zen adapters, docs, and verification.
- **[orders/event-extractor-003.json](orders/event-extractor-003.json)** - Phase 3: implement adapters, update docs, and capture test evidence.
- **[orders/api-connection-test.json](orders/api-connection-test.json)** - Connectivity checks for LangExtract, OpenRouter, and OpenCode Zen.

### Guides
- **[guides/provider_integration_guide.md](guides/provider_integration_guide.md)** - Step-by-step instructions for adding and validating event extractor providers.
- **[guides/opencode_zen_troubleshooting.md](guides/opencode_zen_troubleshooting.md)** - Focused checklist to resolve empty responses and auth issues.

### Reference
- **[reference/configuration.md](reference/configuration.md)** - Environment variables and defaults used across providers.

## 📁 Directory Structure

```
docs/
├── README.md                           # This index file
├── pluggable_extractors_prd.md         # Core PRD
├── adr/                                # Architecture Decision Records
│   ├── ADR-001-pluggable-extractors.md
│   └── ADR-002-observability-with-logfire.md
├── orders/                             # Development task orders
└── reports/                            # Test reports and analysis (planned)
```

## 📝 Planned Documentation

### 🔄 Upcoming Documents
- [ ] **Extractor Provider Matrix** - Comparison of available document and event extraction services
- [ ] **Performance Benchmarking Guide** - Standardized testing procedures and metrics
- [ ] **Troubleshooting Runbook** - Common issues and resolution procedures
- [x] **API Integration Guide** - Instructions for adding new extraction providers (see `docs/guides/provider_integration_guide.md`)
- [x] **Configuration Reference** - Complete environment variable and settings documentation (see `docs/reference/configuration.md`)

### 📊 Test Documentation
- [ ] **Test Report Archive** - Historical test results and performance tracking
- [ ] **Acceptance Criteria Validation** - Verification procedures for feature completion
- [ ] **Load Testing Procedures** - Performance validation under various conditions

### 🚀 Deployment & Operations
- [ ] **Environment Setup Guide** - Production deployment procedures
- [x] **Monitoring & Alerting** - Operational observability specifications (see ADR-002, deferred until production)
- [ ] **Security Guidelines** - API key management and security best practices

## 🔗 Quick Links

- **Main README**: [../README.md](../README.md)
- **Core Source Code**: [../src/](../src/)
- **Test Suites**: [../tests/](../tests/)
- **Demo Applications**: [../examples/](../examples/)

---

*This index is maintained as new documentation is added. Please update this file when creating new documents.*
