# Development Orders - Active Work

**Purpose**: This directory contains active development orders for the docling_langextract_testing project.

**Quick Links**:
- 📋 [Order Template](example-order-template.json) - Blueprint for creating new orders
- 📦 [Archived Orders](../archive/orders/) - Completed and superseded orders
- 📊 [Archive Audit](../reports/archive_audit.md) - Archival status and evidence mapping
- 📘 [Archival Process](#archival-workflow) - How to archive completed orders

---

## Current Active Orders

| Order ID | Priority | Status | Purpose |
|----------|----------|--------|---------|
| **example-order-template.json** | Reference | Template | Blueprint for new development orders |
| **git-housekeeping-001-triage.json** | Medium | Active | Phase 1: Read-only repository triage (file categorization, .gitignore gaps) |
| **git-housekeeping-001-execute.json** | Medium | Active | Phase 2: Execute housekeeping (harden .gitignore, prune refs, cleanup) |
| **git-housekeeping-001.json** | Medium | SUPERSEDED | Original single-phase housekeeping (replaced by triage+execute) |
| **order-archival-001.json** | Medium | In Progress | Automate archival of completed orders |
| **orders-readme-001.json** | Low | Active | Create README guidance for orders directories |
| **event-extractor-001.json** | - | Needs Investigation | Unclear completion status |
| **doc-parsing-fastpath-001.json** | - | Needs Investigation | Unclear completion status |
| **eml-normalization-001.json** | - | Needs Investigation | Email format standardization |

---

## Creating New Orders

### 1. Use the Template
Copy `example-order-template.json` and customize:

```bash
cp docs/orders/example-order-template.json docs/orders/your-feature-001.json
```

### 2. Essential Sections
Every order must include:
- **order_id**: Unique identifier (e.g., `feature-name-001`)
- **supercontext**: Repository mission and context
- **goal**: Clear, measurable objective
- **tasks**: Step-by-step execution plan with task IDs
- **acceptance_criteria**: How success is measured
- **constraints**: What NOT to do and escalation guidance

### 3. Documentation Requirements
Specify in your order:
- Which files need updates (README.md, AGENTS.md, CLAUDE.md, etc.)
- Expected test commands (pytest, run_all_tests.py, manual validation)
- Completion evidence location (docs/reports/your-feature-completion.md)

### 4. Review Checklist
Before committing a new order:
- [ ] Read `CLAUDE.md` design mantras (start small, ship value, prefer simple patterns)
- [ ] Confirm acceptance criteria are measurable
- [ ] List all documentation touch points
- [ ] Specify test/validation approach
- [ ] Reference relevant guardrails (security, configuration, etc.)

---

## Git Housekeeping Workflow

The repository uses a **two-phase housekeeping process** for safe git hygiene maintenance:

### Phase 1: Triage (git-housekeeping-001-triage.json)

**Purpose**: Read-only analysis that can run on repositories with uncommitted work

**Safe to execute**: ✅ YES - Makes NO changes to repository state

**Actions**:
1. Snapshot git status, branches, untracked files
2. Categorize files into:
   - **COMMIT**: Feature work to preserve (scripts, docs, features)
   - **IGNORE**: Transient artifacts (logs, exports, test outputs)
   - **PRESERVE**: Intentionally tracked files (benchmarks, test fixtures)
   - **INVESTIGATE**: Files needing maintainer clarification
3. Security scan for secrets/API keys
4. Identify .gitignore gaps
5. Generate decision matrix with recommended git commands

**Output**: `docs/reports/git-housekeeping-triage-{timestamp}.md`

### Phase 2: Execute (git-housekeeping-001-execute.json)

**Purpose**: Full housekeeping execution on clean baseline

**Prerequisites**:
- ✅ Triage phase completed
- ✅ Feature work committed based on triage guidance
- ✅ Repository in clean state (no uncommitted changes)

**Actions**:
1. Harden .gitignore (add patterns, preserve exception rules)
2. Branch hygiene (prune merged branches, clean remote refs)
3. Artifact cleanup (remove transient files with dry-run verification)
4. Create evergreen housekeeping checklist
5. Validate with test suite

**Output**: `docs/reports/git-housekeeping-001-execution-{timestamp}.md`

### Execution Sequence

```bash
# Step 1: Run triage (can run NOW on dirty repo)
# Execute: git-housekeeping-001-triage.json
# Review: docs/reports/git-housekeeping-triage-{timestamp}.md

# Step 2: Commit feature work (manual, based on triage report)
git add <files from COMMIT category>
git commit -m "feat: <description>"
git push origin main

# Step 3: Run execute (requires clean baseline)
# Execute: git-housekeeping-001-execute.json
# Review: docs/reports/git-housekeeping-001-execution-{timestamp}.md
```

### Cadence

**Recommended**: Monthly (1st of each month)
- Prevents accumulation of untracked files
- Keeps .gitignore patterns current
- Maintains clean contributor experience

See `AGENTS.md` for integration with development workflow.

### Why Two Phases?

**Original Problem**: Single-phase housekeeping assumes clean baseline, but repositories often have uncommitted work in progress.

**Solution**:
- **Triage** provides guidance without requiring clean state
- **Manual commit** gives maintainer control over what to preserve
- **Execute** runs safely on clean baseline with full isolation

---

## Archival Workflow

### When to Archive

Archive an order when:
- ✅ **Completed**: All acceptance criteria met with evidence in `docs/reports/`
- 🔁 **Superseded**: Replaced by a revised version
- ❌ **Cancelled**: Explicitly abandoned with justification documented

### How to Archive

1. **Verify Completion Evidence**
   - Check for completion report in `docs/reports/`
   - Confirm code artifacts exist if no formal report
   - Document implicit completion if functionality is live

2. **Run Archive Audit** (Recommended)
   ```bash
   # Review current orders and evidence mapping
   cat docs/reports/archive_audit.md
   ```

3. **Manual Archival Process**
   ```bash
   # Copy order to archive (preserves Git history)
   cp docs/orders/your-feature-001.json docs/archive/orders/

   # Verify copy succeeded
   ls -l docs/archive/orders/your-feature-001.json

   # Remove from active orders
   rm docs/orders/your-feature-001.json

   # Update archive README
   # Add entry to docs/archive/orders/README.md under appropriate batch
   ```

4. **Update Archive Audit**
   - Add entry to `docs/reports/archive_audit.md`
   - Reference completion evidence
   - Note any special circumstances (superseded, implicit completion)

### What NOT to Archive

❌ **DO NOT ARCHIVE**:
- **Templates** (`example-order-template.json`)
- **Active Work** (orders currently in progress)
- **Orders Without Evidence** (investigate first, document in archive_audit.md)
- **Current Order** (`order-archival-001.json` until it completes)

---

## Order Lifecycle

```
[DRAFT] → [ACTIVE] → [COMPLETED] → [ARCHIVED]
                   ↓
              [SUPERSEDED] → [ARCHIVED]
                   ↓
              [CANCELLED] → [ARCHIVED with note]
```

### State Definitions

- **DRAFT**: Order under review, not yet committed
- **ACTIVE**: Order committed and ready for execution
- **IN PROGRESS**: Agent currently working on order
- **COMPLETED**: All acceptance criteria met, evidence documented
- **SUPERSEDED**: Replaced by revised version (e.g., v1.0 → v1.1)
- **CANCELLED**: Abandoned with justification
- **ARCHIVED**: Moved to `docs/archive/orders/` with evidence trail

---

## Best Practices

### For Order Authors

1. **Be Specific**: Vague tasks lead to incomplete work
2. **Provide Context**: Link to relevant docs, ADRs, PRDs
3. **Measurable Success**: Acceptance criteria should be testable
4. **Document Evidence**: Specify where completion proof will live

### For Order Executors (Agents)

1. **Read Fully**: Review entire order before starting
2. **Follow Sequence**: Execute tasks in order unless explicitly told to skip
3. **Document As You Go**: Create completion report in `docs/reports/`
4. **Escalate When Blocked**: Don't guess - ask for clarification
5. **Reference Evidence**: Link completion reports in final summary

### For Reviewers

1. **Check Evidence**: Verify completion claims with reports/artifacts
2. **Test Claims**: Run specified tests to confirm acceptance criteria
3. **Archive When Ready**: Move completed orders to archive promptly
4. **Monthly Audit**: Review active orders for stale/incomplete work

---

## Integration with Other Docs

| Document | Purpose | Relationship to Orders |
|----------|---------|------------------------|
| **README.md** | Quick start guide | Orders may require README updates |
| **AGENTS.md** | Agent guidance | Archival reminders, order protocols |
| **CLAUDE.md** | Technical mantras | Design principles referenced in orders |
| **docs/reports/** | Completion evidence | Where order outcomes are documented |
| **docs/archive/orders/** | Order history | Completed orders with evidence links |

---

## Troubleshooting

### "Order has no completion evidence"
- Check `docs/reports/` for related files
- Look for code artifacts matching order goals
- Review Git history for merge commits
- If truly incomplete, add to `archive_audit.md` "Needs Investigation" section

### "Order seems complete but no formal report"
- Document implicit completion with code artifact references
- Create lightweight completion note in `docs/reports/`
- Mark as "implicitly completed" in archive audit

### "Multiple orders conflict or duplicate"
- Check archive for superseded versions
- Review order IDs for version numbers (001, 002, -revised)
- Consult `archive_audit.md` for historical context
- Create new order that consolidates requirements

---

## Quick Reference

```bash
# View active orders
ls -1 docs/orders/*.json

# Check archival status
cat docs/reports/archive_audit.md

# View archived orders
ls -1 docs/archive/orders/*.json

# Read archive history
cat docs/archive/orders/README.md

# Create new order from template
cp docs/orders/example-order-template.json docs/orders/new-feature-001.json

# Archive completed order (structured move)
cp docs/orders/completed-order.json docs/archive/orders/ && \
rm docs/orders/completed-order.json
```

---

## Related Resources

- **Order Archival Automation**: See `docs/orders/order-archival-001.json`
- **Template Reference**: See `docs/orders/example-order-template.json`
- **Archive Audit Report**: See `docs/reports/archive_audit.md`
- **AGENTS.md Guidance**: Archival workflow reminders (lines 59-63 in AGENTS.md)

---

*Last Updated: 2025-10-11*
*Next Audit Recommended: 2025-11-11 (monthly cadence)*
