# Pre-Commit Hooks Guide

## Overview

This project includes a lightweight pre-commit hook that validates Python syntax and runs fast unit tests before allowing commits. This helps catch errors early and maintains code quality.

## What the Hook Does

The pre-commit hook (`hooks/pre-commit`) performs two checks:

1. **Python Syntax Validation**: Runs `python -m py_compile` on all staged `.py` files
2. **Fast Unit Tests**: Runs classification prompt tests (`test_classification_prompt.py`)

Total execution time: **~0.1 seconds** (very fast!)

## Installation

### Step 1: Copy the hook

```bash
cp hooks/pre-commit .git/hooks/
chmod +x .git/hooks/pre-commit
```

### Step 2: Verify installation

```bash
# Test the hook manually
.git/hooks/pre-commit
```

You should see: `✅ No Python files staged, skipping checks` (if nothing is staged)

## Usage

### Normal Workflow

The hook runs automatically on every `git commit`:

```bash
git add myfile.py
git commit -m "feat: add new feature"

# Hook runs automatically:
# 🔍 Running pre-commit checks...
# 📝 Staged Python files:
#    myfile.py
# 1️⃣ Checking Python syntax...
#    ✅ All staged files have valid Python syntax
# 2️⃣ Running fast unit tests...
#    ✅ Fast tests passed
# ✅ All pre-commit checks passed!
```

### Bypassing the Hook

**Use sparingly!** Only bypass when necessary (e.g., work-in-progress commits):

```bash
git commit --no-verify -m "wip: incomplete feature"
```

### Testing the Hook

Test without committing:

```bash
# Stage some Python files
git add src/core/my_module.py

# Run hook manually
./hooks/pre-commit
```

## Troubleshooting

### Error: 'uv' command not found

Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Syntax errors in staged files

Fix the Python syntax errors and try again:
```bash
# Hook will tell you which file has errors
❌ Syntax error in: src/core/broken.py

# Fix the file, then:
git add src/core/broken.py
git commit -m "fix: resolve syntax error"
```

### Unit tests failing

Run tests manually to debug:
```bash
uv run python -m unittest discover -s tests -p 'test_classification_prompt.py' -v
```

Fix the failing tests, then commit again.

## What's Checked

### ✅ Syntax Check (Fast)
- Uses Python's built-in `py_compile` module
- Validates basic Python syntax
- Catches: missing colons, unmatched parentheses, invalid indentation

### ✅ Fast Unit Tests
- Runs `test_classification_prompt.py` (0.000s execution time)
- Validates classification prompt construction
- Ensures core functionality works

### ❌ What's NOT Checked
- Type hints (no mypy)
- Code style (no black/flake8)
- Full test suite (only fast tests)
- Import resolution (only syntax)

## Customization

To modify what the hook checks, edit `hooks/pre-commit`:

```bash
# Example: Add more test files
uv run python -m unittest discover -s tests -p "test_classification*.py"

# Example: Add syntax check for a specific directory
for file in $(find src/core -name '*.py'); do
    uv run python -m py_compile "$file"
done
```

After editing, reinstall:
```bash
cp hooks/pre-commit .git/hooks/
chmod +x .git/hooks/pre-commit
```

## Integration with CI/CD

The hook complements CI/CD pipelines:

- **Pre-commit hook**: Fast checks (syntax, core tests) - **0.1s**
- **CI/CD**: Full test suite, linting, type checking - **minutes**

This gives immediate feedback locally while maintaining comprehensive checks in CI.

## Disabling the Hook

To temporarily disable:

```bash
# Option 1: Remove the hook
rm .git/hooks/pre-commit

# Option 2: Rename it
mv .git/hooks/pre-commit .git/hooks/pre-commit.disabled

# Option 3: Use --no-verify for individual commits
git commit --no-verify -m "message"
```

To re-enable:
```bash
cp hooks/pre-commit .git/hooks/
chmod +x .git/hooks/pre-commit
```

## Best Practices

1. **Install immediately**: Set up the hook when you clone the repo
2. **Don't bypass habitually**: Only use `--no-verify` when truly necessary
3. **Keep it fast**: Hook should complete in <5 seconds
4. **Test before pushing**: Hook catches local errors, CI catches the rest
5. **Update together**: When updating `hooks/pre-commit`, reinstall to `.git/hooks/`

## FAQ

**Q: Why not use the `pre-commit` framework?**
A: This is a lightweight solution for a small project. The `pre-commit` framework is great for larger projects with many hooks.

**Q: Can I add more checks?**
A: Yes! Edit `hooks/pre-commit` and add your checks. Keep total time under 5 seconds.

**Q: Does this replace the full test suite?**
A: No! This runs FAST tests only. Always run the full suite before pushing:
```bash
uv run python tests/run_all_tests.py
```

**Q: What if I need to commit broken code temporarily?**
A: Use `git commit --no-verify` but fix it in the next commit.

## Related Documentation

- [Testing Guide](../CLAUDE.md#testing) - Full test suite documentation
- [Development Workflow](../CLAUDE.md#development-commands) - Standard dev commands
- [Git Workflow](../CLAUDE.md#committing-changes-with-git) - Commit message guidelines
