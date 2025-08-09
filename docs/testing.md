## Testing Guide

This document explains the test structure and how to run tests.

### Structure
- `tests/unit/` — unit tests for orchestrator, AML pipeline, and reporting
- `tests/integration/` — integration tests (placeholder)
- `tests/benchmarks/` — performance/benchmark test hooks (placeholder)

### Running Tests
```bash
# Run all tests
pytest -v

# Unit tests only
pytest tests/unit -v

# With coverage
pytest -v --cov=src --cov-report=html
```

### Writing Tests
- Prefer deterministic tests for unit coverage
- For benchmarking code, use reduced iterations and small data sizes (or mocks) to keep tests fast
- Use `pytest.mark.asyncio` for async tests

### Benchmark Tests
- Benchmark tests are separated to avoid long CI runs
- If you add benchmark tests in `tests/benchmarks/`, ensure they are opt-in via markers or environment flags

### Linting & Formatting
```bash
make lint
make format
```

### Troubleshooting
- Import errors: ensure `src/` is on the Python path in tests or use absolute imports (as provided)
- Long runtimes: skip benchmark-heavy tests by default, reduce sample sizes
