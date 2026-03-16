# Testing Policy

## No Automated Tests

This project does **not** use automated tests. Do not:

- Write unit tests, integration tests, or smoke-test scripts
- Suggest adding a test suite or pytest
- Create test files in `/tmp/` or anywhere else as part of implementation work

Verification is done exclusively through **manual testing** — running the dev server and exercising endpoints directly (e.g. `curl`, the FastAPI `/docs` UI, or Neo4j Browser).

When presenting a verification plan, list only manual steps.
