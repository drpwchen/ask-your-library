# Security policy

## Reporting a vulnerability

Please report security issues **privately** through GitHub:
**Security → Report a vulnerability** on this repository
(Private Vulnerability Reporting). Do not open a public issue for them.

Useful in a report: the version or commit, what you did, and what happened.

## Scope notes

- The API server supports an API key (`X-API-Key`). Set one whenever the server is reachable beyond `localhost`.
- Search and indexing run locally (Ollama, LanceDB). Hybrid/Free chat runs the `claude` CLI as a subprocess with a deny-list of tools; a way around that deny-list is in scope.
- `.env`, `plugin/data.json` and `~/.vault-search/` hold keys and your index; they are git-ignored and must never be committed.
