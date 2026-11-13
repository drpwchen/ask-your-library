# obsidian-vault-search — notes for coding agents

Local-first semantic search for an Obsidian vault (MCP server + HTTP API + Obsidian plugin).

## Running the server

- Every interpreter that runs `server/*.py` needs `python-dotenv`: `config.py` loads the dotenv file
  from the directory above `server/`.
- Editing this repo does not change a running deployment. If you run the server from a separate
  checkout, pull there and restart it before checking behaviour against the live API.
