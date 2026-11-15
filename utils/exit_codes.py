"""Process exit codes shared by every CLI command (documented in AGENTS.md)."""

OK = 0            # success (a crawl may still report per-page failures in its JSON)
ERROR = 1         # runtime failure (e.g. crawl fetched nothing, unexpected exception)
USAGE = 2         # bad arguments / invalid option values (Typer also uses 2)
DEPENDENCY = 3    # missing dependency or unreachable service (Ollama, seeds file, package)
NOT_FOUND = 4     # requested session / page / paper does not exist
PARTIAL = 5       # only with --strict: run finished but some pages failed
