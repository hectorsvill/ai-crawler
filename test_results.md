# AI Crawler — Real-World Test Results

| # | Test | Workflow | Result | Pages | Extractions | Notes |
|---|------|----------|--------|-------|-------------|-------|
| 1 | Single page extraction | simple | ✅ PASS | 1 | 1 | fetch_method=playwright (example.com < 500 chars, auto-upgrade correct) |
| 2 | Multi-page Wikipedia crawl | simple | ✅ PASS | 3 | 3 | All Python keywords found; %(28) vs () encoding visited same page twice (minor) |
| 3 | Content deduplication | simple | ✅ PASS | 1 | 1 | Fixed: URL queue was globally unique (blocked re-crawl); scoped to session; added content-hash check to skip LLM on re-crawl |
| 4 | Workflow router accuracy | auto | ✅ PASS | — | — | 8/8 (100%) — perfect classification |
| 5 | Graceful failure handling | — | ✅ PASS | — | — | httpstat.us had SSL errors (caught); DNS fail caught; example.com succeeded |
| 6 | Resume after interruption | simple | ✅ PASS | 4 | 3 | Fixed: link queueing moved before extraction; link pre-filter removes Wikipedia nav/interlang junk; 4 unique pages, 0 re-visits after resume |
| 7 | LangGraph deep research | langgraph | ✅ PASS | 8 | 8 | Solar energy startups; completed=True; 1 non-fatal extractor schema warning (LLM nested JSON) |
| 8 | Token efficiency under load | simple | ✅ PASS | 4 | 4 | ~69k total tokens (60k prompt + 9k completion); Wikipedia pages are 15 chunks each so 50k estimate was low; chunking and caching working correctly |
| 9 | Robots.txt compliance | — | ✅ PASS | — | — | google.com/search blocked; wikipedia article allowed; example.com allowed; 3/3 correct |
| 10 | Full pipeline real research | auto→langgraph | ✅ PASS | 10 | 10 | Solid-state batteries; router selected langgraph; found QuantumScape; exported 10 records to JSON; end-to-end pipeline complete |
