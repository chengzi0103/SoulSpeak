# Repository Guidelines

## Project Structure & Module Organization
SoulSpeak's runtime code lives in `soul_speak/` with domain-specific packages: `asr/`, `llm/`, `tts/`, and `mic/` hosting microphone orchestration (`run_and_speak.py`). Shared helpers are in `utils/` and configuration contracts in `conf/` (split into global, module, and secret YAMLs). Streaming scripts and sample media sit in `soul_speak/modules/`—treat them as reference integrations. Reference docs and architecture notes are in `docs/`, while `install.md` and `README*.md` provide onboarding context.

## Build, Test, and Development Commands
Create a Python 3.10 environment, then install dependencies: `python -m venv .venv && source .venv/bin/activate` followed by `pip install -r requirements.txt`. Use `.env` or `conf/secrets.yaml` to populate tokens before running. Launch the voice loop with `python soul_speak/mic/run_and_speak.py`. Generate static docs via `sphinx-build docs docs/_build/html`. When iterating on streaming components, use the WebSocket scripts in `soul_speak/modules/` to exercise ASR and TTS services individually.

## Coding Style & Naming Conventions
All Python modules use 4-space indentation, `utf-8` encoding, and `snake_case` filenames. Keep public functions typed and documented with concise docstrings; configuration keys mirror lower-case-with-underscores used in existing YAML. Follow the existing module layout by co-locating adapters with their domain (e.g., ASR clients stay in `soul_speak/asr/`). Run `python -m compileall soul_speak` or `python -m py_compile ...` if you need a quick syntax check during CI bring-up.

- SoulTask Orchestrator (`soul_speak/sto/`) 中的类统一使用 `attrs.define`（特殊结构如 `Enum` 与协议接口除外），保持字段通过 attrs 管理，便于后续扩展。

## Testing Guidelines
Pytest is the standard test runner (`pytest`), with coverage tooling available via `pytest --cov=soul_speak --cov-report=term`). Add new suites under a top-level `tests/` package; name files `test_<feature>.py` and use fixtures for external services so runs stay offline by default. For streaming flows, prefer unit tests around protocol helpers and mock WebSocket clients rather than full end-to-end audio pipelines.

## Commit & Pull Request Guidelines
Recent history favors short imperative commit subjects (“add mcp tools”, “完成文档的上传”). Keep messages under 72 characters and elaborate in the body if context is needed; bilingual messages are welcome when clarifying intent. Each pull request should describe the user journey affected, list configuration changes, and attach console logs or audio snippets when touching ASR/TTS behaviour. Link tracking issues and note any follow-up work so release notes remain accurate.

## Security & Configuration Tips
Never commit `conf/secrets.yaml` or other credential files; rely on `.gitignore` and environment variables when scripting. Validate WebSocket endpoints and model paths against `conf/config.yaml` and `conf/globals.yaml` before deployment. When sharing logs, redact tokens and user speech samples, and rotate API keys after public demos.
