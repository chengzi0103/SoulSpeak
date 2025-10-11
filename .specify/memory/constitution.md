<!--
Sync Impact Report:
Version change: 1.0.0 → 1.1.0
Modified principles: Operating Standards → Operating Standards (新增语言与类定义要求)
Added sections: None
Removed sections: None
Templates requiring updates:
- .specify/templates/plan-template.md ✅ updated
- .specify/templates/spec-template.md ✅ updated
- .specify/templates/tasks-template.md ✅ updated
Follow-up TODOs: None
-->

# SoulSpeak Constitution

## Core Principles

### I. Empathy-First Real-Time Loop
- Voice-first features MUST keep the time from user audio to the first spoken response under 1.2s in a local dev profile when measured with `python soul_speak/mic/run_and_speak.py`; if the target cannot be met, the plan/spec MUST document the measured latency and the mitigation (fallback text, buffering, or explicit user prompt) before implementation proceeds.
- Every playback path MUST honor user interruptions by propagating `END_OF_SPEECH` control messages across ASR, LLM, and TTS modules and by draining audio buffers immediately (see `soul_speak/mic/run_and_speak.py`); regressions require a blocking bug.
- Generated speech MUST be chunked into emotion-aware segments (e.g., via `_split_for_tts`) capped at 80 characters unless vocoder requirements dictate otherwise, ensuring emotional cadence survives downstream streaming.
Rationale: Empathetic presence collapses without low-latency, interruption-aware audio delivery that preserves emotional nuance.

### II. Composable Streaming Modules
- ASR, LLM, and TTS integrations MUST expose WebSocket or streaming gRPC contracts that publish `event`-tagged JSON metadata and raw audio/text frames, matching the handshake consumed by `soul_speak/mic/run_and_speak.py`.
- Module swaps MUST be controlled exclusively through Hydra configs under `conf/`; no hard-coded endpoints, credentials, or model IDs belong in runtime code.
- Each new or altered module MUST ship with a runnable harness in `soul_speak/modules/` (or equivalent) that exercises the streaming boundary end-to-end for manual verification.
Rationale: SoulSpeak stays adaptive only when modules remain hot-swappable through declarative contracts and tooling.

### III. Memory & Emotion With Consent
- Long-term state MUST flow through the `soul_speak/llm/memory/` providers; persistence or retrieval MUST be gated by explicit configuration flags and redact personally identifiable content before storage.
- Emotion recognition outputs MUST be labelled with confidence scores and classification provenance, and they CANNOT be persisted without user consent captured via config or task record.
- When memory or emotion subsystems are disabled, the user experience MUST degrade gracefully to stateless replies without raising exceptions.
Rationale: Trustworthy companionship depends on controllable memory and emotion features that respect consent boundaries.

### IV. Proven Real-Time Reliability
- New features touching audio, streaming, or scheduling MUST include deterministic pytest coverage using recorded fixtures or mocked transports so they run offline; network-bound tests belong behind opt-in markers.
- Plans/specs MUST outline how runtime monitoring (structured logs, metrics, or traces) will surface latency breaches or stream desynchronisation before shipping.
- Bug fixes MAY bypass new tests only with an incident reference that documents why regression coverage already exists.
Rationale: Real-time systems tolerate little slack—repeatable tests and observability keep the loop dependable.

### V. Secure Configuration & Telemetry Discipline
- Secrets and API tokens MUST stay out of version control; runtime code retrieves them via `.env`, `conf/secrets.yaml`, or OS environment variables, and any new key MUST be added to the onboarding docs.
- Configuration diffs MUST enumerate expected side effects (model paths, ports, sample rates) in PR notes, and breaking changes require migration steps or roll-back instructions.
- Logging MUST redact user identifiers and audio transcripts by default while still emitting timestamps, module IDs, and latency measurements for diagnosis.
Rationale: A safe, observable system protects users while giving maintainers the insight needed to improve it.

## Operating Standards
- Target runtime is Python 3.10 with dependencies pinned through `requirements.txt`; deviations require prior approval and compatibility notes in `install.md`.
- Default execution profiles rely on local or LAN-accessible services; features that demand public cloud endpoints MUST provide a sandboxed alternative or documented skip path.
- Audio assets committed to the repo stay under `soul_speak/modules/` (reference use only); production prompts, transcripts, or personally identifiable data remain outside the repository.
- Runtime代理与控制台输出在无特别说明时 MUST 使用中文与用户交互；若场景需要其他语言，必须在规划阶段记录理由与回退策略。
- 新增或重构的类 MUST 通过 `from attrs import define, field` 并以 `@define` 声明数据模型；若无法满足，应在评审备注中给出兼容性说明。

## Delivery Workflow
- Every new initiative starts with `/spec` → `/plan` → `/tasks`; skipping a stage demands explicit maintainer approval recorded in the pull request.
- Plans MUST complete the Constitution Check before Phase 0 research begins, and reviewers block if any gate remains unresolved.
- Pull requests MUST link to latency measurements (Principle I), streaming harness updates (Principle II), consent toggles or migration notes (Principle III), test evidence (Principle IV), and configuration summaries (Principle V) in their description.

## Governance
- This constitution supersedes ad-hoc guidelines; conflicting docs must be updated or annotated within the same change set.
- Amendments require consensus from two maintainers, an updated Sync Impact Report, and propagation to dependent templates within the same pull request.
- Version bumps follow semantic rules (Major for principle rewrites/removals, Minor for additional principles or new mandatory workflow, Patch for clarifications); each release logs measurement baselines for Principle I and any new telemetry requirements.

**Version**: 1.1.0 | **Ratified**: 2025-10-10 | **Last Amended**: 2025-10-10
