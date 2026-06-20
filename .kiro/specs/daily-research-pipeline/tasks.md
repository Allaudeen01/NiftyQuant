# Implementation Plan: Daily Research Pipeline

## Overview

This plan implements the Daily Research Pipeline as a staged orchestration layer over the frozen `nifty_quant` v0.2 components, in Python. New code lives in `scripts/daily_research.py` (thin CLI) and a new `nifty_quant/research/pipeline/` sub-package holding the orchestrator, per-stage functions, and pure helper modules (`models.py`, `quality.py`, `evidence.py`, `experiment.py`, `catalog.py`, `stages.py`, `confidence.py`, `regime_view.py`, `cost.py`, `ideas.py`, `report.py`, `roadmap.py`, `orchestrator.py`).

The pipeline runs eight stages in a fixed order: `quality-gate → read → features → validation → comparison → evidence → ideas → report`. A pre-flight Data Quality Gate (Stage 0) runs before the read stage and halts the run on a FAIL verdict. Every stage is wrapped with a cost recorder, every run is framed by a Market Regime classification (reusing the frozen regime detector) and a single Overall Research Confidence Score, and the ROADMAP.md control center is refreshed each run.

Tasks are ordered so each step builds on the previous: data models and pure scoring logic first (quality, evidence, experiment, confidence), then the catalog and stages, then idea/report/roadmap rendering, then orchestration and the CLI that wires everything together. Property-based tests (using `hypothesis`) implement the 27 correctness properties from the design; each property is its own optional sub-task placed next to the code it validates and tagged with its property number and the requirement clauses it checks.

## Tasks

- [x] 1. Set up pipeline package, dependencies, and data models
  - [x] 1.1 Create package skeleton and add the property-test dependency
    - Create `nifty_quant/research/pipeline/__init__.py` establishing the new sub-package
    - Add `hypothesis>=6` to the `dev` optional-dependencies in `pyproject.toml`
    - Confirm the package imports no broker/order modules (read-only constraint)
    - _Requirements: 16.6, 1.6_

  - [x] 1.2 Define pipeline data models
    - In `nifty_quant/research/pipeline/models.py`, define frozen dataclasses: `SessionMetrics`, `MarketSummary`, `EvidenceChange`, `UnusualEvent`, `ComparisonResult`, plus the new `MarketRegime` and `ExperimentRecord`
    - Extend `ResearchIdea` with `information_gain: float = 0.0` and `priority: int | None = None` (Expected_Information_Gain + ranked Priority)
    - Extend `DailyReportModel` with `market_regime: MarketRegime` (Section 0), `confidence: ConfidenceResult`, `quality: QualityReport`, `rejected: list[EvidenceChange]`, and `stage_costs: list[StageCost]`, composing the `QualityReport`/`QualityCheckResult`, `ConfidenceResult`/`ConfidenceBreakdown`, and `StageCost` types defined in their feature modules (tasks 2, 16, 19)
    - Mirror existing codebase dataclass style
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 9.1, 10.1, 5.1, 5.4, 13.4, 19.5, 20.1, 20.2, 21.1, 22.3, 23.1_

- [x] 2. Implement the Data Quality Gate (Stage 0)
  - [x] 2.1 Implement QualityConfig, result models, and the five Quality_Check functions
    - In `nifty_quant/research/pipeline/quality.py`, define frozen `QualityConfig` (pass_threshold, max_gap_minutes, exchange_tz, per-check weights, blocking set), `QualityCheckResult`, and `QualityReport` (with `failing_checks`)
    - Implement `check_api_outage`, `check_duplicate_timestamps`, `check_expiry_mismatch`, `check_holiday`, and `check_timezone_anomaly`, each inspecting Target_Session chains/candles and returning a `QualityCheckResult`
    - Implement `run_quality_checks(chains, candles, session, config)` reading only through the existing `Storage`-sourced inputs (no new backend)
    - _Requirements: 19.1, 19.2, 16.1_

  - [x] 2.2 Implement the pure score_quality aggregation
    - In `nifty_quant/research/pipeline/quality.py`, implement `score_quality(checks, config) -> QualityReport`: `score = clamp(100 - sum(penalties), 0, 100)`; verdict `PASS` iff `score >= pass_threshold` and no blocking check failed, else `FAIL`
    - _Requirements: 19.3_

  - [ ]* 2.3 Write property test for Quality_Score bounds and verdict threshold
    - **Property 20: Quality_Score is bounded and the verdict respects the threshold**
    - **Validates: Requirements 19.3**
    - Tag the test `# Feature: daily-research-pipeline, Property 20` and run `@settings(max_examples=100)`

  - [ ]* 2.4 Write property test for defect detection
    - **Property 21: Each injected data defect is detected and reported by its Quality_Check**
    - **Validates: Requirements 19.2, 19.4**

  - [ ]* 2.5 Write unit tests for the Quality_Checks
    - Test each check against a clean synthetic session and against its single injected defect; assert `detail` text and `blocking` flags
    - _Requirements: 19.2_

- [x] 3. Implement EvidenceEngine scoring logic
  - [x] 3.1 Implement EvidenceConfig, Verdict, and scoring/status functions
    - In `nifty_quant/research/pipeline/evidence.py`, define `EvidenceConfig`, `Verdict` enum, and `EvidenceEngine` with `read_score`, `apply`, `next_status`, and `evaluate`
    - `apply` clamps to [0,100]: SUPPORTING `min(100, s+inc)`, CONTRADICTING `max(0, s-dec)`, ABSENT `max(0, s-decay)`
    - `next_status` maps score to a value in `{open, testing, supported, rejected, inconclusive}` honoring reject/support thresholds
    - `evaluate` deterministically maps a hypothesis (by tags) plus market summary/history to a `Verdict`
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 12.4_

  - [ ]* 3.2 Write property test for Evidence_Score bounds
    - **Property 1: Evidence_Score stays within bounds**
    - **Validates: Requirements 11.1, 11.2, 11.3**

  - [ ]* 3.3 Write property test for Evidence_Score direction and decay
    - **Property 2: Evidence_Score moves in the correct direction**
    - **Validates: Requirements 11.2, 11.3, 11.4**

  - [ ]* 3.4 Write property test for status mapping
    - **Property 3: Status mapping is valid and respects thresholds**
    - **Validates: Requirements 11.5, 11.6, 12.4**

- [x] 4. Implement Evidence_Score journal persistence
  - [x] 4.1 Implement evidence tag encode/decode and journal update integration
    - In `nifty_quant/research/pipeline/evidence.py`, encode score as the `evidence:<NN>` tag plus mirrored `confidence = NN/100`; decode the first `^evidence:(\d{1,3})$` tag, defaulting to `initial_score`
    - Persist score/status/reason changes via the existing `ResearchJournal.update` without modifying its interface
    - _Requirements: 11.7, 12.2, 12.5_

  - [ ]* 4.2 Write property test for journal round-trip
    - **Property 4: Evidence_Score and journal fields round-trip through persistence**
    - **Validates: Requirements 11.7, 12.2**

- [x] 5. Implement the Experiment Registry
  - [x] 5.1 Implement encode_experiment and decode_experiment
    - In `nifty_quant/research/pipeline/experiment.py`, implement pure `encode_experiment(record) -> (hypothesis_text, tags)` and `decode_experiment(h: Hypothesis) -> ExperimentRecord`
    - Map the eight `ExperimentRecord` fields onto the native `Hypothesis.hypothesis` field plus structured tags (`exp.rq`, `exp.dataset`, `exp.feature`, `exp.code`, `exp.result`, `exp.decision`, `exp.next`); base64url-encode free-text fields so they survive the tag-list encoding
    - Persist only through the existing `ResearchJournal` (`add`/`update`/`list`); introduce no separate experiment store
    - _Requirements: 23.1, 23.2_

  - [ ]* 5.2 Write property test for experiment round-trip
    - **Property 26: Experiment_Record round-trips through the journal**
    - **Validates: Requirements 23.1, 23.3**

- [x] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement SessionCatalog
  - [x] 7.1 Implement SessionCatalog over the existing Storage interface
    - In `nifty_quant/research/pipeline/catalog.py`, implement `sessions_up_to`, `collected_count`, `prior_sessions`, plus option-chain and trading-day inventory counts by reading option chains over `[epoch, target_end]` once via `Storage.read_option_chains` and grouping by `snapshot_ts.date()`
    - Use only the existing `Storage` interface; introduce no new backend
    - _Requirements: 5.1, 5.2, 2.5, 16.1, 18.3_

  - [ ]* 7.2 Write property test for collected-session counting
    - **Property 8: Collected_Session_Count and comparison window reflect only collected sessions**
    - **Validates: Requirements 5.1, 5.2**

  - [ ]* 7.3 Write unit tests for SessionCatalog
    - Test distinct-date grouping, ascending order, prior-session derivation, and inventory counts
    - _Requirements: 5.1, 5.2, 18.3_

- [x] 8. Implement Read stage
  - [x] 8.1 Implement read_stage
    - In `nifty_quant/research/pipeline/stages.py`, implement `read_stage` using `Storage.read_option_chains` and `Storage.read_candles` for the Target_Session window; select the latest end-of-day snapshot while keeping the full ordered list
    - Preserve `OptionChain.context` fields (`india_vix`, `days_to_expiry`, `minutes_since_open`, `is_expiry_day`) unchanged downstream
    - Raise `StageError("read", ...)` when the Target_Session has zero snapshots
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 16.1_

  - [ ]* 8.2 Write property test for context preservation
    - **Property 13: Snapshot context is preserved through the read stage**
    - **Validates: Requirements 2.4**

  - [ ]* 8.3 Write edge/example tests for the read stage
    - Test no-option-data path raises and example reads use the existing interfaces
    - _Requirements: 2.1, 2.2, 2.3_

- [x] 9. Implement Feature stage
  - [x] 9.1 Implement feature_stage
    - In `nifty_quant/research/pipeline/stages.py`, replay session events through `FeatureEngine` (`on_option_chain`, `on_candle`) and persist each `FeatureVector` via `FeatureStore.put`, recording `FeatureEngine.version`
    - Reuse `analytics.options` via the engine; do not reimplement option analytics
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 16.2, 16.3_

  - [ ]* 9.2 Write property test for feature refresh idempotence
    - **Property 14: Feature refresh is idempotent**
    - **Validates: Requirements 3.4**

- [x] 10. Implement Validation stage
  - [x] 10.1 Implement validation_stage
    - In `nifty_quant/research/pipeline/stages.py`, build a `Baseline` from prior-session feature values and call `ValidationEngine.validate(...)` with Target_Session features; capture each `Alert` level and message
    - Surface insufficient-data conditions as informational notes rather than failures; do not reimplement drift logic
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 16.4_

  - [ ]* 10.2 Write unit tests for the validation stage
    - Test alert capture and insufficient-data informational note handling
    - _Requirements: 4.2, 4.3_

- [x] 11. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Implement market summary metrics (Report Section 1)
  - [x] 12.1 Implement SessionMetrics and MarketSummary computation
    - In `nifty_quant/research/pipeline/stages.py`, compute NIFTY % change, India VIX % change, PCR, max pain, gamma sign, and OI change per the design's metric-source table, reusing `analytics.options`
    - Add any uncomputable metric to `MarketSummary.unavailable` rather than dropping it
    - _Requirements: 6.2, 6.3, 6.4, 6.5, 3.5, 16.3_

  - [ ]* 12.2 Write unit tests for metric computation
    - Test each metric source and the unavailable-marking path for missing data
    - _Requirements: 6.5_

- [x] 13. Implement Comparison stage (unusual events and small-sample behavior)
  - [x] 13.1 Implement comparison_stage
    - In `nifty_quant/research/pipeline/stages.py`, compute `Collected_Session_Count` from `SessionCatalog`, build per-session metric series over collected Prior_Sessions, detect extremity and gamma-flip unusual events, and annotate statements with the session count when below `min_sample_size`
    - Handle the single-session case with `history_available=False`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 9.2, 9.3, 9.4, 14.1, 14.2, 14.3, 14.4_

  - [ ]* 13.2 Write property test for extreme-metric unusual events
    - **Property 9: Extreme Target_Session metric produces an unusual event with the correct span**
    - **Validates: Requirements 9.2**

  - [ ]* 13.3 Write property test for gamma-flip detection
    - **Property 10: Gamma flip is recorded exactly when the gamma sign changes**
    - **Validates: Requirements 9.3**

  - [ ]* 13.4 Write property test for small-sample annotation
    - **Property 11: Below the minimum sample size, every comparison statement is count-annotated and claims no significance**
    - **Validates: Requirements 5.3, 9.4, 14.2, 14.3, 10.3**

  - [ ]* 13.5 Write property test for graceful small-sample completion
    - **Property 12: Graceful small-sample completion**
    - **Validates: Requirements 14.1, 14.4**

- [x] 14. Implement Journal & Evidence stage
  - [x] 14.1 Implement journal_evidence_stage
    - In `nifty_quant/research/pipeline/stages.py`, load hypotheses via `journal.list()`, evaluate each through `EvidenceEngine`, update score/status, persist via `journal.update`, and add new trackable ideas via `journal.add(..., status="open")`
    - Produce `EvidenceChange` records (prior/updated score, observation, status before/after) for gained, weakened, and rejected hypotheses
    - Write/refresh `Experiment_Record` provenance through `encode_experiment` so status changes to `supported`/`rejected` carry full provenance and trigger a ROADMAP update
    - _Requirements: 7.2, 7.3, 8.2, 8.3, 11.5, 11.6, 12.1, 12.2, 12.3, 12.4, 18.2, 23.1_

  - [ ]* 14.2 Write property test for evidence-change rendering inputs
    - **Property 17: Evidence-change sections render prior and updated scores**
    - **Validates: Requirements 7.2, 7.3, 8.2, 8.3**

  - [ ]* 14.3 Write unit tests for journal load/add/update integration
    - Test hypothesis load, new-idea add with status `open`, update persistence, and Experiment_Record capture of all eight fields
    - _Requirements: 12.1, 12.3, 23.1_

- [x] 15. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 16. Implement the Confidence scorer
  - [x] 16.1 Implement ConfidenceConfig, breakdown models, and compute_confidence
    - In `nifty_quant/research/pipeline/confidence.py`, define frozen `ConfidenceConfig` (five weights summing to 100), `ConfidenceBreakdown`, and `ConfidenceResult` (score + breakdown + `factors` list for rendering)
    - Implement pure `compute_confidence(quality_score, collected_session_count, min_sample_size, drift_severity, alert_severity, evidence_maturity, config)` as a deterministic weighted average of the five normalised factors, clamped to 0..100
    - _Requirements: 21.1, 21.2, 21.3, 21.4_

  - [ ]* 16.2 Write property test for the confidence score
    - **Property 23: Research_Confidence_Score is bounded, deterministic, and shows its factors**
    - **Validates: Requirements 21.1, 21.2, 21.3, 21.4**

- [x] 17. Implement the Market Regime adapter
  - [x] 17.1 Implement the regime_view adapter
    - In `nifty_quant/research/pipeline/regime_view.py`, implement `market_regime(candle_history, summary, config)` that reuses `nifty_quant.research.regime.classify_regime` for trend/volatility and maps the option-derived gamma sign to a gamma regime and PCR to a PCR level, returning a `MarketRegime`
    - Record any classification that cannot be computed as `None` so the renderer marks it `unavailable`; perform no regime math beyond the gamma/PCR labelling
    - _Requirements: 20.2, 20.3, 20.4_

  - [ ]* 17.2 Write a smoke/architectural test for regime reuse
    - Verify the adapter calls the existing `classify_regime` and consumes its `Regime(trend, volatility, tags, stats)` output without modifying the classifier interface
    - _Requirements: 20.3_

- [x] 18. Implement IdeaGenerator and prioritization
  - [x] 18.1 Implement RuleBasedIdeaGenerator (primary/default) and sanitize_idea
    - In `nifty_quant/research/pipeline/ideas.py`, implement the deterministic `RuleBasedIdeaGenerator` as the primary, default hypothesis source, producing `ResearchIdea` records phrased as investigations with occurrence counts relative to collected sessions
    - Implement `sanitize_idea` that strips/raises on trade-recommendation phrasing (buy, sell, enter, exit, go long, go short, take a position)
    - _Requirements: 10.2, 10.5, 10.6, 15.1, 15.2_

  - [x] 18.2 Implement AIIdeaGenerator wrapper (opt-in, with fallback)
    - In `nifty_quant/research/pipeline/ideas.py`, implement `AIIdeaGenerator` that is opt-in only (`use_ai` defaults False), calls the backend, passes output through `sanitize_idea` to supplement the rule-based ideas, and falls back to `RuleBasedIdeaGenerator` on any failure
    - _Requirements: 15.3, 15.4, 10.6_

  - [x] 18.3 Implement estimate_information_gain and prioritize
    - In `nifty_quant/research/pipeline/ideas.py`, implement deterministic `estimate_information_gain(idea, comparison, history)` and `prioritize(ideas)` that sorts ideas by descending `information_gain` (stable on ties) and assigns `priority = 1..N`, with Priority 1 = highest gain
    - Populate the extended `ResearchIdea.information_gain`/`priority` fields
    - _Requirements: 10.3, 10.4_

  - [ ]* 18.4 Write property test for trade-recommendation-free ideas
    - **Property 6: Generated research ideas are never trade recommendations**
    - **Validates: Requirements 10.6, 15.4**

  - [ ]* 18.5 Write property test for AI fallback
    - **Property 7: AI failure falls back to the rule-based result**
    - **Validates: Requirements 15.4**

  - [ ]* 18.6 Write property test for idea prioritization
    - **Property 24: Research ideas are ranked by descending Expected_Information_Gain**
    - **Validates: Requirements 10.3, 10.4**

- [x] 19. Implement per-stage computational cost capture
  - [x] 19.1 Implement the StageCost model
    - In `nifty_quant/research/pipeline/cost.py`, define the frozen `StageCost` dataclass (stage_name, elapsed_seconds, peak_memory_bytes, rows_processed)
    - Provide the helper signature for the cost-capturing wrapper consumed by the orchestrator (the wrapper itself is wired in the orchestrator task), and exclude these instrumentation values from all deterministic computations
    - _Requirements: 22.1, 22.2_

- [x] 20. Implement ReportRenderer
  - [x] 20.1 Implement render and write
    - In `nifty_quant/research/pipeline/report.py`, implement pure `render(DailyReportModel) -> str` producing markdown with the **Market Regime** section first, the **Overall Research Confidence Score** with its contributing factors, the **Quality_Score**, all five required headed sections, and a per-stage **Computational Cost** table; plus a thin `write(path, text)`
    - Name the file `reports/research_<session>.md` and overwrite any existing file for that session
    - Render unavailable metrics/regime fields as `unavailable`; render empty-section messages for gained/weakened/unusual/ideas
    - _Requirements: 6.1, 6.5, 7.1, 7.4, 8.1, 8.4, 9.1, 9.5, 10.1, 10.7, 13.1, 13.2, 13.3, 13.4, 19.5, 20.1, 20.4, 21.1, 21.3, 22.3_

  - [ ]* 20.2 Write property test for regime, confidence, and five-section completeness
    - **Property 15: The report contains the regime section, confidence score, and all five sections**
    - **Validates: Requirements 6.1, 7.1, 8.1, 9.1, 10.1, 13.4, 19.5, 20.1, 21.1**

  - [ ]* 20.3 Write property test for unavailable-metric marking
    - **Property 16: Unavailable market metrics are marked, never omitted**
    - **Validates: Requirements 6.5**

  - [ ]* 20.4 Write property test for report filename
    - **Property 18: Report filename identifies the Target_Session**
    - **Validates: Requirements 13.2**

  - [ ]* 20.5 Write property test for market-regime rendering
    - **Property 27: Market regime renders its classification, marking unavailable fields**
    - **Validates: Requirements 20.2, 20.4**

  - [ ]* 20.6 Write property test for per-stage cost capture and rendering
    - **Property 25: Per-stage computational cost is captured for every stage and rendered**
    - **Validates: Requirements 22.2, 22.3**

  - [ ]* 20.7 Write edge tests for empty sections and overwrite idempotence
    - Test empty gained/weakened/unusual/ideas messages and per-session overwrite
    - _Requirements: 7.4, 8.4, 9.5, 10.7, 13.3_

- [x] 21. Implement RoadmapWriter
  - [x] 21.1 Implement the expanded ROADMAP.md control center
    - In `nifty_quant/research/pipeline/roadmap.py`, render the single root `ROADMAP.md` with Architecture Version, Frozen Components, Current Dataset, Collected Sessions/Collected Option Chains/Collected Trading Days counts, Data Quality (latest Quality_Score + verdict), Open/Supported/Rejected Hypotheses, Top Performing Experiments (ranked by Evidence_Score), Research Priorities (prioritized ResearchIdea list), and Next Milestones
    - Rewrite the three-way hypothesis partition and inventory/quality entries from current journal + warehouse + quality state on each run
    - _Requirements: 18.1, 18.2, 18.3, 18.4_

  - [ ]* 21.2 Write property test for roadmap status partitioning
    - **Property 19: Roadmap hypothesis lists partition by status**
    - **Validates: Requirements 18.2**

- [x] 22. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 23. Implement the Orchestrator
  - [x] 23.1 Implement PipelineConfig, PipelineResult, run_with_cost, and PipelineOrchestrator
    - In `nifty_quant/research/pipeline/orchestrator.py`, run the stages in the fixed new order (quality-gate → read → features → validation → comparison → evidence → ideas → report) threading a `PipelineContext`, injecting dependencies (storage, feature store, journal, idea generator)
    - Implement `run_with_cost(stage_name, fn, ...)` wrapping every executed stage, appending one `StageCost` to `PipelineContext.stage_costs` regardless of outcome, and thread `stage_costs`, `quality_score`, and `confidence_score` into `PipelineResult`
    - Halt before the read stage on a quality-gate FAIL: record every failing `QualityCheckResult`, set `failing_stage="quality_gate"`, exit non-zero; classify other `StageError`s with `failing_stage`/`error_detail` and non-zero exit; recoverable conditions become notes with exit 0
    - Compute the `Research_Confidence_Score` via `compute_confidence` after validation/evidence are known; defer journal flush until the journal/evidence stage completes
    - _Requirements: 1.1, 1.4, 1.5, 1.6, 16.5, 16.6, 19.1, 19.4, 19.5, 21.1, 22.2_

  - [ ]* 23.2 Write property test for quality-gate halt before read
    - **Property 22: A FAIL verdict halts the pipeline before the read stage**
    - **Validates: Requirements 19.1, 19.4**

  - [ ]* 23.3 Write unit tests for orchestration and exit codes
    - Test fixed stage order with the quality gate first, success exit 0, quality-gate FAIL halt-before-read path, unrecoverable stage-error path with failing-stage detail, recoverable-note handling, and one StageCost per executed stage
    - _Requirements: 1.1, 1.4, 1.5, 19.1, 22.2_

- [x] 24. Implement the CLI entry point
  - [x] 24.1 Implement scripts/daily_research.py main()
    - Parse arguments (`--session`, `--underlying`, `--timeframe`, dirs, `--journal`, `--roadmap`, `--ai/--no-ai` defaulting to `--no-ai`, `--min-sample`, `--quality-threshold`), build `PipelineConfig`, construct frozen components and orchestrator, and `sys.exit(main())`
    - Default Target_Session to the most recent collected session; use explicit `--session` when provided; return exit code 2 on malformed arguments
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 15.1_

  - [ ]* 24.2 Write unit tests for CLI argument parsing
    - Test default vs explicit session selection, AI-off default, and malformed-argument exit code
    - _Requirements: 1.2, 1.3, 15.1_

- [ ] 25. Final integration and determinism verification
  - [ ]* 25.1 Write property test for end-to-end determinism
    - **Property 5: Determinism with AI disabled**
    - **Validates: Requirements 15.1, 17.1**

  - [ ]* 25.2 Write smoke and architectural tests
    - Verify read-only imports (no broker/order modules); reuse of frozen `Storage`/`FeatureEngine`/`FeatureStore`/analytics/`ValidationEngine`/`ResearchJournal` and the `nifty_quant.research.regime` classifier without interface changes; experiment provenance persisted only through the existing journal; and per-stage independent testability with synthetic inputs
    - _Requirements: 1.6, 2.5, 3.5, 4.4, 12.5, 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 17.2, 17.3, 20.3, 23.2_

- [x] 26. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP; they cover unit, property, and integration tests.
- Each task references specific requirement sub-clauses for traceability.
- Property-based tests use `hypothesis` (min 100 examples each) and each carries a `# Feature: daily-research-pipeline, Property N` tag. All 27 correctness properties (1-27) are covered as optional property-test sub-tasks.
- The Data Quality Gate is Stage 0 and halts the run before the read stage on a FAIL verdict; the orchestrator wraps every stage with `run_with_cost` and computes the Overall Research Confidence Score.
- Checkpoints (tasks 6, 11, 15, 22, 26) provide incremental validation breaks.
- All stage logic lives in `stages.py` and is wired by the orchestrator; tasks touching `stages.py` (8.1, 9.1, 10.1, 12.1, 13.1, 14.1) and tasks touching `ideas.py` (18.1, 18.2, 18.3) and `quality.py` (2.1, 2.2) are scheduled in separate waves to avoid write conflicts.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["2.1", "3.1", "5.1", "7.1", "16.1", "19.1"] },
    { "id": 3, "tasks": ["2.2", "4.1", "8.1", "17.1", "3.2", "3.3", "3.4", "5.2", "16.2"] },
    { "id": 4, "tasks": ["9.1", "2.3", "2.4", "2.5", "4.2", "8.2", "8.3", "17.2", "7.2", "7.3"] },
    { "id": 5, "tasks": ["10.1", "9.2"] },
    { "id": 6, "tasks": ["12.1", "10.2"] },
    { "id": 7, "tasks": ["13.1", "12.2"] },
    { "id": 8, "tasks": ["14.1", "13.2", "13.3", "13.4", "13.5"] },
    { "id": 9, "tasks": ["18.1", "14.2", "14.3"] },
    { "id": 10, "tasks": ["18.2"] },
    { "id": 11, "tasks": ["18.3"] },
    { "id": 12, "tasks": ["20.1", "21.1", "18.4", "18.5", "18.6"] },
    { "id": 13, "tasks": ["23.1", "20.2", "20.3", "20.4", "20.5", "20.6", "20.7", "21.2"] },
    { "id": 14, "tasks": ["24.1", "23.2", "23.3"] },
    { "id": 15, "tasks": ["24.2", "25.1", "25.2"] }
  ]
}
```
