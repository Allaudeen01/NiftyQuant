# Requirements Document

## Introduction

The Daily Research Pipeline is a single evening command (`python scripts/daily_research.py`) that converts the trading day's collected option-chain and candle data into structured research output and maintains an evidence-based view of which research hypotheses are surviving. It is not a dashboard, not a UI, and not a trade-signal generator.

The pipeline orchestrates existing, frozen `nifty_quant` components: it first runs a pre-flight Data Quality Gate that validates input-data integrity before any research begins, then reads option-chain snapshots and candles from the Parquet warehouse, refreshes the feature store, runs validation/drift health checks, updates the research journal, compares the current session against previously collected sessions, and produces one concise, self-contained daily research report. Every report opens with a prominent Market Regime summary and a single Overall Research Confidence Score, and every stage logs its computational cost for later analysis.

Two cross-cutting principles shape every requirement:

1. **Honest data constraints.** Option-chain history is forward-collected only and cannot be backfilled. All "N sessions" comparisons operate on however many sessions have actually been collected. The pipeline must behave gracefully with very few sessions (e.g., 1-5) and must not overclaim statistical significance on small samples.
2. **Research, not trading.** All generated suggestions are research ideas/hypotheses, never trade recommendations. The pipeline is read-only analysis, deterministic by default. AI-generated ideas are disabled by default: the primary hypothesis source is the rule-based generator feeding the research journal, evidence, and manual review, with AI available only as an optional, off-by-default assistant that produces research ideas and explanations.

This feature reuses the architecture frozen at v0.2 and introduces no major new architecture.

## Glossary

- **Daily_Research_Pipeline**: The orchestrating system invoked by `python scripts/daily_research.py` that runs the end-to-end evening research workflow and is the subject of most requirements in this document.
- **Data_Quality_Gate**: The pre-flight validation stage that runs before the read stage to verify input-data integrity, producing a Quality_Score and a PASS/FAIL verdict that gates the rest of the pipeline.
- **Quality_Score**: An integer percentage in the range 0 to 100 produced by the Data_Quality_Gate summarizing the integrity of the Target_Session input data.
- **Quality_Check**: One individual data-integrity check performed by the Data_Quality_Gate (for example: API-outage/missing-candle detection, duplicate-timestamp detection, expiry-mismatch detection, holiday/non-trading-day detection, timezone-anomaly detection).
- **Session**: One collected trading day of market data, identified by `session_id` (the date in ISO format), as defined by `nifty_quant.data.session`.
- **Target_Session**: The session the pipeline is processing on a given run (by default the most recent collected session).
- **Prior_Sessions**: The set of sessions collected before the Target_Session, used for historical comparison.
- **Collected_Session_Count**: The number of sessions available in the warehouse at or before the Target_Session, used to gate statistical claims.
- **Warehouse**: The Parquet-backed `Storage` layer (`nifty_quant.data.storage.ParquetStorage`) holding option-chain snapshots and candles.
- **Snapshot**: One `OptionChain` snapshot (with synchronized `context`, e.g. `india_vix`, and session metadata such as `days_to_expiry`, `minutes_since_open`, `is_expiry_day`).
- **Feature_Store**: The versioned feature persistence layer (`nifty_quant.features.store.FeatureStore`, default `ParquetFeatureStore`).
- **Feature_Engine**: The deterministic feature computation component (`nifty_quant.features.engine.FeatureEngine`).
- **Validation_Engine**: The existing drift/health-check component (`nifty_quant.validation.engine.ValidationEngine`).
- **Research_Journal**: The hypothesis knowledge base (`nifty_quant.research.journal.ResearchJournal`) storing `Hypothesis` records with `id`, `status`, `confidence`, `reason`, and `tags`.
- **Hypothesis**: A single research idea recorded in the Research_Journal, with a lifecycle status in `{open, testing, supported, rejected, inconclusive}`.
- **Evidence_Score**: An integer score in the range 0 to 100 maintained per Hypothesis that rises as supporting evidence accumulates across sessions and decays toward rejection as evidence weakens or is absent.
- **Daily_Research_Report**: The concise, self-contained artifact (text/markdown/HTML) written under `reports/` for the Target_Session, answering the five required questions.
- **Market_Summary**: The computed set of day metrics (NIFTY % change, India VIX % change, PCR, max pain, gamma sign, OI change) for the Target_Session.
- **Market_Regime**: The classification of prevailing market conditions for the Target_Session (for example: Trending vs Range-Bound, High vs Low Volatility, Bullish/Neutral/Bearish Gamma, PCR level), obtained by reusing the project's existing regime detection.
- **Research_Confidence_Score**: A single integer percentage in the range 0 to 100 reported for the Target_Session that combines data quality, sample size, feature drift, validation alerts, and evidence maturity into one overall measure of how much trust to place in the run's research output.
- **Expected_Information_Gain**: The estimated amount a proposed investigation is expected to teach, used to rank Research_Ideas by Priority.
- **Computational_Cost**: The per-stage execution metrics captured for each pipeline run, comprising execution time, memory usage, and rows processed.
- **Experiment_Record**: An entry in the Research_Journal capturing a reproducible unit of research work, including its research question, hypothesis, dataset version, feature version, code version, result, decision, and next action.
- **Unusual_Event**: A detected occurrence where a Target_Session metric is extreme relative to the Prior_Sessions actually collected (e.g., "largest IV expansion in 18 sessions", "gamma flip detected").
- **Research_Idea**: An evidence-based suggestion of what to investigate next, derived from observed data patterns; never a trade recommendation.
- **AI_Enhancement**: The optional component that generates Research_Idea text and explanations using an AI model when configured and available.
- **Roadmap_Document**: The single-source-of-truth `ROADMAP.md` control center recording architecture version, frozen components, current dataset and data inventory (collected sessions, option chains, trading days), data quality, open/supported/rejected hypotheses, top performing experiments, research priorities, and next milestones.
- **Analytics_Module**: The existing option analytics functions (`nifty_quant.analytics.options`: `put_call_ratio`, `max_pain`, `atm_iv`, `gamma_exposure`).

## Requirements

### Requirement 1: Single-Command Pipeline Orchestration

**User Story:** As a quantitative researcher, I want one evening command to run the entire research workflow end to end, so that I can produce my daily research output without manual steps or clicks.

#### Acceptance Criteria

1. WHEN `python scripts/daily_research.py` is invoked, THE Daily_Research_Pipeline SHALL execute the following stages in order: run the Data_Quality_Gate, read warehouse data, refresh features, run validation, compare against prior sessions, accumulate evidence, generate research ideas, and produce the Daily_Research_Report.
2. WHEN no Target_Session is specified by the operator, THE Daily_Research_Pipeline SHALL select the most recent collected Session as the Target_Session.
3. WHERE the operator supplies an explicit session date, THE Daily_Research_Pipeline SHALL use that date as the Target_Session.
4. WHEN every stage completes without an unrecoverable error, THE Daily_Research_Pipeline SHALL exit with a success status code of 0.
5. IF a stage raises an unrecoverable error, THEN THE Daily_Research_Pipeline SHALL record the failing stage name and error detail in its output and exit with a non-zero status code.
6. THE Daily_Research_Pipeline SHALL perform read-only analysis and SHALL NOT place, modify, or simulate any live trade or order.

### Requirement 2: Read Warehouse Data for the Target Session

**User Story:** As a researcher, I want the pipeline to load the day's option-chain snapshots, spot, and candles from the warehouse, so that all downstream analysis uses the collected data.

#### Acceptance Criteria

1. WHEN the read stage runs, THE Daily_Research_Pipeline SHALL read the Target_Session option-chain Snapshots from the Warehouse using the existing `Storage.read_option_chains` interface.
2. WHEN the read stage runs, THE Daily_Research_Pipeline SHALL read the Target_Session candles from the Warehouse using the existing `Storage.read_candles` interface.
3. IF no option-chain Snapshot exists for the Target_Session, THEN THE Daily_Research_Pipeline SHALL report that the Target_Session has no option-chain data and SHALL exit with a non-zero status code.
4. WHERE a Snapshot includes `context` fields (such as `india_vix`) and session metadata (such as `days_to_expiry`, `minutes_since_open`, `is_expiry_day`), THE Daily_Research_Pipeline SHALL preserve those fields for use in downstream stages.
5. THE Daily_Research_Pipeline SHALL read warehouse data through the existing `Storage` interface and SHALL NOT introduce a new storage backend.

### Requirement 3: Refresh Features and Update the Feature Store

**User Story:** As a researcher, I want the pipeline to compute and persist the day's features, so that the feature store stays current and reproducible.

#### Acceptance Criteria

1. WHEN the feature stage runs, THE Daily_Research_Pipeline SHALL compute features for the Target_Session using the existing Feature_Engine.
2. WHEN features for the Target_Session are computed, THE Daily_Research_Pipeline SHALL persist them to the Feature_Store using the existing `FeatureStore.put` interface.
3. THE Daily_Research_Pipeline SHALL record the Feature_Engine version associated with the persisted features so that incompatible feature sets are not mixed.
4. WHEN features for the Target_Session already exist in the Feature_Store, THE Daily_Research_Pipeline SHALL refresh them such that re-running the stage for the same Target_Session produces the same stored feature values.
5. THE Daily_Research_Pipeline SHALL compute option-derived features by reusing the Analytics_Module functions and SHALL NOT reimplement put/call ratio, max pain, ATM IV, or gamma exposure.

### Requirement 4: Run Validation and Health Checks

**User Story:** As a researcher, I want the pipeline to run drift and health checks each evening, so that I am alerted when the day's data or features diverge from expectations.

#### Acceptance Criteria

1. WHEN the validation stage runs, THE Daily_Research_Pipeline SHALL invoke the existing Validation_Engine against the Target_Session features.
2. WHEN the Validation_Engine produces alerts, THE Daily_Research_Pipeline SHALL capture each alert's level and description for inclusion in the Daily_Research_Report.
3. WHERE the Validation_Engine reports an insufficient-data condition, THE Daily_Research_Pipeline SHALL include that condition in the Daily_Research_Report as an informational note rather than treating it as a failure.
4. THE Daily_Research_Pipeline SHALL perform validation by reusing the existing Validation_Engine and SHALL NOT reimplement drift or health-check logic.

### Requirement 5: Compare the Target Session Against Prior Sessions

**User Story:** As a researcher, I want the pipeline to compare today against previously collected sessions, so that I can see how the day stands relative to recent history.

#### Acceptance Criteria

1. WHEN the comparison stage runs, THE Daily_Research_Pipeline SHALL compute the Collected_Session_Count from the sessions available in the Warehouse.
2. WHEN comparing the Target_Session against history, THE Daily_Research_Pipeline SHALL use only the Prior_Sessions actually collected and SHALL NOT assume any backfilled history.
3. WHEN a comparison metric is computed, THE Daily_Research_Pipeline SHALL express the historical window as the number of Prior_Sessions used in that comparison.
4. IF the Target_Session is the only collected Session, THEN THE Daily_Research_Pipeline SHALL produce the Daily_Research_Report using single-session values and SHALL state that no historical comparison is available.

### Requirement 6: Report Section 1 — What Happened Today

**User Story:** As a researcher, I want a concise summary of the day's headline metrics, so that I can quickly understand market conditions for the session.

#### Acceptance Criteria

1. THE Daily_Research_Report SHALL include a "What happened today" section for the Target_Session.
2. WHEN the Market_Summary is produced, THE Daily_Research_Report SHALL state the NIFTY percentage change for the Target_Session.
3. WHEN the Market_Summary is produced, THE Daily_Research_Report SHALL state the India VIX percentage change for the Target_Session.
4. WHEN the Market_Summary is produced, THE Daily_Research_Report SHALL state the put/call ratio, the max pain strike, the gamma exposure sign, and the open-interest change for the Target_Session.
5. IF a Market_Summary metric cannot be computed from the available data, THEN THE Daily_Research_Report SHALL mark that metric as unavailable rather than omitting it silently.

### Requirement 7: Report Section 2 — Hypotheses That Gained Evidence

**User Story:** As a researcher, I want to see which hypotheses strengthened today and by how much, so that I can focus on ideas that are accumulating support.

#### Acceptance Criteria

1. THE Daily_Research_Report SHALL include a "Hypotheses that gained evidence" section.
2. WHEN a Hypothesis Evidence_Score increased for the Target_Session, THE Daily_Research_Report SHALL list that Hypothesis with its prior Evidence_Score and its updated Evidence_Score.
3. WHEN a Hypothesis gained evidence, THE Daily_Research_Report SHALL state the observation that supported the increase.
4. IF no Hypothesis gained evidence for the Target_Session, THEN THE Daily_Research_Report SHALL state that no hypotheses gained evidence.

### Requirement 8: Report Section 3 — Hypotheses That Weakened

**User Story:** As a researcher, I want to see which hypotheses lost evidence and which moved toward rejection, so that I stop holding dead ideas.

#### Acceptance Criteria

1. THE Daily_Research_Report SHALL include a "Hypotheses that weakened" section.
2. WHEN a Hypothesis Evidence_Score decreased for the Target_Session, THE Daily_Research_Report SHALL list that Hypothesis with its prior Evidence_Score and its updated Evidence_Score.
3. WHEN a Hypothesis status changed to `rejected` during the run, THE Daily_Research_Report SHALL list that Hypothesis as moved to rejected with the reason recorded.
4. IF no Hypothesis weakened for the Target_Session, THEN THE Daily_Research_Report SHALL state that no hypotheses weakened.

### Requirement 9: Report Section 4 — Unusual Events

**User Story:** As a researcher, I want the pipeline to flag unusual market events relative to collected history, so that I notice extremes without manual scanning.

#### Acceptance Criteria

1. THE Daily_Research_Report SHALL include an "Unusual events" section.
2. WHEN a Target_Session metric is the most extreme value among the collected sessions for that metric, THE Daily_Research_Pipeline SHALL record an Unusual_Event that states the metric and the number of sessions spanned (for example, "largest IV expansion in 18 sessions").
3. WHEN a gamma flip is detected for the Target_Session relative to the prior Session, THE Daily_Research_Pipeline SHALL record an Unusual_Event noting the gamma flip.
4. WHILE the Collected_Session_Count is below a configured minimum sample size, THE Daily_Research_Pipeline SHALL qualify each Unusual_Event with the actual session count and SHALL NOT claim statistical significance.
5. IF no Unusual_Event is detected for the Target_Session, THEN THE Daily_Research_Report SHALL state that no unusual events were detected.

### Requirement 10: Report Section 5 — What to Investigate Tomorrow

**User Story:** As a researcher, I want evidence-based research ideas for tomorrow, so that I can plan investigations grounded in observed data patterns rather than trade calls.

#### Acceptance Criteria

1. THE Daily_Research_Report SHALL include a "What to investigate tomorrow" section.
2. WHEN the suggestion stage runs, THE Daily_Research_Pipeline SHALL generate one or more Research_Ideas derived from observed Target_Session data patterns.
3. WHEN two or more Research_Ideas are generated, THE Daily_Research_Pipeline SHALL rank the Research_Ideas by descending Expected_Information_Gain and SHALL assign each a Priority label (Priority 1, Priority 2, Priority 3, and so on) in that ranked order.
4. THE Daily_Research_Pipeline SHALL present the Research_Idea with the highest Expected_Information_Gain as Priority 1 so that the first suggested experiment is the one expected to teach the most.
5. WHEN a Research_Idea references how often a pattern has occurred, THE Daily_Research_Pipeline SHALL state the count relative to the collected sessions (for example, "this combination occurred only 11 times in the dataset").
6. THE Daily_Research_Pipeline SHALL phrase each Research_Idea as an investigation to test and SHALL NOT phrase any Research_Idea as a trade recommendation.
7. IF no Research_Idea can be derived from the Target_Session data, THEN THE Daily_Research_Report SHALL state that no research ideas were generated.

### Requirement 11: Evidence Score Mechanism

**User Story:** As a researcher, I want each hypothesis to carry a 0-100 evidence score that updates as evidence accumulates, so that weak ideas decay toward rejection and strong ideas rise, preventing me from emotionally holding dead ideas.

#### Acceptance Criteria

1. THE Daily_Research_Pipeline SHALL maintain an Evidence_Score in the range 0 to 100 inclusive for each tracked Hypothesis.
2. WHEN evidence supporting a Hypothesis is observed for the Target_Session, THE Daily_Research_Pipeline SHALL increase that Hypothesis Evidence_Score, bounded at a maximum of 100.
3. WHEN evidence contradicting a Hypothesis is observed for the Target_Session, THE Daily_Research_Pipeline SHALL decrease that Hypothesis Evidence_Score, bounded at a minimum of 0.
4. WHILE a Hypothesis receives no supporting evidence across successive sessions, THE Daily_Research_Pipeline SHALL decay that Hypothesis Evidence_Score toward 0.
5. WHEN a Hypothesis Evidence_Score reaches or falls below a configured rejection threshold, THE Daily_Research_Pipeline SHALL set that Hypothesis status to `rejected` and SHALL record the reason in the Research_Journal.
6. WHEN a Hypothesis Evidence_Score reaches or exceeds a configured support threshold, THE Daily_Research_Pipeline SHALL set that Hypothesis status to `supported` and SHALL record the reason in the Research_Journal.
7. WHEN the Daily_Research_Pipeline updates an Evidence_Score, THE Daily_Research_Pipeline SHALL persist the updated score so that the next run continues from it.

### Requirement 12: Research Journal Integration

**User Story:** As a researcher, I want the pipeline to read and update the research journal, so that my hypothesis knowledge base stays current and auditable across sessions.

#### Acceptance Criteria

1. WHEN the journal stage runs, THE Daily_Research_Pipeline SHALL load existing hypotheses from the Research_Journal using the existing `ResearchJournal` interface.
2. WHEN a Hypothesis Evidence_Score, status, or reason changes during the run, THE Daily_Research_Pipeline SHALL persist that change to the Research_Journal using the existing `ResearchJournal.update` interface.
3. WHERE a new rule-based Research_Idea qualifies as a trackable Hypothesis, THE Daily_Research_Pipeline SHALL add it to the Research_Journal using the existing `ResearchJournal.add` interface with status `open`.
4. THE Daily_Research_Pipeline SHALL set each updated Hypothesis status only to a value in `{open, testing, supported, rejected, inconclusive}`.
5. THE Daily_Research_Pipeline SHALL integrate with the existing Research_Journal and SHALL NOT introduce a separate hypothesis store.

### Requirement 13: Report Artifact Output

**User Story:** As a researcher, I want the daily report written as a concise self-contained file I can open each evening, so that I have a durable record per session.

#### Acceptance Criteria

1. WHEN the report stage completes, THE Daily_Research_Pipeline SHALL write the Daily_Research_Report as a self-contained file under the `reports/` directory.
2. THE Daily_Research_Pipeline SHALL name the Daily_Research_Report file so that the Target_Session is identifiable from the filename.
3. WHEN a Daily_Research_Report already exists for the Target_Session, THE Daily_Research_Pipeline SHALL overwrite it so that re-running the pipeline produces a single current report per session.
4. THE Daily_Research_Report SHALL contain all five required sections in a single artifact without requiring external resources to read.

### Requirement 14: Graceful Small-Sample Behavior

**User Story:** As a researcher, I want the pipeline to behave correctly when only a few sessions have been collected, so that early-stage output is honest and never misleading.

#### Acceptance Criteria

1. WHILE the Collected_Session_Count is between 1 and 5 inclusive, THE Daily_Research_Pipeline SHALL complete all stages and produce a Daily_Research_Report.
2. WHEN the Collected_Session_Count is below a configured minimum sample size, THE Daily_Research_Pipeline SHALL annotate comparison and Unusual_Event statements with the actual number of sessions used.
3. THE Daily_Research_Pipeline SHALL NOT claim statistical significance for any comparison computed from fewer sessions than the configured minimum sample size.
4. IF a metric requires more Prior_Sessions than are collected, THEN THE Daily_Research_Pipeline SHALL report that metric as based on the available sessions rather than failing.

### Requirement 15: Optional, Off-by-Default AI Enhancement for Research Ideas

**User Story:** As a researcher, I want AI-generated research ideas to remain disabled by default with the rule-based generator as the primary hypothesis source, so that the pipeline stays deterministic and trade-free while AI remains an optional assistant once many experiments have accumulated.

#### Acceptance Criteria

1. THE Daily_Research_Pipeline SHALL keep the AI_Enhancement disabled by default and SHALL generate Research_Ideas through the primary flow of rule-based generator, Research_Journal, Evidence, and manual review.
2. WHERE the AI_Enhancement is disabled, THE Daily_Research_Pipeline SHALL generate Research_Ideas using rule-based logic and SHALL complete the full run deterministically.
3. WHERE the operator explicitly enables the AI_Enhancement and the AI_Enhancement is available, THE Daily_Research_Pipeline SHALL use it to summarize accumulated experiments and propose additional investigations to supplement the rule-based Research_Ideas.
4. THE AI_Enhancement SHALL produce only research ideas, hypotheses, and explanations and SHALL NOT produce trade recommendations.
5. IF the AI_Enhancement is enabled but unavailable or fails, THEN THE Daily_Research_Pipeline SHALL fall back to rule-based Research_Idea generation and SHALL complete the run.

### Requirement 16: Reuse of Existing Frozen Components

**User Story:** As a maintainer, I want the pipeline to reuse the frozen v0.2 architecture, so that no major new architecture is introduced and the system stays maintainable.

#### Acceptance Criteria

1. THE Daily_Research_Pipeline SHALL read market data through the existing `Storage` interface.
2. THE Daily_Research_Pipeline SHALL compute features through the existing Feature_Engine and persist them through the existing Feature_Store.
3. THE Daily_Research_Pipeline SHALL compute option analytics through the existing Analytics_Module.
4. THE Daily_Research_Pipeline SHALL run health checks through the existing Validation_Engine.
5. THE Daily_Research_Pipeline SHALL record hypotheses through the existing Research_Journal.
6. THE Daily_Research_Pipeline SHALL be implemented as new orchestration code in `scripts/daily_research.py` and supporting `nifty_quant` modules without modifying the frozen component interfaces.

### Requirement 17: Determinism and Testability

**User Story:** As a maintainer, I want the pipeline to be deterministic and unit-tested with synthetic data, so that it fits the project's tested, reproducible engineering standard.

#### Acceptance Criteria

1. WHEN the Daily_Research_Pipeline is run twice on identical warehouse data, identical journal state, and the AI_Enhancement disabled, THE Daily_Research_Pipeline SHALL produce identical Evidence_Score updates and identical Daily_Research_Report content.
2. THE Daily_Research_Pipeline SHALL be unit-testable using synthetic Snapshot data without requiring live market access.
3. THE Daily_Research_Pipeline SHALL expose its stages such that each stage can be tested independently with synthetic inputs.

### Requirement 18: Roadmap Single-Source-of-Truth Control Center

**User Story:** As a maintainer, I want a single ROADMAP.md control center capturing architecture state, data inventory, hypothesis status, top experiments, priorities, and milestones, so that the project state is recorded in one authoritative place.

#### Acceptance Criteria

1. THE Roadmap_Document SHALL record the Architecture Version, the Frozen Components, the Current Dataset, the Collected Sessions count, the Collected Option Chains count, the Collected Trading Days count, the Data Quality, the Open Hypotheses, the Supported Hypotheses, the Rejected Hypotheses, the Top Performing Experiments, the Research Priorities, and the Next Milestones.
2. WHEN a Hypothesis status changes to `supported` or `rejected` during a pipeline run, THE Daily_Research_Pipeline SHALL update the Open Hypotheses, Supported Hypotheses, and Rejected Hypotheses listed in the Roadmap_Document to reflect the change.
3. WHEN a pipeline run completes, THE Daily_Research_Pipeline SHALL refresh the Current Dataset, Collected Sessions count, Collected Option Chains count, Collected Trading Days count, and Data Quality entries in the Roadmap_Document from the current Warehouse and Data_Quality_Gate state.
4. THE Roadmap_Document SHALL be maintained as a single file at the repository root named `ROADMAP.md`.

### Requirement 19: Pre-Flight Data Quality Gate

**User Story:** As a researcher, I want a data quality gate to validate the day's input data before any research begins, so that I never build analysis on broken, missing, or misaligned data.

#### Acceptance Criteria

1. WHEN the Daily_Research_Pipeline starts a run, THE Data_Quality_Gate SHALL execute before the read stage and before any research computation begins.
2. WHEN the Data_Quality_Gate runs, THE Data_Quality_Gate SHALL perform Quality_Checks that detect API-outage or missing-candle conditions, duplicate timestamps, expiry mismatches, holiday or non-trading-day conditions, and timezone anomalies in the Target_Session input data.
3. WHEN the Quality_Checks complete, THE Data_Quality_Gate SHALL compute a Quality_Score in the range 0 to 100 inclusive and SHALL assign a verdict of PASS or FAIL for the Target_Session.
4. IF the Data_Quality_Gate verdict is FAIL, THEN THE Daily_Research_Pipeline SHALL halt before the read stage, SHALL report each failing Quality_Check, and SHALL exit with a non-zero status code.
5. WHEN the Data_Quality_Gate verdict is PASS, THE Daily_Research_Pipeline SHALL proceed to the read stage and SHALL include the Quality_Score in the Daily_Research_Report.

### Requirement 20: Report Market Regime Summary

**User Story:** As a researcher, I want every report to open with a prominent market regime summary, so that I immediately understand the prevailing conditions before reading any detail.

#### Acceptance Criteria

1. THE Daily_Research_Report SHALL include a Market_Regime section presented as the first section of the report, above the "What happened today" section.
2. WHEN the Market_Regime section is produced, THE Daily_Research_Report SHALL state the regime classification including the trend regime, the volatility regime, the gamma regime, and the PCR level for the Target_Session.
3. THE Daily_Research_Pipeline SHALL obtain the Market_Regime by reusing the project's existing regime detection and SHALL NOT reimplement regime-classification logic.
4. IF a Market_Regime classification cannot be computed from the available data, THEN THE Daily_Research_Report SHALL mark that classification as unavailable rather than omitting it silently.

### Requirement 21: Overall Research Confidence Score

**User Story:** As a researcher, I want a single overall confidence score with its contributing factors, so that I know how much to trust each day's research output at a glance.

#### Acceptance Criteria

1. THE Daily_Research_Report SHALL include a single Research_Confidence_Score in the range 0 to 100 inclusive for the Target_Session.
2. WHEN the Research_Confidence_Score is computed, THE Daily_Research_Pipeline SHALL combine the data quality, the sample size, the feature drift, the validation alerts, and the evidence maturity into that score.
3. WHEN the Research_Confidence_Score is reported, THE Daily_Research_Report SHALL show the contributing factors that determined the score.
4. WHEN the Daily_Research_Pipeline is run twice on identical warehouse data, identical journal state, and the AI_Enhancement disabled, THE Daily_Research_Pipeline SHALL produce an identical Research_Confidence_Score.

### Requirement 22: Per-Stage Computational Cost Logging

**User Story:** As a maintainer, I want each pipeline stage to log its computational cost, so that I can analyze and optimize the pipeline's performance over time.

#### Acceptance Criteria

1. WHEN a pipeline stage completes, THE Daily_Research_Pipeline SHALL log the Computational_Cost for that stage, comprising its execution time, its memory usage, and the number of rows processed.
2. THE Daily_Research_Pipeline SHALL capture the Computational_Cost of every stage for each run so that the metrics are retained for later analysis.
3. WHEN the Daily_Research_Report is produced, THE Daily_Research_Report SHALL include the per-stage Computational_Cost metrics for the run.

### Requirement 23: Experiment Registry Expansion

**User Story:** As a researcher, I want each experiment record to capture full provenance and outcome, so that my work is reproducible and not repeated months later.

#### Acceptance Criteria

1. WHEN the Daily_Research_Pipeline records an Experiment_Record in the Research_Journal, THE Daily_Research_Pipeline SHALL capture the Research Question, the Hypothesis, the Dataset Version, the Feature Version, the Code Version, the Result, the Decision, and the Next Action for that record.
2. THE Daily_Research_Pipeline SHALL persist the Experiment_Record fields through the existing Research_Journal interface and SHALL NOT introduce a separate experiment store.
3. WHEN an Experiment_Record is reloaded from the Research_Journal, THE Daily_Research_Pipeline SHALL recover the identical Research Question, Hypothesis, Dataset Version, Feature Version, Code Version, Result, Decision, and Next Action that were persisted.
