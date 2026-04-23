// ─── AEGIS Pipeline Types ───────────────────────────────────────────
// Mirrors the JSON output from core/pipeline.py

export interface AegisResult {
  dataset_analysis: DatasetAnalysis;
  model_analysis: ModelAnalysis;
  explanations: Explanations;
  metadata: PipelineMetadata;
}

// ── Dataset Analysis ────────────────────────────────────────────────

export interface DatasetAnalysis {
  bias_report: BiasReport;
  dataset_comparison: DatasetComparison;
  bias_tags: Record<string, boolean>;
}

export interface BiasReport {
  distribution_bias: Record<string, DistributionBias>;
  outcome_bias: Record<string, OutcomeBias>;
  fairness_metrics: Record<string, FairnessMetrics>;
  advanced_bias: AdvancedBias;
  insights: string[];
}

export interface DistributionBias {
  group_proportions: Record<string, number>;
  imbalance_ratio: number;
}

export interface OutcomeBias {
  outcome_rates: Record<string, number>;
  disparity: number;
}

export interface FairnessMetrics {
  demographic_parity_difference: number;
  disparate_impact_ratio: number;
}

export interface AdvancedBias {
  proxy_bias: Record<string, { correlation: number; correlated_with: string }>;
  intersectional_bias: {
    features?: string[];
    max_disparity?: number;
    group_rates?: Record<string, number>;
  };
  label_bias: Record<string, unknown>;
}

export interface DatasetComparison {
  baseline_stats: DatasetStats;
  debiased_stats: DatasetStats;
  representation_shift: Record<string, RepresentationShift>;
  outcome_rate_change: Record<string, OutcomeRateChange>;
  fairness_deltas: Record<string, FairnessDelta>;
  statistical_tests: Record<string, StatisticalTest>;
  summary: ComparisonSummary;
}

export interface DatasetStats {
  shape: [number, number];
  columns: string[];
  dtypes: Record<string, string>;
  missing_values: number;
  target_distribution: Record<string, number>;
  target_positive_rate: number;
}

export interface RepresentationShift {
  groups: Record<string, { before: number; after: number; delta: number }>;
  imbalance_before: number;
  imbalance_after: number;
  imbalance_improved: boolean;
}

export interface OutcomeRateChange {
  groups: Record<string, { before: number; after: number; delta: number }>;
  disparity_before: number;
  disparity_after: number;
  disparity_improved: boolean;
}

export interface FairnessDelta {
  before: { dp_diff: number; di_ratio: number };
  after: { dp_diff: number; di_ratio: number };
  dp_diff_delta: number;
  di_ratio_delta: number;
  dp_improved: boolean;
  di_improved: boolean;
}

export interface StatisticalTest {
  ks_statistic?: number;
  p_value?: number;
  significantly_different?: boolean;
  mean_before?: number;
  mean_after?: number;
  mean_delta?: number;
}

export interface ComparisonSummary {
  features_improved: string[];
  features_worsened: string[];
  dataset_size_change: { before: number; after: number };
  statistical_tests_run: number;
  statistically_significant_changes: number;
  overall_assessment: string;
}

// ── Model Analysis ──────────────────────────────────────────────────

export interface ModelAnalysis {
  model_output: ModelOutput;
  ranking: Ranking;
  explainability: Explainability;
}

export interface ModelOutput {
  best_strategy: string;
  accuracy_drop: number;
  comparison_table: ComparisonRow[];
  metrics_before_after: Record<string, unknown>;
  [key: string]: unknown;
}

export interface ComparisonRow {
  pipeline: string;
  strategy: string;
  rank: number;
  score: number;
  accuracy: number;
  demographic_parity_diff: number;
  disparate_impact: number;
  [key: string]: unknown;
}

export interface Ranking {
  best_strategy: string;
  best_score: number;
  ranking_table: ComparisonRow[];
}

export interface Explainability {
  feature_importance: {
    baseline: Record<string, number>;
    best: Record<string, number>;
  };
  model_comparison: {
    baseline_metrics: Record<string, unknown>;
    best_metrics: Record<string, unknown>;
    strategies_evaluated: number;
    comparison_table: ComparisonRow[];
  };
  predictions_analysis: {
    baseline?: PredictionStats;
    best?: PredictionStats;
    prediction_agreement?: number;
    predictions_changed?: number;
  };
  shap_summary: ShapSummary | null;
  explanation: string;
}

export interface PredictionStats {
  total: number;
  positive_count: number;
  negative_count: number;
  positive_rate: number;
}

export interface ShapSummary {
  top_features: Record<string, number>;
  sample_size: number;
  method: string;
}

// ── Explanations ────────────────────────────────────────────────────

export interface Explanations {
  summary: string;
  bias_explanation: string;
  strategy_justification: string;
  tradeoff_analysis: string;
  recommendation: string;
  gemini_used: boolean;
}

// ── Pipeline Metadata ───────────────────────────────────────────────

export interface PipelineMetadata {
  mode: string;
  strategy_used: string;
  timestamp: string;
  elapsed_seconds: number;
  config: {
    alpha: number;
    beta: number;
    model_type: string;
    gemini_enabled: boolean;
  };
}

// ── Pipeline Modes ──────────────────────────────────────────────────

export type PipelineMode = "analysis" | "train" | "full_pipeline";

export interface PipelineInput {
  dataset: File | null;
  mode: PipelineMode;
  model: File | null;
}
