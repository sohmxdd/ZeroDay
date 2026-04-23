"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  Shield, Target, Scale, Sparkles, TrendingUp, AlertTriangle,
  Search, Wrench, GitCompare, Brain, Download, ChevronRight,
  BarChart3, Activity, Users, FileText, Lightbulb, Info,
} from "lucide-react";
import Navbar from "@/components/layout/Navbar";
import { MetricCard, SectionCard, InsightCard } from "@/components/cards/Cards";
import { DistributionChart } from "@/components/charts/DistributionChart";
import { ComparisonBar } from "@/components/charts/ComparisonBar";
import { FeatureImportance } from "@/components/charts/FeatureImportance";
import { TradeoffChart } from "@/components/charts/TradeoffChart";
import { RankingTable } from "@/components/tables/RankingTable";
import { MOCK_RESULT } from "@/lib/mock-data";
import { cn, formatPercent } from "@/lib/utils";

const TABS = [
  { id: "overview", label: "Overview", icon: BarChart3 },
  { id: "detection", label: "Bias Detection", icon: Search },
  { id: "mitigation", label: "Mitigation", icon: Wrench },
  { id: "comparison", label: "Comparison", icon: GitCompare },
  { id: "explainability", label: "Explainability", icon: Brain },
];

export default function ResultsPage() {
  const [activeTab, setActiveTab] = useState("overview");
  const data = MOCK_RESULT;
  const { bias_report, dataset_comparison, bias_tags } = data.dataset_analysis;
  const { ranking, explainability } = data.model_analysis;
  const { explanations, metadata } = data;

  return (
    <div className="min-h-screen bg-[var(--color-background)]">
      <Navbar />

      <main className="max-w-7xl mx-auto px-6 pt-24 pb-20">
        {/* Page Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-start justify-between mb-8"
        >
          <div>
            <h1 className="text-2xl font-bold mb-1">Analysis Results</h1>
            <p className="text-sm text-[var(--color-text-muted)]">
              UCI Adult Dataset &middot; {metadata.mode.replace("_", " ")} &middot; {metadata.elapsed_seconds}s
            </p>
          </div>
          <button className="flex items-center gap-2 px-5 py-2.5 rounded-xl border border-[var(--color-border)] hover:border-[var(--color-accent)]/30 text-sm font-medium transition-colors">
            <Download className="w-4 h-4" /> Export Report
          </button>
        </motion.div>

        {/* Tabs */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="flex items-center gap-1 p-1 rounded-xl bg-[var(--color-surface)] border border-[var(--color-border)] mb-8 overflow-x-auto"
        >
          {TABS.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  "relative flex items-center gap-2 px-4 py-2.5 rounded-lg text-xs font-medium whitespace-nowrap transition-colors",
                  isActive ? "text-[var(--color-text-primary)]" : "text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)]"
                )}
              >
                {isActive && (
                  <motion.div
                    layoutId="activeTab"
                    className="absolute inset-0 bg-[var(--color-accent-muted)] rounded-lg"
                    transition={{ type: "spring", stiffness: 400, damping: 30 }}
                  />
                )}
                <Icon className="w-3.5 h-3.5 relative z-10" />
                <span className="relative z-10">{tab.label}</span>
              </button>
            );
          })}
        </motion.div>

        {/* Tab Content */}
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2 }}
        >
          {activeTab === "overview" && <OverviewTab data={data} />}
          {activeTab === "detection" && <DetectionTab biasReport={bias_report} />}
          {activeTab === "mitigation" && <MitigationTab ranking={ranking} explanations={explanations} />}
          {activeTab === "comparison" && <ComparisonTab comparison={dataset_comparison} />}
          {activeTab === "explainability" && <ExplainabilityTab explainability={explainability} explanations={explanations} />}
        </motion.div>
      </main>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
// TAB: Overview
// ═══════════════════════════════════════════════════════════════════

function OverviewTab({ data }: { data: typeof MOCK_RESULT }) {
  const { ranking, explainability } = data.model_analysis;
  const { bias_report, bias_tags } = data.dataset_analysis;
  const biasCount = Object.values(bias_tags).filter(Boolean).length;

  return (
    <div className="space-y-6">
      {/* Metric Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Accuracy"
          value={formatPercent(ranking.ranking_table[0]?.accuracy || 0)}
          subtitle="Best model"
          icon={Target}
          trend="up"
          trendValue="+4.85%"
          variant="accent"
        />
        <MetricCard
          title="Fairness Score"
          value={ranking.ranking_table[0]?.demographic_parity_diff.toFixed(4) || "N/A"}
          subtitle="DP Difference"
          icon={Scale}
          trend="down"
          trendValue="0.0000"
          variant="success"
        />
        <MetricCard
          title="Bias Detected"
          value={`${biasCount} types`}
          subtitle={`${bias_report.insights.length} insights`}
          icon={AlertTriangle}
          variant="warning"
        />
        <MetricCard
          title="Strategy Used"
          value="Resampling"
          subtitle={`Score: ${ranking.best_score.toFixed(4)}`}
          icon={Sparkles}
          variant="default"
        />
      </div>

      {/* AI Summary */}
      <SectionCard title="AI-Generated Summary" icon={Lightbulb} badge={data.explanations.gemini_used ? "Gemini" : "Fallback"} badgeVariant="success">
        <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">
          {data.explanations.summary}
        </p>
      </SectionCard>

      {/* Strategy Ranking + Tradeoff */}
      <div className="grid lg:grid-cols-2 gap-6">
        <SectionCard title="Strategy Ranking" icon={TrendingUp} description="All evaluated mitigation strategies">
          <RankingTable data={ranking.ranking_table} />
        </SectionCard>
        <SectionCard title="Accuracy vs Fairness" icon={Activity} description="Strategy tradeoff visualization">
          <TradeoffChart data={ranking.ranking_table} />
        </SectionCard>
      </div>

      {/* Key Insights */}
      <SectionCard title="Key Insights" icon={Info} badge={`${bias_report.insights.length}`}>
        <div className="space-y-2">
          {bias_report.insights.slice(0, 5).map((insight, i) => (
            <InsightCard key={i} insight={insight} index={i} />
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
// TAB: Bias Detection (Phase 1)
// ═══════════════════════════════════════════════════════════════════

function DetectionTab({ biasReport }: { biasReport: typeof MOCK_RESULT.dataset_analysis.bias_report }) {
  const sensitiveFeatures = Object.keys(biasReport.distribution_bias);

  return (
    <div className="space-y-6">
      {/* Distribution Charts */}
      <div className="grid lg:grid-cols-2 gap-6">
        {sensitiveFeatures.map((feat) => (
          <SectionCard key={feat} title={`Distribution: ${feat}`} icon={Users} badge={`Ratio: ${biasReport.distribution_bias[feat].imbalance_ratio.toFixed(1)}`} badgeVariant={biasReport.distribution_bias[feat].imbalance_ratio > 5 ? "danger" : "warning"}>
            <DistributionChart data={biasReport.distribution_bias[feat].group_proportions} />
          </SectionCard>
        ))}
      </div>

      {/* Outcome Disparity */}
      <div className="grid lg:grid-cols-2 gap-6">
        {sensitiveFeatures.map((feat) => (
          <SectionCard key={feat} title={`Outcome Rates: ${feat}`} icon={BarChart3} badge={`Gap: ${formatPercent(biasReport.outcome_bias[feat]?.disparity || 0)}`} badgeVariant={
            (biasReport.outcome_bias[feat]?.disparity || 0) > 0.15 ? "danger" : "warning"
          }>
            <DistributionChart data={biasReport.outcome_bias[feat]?.outcome_rates || {}} color="#f59e0b" />
          </SectionCard>
        ))}
      </div>

      {/* Fairness Metrics Table */}
      <SectionCard title="Fairness Metrics" icon={Scale} description="Demographic parity and disparate impact per attribute">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--color-border)]">
                <th className="text-left py-3 px-3 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Feature</th>
                <th className="text-right py-3 px-3 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">DP Difference</th>
                <th className="text-right py-3 px-3 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Disparate Impact</th>
                <th className="text-right py-3 px-3 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Status</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(biasReport.fairness_metrics).map(([feat, metrics]) => (
                <tr key={feat} className="border-b border-[var(--color-border-subtle)]">
                  <td className="py-3 px-3 font-medium">{feat}</td>
                  <td className="py-3 px-3 text-right font-mono text-xs">
                    <span className={cn(
                      metrics.demographic_parity_difference > 0.1 ? "text-[var(--color-danger)]" : "text-[var(--color-success)]"
                    )}>
                      {metrics.demographic_parity_difference.toFixed(4)}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-right font-mono text-xs text-[var(--color-text-secondary)]">
                    {metrics.disparate_impact_ratio.toFixed(4)}
                  </td>
                  <td className="py-3 px-3 text-right">
                    <span className={cn(
                      "px-2 py-0.5 rounded-full text-[10px] font-semibold",
                      metrics.demographic_parity_difference > 0.15
                        ? "bg-[var(--color-danger-muted)] text-[var(--color-danger)]"
                        : metrics.demographic_parity_difference > 0.05
                        ? "bg-[var(--color-warning-muted)] text-[var(--color-warning)]"
                        : "bg-[var(--color-success-muted)] text-[var(--color-success)]"
                    )}>
                      {metrics.demographic_parity_difference > 0.15 ? "UNFAIR" : metrics.demographic_parity_difference > 0.05 ? "RISK" : "FAIR"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Advanced Bias */}
      <div className="grid lg:grid-cols-2 gap-6">
        <SectionCard title="Proxy Bias" icon={AlertTriangle} badge="Detected" badgeVariant="danger">
          {Object.entries(biasReport.advanced_bias.proxy_bias).map(([feature, info]) => (
            <div key={feature} className="flex items-center justify-between p-3 rounded-lg bg-[var(--color-danger-muted)] border border-[var(--color-danger)]/20">
              <div>
                <p className="text-sm font-medium">&apos;{feature}&apos; acts as proxy</p>
                <p className="text-xs text-[var(--color-text-muted)] mt-0.5">
                  Correlates with &apos;{info.correlated_with}&apos;
                </p>
              </div>
              <span className="font-mono text-sm text-[var(--color-danger)] font-bold">
                r = {info.correlation.toFixed(3)}
              </span>
            </div>
          ))}
        </SectionCard>
        <SectionCard title="Intersectional Bias" icon={Users} badge={`Max: ${formatPercent(biasReport.advanced_bias.intersectional_bias.max_disparity || 0)}`} badgeVariant="danger">
          <div className="space-y-2">
            {Object.entries(biasReport.advanced_bias.intersectional_bias.group_rates || {}).map(([group, rate]) => (
              <div key={group} className="flex items-center justify-between py-1.5 text-xs">
                <span className="text-[var(--color-text-secondary)]">{group}</span>
                <div className="flex items-center gap-2">
                  <div className="w-24 h-1.5 rounded-full bg-[var(--color-surface-elevated)] overflow-hidden">
                    <div className="h-full rounded-full bg-[var(--color-accent)]" style={{ width: `${rate * 100}%` }} />
                  </div>
                  <span className="font-mono text-[var(--color-text-muted)] w-12 text-right">{formatPercent(rate)}</span>
                </div>
              </div>
            ))}
          </div>
        </SectionCard>
      </div>

      {/* All Insights */}
      <SectionCard title="All Insights" icon={Info}>
        <div className="space-y-2">
          {biasReport.insights.map((insight, i) => (
            <InsightCard key={i} insight={insight} index={i} />
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
// TAB: Mitigation (Phase 2)
// ═══════════════════════════════════════════════════════════════════

function MitigationTab({ ranking, explanations }: { ranking: typeof MOCK_RESULT.model_analysis.ranking; explanations: typeof MOCK_RESULT.explanations }) {
  return (
    <div className="space-y-6">
      {/* Winner Card */}
      <div className="p-6 rounded-xl border border-[var(--color-success)]/30 bg-[var(--color-success-muted)]">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-10 h-10 rounded-xl bg-[var(--color-success)] flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <p className="text-xs text-[var(--color-success)] font-semibold uppercase tracking-wider">Best Strategy</p>
            <h3 className="text-lg font-bold capitalize">{ranking.best_strategy.replace("_", " ")}</h3>
          </div>
          <div className="ml-auto text-right">
            <p className="text-2xl font-bold text-[var(--color-success)]">{ranking.best_score.toFixed(4)}</p>
            <p className="text-xs text-[var(--color-text-muted)]">Tradeoff Score</p>
          </div>
        </div>
      </div>

      {/* Strategy Justification */}
      <SectionCard title="Strategy Justification" icon={Lightbulb} badge="Gemini" badgeVariant="success">
        <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">
          {explanations.strategy_justification}
        </p>
      </SectionCard>

      {/* Ranking + Tradeoff */}
      <div className="grid lg:grid-cols-2 gap-6">
        <SectionCard title="Full Strategy Ranking" icon={TrendingUp}>
          <RankingTable data={ranking.ranking_table} />
        </SectionCard>
        <SectionCard title="Tradeoff Visualization" icon={Activity}>
          <TradeoffChart data={ranking.ranking_table} />
        </SectionCard>
      </div>

      {/* Tradeoff Analysis */}
      <SectionCard title="Tradeoff Analysis" icon={Scale} badge="AI Analysis" badgeVariant="success">
        <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">
          {explanations.tradeoff_analysis}
        </p>
      </SectionCard>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
// TAB: Dataset Comparison (Phase 3)
// ═══════════════════════════════════════════════════════════════════

function ComparisonTab({ comparison }: { comparison: typeof MOCK_RESULT.dataset_analysis.dataset_comparison }) {
  return (
    <div className="space-y-6">
      {/* Dataset Size Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard title="Baseline Rows" value={comparison.baseline_stats.shape[0].toLocaleString()} subtitle={`${comparison.baseline_stats.shape[1]} columns`} icon={FileText} />
        <MetricCard title="Debiased Rows" value={comparison.debiased_stats.shape[0].toLocaleString()} subtitle={`${comparison.debiased_stats.shape[1]} columns`} icon={FileText} variant="success" />
        <MetricCard title="Features Improved" value={comparison.summary.features_improved.length} subtitle={comparison.summary.features_improved.join(", ") || "None"} icon={TrendingUp} variant="success" />
        <MetricCard title="Stat. Significant" value={`${comparison.summary.statistically_significant_changes}/${comparison.summary.statistical_tests_run}`} subtitle="distribution changes" icon={Activity} variant="accent" />
      </div>

      {/* Representation Shift Charts */}
      <div className="grid lg:grid-cols-2 gap-6">
        {Object.entries(comparison.representation_shift).map(([feat, shift]) => (
          <SectionCard key={feat} title={`Representation Shift: ${feat}`} icon={Users} badge={shift.imbalance_improved ? "Improved" : "Worsened"} badgeVariant={shift.imbalance_improved ? "success" : "warning"}>
            <ComparisonBar data={shift.groups} labelBefore="Original" labelAfter="Debiased" />
          </SectionCard>
        ))}
      </div>

      {/* Outcome Rate Changes */}
      <div className="grid lg:grid-cols-2 gap-6">
        {Object.entries(comparison.outcome_rate_change).map(([feat, change]) => (
          <SectionCard key={feat} title={`Outcome Rates: ${feat}`} icon={BarChart3} badge={change.disparity_improved ? "Improved" : "Worsened"} badgeVariant={change.disparity_improved ? "success" : "warning"}>
            <ComparisonBar data={change.groups} labelBefore="Original" labelAfter="Debiased" />
            <div className="flex items-center justify-between mt-4 pt-3 border-t border-[var(--color-border)]">
              <span className="text-xs text-[var(--color-text-muted)]">Disparity Before</span>
              <span className="text-xs font-mono">{formatPercent(change.disparity_before)}</span>
              <span className="text-xs text-[var(--color-text-muted)]">Disparity After</span>
              <span className={cn("text-xs font-mono", change.disparity_improved ? "text-[var(--color-success)]" : "text-[var(--color-danger)]")}>
                {formatPercent(change.disparity_after)}
              </span>
            </div>
          </SectionCard>
        ))}
      </div>

      {/* Statistical Tests */}
      <SectionCard title="Statistical Tests (KS)" icon={Activity} description="Kolmogorov-Smirnov tests on numeric features">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--color-border)]">
                <th className="text-left py-3 px-3 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Feature</th>
                <th className="text-right py-3 px-3 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">KS Statistic</th>
                <th className="text-right py-3 px-3 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">P-Value</th>
                <th className="text-right py-3 px-3 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Significant</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(comparison.statistical_tests).map(([feat, test]) => (
                <tr key={feat} className="border-b border-[var(--color-border-subtle)]">
                  <td className="py-3 px-3 font-medium">{feat}</td>
                  <td className="py-3 px-3 text-right font-mono text-xs">{test.ks_statistic?.toFixed(4)}</td>
                  <td className="py-3 px-3 text-right font-mono text-xs">{test.p_value?.toFixed(4)}</td>
                  <td className="py-3 px-3 text-right">
                    <span className={cn(
                      "px-2 py-0.5 rounded-full text-[10px] font-semibold",
                      test.significantly_different ? "bg-[var(--color-warning-muted)] text-[var(--color-warning)]" : "bg-[var(--color-success-muted)] text-[var(--color-success)]"
                    )}>
                      {test.significantly_different ? "YES" : "NO"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
// TAB: Explainability (Phase 4)
// ═══════════════════════════════════════════════════════════════════

function ExplainabilityTab({ explainability, explanations }: { explainability: typeof MOCK_RESULT.model_analysis.explainability; explanations: typeof MOCK_RESULT.explanations }) {
  const preds = explainability.predictions_analysis;

  return (
    <div className="space-y-6">
      {/* Prediction Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard title="Prediction Agreement" value={formatPercent(preds.prediction_agreement || 0)} subtitle="with baseline" icon={Target} variant="accent" />
        <MetricCard title="Predictions Changed" value={preds.predictions_changed?.toLocaleString() || "0"} subtitle={`of ${preds.baseline?.total.toLocaleString() || 0}`} icon={Activity} variant="warning" />
        <MetricCard title="Baseline +Rate" value={formatPercent(preds.baseline?.positive_rate || 0)} subtitle={`${preds.baseline?.positive_count.toLocaleString()} positive`} icon={BarChart3} />
        <MetricCard title="Mitigated +Rate" value={formatPercent(preds.best?.positive_rate || 0)} subtitle={`${preds.best?.positive_count.toLocaleString()} positive`} icon={BarChart3} variant="success" />
      </div>

      {/* Feature Importance */}
      <SectionCard title="Feature Importance" icon={BarChart3} description="Comparing feature weights: baseline vs mitigated model">
        <FeatureImportance baseline={explainability.feature_importance.baseline} best={explainability.feature_importance.best} />
      </SectionCard>

      {/* AI Explanation */}
      <SectionCard title="Why Bias Happened" icon={Brain} badge="Gemini" badgeVariant="success">
        <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">
          {explanations.bias_explanation}
        </p>
      </SectionCard>

      <SectionCard title="How It Was Fixed" icon={Wrench} badge="Gemini" badgeVariant="success">
        <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">
          {explanations.strategy_justification}
        </p>
      </SectionCard>

      {/* Model Explanation */}
      <SectionCard title="Model Behavior Analysis" icon={Lightbulb} badge="AI Generated" badgeVariant="success">
        <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">
          {explainability.explanation}
        </p>
      </SectionCard>

      {/* Recommendation */}
      <div className="p-5 rounded-xl border border-[var(--color-accent)]/30 bg-[var(--color-accent-muted)]">
        <div className="flex items-start gap-3">
          <Sparkles className="w-5 h-5 text-[var(--color-accent)] mt-0.5 shrink-0" />
          <div>
            <p className="text-xs font-semibold text-[var(--color-accent)] uppercase tracking-wider mb-1">Recommendation</p>
            <p className="text-sm text-[var(--color-text-secondary)]">{explanations.recommendation}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
