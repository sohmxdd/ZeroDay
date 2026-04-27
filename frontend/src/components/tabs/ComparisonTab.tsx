"use client";

import { MetricCard, SectionCard } from "@/components/cards/Cards";
import { ComparisonBar } from "@/components/charts/ComparisonBar";
import { cn, formatPercent } from "@/lib/utils";
import { FileText, TrendingUp, Activity, Users, BarChart3, GitCompare } from "lucide-react";

interface ComparisonTabProps {
  comparison: any;
}

export function ComparisonTab({ comparison }: ComparisonTabProps) {
  if (!comparison || !comparison.baseline_stats || !comparison.debiased_stats) {
    return (
      <div className="p-12 text-center rounded-2xl border border-dashed border-[var(--color-border)]">
        <GitCompare className="w-8 h-8 text-[var(--color-text-muted)] mx-auto mb-4 opacity-20" />
        <p className="text-sm text-[var(--color-text-secondary)] font-medium">No comparison data available.</p>
        <p className="text-xs text-[var(--color-text-muted)] mt-1">Ensure the full pipeline was executed with successful mitigation.</p>
      </div>
    );
  }

  const hasShifts = Object.keys(comparison.representation_shift || {}).length > 0 ||
    Object.keys(comparison.outcome_rate_change || {}).length > 0;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard title="Baseline Dataset" value={comparison.baseline_stats.shape?.[0]?.toLocaleString() || "0"} subtitle={`${comparison.baseline_stats.shape?.[1] || 0} features`} icon={FileText} />
        <MetricCard title="Debiased Dataset" value={comparison.debiased_stats.shape?.[0]?.toLocaleString() || "0"} subtitle={`${comparison.debiased_stats.shape?.[1] || 0} features`} icon={FileText} variant="success" />
        <MetricCard title="Attributes Improved" value={comparison.summary?.features_improved?.length || 0} subtitle={comparison.summary?.features_improved?.join(", ") || "None"} icon={TrendingUp} variant="success" />
        <MetricCard title="Significant Shifts" value={`${comparison.summary?.statistically_significant_changes || 0}/${comparison.summary?.statistical_tests_run || 0}`} subtitle="Distribution changes detected" icon={Activity} variant="accent" />
      </div>

      {!hasShifts && (
        <div className="p-8 text-center rounded-xl bg-[var(--color-surface)] border border-[var(--color-border)]">
          <Users className="w-6 h-6 text-[var(--color-text-muted)] mx-auto mb-3 opacity-30" />
          <p className="text-xs text-[var(--color-text-secondary)]">No sensitive feature shifts were detected.</p>
        </div>
      )}

      <div className="grid lg:grid-cols-2 gap-6">
        {Object.entries(comparison.representation_shift || {}).map(([feat, shift]: [string, any]) => (
          <SectionCard key={feat} title={`Representation Shift: ${feat}`} icon={Users} badge={shift.imbalance_improved ? "Improved" : "Worsened"} badgeVariant={shift.imbalance_improved ? "success" : "warning"}>
            <ComparisonBar data={shift.groups} labelBefore="Original" labelAfter="Debiased" />
          </SectionCard>
        ))}
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {Object.entries(comparison.outcome_rate_change || {}).map(([feat, change]: [string, any]) => (
          <SectionCard key={feat} title={`Outcome Rate Shift: ${feat}`} icon={BarChart3} badge={change.disparity_improved ? "Improved" : "Worsened"} badgeVariant={change.disparity_improved ? "success" : "warning"}>
            <ComparisonBar data={change.groups} labelBefore="Original" labelAfter="Debiased" />
            <div className="flex items-center justify-between mt-4 pt-3 border-t border-[var(--color-border)]">
              <span className="text-xs text-[var(--color-text-muted)]">Disparity Before</span>
              <span className="text-xs font-mono">{formatPercent(change.disparity_before)}</span>
              <span className="text-xs text-[var(--color-text-muted)]">Disparity After</span>
              <span className={cn("text-xs font-mono", change.disparity_improved ? "text-[var(--color-success)]" : "text-[var(--color-danger)]")}>{formatPercent(change.disparity_after)}</span>
            </div>
          </SectionCard>
        ))}
      </div>

      {Object.keys(comparison.statistical_tests || {}).length > 0 && (
        <SectionCard title="Statistical Distribution Tests (KS)" icon={Activity} description="Kolmogorov-Smirnov tests comparing baseline vs. debiased feature distributions">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[var(--color-border)]">
                  {["Feature", "KS Statistic", "P-Value", "Significant"].map((h, i) => (
                    <th key={h} className={cn("py-3 px-3 text-[10px] font-bold uppercase tracking-widest text-[var(--color-text-muted)]", i === 0 ? "text-left" : "text-right")}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(comparison.statistical_tests).map(([feat, test]: [string, any]) => (
                  <tr key={feat} className="border-b border-[var(--color-border-subtle)] hover:bg-[var(--color-surface-hover)] transition-colors">
                    <td className="py-3 px-3 font-medium">{feat}</td>
                    <td className="py-3 px-3 text-right font-mono text-xs">{test.ks_statistic?.toFixed(4) || "0.0000"}</td>
                    <td className="py-3 px-3 text-right font-mono text-xs">{test.p_value?.toFixed(4) || "0.0000"}</td>
                    <td className="py-3 px-3 text-right">
                      <span className={cn("px-2.5 py-0.5 rounded-full text-[10px] font-bold", test.significantly_different ? "bg-[var(--color-warning-muted)] text-[var(--color-warning)]" : "bg-[var(--color-success-muted)] text-[var(--color-success)]")}>{test.significantly_different ? "YES" : "NO"}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}
    </div>
  );
}
