"use client";

import { motion } from "framer-motion";
import { ComparisonRow } from "@/lib/types";
import { cn } from "@/lib/utils";

interface RankingTableProps {
  data: ComparisonRow[];
}

export function RankingTable({ data }: RankingTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-[var(--color-border)]">
            <th className="text-left py-3 px-3 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Rank</th>
            <th className="text-left py-3 px-3 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Strategy</th>
            <th className="text-right py-3 px-3 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Score</th>
            <th className="text-right py-3 px-3 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Accuracy</th>
            <th className="text-right py-3 px-3 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">DP Diff</th>
            <th className="text-right py-3 px-3 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">DI Ratio</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <motion.tr
              key={row.pipeline}
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.03 }}
              className={cn(
                "border-b border-[var(--color-border-subtle)] transition-colors hover:bg-[var(--color-surface-hover)]",
                row.rank === 1 && "bg-[var(--color-success-muted)]"
              )}
            >
              <td className="py-3 px-3">
                <span className={cn(
                  "inline-flex items-center justify-center w-6 h-6 rounded-md text-xs font-bold",
                  row.rank === 1
                    ? "bg-[var(--color-success)] text-white"
                    : row.strategy === "baseline"
                    ? "bg-[var(--color-danger-muted)] text-[var(--color-danger)]"
                    : "bg-[var(--color-surface-elevated)] text-[var(--color-text-muted)]"
                )}>
                  {row.rank}
                </span>
              </td>
              <td className="py-3 px-3">
                <span className="font-medium text-[var(--color-text-primary)]">{row.pipeline}</span>
              </td>
              <td className="py-3 px-3 text-right font-mono text-xs text-[var(--color-accent)]">
                {row.score.toFixed(4)}
              </td>
              <td className="py-3 px-3 text-right font-mono text-xs text-[var(--color-text-secondary)]">
                {(row.accuracy * 100).toFixed(1)}%
              </td>
              <td className="py-3 px-3 text-right font-mono text-xs">
                <span className={cn(
                  row.demographic_parity_diff < 0.05
                    ? "text-[var(--color-success)]"
                    : row.demographic_parity_diff < 0.15
                    ? "text-[var(--color-warning)]"
                    : "text-[var(--color-danger)]"
                )}>
                  {row.demographic_parity_diff.toFixed(4)}
                </span>
              </td>
              <td className="py-3 px-3 text-right font-mono text-xs text-[var(--color-text-muted)]">
                {row.disparate_impact.toFixed(4)}
              </td>
            </motion.tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
