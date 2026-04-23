"use client";

import { ReactNode } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

// ─── MetricCard ─────────────────────────────────────────────────────

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: LucideIcon;
  trend?: "up" | "down" | "neutral";
  trendValue?: string;
  variant?: "default" | "accent" | "success" | "warning" | "danger";
  className?: string;
}

export function MetricCard({
  title, value, subtitle, icon: Icon, trend, trendValue, variant = "default", className,
}: MetricCardProps) {
  const variantStyles = {
    default: "bg-[var(--color-surface)] border-[var(--color-border)]",
    accent: "bg-[var(--color-accent-muted)] border-[var(--color-accent)]/20",
    success: "bg-[var(--color-success-muted)] border-[var(--color-success)]/20",
    warning: "bg-[var(--color-warning-muted)] border-[var(--color-warning)]/20",
    danger: "bg-[var(--color-danger-muted)] border-[var(--color-danger)]/20",
  };

  const trendColors = {
    up: "text-[var(--color-success)]",
    down: "text-[var(--color-danger)]",
    neutral: "text-[var(--color-text-muted)]",
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "rounded-xl border p-5 transition-all duration-200 hover:border-[var(--color-accent)]/30",
        variantStyles[variant],
        className
      )}
    >
      <div className="flex items-start justify-between mb-3">
        <span className="text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)]">
          {title}
        </span>
        {Icon && (
          <div className="w-8 h-8 rounded-lg bg-[var(--color-surface-elevated)] flex items-center justify-center">
            <Icon className="w-4 h-4 text-[var(--color-text-secondary)]" />
          </div>
        )}
      </div>
      <div className="text-2xl font-bold tracking-tight">{value}</div>
      <div className="flex items-center gap-2 mt-1.5">
        {trend && trendValue && (
          <span className={cn("text-xs font-medium", trendColors[trend])}>
            {trend === "up" ? "↑" : trend === "down" ? "↓" : "→"} {trendValue}
          </span>
        )}
        {subtitle && (
          <span className="text-xs text-[var(--color-text-muted)]">{subtitle}</span>
        )}
      </div>
    </motion.div>
  );
}

// ─── SectionCard ────────────────────────────────────────────────────

interface SectionCardProps {
  title: string;
  description?: string;
  icon?: LucideIcon;
  badge?: string;
  badgeVariant?: "default" | "success" | "warning" | "danger";
  children: ReactNode;
  className?: string;
}

export function SectionCard({
  title, description, icon: Icon, badge, badgeVariant = "default", children, className,
}: SectionCardProps) {
  const badgeColors = {
    default: "bg-[var(--color-surface-elevated)] text-[var(--color-text-secondary)]",
    success: "bg-[var(--color-success-muted)] text-[var(--color-success)]",
    warning: "bg-[var(--color-warning-muted)] text-[var(--color-warning)]",
    danger: "bg-[var(--color-danger-muted)] text-[var(--color-danger)]",
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] overflow-hidden",
        className
      )}
    >
      <div className="px-6 py-4 border-b border-[var(--color-border)] flex items-center gap-3">
        {Icon && (
          <div className="w-8 h-8 rounded-lg bg-[var(--color-accent-muted)] flex items-center justify-center shrink-0">
            <Icon className="w-4 h-4 text-[var(--color-accent)]" />
          </div>
        )}
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold">{title}</h3>
          {description && (
            <p className="text-xs text-[var(--color-text-muted)] mt-0.5 truncate">{description}</p>
          )}
        </div>
        {badge && (
          <span className={cn("px-2.5 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider", badgeColors[badgeVariant])}>
            {badge}
          </span>
        )}
      </div>
      <div className="p-6">{children}</div>
    </motion.div>
  );
}

// ─── InsightCard ────────────────────────────────────────────────────

interface InsightCardProps {
  insight: string;
  index: number;
}

export function InsightCard({ insight, index }: InsightCardProps) {
  const isProxy = insight.toLowerCase().includes("proxy");
  const isIntersectional = insight.toLowerCase().includes("intersectional");
  const isDisparity = insight.toLowerCase().includes("disparity");

  let color = "border-[var(--color-warning)]/30 bg-[var(--color-warning-muted)]";
  if (isProxy || isIntersectional)
    color = "border-[var(--color-danger)]/30 bg-[var(--color-danger-muted)]";
  else if (!isDisparity)
    color = "border-[var(--color-info)]/20 bg-[var(--color-info-muted)]";

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.05 }}
      className={cn("flex items-start gap-3 px-4 py-3 rounded-lg border text-sm", color)}
    >
      <span className="text-[var(--color-text-muted)] font-mono text-xs mt-0.5 shrink-0">
        {String(index + 1).padStart(2, "0")}
      </span>
      <span className="text-[var(--color-text-secondary)] leading-relaxed">{insight}</span>
    </motion.div>
  );
}

// ─── GlassCard ──────────────────────────────────────────────────────

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  hover?: boolean;
}

export function GlassCard({ children, className, hover = false }: GlassCardProps) {
  return (
    <div className={cn("glass rounded-xl p-6", hover && "glass-hover cursor-pointer", className)}>
      {children}
    </div>
  );
}
