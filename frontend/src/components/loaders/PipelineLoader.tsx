"use client";

import { motion } from "framer-motion";
import { Shield, Search, Wrench, GitCompare, Brain, CheckCircle } from "lucide-react";
import { cn } from "@/lib/utils";

const STEPS = [
  { id: 0, label: "Loading Data", icon: Shield, description: "Preprocessing dataset..." },
  { id: 1, label: "Detecting Bias", icon: Search, description: "Scanning for disparities..." },
  { id: 2, label: "Mitigating", icon: Wrench, description: "Training candidate strategies..." },
  { id: 3, label: "Comparing", icon: GitCompare, description: "Evaluating before & after..." },
  { id: 4, label: "Explaining", icon: Brain, description: "Generating AI explanations..." },
  { id: 5, label: "Complete", icon: CheckCircle, description: "Pipeline finished!" },
];

interface PipelineLoaderProps {
  currentStep: number;
  className?: string;
}

export function PipelineLoader({ currentStep, className }: PipelineLoaderProps) {
  return (
    <div className={cn("w-full max-w-2xl mx-auto", className)}>
      {/* Progress bar */}
      <div className="relative mb-10">
        <div className="h-1 bg-[var(--color-surface-elevated)] rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-[var(--color-accent)] to-[var(--color-accent-hover)] rounded-full"
            initial={{ width: "0%" }}
            animate={{ width: `${Math.min((currentStep / (STEPS.length - 1)) * 100, 100)}%` }}
            transition={{ duration: 0.6, ease: "easeInOut" }}
          />
        </div>
      </div>

      {/* Steps */}
      <div className="space-y-3">
        {STEPS.map((step) => {
          const Icon = step.icon;
          const isActive = step.id === currentStep;
          const isDone = step.id < currentStep;
          const isPending = step.id > currentStep;

          return (
            <motion.div
              key={step.id}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: step.id * 0.08 }}
              className={cn(
                "flex items-center gap-4 px-5 py-3.5 rounded-xl border transition-all duration-300",
                isActive && "border-[var(--color-accent)]/40 bg-[var(--color-accent-muted)] glow-sm",
                isDone && "border-[var(--color-success)]/20 bg-[var(--color-success-muted)]",
                isPending && "border-[var(--color-border-subtle)] bg-[var(--color-surface)] opacity-40"
              )}
            >
              <div className={cn(
                "w-9 h-9 rounded-lg flex items-center justify-center shrink-0",
                isActive && "bg-[var(--color-accent)] text-white",
                isDone && "bg-[var(--color-success)] text-white",
                isPending && "bg-[var(--color-surface-elevated)] text-[var(--color-text-muted)]"
              )}>
                {isDone ? (
                  <CheckCircle className="w-4.5 h-4.5" />
                ) : (
                  <Icon className="w-4.5 h-4.5" />
                )}
              </div>

              <div className="flex-1 min-w-0">
                <p className={cn(
                  "text-sm font-medium",
                  isActive && "text-[var(--color-text-primary)]",
                  isDone && "text-[var(--color-success)]",
                  isPending && "text-[var(--color-text-muted)]"
                )}>
                  {step.label}
                </p>
                <p className="text-xs text-[var(--color-text-muted)] mt-0.5">
                  {step.description}
                </p>
              </div>

              {isActive && (
                <div className="flex gap-1">
                  {[0, 1, 2].map((i) => (
                    <motion.div
                      key={i}
                      className="w-1.5 h-1.5 rounded-full bg-[var(--color-accent)]"
                      animate={{ opacity: [0.2, 1, 0.2] }}
                      transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }}
                    />
                  ))}
                </div>
              )}

              {isDone && (
                <span className="text-[10px] font-medium text-[var(--color-success)] uppercase tracking-wider">
                  Done
                </span>
              )}
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
