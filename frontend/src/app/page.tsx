"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import {
  Shield, ArrowRight, Search, Wrench, GitCompare, Brain,
  ChevronRight, Zap, Lock, BarChart3,
} from "lucide-react";

const FEATURES = [
  {
    icon: Search,
    title: "Bias Detection",
    desc: "Scan datasets for distribution, outcome, proxy, and intersectional biases across protected attributes.",
  },
  {
    icon: Wrench,
    title: "Bias Mitigation",
    desc: "Automatically evaluate 10+ strategies including resampling, reweighting, and fairlearn reduction.",
  },
  {
    icon: GitCompare,
    title: "Dataset Comparison",
    desc: "Compare baseline vs debiased datasets with statistical tests and representation shift analysis.",
  },
  {
    icon: Brain,
    title: "AI Explainability",
    desc: "Gemini-powered explanations of model behavior, feature importance, and fairness tradeoffs.",
  },
];

const STATS = [
  { value: "10+", label: "Mitigation Strategies" },
  { value: "5", label: "Bias Dimensions" },
  { value: "4", label: "Pipeline Phases" },
  { value: "< 30s", label: "Full Analysis" },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-[var(--color-background)] relative overflow-hidden">
      {/* Background grid */}
      <div className="absolute inset-0 animate-grid" />
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[600px] bg-[var(--color-accent)] opacity-[0.03] blur-[120px] rounded-full" />

      {/* Header */}
      <header className="relative z-10 max-w-7xl mx-auto px-6 py-6 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-xl bg-[var(--color-accent)] flex items-center justify-center">
            <Shield className="w-5 h-5 text-white" />
          </div>
          <span className="text-xl font-bold tracking-tight">AEGIS</span>
        </div>
        <Link
          href="/upload"
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-[var(--color-accent)] hover:bg-[var(--color-accent-hover)] text-white text-sm font-medium transition-colors"
        >
          Start Analysis <ArrowRight className="w-4 h-4" />
        </Link>
      </header>

      {/* Hero */}
      <section className="relative z-10 max-w-7xl mx-auto px-6 pt-24 pb-20">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="text-center max-w-3xl mx-auto"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 }}
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-[var(--color-accent)]/20 bg-[var(--color-accent-muted)] text-[var(--color-accent)] text-xs font-medium mb-8"
          >
            <Zap className="w-3.5 h-3.5" />
            Powered by Gemini AI
          </motion.div>

          <h1 className="text-5xl md:text-7xl font-bold tracking-tight leading-[1.1] mb-6">
            <span className="gradient-text">Detect, Fix &</span>
            <br />
            <span className="text-[var(--color-text-primary)]">Explain Bias in AI</span>
          </h1>

          <p className="text-lg text-[var(--color-text-secondary)] max-w-xl mx-auto mb-10 leading-relaxed text-balance">
            An enterprise-grade AI auditing platform that ensures fairness
            and accountability in machine learning systems.
          </p>

          <div className="flex items-center justify-center gap-4">
            <Link
              href="/upload"
              className="group flex items-center gap-2 px-8 py-3.5 rounded-xl bg-[var(--color-accent)] hover:bg-[var(--color-accent-hover)] text-white font-medium transition-all glow"
            >
              Start Analysis
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              href="/results"
              className="flex items-center gap-2 px-8 py-3.5 rounded-xl border border-[var(--color-border)] hover:border-[var(--color-accent)]/30 text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] font-medium transition-all"
            >
              View Demo
              <ChevronRight className="w-4 h-4" />
            </Link>
          </div>
        </motion.div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-2xl mx-auto mt-20"
        >
          {STATS.map((stat, i) => (
            <div key={i} className="text-center">
              <div className="text-2xl font-bold gradient-text">{stat.value}</div>
              <div className="text-xs text-[var(--color-text-muted)] mt-1">{stat.label}</div>
            </div>
          ))}
        </motion.div>
      </section>

      {/* Features */}
      <section className="relative z-10 max-w-7xl mx-auto px-6 pb-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="text-center mb-14"
        >
          <h2 className="text-2xl font-bold mb-3">Four-Phase Pipeline</h2>
          <p className="text-sm text-[var(--color-text-muted)]">
            End-to-end bias governance from detection to explainability
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          {FEATURES.map((feat, i) => {
            const Icon = feat.icon;
            return (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 + i * 0.1 }}
                className="group glass glass-hover rounded-xl p-6 transition-all duration-300"
              >
                <div className="w-10 h-10 rounded-xl bg-[var(--color-accent-muted)] flex items-center justify-center mb-4 group-hover:bg-[var(--color-accent)] transition-colors">
                  <Icon className="w-5 h-5 text-[var(--color-accent)] group-hover:text-white transition-colors" />
                </div>
                <h3 className="text-sm font-semibold mb-2">{feat.title}</h3>
                <p className="text-xs text-[var(--color-text-muted)] leading-relaxed">
                  {feat.desc}
                </p>
              </motion.div>
            );
          })}
        </div>
      </section>

      {/* Trust bar */}
      <section className="relative z-10 border-t border-[var(--color-border)] py-10">
        <div className="max-w-7xl mx-auto px-6 flex items-center justify-center gap-8 text-xs text-[var(--color-text-muted)]">
          <div className="flex items-center gap-2">
            <Lock className="w-3.5 h-3.5" />
            SOC 2 Compliant Architecture
          </div>
          <div className="w-px h-4 bg-[var(--color-border)]" />
          <div className="flex items-center gap-2">
            <BarChart3 className="w-3.5 h-3.5" />
            UCI Adult Dataset Validated
          </div>
          <div className="w-px h-4 bg-[var(--color-border)]" />
          <div className="flex items-center gap-2">
            <Shield className="w-3.5 h-3.5" />
            Deterministic Decisions
          </div>
        </div>
      </section>
    </div>
  );
}
