"use client";

import { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Search, Wrench, GitCompare, Brain, Download, BarChart3,
  FileSpreadsheet, Box, FileText, FlaskConical, AlertTriangle,
} from "lucide-react";
import Navbar from "@/components/layout/Navbar";
import { OverviewTab } from "@/components/tabs/OverviewTab";
import { DetectionTab } from "@/components/tabs/DetectionTab";
import { MitigationTab } from "@/components/tabs/MitigationTab";
import { ComparisonTab } from "@/components/tabs/ComparisonTab";
import { ExplainabilityTab } from "@/components/tabs/ExplainabilityTab";
import { MOCK_RESULT } from "@/lib/mock-data";
import { downloadJSON, downloadPDF, downloadCSV } from "@/lib/exports";
import { cn } from "@/lib/utils";
import type { AegisResult } from "@/lib/types";

const TABS = [
  { id: "overview", label: "Overview", icon: BarChart3 },
  { id: "detection", label: "Bias Detection", icon: Search },
  { id: "mitigation", label: "Mitigation", icon: Wrench },
  { id: "comparison", label: "Comparison", icon: GitCompare },
  { id: "explainability", label: "Explainability", icon: Brain },
];

export default function ResultsPage() {
  const [activeTab, setActiveTab] = useState("overview");
  const [showExport, setShowExport] = useState(false);
  const exportRef = useRef<HTMLDivElement>(null);
  const [data, setData] = useState<AegisResult | null>(null);
  const [isDemo, setIsDemo] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Load data: real pipeline result from localStorage, or fallback to demo
  useEffect(() => {
    try {
      const stored = localStorage.getItem("aegis_result");
      const demoFlag = localStorage.getItem("aegis_demo");

      if (stored && !demoFlag) {
        setData(JSON.parse(stored));
        setIsDemo(false);
      } else {
        setData(MOCK_RESULT);
        setIsDemo(true);
      }
    } catch {
      setData(MOCK_RESULT);
      setIsDemo(true);
    }
    setIsLoading(false);
  }, []);

  // Close export dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (exportRef.current && !exportRef.current.contains(e.target as Node)) {
        setShowExport(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  if (isLoading || !data) {
    return (
      <div className="min-h-screen bg-[var(--color-background)] flex items-center justify-center">
        <Navbar />
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-[var(--color-accent)] border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-sm text-[var(--color-text-muted)]">Loading results...</p>
        </div>
      </div>
    );
  }

  const handleExportReport = () => {
    downloadPDF(data, "aegis_bias_report.pdf");
    setShowExport(false);
  };

  const handleExportDataset = () => {
    // We now download directly from the backend's static artifacts
    // This is more robust for large datasets than client-side blob generation
    const downloadUrl = "http://localhost:8000/api/exports/debiased_dataset.csv";
    
    const a = document.createElement("a");
    a.href = downloadUrl;
    a.download = "aegis_debiased_dataset.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    setShowExport(false);
  };

  const handleExportModel = () => {
    downloadJSON({
      strategy: data.metadata?.strategy_used,
      model_type: data.metadata?.config?.model_type,
      accuracy: data.model_analysis?.ranking?.ranking_table?.[0]?.accuracy,
      exported_at: new Date().toISOString(),
    }, "aegis_debiased_model.json");
    setShowExport(false);
  };

  const handleExportFullJSON = () => {
    downloadJSON(data, "aegis_full_report.json");
    setShowExport(false);
  };

  // Derive display info
  const modeName = data.metadata?.mode?.replace("_", " ") || "pipeline";
  const elapsed = data.metadata?.elapsed_seconds || 0;
  const strategyUsed = data.metadata?.strategy_used || "none";

  return (
    <div className="min-h-screen bg-[var(--color-background)]">
      <Navbar />

      <main className="max-w-7xl mx-auto px-6 pt-24 pb-20">
        {/* Demo banner */}
        {isDemo && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 p-3 rounded-xl border border-[var(--color-warning)]/30 bg-[var(--color-warning-muted)] flex items-center gap-3"
          >
            <FlaskConical className="w-4 h-4 text-[var(--color-warning)] shrink-0" />
            <p className="text-xs text-[var(--color-text-secondary)]">
              <span className="font-bold text-[var(--color-warning)]">Demo Mode</span> — You are viewing pre-computed results from the UCI Adult dataset.
              <a href="/upload" className="ml-1 underline hover:text-[var(--color-accent)]">Upload your own dataset</a> to run a real analysis.
            </p>
          </motion.div>
        )}

        {/* Page header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-start justify-between mb-8"
        >
          <div>
            <div className="flex items-center gap-3 mb-1">
              <h1 className="text-2xl font-bold">Governance Dashboard</h1>
              {!isDemo && (
                <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-[var(--color-success-muted)] text-[var(--color-success)] border border-[var(--color-success)]/20">
                  LIVE
                </span>
              )}
            </div>
            <p className="text-sm text-[var(--color-text-muted)]">
              {modeName} &middot; {strategyUsed !== "none" ? strategyUsed.replace("_", " ") : "analysis"} &middot; {elapsed}s elapsed
            </p>
          </div>

          {/* Export dropdown */}
          <div ref={exportRef} className="relative">
            <button
              onClick={() => setShowExport(!showExport)}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl border border-[var(--color-border)] hover:border-[var(--color-accent)]/30 text-sm font-semibold transition-colors"
            >
              <Download className="w-4 h-4" /> Export
            </button>

            {showExport && (
              <motion.div
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                className="absolute right-0 top-12 w-64 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] shadow-2xl overflow-hidden z-50"
              >
                <div className="p-2 space-y-0.5">
                  <button onClick={handleExportReport} className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-[var(--color-surface-hover)] transition-colors text-left">
                    <FileText className="w-4 h-4 text-[var(--color-accent)] shrink-0" />
                    <div>
                      <p className="text-xs font-semibold">Export Bias Analysis Report</p>
                      <p className="text-[10px] text-[var(--color-text-muted)]">PDF report (.pdf)</p>
                    </div>
                  </button>
                  <button onClick={handleExportDataset} className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-[var(--color-surface-hover)] transition-colors text-left">
                    <FileSpreadsheet className="w-4 h-4 text-[var(--color-success)] shrink-0" />
                    <div>
                      <p className="text-xs font-semibold">Download Debiased Dataset</p>
                      <p className="text-[10px] text-[var(--color-text-muted)]">CSV format (.csv)</p>
                    </div>
                  </button>
                  <button onClick={handleExportModel} className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-[var(--color-surface-hover)] transition-colors text-left">
                    <Box className="w-4 h-4 text-[var(--color-warning)] shrink-0" />
                    <div>
                      <p className="text-xs font-semibold">Download Debiased Model</p>
                      <p className="text-[10px] text-[var(--color-text-muted)]">Model config (.json)</p>
                    </div>
                  </button>
                  <div className="border-t border-[var(--color-border)] my-1" />
                  <button onClick={handleExportFullJSON} className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-[var(--color-surface-hover)] transition-colors text-left">
                    <Download className="w-4 h-4 text-[var(--color-text-muted)] shrink-0" />
                    <div>
                      <p className="text-xs font-semibold">Export Raw Pipeline JSON</p>
                      <p className="text-[10px] text-[var(--color-text-muted)]">Full data (.json)</p>
                    </div>
                  </button>
                </div>
              </motion.div>
            )}
          </div>
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
                  "relative flex items-center gap-2 px-4 py-2.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-colors",
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

        {/* Active tab content */}
        <motion.div key={activeTab} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
          {activeTab === "overview" && <OverviewTab data={data} />}
          {activeTab === "detection" && <DetectionTab biasReport={data.dataset_analysis?.bias_report} />}
          {activeTab === "mitigation" && <MitigationTab ranking={data.model_analysis?.ranking} explanations={data.explanations} />}
          {activeTab === "comparison" && <ComparisonTab comparison={data.dataset_analysis?.dataset_comparison} />}
          {activeTab === "explainability" && <ExplainabilityTab explainability={data.model_analysis?.explainability} explanations={data.explanations} />}
        </motion.div>
      </main>
    </div>
  );
}
