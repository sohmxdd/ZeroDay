"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useRouter } from "next/navigation";
import {
  Upload, FileSpreadsheet, X, Play, Settings2,
  ChevronDown, Database, Cpu, Box, Check, Loader2, AlertCircle,
} from "lucide-react";
import Navbar from "@/components/layout/Navbar";
import { PipelineLoader } from "@/components/loaders/PipelineLoader";
import { cn } from "@/lib/utils";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";


export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [mode] = useState("full_pipeline");
  const [isDragging, setIsDragging] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [selectedModelType, setSelectedModelType] = useState("logistic_regression");
  const [alpha, setAlpha] = useState(0.6);
  const [beta, setBeta] = useState(0.4);

  // Pipeline execution state
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentPhase, setCurrentPhase] = useState(-1); // -1=idle, 0-5=phases

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped && (dropped.name.endsWith(".csv") || dropped.name.endsWith(".xlsx"))) {
      setFile(dropped);
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) setFile(e.target.files[0]);
  };

  const handleModelSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f && f.name.endsWith(".pkl")) setModelFile(f);
  };

  const handleRun = async () => {
    setIsRunning(true);
    setError(null);
    setCurrentPhase(0); // Loading Data

    try {
      const formData = new FormData();
      if (file) formData.append("dataset", file);
      formData.append("mode", mode);
      formData.append("speed", "fast");
      formData.append("model_type", selectedModelType);
      formData.append("alpha", alpha.toString());
      formData.append("beta", beta.toString());
      if (modelFile) formData.append("model", modelFile);

      // Simulate phase progression while waiting for response
      const phaseTimer = setInterval(() => {
        setCurrentPhase((prev) => (prev < 4 ? prev + 1 : prev));
      }, 1800);

      const res = await fetch(`${API_URL}/api/run-pipeline`, {
        method: "POST",
        body: formData,
      });

      clearInterval(phaseTimer);

      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: "Server error" }));
        throw new Error(err.error || `Server returned ${res.status}`);
      }

      const result = await res.json();

      // Show completion phase
      setCurrentPhase(5);
      localStorage.setItem("aegis_result", JSON.stringify(result));
      localStorage.removeItem("aegis_demo");

      // Brief pause to show "Complete" then navigate
      await new Promise((r) => setTimeout(r, 800));
      router.push("/results");
    } catch (err: any) {
      console.error("Pipeline error:", err);
      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to AEGIS backend. Please ensure the server is running (python server.py)."
          : err.message || "An unexpected error occurred."
      );
      setIsRunning(false);
      setCurrentPhase(-1);
    }
  };

  const activeModelSource = modelFile ? "uploaded" : "selected";

  return (
    <div className="min-h-screen bg-[var(--color-background)]">
      <Navbar />

      <main className="max-w-4xl mx-auto px-6 pt-28 pb-20">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-10"
        >
          <h1 className="text-3xl font-bold mb-2">Configure Pipeline</h1>
          <p className="text-sm text-[var(--color-text-muted)] max-w-md mx-auto">
            Upload your training dataset, configure the pipeline mode, and optionally provide a pre-trained model
          </p>
        </motion.div>

        {/* ──── Dataset Upload Zone ──── */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}>
          <label className="text-[10px] font-bold uppercase tracking-widest text-[var(--color-text-muted)] block mb-3">
            Training Dataset
          </label>
          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={cn(
              "relative rounded-2xl border-2 border-dashed transition-all duration-300 overflow-hidden",
              isDragging
                ? "border-[var(--color-accent)] bg-[var(--color-accent-muted)] scale-[1.005]"
                : file
                ? "border-[var(--color-success)]/30 bg-[var(--color-success-muted)]"
                : "border-[var(--color-border)] bg-[var(--color-surface)] hover:border-[var(--color-accent)]/30"
            )}
          >
            <div className="absolute inset-0 bg-gradient-to-br from-white/[0.015] to-transparent pointer-events-none" />
            <div className="relative px-8 py-14 text-center">
              <AnimatePresence mode="wait">
                {file ? (
                  <motion.div key="file" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }} className="flex flex-col items-center gap-3">
                    <div className="w-12 h-12 rounded-xl bg-[var(--color-success-muted)] border border-[var(--color-success)]/20 flex items-center justify-center">
                      <FileSpreadsheet className="w-6 h-6 text-[var(--color-success)]" />
                    </div>
                    <div>
                      <p className="text-sm font-semibold">{file.name}</p>
                      <p className="text-xs text-[var(--color-text-muted)] mt-0.5">
                        {(file.size / 1024).toFixed(1)} KB &middot; Ready for analysis
                      </p>
                    </div>
                    <button onClick={() => setFile(null)} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-[var(--color-text-muted)] hover:text-[var(--color-danger)] hover:bg-[var(--color-danger-muted)] transition-colors">
                      <X className="w-3.5 h-3.5" /> Remove File
                    </button>
                  </motion.div>
                ) : (
                  <motion.div key="empty" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }} className="flex flex-col items-center gap-3">
                    <div className="w-12 h-12 rounded-xl bg-[var(--color-accent-muted)] border border-[var(--color-accent)]/20 flex items-center justify-center">
                      <Upload className="w-6 h-6 text-[var(--color-accent)]" />
                    </div>
                    <div>
                      <p className="text-sm font-semibold">Drop your CSV or XLSX file here</p>
                      <p className="text-xs text-[var(--color-text-muted)] mt-0.5">or click below to browse your filesystem</p>
                    </div>
                    <label className="px-5 py-2 rounded-xl bg-[var(--color-surface-elevated)] hover:bg-[var(--color-surface-hover)] border border-[var(--color-border)] text-xs font-semibold cursor-pointer transition-colors">
                      Browse Files
                      <input type="file" accept=".csv,.xlsx" onChange={handleFileSelect} className="hidden" />
                    </label>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </motion.div>

        {/* ──── Model Upload ──── */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="mt-8">
          <label className="text-[10px] font-bold uppercase tracking-widest text-[var(--color-text-muted)] block mb-3">
            Pre-trained Model <span className="text-[var(--color-text-muted)] font-normal normal-case tracking-normal">(optional)</span>
          </label>
          <div className="flex items-center gap-4">
            {modelFile ? (
              <div className="flex-1 flex items-center gap-3 p-4 rounded-xl border border-[var(--color-success)]/30 bg-[var(--color-success-muted)]">
                <Box className="w-5 h-5 text-[var(--color-success)] shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold truncate">{modelFile.name}</p>
                  <p className="text-[10px] text-[var(--color-success)]">Using Uploaded Model</p>
                </div>
                <button onClick={() => setModelFile(null)} className="p-1.5 rounded-lg hover:bg-[var(--color-danger-muted)] transition-colors">
                  <X className="w-3.5 h-3.5 text-[var(--color-text-muted)] hover:text-[var(--color-danger)]" />
                </button>
              </div>
            ) : (
              <label className="flex-1 flex items-center gap-3 p-4 rounded-xl border border-dashed border-[var(--color-border)] bg-[var(--color-surface)] hover:border-[var(--color-accent)]/30 cursor-pointer transition-colors">
                <Box className="w-5 h-5 text-[var(--color-text-muted)] shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-[var(--color-text-secondary)]">Upload Model (.pkl)</p>
                  <p className="text-[10px] text-[var(--color-text-muted)]">Using Selected Model: {selectedModelType.replace("_", " ")}</p>
                </div>
                <span className="px-3 py-1 rounded-lg bg-[var(--color-surface-elevated)] border border-[var(--color-border)] text-[10px] font-semibold">Browse</span>
                <input type="file" accept=".pkl" onChange={handleModelSelect} className="hidden" />
              </label>
            )}
          </div>
          <div className="flex items-center gap-2 mt-2 text-[10px] text-[var(--color-text-muted)]">
            <Check className="w-3 h-3 text-[var(--color-success)]" />
            Active: {activeModelSource === "uploaded" ? `Uploaded model (${modelFile?.name})` : `${selectedModelType.replace("_", " ")} (built-in)`}
          </div>
        </motion.div>

        {/* ──── Advanced Settings ──── */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }} className="mt-8">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className={cn(
              "w-full flex items-center gap-3 px-5 py-4 rounded-xl border text-left transition-all duration-200",
              showSettings
                ? "border-[var(--color-accent)]/40 bg-[var(--color-accent-muted)]"
                : "border-[var(--color-border)] bg-[var(--color-surface)] hover:border-[var(--color-accent)]/30 hover:bg-[var(--color-surface-hover)]"
            )}
          >
            <div className={cn(
              "w-8 h-8 rounded-lg flex items-center justify-center shrink-0 transition-colors",
              showSettings ? "bg-[var(--color-accent)] text-white" : "bg-[var(--color-surface-elevated)] text-[var(--color-text-muted)]"
            )}>
              <Settings2 className="w-4 h-4" />
            </div>
            <div className="flex-1">
              <p className="text-sm font-semibold">Advanced Configuration</p>
              <p className="text-[10px] text-[var(--color-text-muted)] mt-0.5">Accuracy-fairness weights, model architecture, and AI settings</p>
            </div>
            <ChevronDown className={cn("w-4 h-4 text-[var(--color-text-muted)] transition-transform duration-200", showSettings && "rotate-180")} />
          </button>

          <AnimatePresence>
            {showSettings && (
              <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                <div className="grid grid-cols-2 gap-4 mt-4 p-5 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)]">
                  <div>
                    <label className="text-[10px] font-bold uppercase tracking-widest text-[var(--color-text-muted)] block mb-2">Accuracy Weight (α)</label>
                    <input type="number" value={alpha} onChange={(e) => setAlpha(parseFloat(e.target.value) || 0.6)} step={0.1} min={0} max={1} className="w-full px-3 py-2.5 rounded-lg bg-[var(--color-surface-elevated)] border border-[var(--color-border)] text-sm text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-accent)] transition-colors" />
                  </div>
                  <div>
                    <label className="text-[10px] font-bold uppercase tracking-widest text-[var(--color-text-muted)] block mb-2">Fairness Weight (β)</label>
                    <input type="number" value={beta} onChange={(e) => setBeta(parseFloat(e.target.value) || 0.4)} step={0.1} min={0} max={1} className="w-full px-3 py-2.5 rounded-lg bg-[var(--color-surface-elevated)] border border-[var(--color-border)] text-sm text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-accent)] transition-colors" />
                  </div>
                  <div>
                    <label className="text-[10px] font-bold uppercase tracking-widest text-[var(--color-text-muted)] block mb-2">Model Architecture</label>
                    <select value={selectedModelType} onChange={(e) => setSelectedModelType(e.target.value)} className="w-full px-3 py-2.5 rounded-lg bg-[var(--color-surface-elevated)] border border-[var(--color-border)] text-sm text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-accent)] transition-colors">
                      <option value="logistic_regression">Logistic Regression</option>
                      <option value="random_forest">Random Forest</option>
                      <option value="xgboost">XGBoost</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[10px] font-bold uppercase tracking-widest text-[var(--color-text-muted)] block mb-2">Gemini AI Explanations</label>
                    <select className="w-full px-3 py-2.5 rounded-lg bg-[var(--color-surface-elevated)] border border-[var(--color-border)] text-sm text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-accent)] transition-colors">
                      <option value="enabled">Enabled</option>
                      <option value="disabled">Disabled</option>
                    </select>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        {/* ──── Error Banner ──── */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mt-6 p-4 rounded-xl border border-[var(--color-danger)]/30 bg-[var(--color-danger-muted)] flex items-start gap-3"
            >
              <AlertCircle className="w-5 h-5 text-[var(--color-danger)] shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="text-sm font-semibold text-[var(--color-danger)]">Pipeline Error</p>
                <p className="text-xs text-[var(--color-text-secondary)] mt-1">{error}</p>
              </div>
              <button onClick={() => setError(null)} className="p-1 rounded hover:bg-white/5">
                <X className="w-4 h-4 text-[var(--color-text-muted)]" />
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ──── Execute Button ──── */}
        {!isRunning && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }} className="mt-10 flex flex-col items-center gap-3">
            <button
              onClick={handleRun}
              className="group flex items-center gap-3 px-10 py-4 rounded-2xl font-bold text-base transition-all bg-[var(--color-accent)] hover:bg-[var(--color-accent-hover)] text-white glow"
            >
              <Play className="w-5 h-5" />
              Execute Bias Analysis Pipeline
            </button>
          </motion.div>
        )}

        {/* ──── Info Footer ──── */}
        {!isRunning && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }} className="grid grid-cols-2 gap-4 mt-12">
            <div className="flex items-start gap-3 p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)]">
              <Database className="w-4 h-4 text-[var(--color-accent)] mt-0.5 shrink-0" />
              <div>
                <p className="text-xs font-semibold mb-0.5">No dataset available?</p>
                <p className="text-[10px] text-[var(--color-text-muted)] leading-relaxed">
                  Execute without uploading to use the built-in UCI Adult dataset for demonstration purposes.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)]">
              <Cpu className="w-4 h-4 text-[var(--color-accent)] mt-0.5 shrink-0" />
              <div>
                <p className="text-xs font-semibold mb-0.5">Estimated Processing Time</p>
                <p className="text-[10px] text-[var(--color-text-muted)] leading-relaxed">
                  Full pipeline completes in ~5 seconds. Analysis-only mode completes in under 2 seconds.
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </main>

      {/* ══════ FULL-SCREEN PROCESSING OVERLAY ══════ */}
      <AnimatePresence>
        {isRunning && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-[var(--color-background)]"
          >
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-[var(--color-accent)] opacity-[0.04] blur-[140px] rounded-full pointer-events-none" />

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="relative z-10 w-full max-w-xl px-6"
            >
              <div className="text-center mb-10">
                <motion.div
                  initial={{ scale: 0.9 }}
                  animate={{ scale: 1 }}
                  className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[var(--color-accent-muted)] border border-[var(--color-accent)]/20 text-[var(--color-accent)] text-[11px] font-bold uppercase tracking-widest mb-4"
                >
                  <Loader2 className="w-3 h-3 animate-spin" />
                  Pipeline Active
                </motion.div>
                <h1 className="text-2xl font-bold mb-2">Executing Analysis</h1>
                <p className="text-sm text-[var(--color-text-muted)]">
                  Analyzing your dataset for bias patterns and generating corrective insights...
                </p>
              </div>

              <PipelineLoader currentStep={currentPhase} />

              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }} className="text-center mt-10">
                <p className="text-[11px] text-[var(--color-text-muted)]">
                  Optimized fast mode &bull; Estimated completion: ~5 seconds
                </p>
              </motion.div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
