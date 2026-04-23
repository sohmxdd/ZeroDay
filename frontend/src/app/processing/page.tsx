"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { useRouter } from "next/navigation";
import { Shield } from "lucide-react";
import { PipelineLoader } from "@/components/loaders/PipelineLoader";

export default function ProcessingPage() {
  const router = useRouter();
  const [step, setStep] = useState(0);

  useEffect(() => {
    const timings = [1500, 2500, 3000, 2000, 2000, 1000];
    let timeout: NodeJS.Timeout;

    const advance = (current: number) => {
      if (current < 5) {
        timeout = setTimeout(() => {
          setStep(current + 1);
          advance(current + 1);
        }, timings[current]);
      } else {
        timeout = setTimeout(() => {
          router.push("/results");
        }, 1500);
      }
    };

    advance(0);
    return () => clearTimeout(timeout);
  }, [router]);

  return (
    <div className="min-h-screen bg-[var(--color-background)] flex flex-col items-center justify-center relative overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0 animate-grid" />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-[var(--color-accent)] opacity-[0.03] blur-[120px] rounded-full" />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative z-10 w-full max-w-xl px-6"
      >
        {/* Logo */}
        <div className="flex items-center justify-center gap-2.5 mb-12">
          <div className="w-10 h-10 rounded-xl bg-[var(--color-accent)] flex items-center justify-center">
            <Shield className="w-5 h-5 text-white" />
          </div>
          <span className="text-2xl font-bold tracking-tight">AEGIS</span>
        </div>

        {/* Title */}
        <div className="text-center mb-10">
          <h1 className="text-xl font-bold mb-2">Running Pipeline</h1>
          <p className="text-sm text-[var(--color-text-muted)]">
            Analyzing your dataset for bias and generating insights...
          </p>
        </div>

        {/* Pipeline Loader */}
        <PipelineLoader currentStep={step} />

        {/* Elapsed time */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="text-center mt-10"
        >
          <p className="text-xs text-[var(--color-text-muted)]">
            Estimated: ~30 seconds for full pipeline
          </p>
        </motion.div>
      </motion.div>
    </div>
  );
}
