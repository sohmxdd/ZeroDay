// ─── Export Utilities ─────────────────────────────────────────────
// Client-side download triggers for reports, datasets, and models

import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ─── PDF Report ──────────────────────────────────────────────────

export function downloadPDF(data: any, filename: string) {
  const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  const pageWidth = doc.internal.pageSize.getWidth();
  let y = 15;

  // ── Header ──
  doc.setFillColor(15, 15, 25);
  doc.rect(0, 0, pageWidth, 40, "F");
  doc.setTextColor(120, 180, 255);
  doc.setFontSize(22);
  doc.setFont("helvetica", "bold");
  doc.text("AEGIS", 14, 20);
  doc.setFontSize(10);
  doc.setTextColor(180, 180, 200);
  doc.text("AI Bias Governance Report", 14, 28);
  doc.setFontSize(8);
  doc.text(`Generated: ${new Date().toLocaleString()}`, 14, 35);

  y = 50;
  doc.setTextColor(40, 40, 40);

  // ── Pipeline Metadata ──
  const mode = data?.metadata?.mode || "full_pipeline";
  const strategy = data?.metadata?.strategy_used || "N/A";
  const elapsed = data?.metadata?.elapsed_seconds?.toFixed(1) || "0";

  doc.setFontSize(13);
  doc.setFont("helvetica", "bold");
  doc.text("Pipeline Summary", 14, y);
  y += 8;

  doc.setFontSize(9);
  doc.setFont("helvetica", "normal");
  const metaLines = [
    `Mode: ${mode.replace("_", " ")}`,
    `Best Strategy: ${strategy.replace("_", " ")}`,
    `Elapsed: ${elapsed}s`,
  ];
  metaLines.forEach((line) => {
    doc.text(line, 14, y);
    y += 5;
  });
  y += 4;

  // ── Bias Detection Summary ──
  const biasReport = data?.dataset_analysis?.bias_report;
  if (biasReport) {
    doc.setFontSize(13);
    doc.setFont("helvetica", "bold");
    doc.text("Bias Detection", 14, y);
    y += 8;

    const biasRows: string[][] = [];

    // Distribution bias
    const dist = biasReport.distribution_bias;
    if (dist && typeof dist === "object") {
      Object.entries(dist).forEach(([feature, info]: [string, any]) => {
        biasRows.push([feature, "Distribution", `Disparity: ${(info?.max_disparity ?? 0).toFixed(3)}`]);
      });
    }

    // Outcome bias
    const outcome = biasReport.outcome_bias;
    if (outcome && typeof outcome === "object") {
      Object.entries(outcome).forEach(([feature, info]: [string, any]) => {
        biasRows.push([feature, "Outcome", `Gap: ${(info?.max_gap ?? 0).toFixed(3)}`]);
      });
    }

    if (biasRows.length > 0) {
      autoTable(doc, {
        startY: y,
        head: [["Feature", "Bias Type", "Metric"]],
        body: biasRows,
        theme: "striped",
        headStyles: { fillColor: [30, 60, 120], fontSize: 8 },
        bodyStyles: { fontSize: 8 },
        margin: { left: 14, right: 14 },
      });
      y = (doc as any).lastAutoTable.finalY + 10;
    }
  }

  // ── Ranking Table ──
  const rankingTable = data?.model_analysis?.ranking?.ranking_table;
  if (rankingTable && rankingTable.length > 0) {
    if (y > 240) { doc.addPage(); y = 15; }

    doc.setFontSize(13);
    doc.setFont("helvetica", "bold");
    doc.text("Strategy Ranking", 14, y);
    y += 8;

    const rankRows = rankingTable.map((row: any) => [
      `#${row.rank}`,
      row.pipeline?.replace(/_/g, " ") || "",
      (row.accuracy ?? 0).toFixed(4),
      (row.demographic_parity_diff ?? 0).toFixed(4),
      (row.score ?? 0).toFixed(4),
    ]);

    autoTable(doc, {
      startY: y,
      head: [["Rank", "Strategy", "Accuracy", "DP Diff", "Score"]],
      body: rankRows,
      theme: "striped",
      headStyles: { fillColor: [30, 60, 120], fontSize: 8 },
      bodyStyles: { fontSize: 7 },
      margin: { left: 14, right: 14 },
    });
    y = (doc as any).lastAutoTable.finalY + 10;
  }

  // ── Explanations ──
  const expl = data?.explanations;
  if (expl) {
    if (y > 220) { doc.addPage(); y = 15; }

    doc.setFontSize(13);
    doc.setFont("helvetica", "bold");
    doc.text("AI Explanations", 14, y);
    y += 8;

    const sections = [
      { title: "Summary", content: expl.summary },
      { title: "Bias Explanation", content: expl.bias_explanation },
      { title: "Strategy Justification", content: expl.strategy_justification },
      { title: "Tradeoff Analysis", content: expl.tradeoff_analysis },
      { title: "Recommendation", content: expl.recommendation },
    ];

    sections.forEach((section) => {
      if (!section.content) return;
      if (y > 260) { doc.addPage(); y = 15; }

      doc.setFontSize(10);
      doc.setFont("helvetica", "bold");
      doc.text(section.title, 14, y);
      y += 5;

      doc.setFontSize(8);
      doc.setFont("helvetica", "normal");
      const lines = doc.splitTextToSize(section.content, pageWidth - 28);
      doc.text(lines, 14, y);
      y += lines.length * 3.5 + 6;
    });
  }

  // ── Footer ──
  const pageCount = doc.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    doc.setFontSize(7);
    doc.setTextColor(150, 150, 150);
    doc.text(
      `AEGIS Bias Governance Report — Page ${i}/${pageCount}`,
      pageWidth / 2,
      doc.internal.pageSize.getHeight() - 8,
      { align: "center" }
    );
  }

  doc.save(filename);
}

// ─── JSON Export ─────────────────────────────────────────────────

export function downloadJSON(data: unknown, filename: string) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  triggerDownload(blob, filename);
}

// ─── CSV Export (from backend) ───────────────────────────────────

export async function downloadDatasetCSV(filename: string) {
  try {
    const res = await fetch(`${API_BASE}/api/download-dataset`);
    if (!res.ok) {
      // Fallback: no dataset on server, download from localStorage
      const stored = localStorage.getItem("aegis_result");
      if (stored) {
        const data = JSON.parse(stored);
        const comparison = data?.dataset_analysis?.dataset_comparison;
        const columns = comparison?.debiased_stats?.columns || comparison?.original_stats?.columns;
        if (columns && columns.length > 0) {
          const csv = columns.join(",");
          const blob = new Blob([csv], { type: "text/csv" });
          triggerDownload(blob, filename);
          return;
        }
      }
      alert("No debiased dataset available. Run the pipeline first.");
      return;
    }
    const blob = await res.blob();
    triggerDownload(blob, filename);
  } catch {
    alert("Could not download dataset. Make sure the AEGIS backend is running.");
  }
}

// ─── Model PKL Export (from backend) ─────────────────────────────

export async function downloadModelPKL(filename: string) {
  try {
    const res = await fetch(`${API_BASE}/api/download-model`);
    if (!res.ok) {
      alert("No trained model available. Run the full pipeline first.");
      return;
    }
    const blob = await res.blob();
    triggerDownload(blob, filename);
  } catch {
    alert("Could not download model. Make sure the AEGIS backend is running.");
  }
}

// ─── Legacy CSV (client-side) ────────────────────────────────────

export function downloadCSV(headers: string[], rows: string[][], filename: string) {
  const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  triggerDownload(blob, filename);
}

// ─── Helpers ─────────────────────────────────────────────────────

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
