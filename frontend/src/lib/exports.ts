// ─── Export Utilities ─────────────────────────────────────────────
// Client-side download triggers for reports, datasets, and models

import { jsPDF } from "jspdf";

export function downloadJSON(data: unknown, filename: string) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  triggerDownload(blob, filename);
}

export function downloadCSV(headers: string[], rows: any[][], filename: string) {
  const escape = (val: any) => {
    const str = (val === null || val === undefined) ? "" : String(val);
    if (str.includes(",") || str.includes('"') || str.includes("\n")) {
      return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
  };

  const csv = [
    headers.map(escape).join(","),
    ...rows.map((r) => r.map(escape).join(","))
  ].join("\n");

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  triggerDownload(blob, filename);
}

export function downloadPDF(data: any, filename: string) {
  const doc = new jsPDF();
  const report = generateTextReport(data);
  
  // Set fonts and styles
  doc.setFont("helvetica", "bold");
  doc.setFontSize(18);
  doc.text("AEGIS — AI Bias Governance Report", 20, 20);
  
  doc.setFont("helvetica", "normal");
  doc.setFontSize(10);
  doc.setTextColor(100);
  doc.text(`Generated on: ${new Date().toLocaleString()}`, 20, 28);
  
  doc.setDrawColor(200);
  doc.line(20, 32, 190, 32);
  
  doc.setFontSize(11);
  doc.setTextColor(0);
  
  let y = 42;
  const margin = 20;
  const pageWidth = 170;
  
  const addSection = (title: string, content: string) => {
    if (y > 250) {
      doc.addPage();
      y = 20;
    }
    doc.setFont("helvetica", "bold");
    doc.text(title, margin, y);
    y += 7;
    doc.setFont("helvetica", "normal");
    const splitText = doc.splitTextToSize(content || "No data available.", pageWidth);
    doc.text(splitText, margin, y);
    y += (splitText.length * 6) + 10;
  };

  addSection("Pipeline Mode", data?.metadata?.mode || "full_pipeline");
  addSection("Strategy Used", data?.metadata?.strategy_used || "N/A");
  addSection("Summary", data?.explanations?.summary);
  addSection("Bias Explanation", data?.explanations?.bias_explanation);
  addSection("Strategy Justification", data?.explanations?.strategy_justification);
  addSection("Tradeoff Analysis", data?.explanations?.tradeoff_analysis);
  addSection("Recommendation", data?.explanations?.recommendation);

  doc.save(filename.endsWith(".pdf") ? filename : `${filename}.pdf`);
}

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

function generateTextReport(data: any): string {
  const lines: string[] = [
    "═══════════════════════════════════════════════════════════════",
    "  AEGIS — AI Bias Governance Report",
    "═══════════════════════════════════════════════════════════════",
    "",
    `Date: ${new Date().toISOString()}`,
    `Mode: ${data?.metadata?.mode || "full_pipeline"}`,
    `Strategy: ${data?.metadata?.strategy_used || "N/A"}`,
    `Elapsed: ${data?.metadata?.elapsed_seconds || 0}s`,
    "",
    "─── Summary ───",
    data?.explanations?.summary || "",
    "",
    "─── Bias Explanation ───",
    data?.explanations?.bias_explanation || "",
    "",
    "─── Strategy Justification ───",
    data?.explanations?.strategy_justification || "",
    "",
    "─── Tradeoff Analysis ───",
    data?.explanations?.tradeoff_analysis || "",
    "",
    "─── Recommendation ───",
    data?.explanations?.recommendation || "",
    "",
    "═══════════════════════════════════════════════════════════════",
  ];
  return lines.join("\n");
}
