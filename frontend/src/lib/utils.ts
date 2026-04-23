import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatScore(value: number): string {
  return value.toFixed(4);
}

export function getBiasLevel(disparity: number): {
  level: "low" | "medium" | "high" | "critical";
  color: string;
  label: string;
} {
  if (disparity < 0.05)
    return { level: "low", color: "text-[var(--color-success)]", label: "Low" };
  if (disparity < 0.15)
    return { level: "medium", color: "text-[var(--color-warning)]", label: "Medium" };
  if (disparity < 0.3)
    return { level: "high", color: "text-[var(--color-danger)]", label: "High" };
  return { level: "critical", color: "text-[var(--color-danger)]", label: "Critical" };
}
