"use client";

// Shared UI primitives. ALL panels should build from these so the dashboard
// stays visually consistent.
//
// Color tokens (from tailwind.config.ts): bg, surface / surface-raised /
// surface-hi, border-line, text-ink / ink-dim / ink-faint, text-up (green,
// undervalued/positive), text-down (rose, overvalued/negative), text-flat
// (amber, neutral), brand (blue accent). Use `.num` (in globals.css) on any
// numeric figure for tabular mono digits.

import React from "react";
import { setsTextColor } from "@/lib/format";

export function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

// --- Card / section ------------------------------------------------------- //
export function Card({
  title,
  subtitle,
  right,
  className,
  bodyClassName,
  children,
}: {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  right?: React.ReactNode;
  className?: string;
  bodyClassName?: string;
  children: React.ReactNode;
}) {
  return (
    <section
      className={cx(
        "rounded-xl border border-line bg-surface shadow-sm",
        className
      )}
    >
      {(title || right) && (
        <header className="flex flex-wrap items-start justify-between gap-3 border-b border-line px-4 py-3">
          <div className="min-w-0">
            {title && (
              <h3 className="text-sm font-semibold tracking-wide text-ink">
                {title}
              </h3>
            )}
            {subtitle && (
              <p className="mt-0.5 text-xs text-ink-faint">{subtitle}</p>
            )}
          </div>
          {/* Wraps under the title on narrow screens instead of overflowing. */}
          {right && <div className="min-w-0 max-w-full">{right}</div>}
        </header>
      )}
      <div className={cx("px-4 py-3", bodyClassName)}>{children}</div>
    </section>
  );
}

// --- Stat ----------------------------------------------------------------- //
export function Stat({
  label,
  value,
  sub,
  tone,
  className,
  title,
}: {
  label: React.ReactNode;
  value: React.ReactNode;
  sub?: React.ReactNode;
  tone?: string; // a text-* class, e.g. "text-up"
  className?: string;
  title?: string; // hover tooltip for the whole stat
}) {
  return (
    <div className={cx("min-w-0", className)} title={title}>
      <div className="text-[11px] uppercase tracking-wider text-ink-faint">
        {label}
      </div>
      <div className={cx("num mt-0.5 text-lg font-semibold", tone || "text-ink")}>
        {value}
      </div>
      {sub != null && <div className="num text-xs text-ink-dim">{sub}</div>}
    </div>
  );
}

// --- Badge / Pill --------------------------------------------------------- //
export function Badge({
  children,
  tone = "neutral",
}: {
  children: React.ReactNode;
  tone?: "up" | "down" | "flat" | "neutral" | "brand";
}) {
  const map: Record<string, string> = {
    up: "bg-up/10 text-up border-up/30",
    down: "bg-down/10 text-down border-down/30",
    flat: "bg-flat/10 text-flat border-flat/30",
    brand: "bg-brand/10 text-brand border-brand/30",
    neutral: "bg-surface-hi text-ink-dim border-line",
  };
  return (
    <span
      className={cx(
        "inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium",
        map[tone]
      )}
    >
      {children}
    </span>
  );
}

// --- Table primitives ----------------------------------------------------- //
export function Table({ children }: { children: React.ReactNode }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-sm">{children}</table>
    </div>
  );
}

export function TH({
  children,
  className,
  align = "right",
}: {
  children?: React.ReactNode;
  className?: string;
  align?: "left" | "right" | "center";
}) {
  return (
    <th
      className={cx(
        "whitespace-nowrap border-b border-line px-2.5 py-2 text-[11px] font-semibold uppercase tracking-wider text-ink-faint",
        align === "left" && "text-left",
        align === "right" && "text-right",
        align === "center" && "text-center",
        className
      )}
    >
      {children}
    </th>
  );
}

export function TD({
  children,
  className,
  align = "right",
  num = false,
}: {
  children?: React.ReactNode;
  className?: string;
  align?: "left" | "right" | "center";
  num?: boolean;
}) {
  return (
    <td
      className={cx(
        "whitespace-nowrap border-b border-line/60 px-2.5 py-1.5",
        // A colour in className (an upside tone) replaces the default ink.
        !setsTextColor(className) && "text-ink",
        num && "num",
        align === "left" && "text-left",
        align === "right" && "text-right",
        align === "center" && "text-center",
        className
      )}
    >
      {children}
    </td>
  );
}

// --- Spinner / loading ---------------------------------------------------- //
export function Spinner({ className }: { className?: string }) {
  return (
    <span
      className={cx(
        "inline-block h-4 w-4 animate-spin rounded-full border-2 border-ink-faint border-t-brand",
        className
      )}
      aria-label="loading"
    />
  );
}

// --- Empty / hint state --------------------------------------------------- //
export function EmptyState({
  title,
  hint,
  className,
}: {
  title: React.ReactNode;
  hint?: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cx(
        "rounded-lg border border-dashed border-line bg-surface-raised/40 px-4 py-8 text-center",
        className
      )}
    >
      <div className="text-sm font-medium text-ink-dim">{title}</div>
      {hint && <div className="mt-1 text-xs text-ink-faint">{hint}</div>}
    </div>
  );
}

// --- Assumption controls (used by the DCF panel) -------------------------- //
export function AssumptionSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
  hint,
}: {
  label: React.ReactNode;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format: (v: number) => string;
  hint?: React.ReactNode;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <label className="text-xs font-medium text-ink-dim">{label}</label>
        <span className="num text-sm font-semibold text-ink">
          {format(value)}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="mt-1 w-full"
      />
      {hint && <p className="mt-0.5 text-[11px] text-ink-faint">{hint}</p>}
    </div>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: { label: string; value: T }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="inline-flex rounded-lg border border-line bg-surface-raised p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          className={cx(
            "rounded-md px-3 py-1 text-xs font-medium transition",
            value === o.value
              ? "bg-brand text-white"
              : "text-ink-dim hover:text-ink"
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

// --- Buttons -------------------------------------------------------------- //
export function Button({
  children,
  onClick,
  disabled,
  variant = "primary",
  className,
  type = "button",
}: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "ghost" | "subtle";
  className?: string;
  type?: "button" | "submit";
}) {
  const map: Record<string, string> = {
    primary: "bg-brand text-white hover:bg-brand-dim disabled:opacity-50",
    ghost: "border border-line text-ink-dim hover:text-ink hover:border-ink-faint",
    subtle: "bg-surface-hi text-ink hover:bg-line",
  };
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={cx(
        "inline-flex items-center justify-center gap-2 rounded-lg px-3 py-1.5 text-sm font-medium transition disabled:cursor-not-allowed",
        map[variant],
        className
      )}
    >
      {children}
    </button>
  );
}
