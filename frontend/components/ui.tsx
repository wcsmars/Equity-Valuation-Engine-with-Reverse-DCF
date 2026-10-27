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
import { sliderBounds } from "@/lib/assumptions";
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
  title,
}: {
  children: React.ReactNode;
  tone?: "up" | "down" | "flat" | "neutral" | "brand";
  title?: string; // hover tooltip
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
      title={title}
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

// --- Alert banner ---------------------------------------------------------- //
// An error ("down") or a warning ("flat") the user must act on. role="alert"
// has screen readers announce it, wherever focus is.
export function Alert({
  tone = "down",
  className,
  children,
}: {
  tone?: "down" | "flat";
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <div
      role="alert"
      className={cx(
        "rounded-lg border px-4 py-3 text-sm",
        tone === "down"
          ? "border-down/40 bg-down/10 text-down"
          : "border-flat/40 bg-flat/10 text-flat",
        className
      )}
    >
      {children}
    </div>
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
// The label is tied to the slider and the value is read out as shown ("2.5%",
// not "0.025"). The range always takes in the value and `anchors` (the
// engine's own figures for this driver), so the thumb never sits pinned at an
// end while the label shows a value beyond it; anchors keep that wider range
// steady while the user drags away from such a value. A `changed` value (an
// edit the report on screen does not reflect yet) is drawn in amber and read
// out as not applied.
export function AssumptionSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
  hint,
  anchors,
  valueText,
  changed = false,
}: {
  label: React.ReactNode;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format: (v: number) => string;
  hint?: React.ReactNode;
  anchors?: (number | null | undefined)[];
  valueText?: (v: number) => string; // spoken value, when `format` is terse
  changed?: boolean;
}) {
  const id = React.useId();
  const [lo, hi] = sliderBounds(min, max, step, [...(anchors ?? []), value]);
  const spoken = (valueText ?? format)(value);
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <label htmlFor={id} className="text-xs font-medium text-ink-dim">
          {label}
        </label>
        <span
          className={cx("num text-sm font-semibold", changed ? "text-flat" : "text-ink")}
          title={changed ? "Changed: recompute to apply" : undefined}
        >
          {format(value)}
        </span>
      </div>
      <input
        id={id}
        type="range"
        min={lo}
        max={hi}
        step={step}
        value={value}
        aria-valuetext={changed ? `${spoken}, not applied` : spoken}
        aria-describedby={hint ? `${id}-hint` : undefined}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="mt-1 w-full"
      />
      {hint && (
        <p id={`${id}-hint`} className="mt-0.5 text-[11px] text-ink-faint">
          {hint}
        </p>
      )}
    </div>
  );
}

// --- Unapplied edits ------------------------------------------------------- //
// A sticky amber bar listing the draft's edits that the report on screen does
// not reflect yet (lib/assumptions.ts diffAssumptions), with Recompute and
// Discard. It stays in view at the bottom of the window while its card is.
export function UnappliedBar({
  changes,
  onRecompute,
  onDiscard,
  recomputing,
  shown = 3,
}: {
  changes: { key: string; label: string; from: string; to: string }[];
  onRecompute: () => void;
  onDiscard: () => void;
  recomputing: boolean;
  shown?: number; // changes listed before an ellipsis
}) {
  const n = changes.length;
  const count = n === 0 ? "" : `${n} unapplied change${n === 1 ? "" : "s"}`;
  const item = (c: { label: string; from: string; to: string }) =>
    `${c.label} ${c.from} → ${c.to}`;
  const list = changes.slice(0, shown).map(item).join("; ") + (n > shown ? " …" : "");
  // Only the count is announced (a polite live region), so dragging a slider,
  // which rewrites the list on every tick, stays quiet until the number of
  // changes moves. The region stays mounted, empty without changes, so the
  // first change is announced too; the bar's visible count is hidden from
  // screen readers so it is not read twice.
  return (
    <>
      <span role="status" className="sr-only">
        {count}
      </span>
      {n > 0 && (
        <div className="sticky bottom-2 z-10 rounded-lg border border-flat/50 bg-surface shadow-lg">
          <div
            className="flex flex-wrap items-center gap-2 rounded-lg bg-flat/10 px-3 py-2"
            title={changes.map(item).join("\n")}
          >
            <p className="min-w-[12rem] flex-1 text-xs text-flat">
              <span className="font-semibold" aria-hidden="true">{`${count}:`}</span>
              {` ${list}`}
            </p>
            <Button onClick={onRecompute} disabled={recomputing} className="px-2.5 py-1 text-xs">
              {recomputing ? (
                <>
                  <Spinner className="h-3 w-3" /> Recomputing
                </>
              ) : (
                "Recompute"
              )}
            </Button>
            <Button variant="ghost" onClick={onDiscard} disabled={recomputing}
              className="px-2.5 py-1 text-xs">
              Discard
            </Button>
          </div>
        </div>
      )}
    </>
  );
}

// A row of mutually exclusive options; the selected one is announced as
// pressed, not only coloured.
export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { label: string; value: T }[];
  onChange: (v: T) => void;
  label?: string; // accessible name of the group
}) {
  return (
    <div
      role="group"
      aria-label={label}
      className="inline-flex rounded-lg border border-line bg-surface-raised p-0.5"
    >
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={value === o.value}
          onClick={() => onChange(o.value)}
          className={cx(
            "rounded-md px-3 py-1 text-xs font-medium transition",
            value === o.value
              ? "bg-brand-dim text-white"
              : "text-ink-dim hover:text-ink"
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

// A row of section tabs. Plain buttons that mark the open section as current
// (aria-current), so it is announced and not shown by colour alone.
export function TabBar<T extends string>({
  tabs,
  value,
  onChange,
  label,
  badge,
  className,
}: {
  tabs: readonly T[];
  value: T;
  onChange: (t: T) => void;
  label: string; // accessible name of the tab row
  badge?: (t: T) => React.ReactNode;
  className?: string;
}) {
  return (
    <div
      role="group"
      aria-label={label}
      className={cx("flex flex-wrap gap-1 border-b border-line", className)}
    >
      {tabs.map((t) => (
        <button
          key={t}
          type="button"
          aria-current={value === t ? "true" : undefined}
          onClick={() => onChange(t)}
          className={cx(
            "-mb-px border-b-2 px-3 py-2 text-sm font-medium transition",
            value === t
              ? "border-brand text-ink"
              : "border-transparent text-ink-faint hover:text-ink-dim"
          )}
        >
          {t}
          {badge?.(t)}
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
    // White on brand-dim (4.8:1) and brand-deep (5.7:1) meets WCAG AA for
    // small text; white on the lighter accent brand would be 2.7:1.
    primary: "bg-brand-dim text-white hover:bg-brand-deep disabled:opacity-50",
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
