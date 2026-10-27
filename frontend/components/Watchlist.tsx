"use client";

// Watchlist cards (landing screen) and the watch toggle (company header).
// Each card opens its ticker from the mouse or the keyboard; its remove
// button is a separate control that shows on hover and on keyboard focus.

import React from "react";
import type { WatchlistItem } from "@/lib/types";
import {
  fmtDate,
  fmtMoney,
  fmtPct,
  toneForRecommendation,
  toneForUpside,
  watchlistUpside,
} from "@/lib/format";
import { cx } from "@/components/ui";

export function WatchlistGrid({
  items,
  onOpen,
  onRemove,
}: {
  items: WatchlistItem[];
  onOpen: (ticker: string) => void;
  onRemove: (ticker: string) => void;
}) {
  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
      {items.map((w) => {
        // No upside (and no colour) without a target or when the verdict
        // was withheld.
        const up = watchlistUpside(w);
        const hasTarget = Number.isFinite(w.blended_target as number);
        return (
          <div
            key={w.ticker}
            className="group relative rounded-xl border border-line bg-surface transition focus-within:border-brand hover:border-brand"
          >
            {/* The opener covers the card; the remove button sits above it. */}
            <button
              type="button"
              onClick={() => onOpen(w.ticker)}
              className="block w-full rounded-xl px-4 py-3 text-left focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand"
            >
              <span className="flex items-center gap-2 pr-6">
                <span className="font-semibold text-ink">{w.ticker}</span>
                <span className="truncate text-xs text-ink-faint">{w.name}</span>
              </span>
              <span className="num mt-1 flex items-baseline gap-2 text-sm">
                <span className="text-ink">{fmtMoney(w.price, w.currency)}</span>
                <span className="text-ink-faint" aria-hidden>
                  →
                </span>
                <span className={toneForUpside(up)}>
                  {hasTarget
                    ? `${fmtMoney(w.blended_target, w.currency)} (${
                        up == null ? "n/a" : fmtPct(up, { signed: true })
                      })`
                    : "no target"}
                </span>
              </span>
              <span className="mt-1 flex items-center justify-between text-[11px]">
                <span className={toneForRecommendation(w.recommendation)}>
                  {w.recommendation}
                </span>
                <span className="text-ink-faint">{fmtDate(w.updated_at)}</span>
              </span>
            </button>
            <button
              type="button"
              onClick={() => onRemove(w.ticker)}
              aria-label={`Remove ${w.ticker} from watchlist`}
              title="Remove"
              className="absolute right-3 top-2.5 px-1 text-ink-faint opacity-0 transition hover:text-down focus-visible:opacity-100 group-focus-within:opacity-100 group-hover:opacity-100"
            >
              <span aria-hidden>×</span>
            </button>
          </div>
        );
      })}
    </div>
  );
}

// The star in the company header: a toggle named for what it does, with its
// state announced (pressed = on the watchlist).
export function WatchToggle({
  ticker,
  watching,
  onToggle,
}: {
  ticker: string;
  watching: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-pressed={watching}
      aria-label={`Watch ${ticker}`}
      className={cx(
        "text-lg leading-none transition",
        watching ? "text-flat" : "text-ink-faint hover:text-flat"
      )}
      title={watching ? "Remove from watchlist" : "Add to watchlist"}
    >
      <span aria-hidden>{watching ? "★" : "☆"}</span>
    </button>
  );
}
