"use client";

// One line under the header verdict saying why there is no verdict, no
// blended target, or a blend without some methods (a bank's DCF and FCFE, a
// low-payout DDM). Built from the summary's structured fields only
// (lib/format.ts verdictExplanation). When peers would give a comps value,
// an "Add peers" action opens the comps card's peer input.

import React from "react";
import type { Comps, Summary } from "@/lib/types";
import { verdictExplanation } from "@/lib/format";

export default function VerdictExplainer({
  summary,
  comps,
  onAddPeers,
}: {
  summary: Summary;
  comps: Comps | null | undefined;
  onAddPeers?: () => void;
}) {
  const why = verdictExplanation(summary, comps);
  if (!why) return null;
  return (
    <p className="max-w-md text-right text-[11px] leading-snug text-ink-faint">
      {why.text}
      {why.addPeers && onAddPeers && (
        <button
          type="button"
          onClick={onAddPeers}
          className="ml-2 rounded-md border border-brand/40 px-1.5 py-0.5 text-[11px] font-medium text-brand transition hover:border-brand"
        >
          Add peers
        </button>
      )}
    </p>
  );
}
