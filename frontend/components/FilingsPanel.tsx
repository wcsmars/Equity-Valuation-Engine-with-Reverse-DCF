"use client";

// SEC filings + earnings-call transcripts the AI can read on demand. Each row
// has a "Digest" action that sends the document to the analyst, which returns
// a cited brief (Digest) the orchestrator appends to AI Research.

import React, { useEffect, useState } from "react";
import type { Digest, Filing, FilingsList, Report, TranscriptMeta } from "@/lib/types";
import {
  digestFiling,
  digestTranscript,
  fetchFilings,
  fetchTranscripts,
} from "@/lib/api";
import { fmtDate } from "@/lib/format";
import {
  Badge,
  Button,
  Card,
  EmptyState,
  Spinner,
  Table,
  TD,
  TH,
} from "@/components/ui";

export default function FilingsPanel({
  ticker,
  report,
  extraContext,
  aiEnabled,
  fmpEnabled,
  onDigested,
}: {
  ticker: string;
  report: Report;
  extraContext: string;
  aiEnabled: boolean;
  fmpEnabled: boolean;
  onDigested: (source: string, digest: Digest) => void;
}) {
  const [filings, setFilings] = useState<FilingsList | null>(null);
  const [filingsError, setFilingsError] = useState<string | null>(null);
  const [filingsLoading, setFilingsLoading] = useState(false);
  const [transcripts, setTranscripts] = useState<{
    enabled: boolean;
    transcripts: TranscriptMeta[];
  } | null>(null);
  const [digestingKey, setDigestingKey] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!ticker) return;
    let alive = true;

    setFilings(null);
    setFilingsError(null);
    setFilingsLoading(true);
    setErr(null);

    fetchFilings(ticker)
      .then((f) => {
        if (alive) setFilings(f);
      })
      .catch((e: unknown) => {
        if (alive)
          setFilingsError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (alive) setFilingsLoading(false);
      });

    return () => {
      alive = false;
    };
  }, [ticker]);

  // Separate effect so saving an FMP key in the app re-lists transcripts
  // without reloading the ticker.
  useEffect(() => {
    if (!ticker) return;
    let alive = true;
    setTranscripts(null);

    fetchTranscripts(ticker)
      .then((t) => {
        if (alive) setTranscripts(t);
      })
      .catch(() => {
        if (alive) setTranscripts({ enabled: false, transcripts: [] });
      });

    return () => {
      alive = false;
    };
  }, [ticker, fmpEnabled]);

  async function runFilingDigest(f: Filing): Promise<void> {
    setDigestingKey(f.accession_number);
    setErr(null);
    try {
      const r = await digestFiling({
        ticker,
        filing: f,
        report,
        extra_context: extraContext,
      });
      onDigested(r.source ?? `${f.form} filed ${f.filed}`, r.digest);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setDigestingKey(null);
    }
  }

  async function runTranscriptDigest(t: TranscriptMeta): Promise<void> {
    const key = `${t.year}-${t.quarter}`;
    setDigestingKey(key);
    setErr(null);
    try {
      const r = await digestTranscript({
        ticker,
        year: t.year,
        quarter: t.quarter,
        report,
        extra_context: extraContext,
      });
      onDigested(
        r.source ?? `Q${t.quarter} FY${t.year} earnings call`,
        r.digest
      );
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setDigestingKey(null);
    }
  }

  const filingRows = (filings?.filings ?? []).slice(0, 20);
  const transcriptRows = transcripts?.transcripts ?? [];

  return (
    <div>
      {err && (
        <div className="mb-4 rounded-lg border border-down/40 bg-down/10 px-4 py-3 text-sm text-down">
          {err}
        </div>
      )}

      <div className="grid grid-cols-1 gap-4">
        <Card
          title="SEC filings"
          subtitle="Primary sources, read directly from EDGAR — the analyst extracts Risk Factors, MD&A and Business sections"
          right={
            !aiEnabled ? (
              <Badge tone="flat">
                AI off — set ANTHROPIC_API_KEY to digest filings
              </Badge>
            ) : undefined
          }
        >
          {filingsLoading ? (
            <div className="flex items-center justify-center py-10">
              <Spinner />
            </div>
          ) : filingsError ? (
            <EmptyState title="Couldn't load filings" hint={filingsError} />
          ) : filingRows.length === 0 ? (
            <EmptyState title="No filings found" />
          ) : (
            <>
              <Table>
                <thead>
                  <tr>
                    <TH align="left">Form</TH>
                    <TH align="left">Filed</TH>
                    <TH align="left">Period</TH>
                    <TH align="left">Description</TH>
                    <TH align="left">Doc</TH>
                    <TH align="right">Action</TH>
                  </tr>
                </thead>
                <tbody>
                  {filingRows.map((f, i) => {
                    const key = f.accession_number || `${f.form}-${f.filed}-${i}`;
                    const isDigesting =
                      digestingKey !== null &&
                      digestingKey === f.accession_number;
                    const is10K = (f.form ?? "").startsWith("10-K");
                    return (
                      <tr key={key}>
                        <TD align="left">
                          <Badge tone={is10K ? "brand" : "neutral"}>
                            {f.form || "—"}
                          </Badge>
                        </TD>
                        <TD align="left" num>
                          {fmtDate(f.filed) || "—"}
                        </TD>
                        <TD align="left" num>
                          {fmtDate(f.report_date) || "—"}
                        </TD>
                        <TD align="left" className="text-ink-dim">
                          <div
                            className="max-w-[280px] truncate"
                            title={f.description || undefined}
                          >
                            {f.description || "—"}
                          </div>
                        </TD>
                        <TD align="left">
                          {f.url ? (
                            <a
                              href={f.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-brand hover:underline"
                            >
                              view
                            </a>
                          ) : (
                            <span className="text-ink-faint">—</span>
                          )}
                        </TD>
                        <TD align="right">
                          <Button
                            variant="subtle"
                            disabled={!aiEnabled || digestingKey !== null}
                            onClick={() => runFilingDigest(f)}
                          >
                            {isDigesting ? (
                              <>
                                <Spinner className="h-3 w-3" /> Reading…
                              </>
                            ) : (
                              "Digest"
                            )}
                          </Button>
                        </TD>
                      </tr>
                    );
                  })}
                </tbody>
              </Table>
              <p className="mt-2 text-[11px] text-ink-faint">
                Digest = the AI reads the filing against your live model and
                returns a cited brief with suggested assumption changes.
              </p>
            </>
          )}
        </Card>

        <Card title="Earnings calls" subtitle="Transcripts via FMP">
          {transcripts === null ? (
            <div className="flex items-center justify-center py-6">
              <Spinner />
            </div>
          ) : !transcripts.enabled ? (
            <EmptyState
              title="Transcripts need FMP"
              hint="Set FMP_API_KEY to list and digest earnings-call transcripts."
            />
          ) : transcriptRows.length === 0 ? (
            <EmptyState title="No transcripts available" />
          ) : (
            <Table>
              <thead>
                <tr>
                  <TH align="left">Quarter</TH>
                  <TH align="left">Date</TH>
                  <TH align="right">Action</TH>
                </tr>
              </thead>
              <tbody>
                {transcriptRows.map((t, i) => {
                  const key = `${t.year}-${t.quarter}`;
                  const isDigesting = digestingKey === key;
                  return (
                    <tr key={`${key}-${i}`}>
                      <TD align="left" num>
                        Q{t.quarter ?? "—"} FY{t.year ?? "—"}
                      </TD>
                      <TD align="left" num>
                        {fmtDate(t.date) || "—"}
                      </TD>
                      <TD align="right">
                        <Button
                          variant="subtle"
                          disabled={
                            !aiEnabled ||
                            digestingKey !== null ||
                            t.year == null ||
                            t.quarter == null
                          }
                          onClick={() => runTranscriptDigest(t)}
                        >
                          {isDigesting ? (
                            <>
                              <Spinner className="h-3 w-3" /> Reading…
                            </>
                          ) : (
                            "Digest"
                          )}
                        </Button>
                      </TD>
                    </tr>
                  );
                })}
              </tbody>
            </Table>
          )}
        </Card>
      </div>
    </div>
  );
}
