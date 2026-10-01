"use client";

// AI equity researcher panel (v2). Feed material (text + PDFs), accumulate a
// history of cited digests with one-click assumption application, generate a
// full cited research note, and chat grounded in the model + everything fed.
// Built only from the shared primitives + api helpers.

import React, { useEffect, useRef, useState } from "react";
import type {
  AssumptionSuggestion,
  ChatTurn,
  CitedFact,
  Digest,
  DigestEntry,
  PdfAttachment,
  Report,
  ResearchNote,
} from "@/lib/types";
import {
  MAX_UPLOAD_BYTES,
  fileToBase64,
  jsonBytes,
  planPdfAttachments,
  postChat,
  postDigest,
} from "@/lib/api";
import { fmtBytes, fmtDate, fmtMult, fmtNum, fmtPct } from "@/lib/format";
import { Badge, Button, Card, EmptyState, Spinner, cx } from "@/components/ui";

// Render a suggestion value according to its declared unit. Rates are decimals.
function fmtByUnit(
  v: number | null | undefined,
  unit: AssumptionSuggestion["unit"] | null | undefined
): string {
  if (v == null || !Number.isFinite(v)) return "—";
  switch (unit) {
    case "percent":
      return fmtPct(v);
    case "multiple":
      return fmtMult(v);
    case "years":
      return `${Math.round(v)}y`;
    default:
      return fmtNum(v, 2);
  }
}

function sentimentTone(
  s: Digest["sentiment"] | null | undefined
): "up" | "down" | "flat" {
  switch (s) {
    case "bullish":
      return "up";
    case "bearish":
      return "down";
    default:
      return "flat";
  }
}

function stanceTone(
  s: ResearchNote["stance"] | null | undefined
): "up" | "down" | "flat" {
  switch (s) {
    case "constructive":
      return "up";
    case "cautious":
      return "down";
    default:
      return "flat";
  }
}

// A titled bullet list used by the research note sections.
function NoteSection({
  title,
  items,
  tone,
}: {
  title: string;
  items: string[] | null | undefined;
  tone?: string; // text-* class for the bullets (e.g. "text-down")
}) {
  const list = (items ?? []).filter(
    (x): x is string => typeof x === "string" && x.length > 0
  );
  return (
    <div>
      <div className="text-[11px] font-semibold uppercase tracking-wider text-ink-faint">
        {title}
      </div>
      {list.length > 0 ? (
        <ul
          className={cx(
            "mt-1 list-disc space-y-1 pl-5 text-sm",
            tone || "text-ink-dim"
          )}
        >
          {list.map((it: string, i: number) => (
            <li key={i}>{it}</li>
          ))}
        </ul>
      ) : (
        <p className="mt-1 text-sm text-ink-faint">—</p>
      )}
    </div>
  );
}

// One suggested assumption change with an Apply button.
function SuggestionRow({
  s,
  onApply,
}: {
  s: AssumptionSuggestion;
  onApply: (s: AssumptionSuggestion) => void;
}) {
  return (
    <div className="flex items-start justify-between gap-3 rounded-lg border border-line p-3">
      <div className="min-w-0">
        <div className="text-sm font-medium text-ink">{s.label || s.field}</div>
        <div className="num mt-0.5 text-sm text-ink-dim">
          {s.current_value == null
            ? "model default"
            : fmtByUnit(s.current_value, s.unit)}{" "}
          <span className="text-ink-faint">→</span>{" "}
          <span className="font-semibold text-ink">
            {fmtByUnit(s.suggested_value, s.unit)}
          </span>
        </div>
        {s.rationale && <p className="mt-1 text-xs text-ink-dim">{s.rationale}</p>}
      </div>
      <Button variant="subtle" onClick={() => onApply(s)}>
        Apply
      </Button>
    </div>
  );
}

const INPUT_CLS =
  "w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-faint focus:border-brand focus:outline-none";

export default function AIResearchPanel({
  report,
  researchNotes,
  setResearchNotes,
  onApplySuggestion,
  aiEnabled,
  digests,
  onDigested,
  note,
  onGenerateNote,
  noteLoading,
  onExportMemo,
}: {
  report: Report;
  researchNotes: string;
  setResearchNotes: (s: string) => void;
  onApplySuggestion: (s: AssumptionSuggestion) => void;
  aiEnabled: boolean;
  digests: DigestEntry[];
  onDigested: (source: string, digest: Digest) => void;
  note: ResearchNote | null;
  onGenerateNote: () => void;
  noteLoading: boolean;
  onExportMemo: () => void;
}) {
  const [material, setMaterial] = useState<string>("");
  const [pdfs, setPdfs] = useState<PdfAttachment[]>([]);
  const [digesting, setDigesting] = useState<boolean>(false);
  const [readingPdfs, setReadingPdfs] = useState(false);
  const readInProgress = useRef(false);
  const digestInProgress = useRef(false);
  const chatInProgress = useRef(false);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [chatInput, setChatInput] = useState<string>("");
  const [chatting, setChatting] = useState<boolean>(false);
  const [err, setErr] = useState<string | null>(null);
  const chatScrollRef = useRef<HTMLDivElement | null>(null);

  // Keep the newest message in view — without this, replies render below the
  // fold once the thread exceeds the container height.
  useEffect(() => {
    const el = chatScrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [turns, chatting]);
  const [expandedDigest, setExpandedDigest] = useState<number | null>(
    digests.length > 0 ? digests.length - 1 : null
  );

  // Default the newest digest expanded whenever the history grows.
  useEffect(() => {
    setExpandedDigest(digests.length > 0 ? digests.length - 1 : null);
  }, [digests.length]);

  async function runDigest(): Promise<void> {
    if (!aiEnabled || digestInProgress.current || readInProgress.current) return;
    digestInProgress.current = true;
    setDigesting(true);
    setErr(null);
    try {
      const d = await postDigest({
        report,
        extra_context: researchNotes,
        material_text: material,
        pdfs,
      });
      const src = material.trim()
        ? "Pasted material"
        : pdfs.map((p: PdfAttachment) => p.name).join(", ") || "Attached PDFs";
      // The page callback appends to the digest history AND the running
      // research notes — do not also call setResearchNotes here.
      onDigested(src, d);
      setMaterial("");
      setPdfs([]);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      digestInProgress.current = false;
      setDigesting(false);
    }
  }

  async function onPickPdfs(
    e: React.ChangeEvent<HTMLInputElement>
  ): Promise<void> {
    if (readInProgress.current || digestInProgress.current) return;
    const files = Array.from(e.target.files ?? []);
    // Reset so re-selecting the same file fires onChange again.
    e.target.value = "";
    setErr(null);
    readInProgress.current = true;
    setReadingPdfs(true);

    // The digest request carries the model, the research log, the pasted text
    // and every PDF (base64, about 4/3 of the file size) and is capped at
    // MAX_UPLOAD_BYTES. Sizes are checked before anything is read, so an
    // oversized file is refused with a clear message instead of failing
    // after the upload.
    const { accepted, refused } = planPdfAttachments(
      files,
      jsonBytes({
        report,
        extra_context: researchNotes,
        material_text: material,
        pdfs,
      })
    );

    try {
      const added = await Promise.all(
        accepted.map(
          async (f: File): Promise<PdfAttachment> => ({
            name: f.name,
            data_base64: await fileToBase64(f),
          })
        )
      );
      if (added.length > 0) setPdfs((p: PdfAttachment[]) => [...p, ...added]);
      if (refused.length > 0) setErr(refused.join(" "));
    } catch (e2) {
      setErr(e2 instanceof Error ? e2.message : String(e2));
    } finally {
      readInProgress.current = false;
      setReadingPdfs(false);
    }
  }

  function removePdf(idx: number): void {
    setPdfs((p: PdfAttachment[]) =>
      p.filter((_: PdfAttachment, i: number) => i !== idx)
    );
  }

  async function send(): Promise<void> {
    const q = chatInput.trim();
    if (!q || !aiEnabled || chatInProgress.current) return;
    chatInProgress.current = true;
    const nextTurns: ChatTurn[] = [...turns, { role: "user" as const, content: q }];
    setTurns(nextTurns);
    setChatInput("");
    setChatting(true);
    setErr(null);
    try {
      const reply = await postChat({
        report,
        extra_context: researchNotes,
        turns: nextTurns,
      });
      setTurns((t: ChatTurn[]) => [
        ...t,
        { role: "assistant" as const, content: reply },
      ]);
    } catch (e) {
      const m = e instanceof Error ? e.message : String(e);
      setErr(m);
      // Keep transport errors out of the model's conversation history, and
      // preserve a new draft typed while this request was in progress.
      setTurns(turns);
      setChatInput((current) => current || q);
    } finally {
      chatInProgress.current = false;
      setChatting(false);
    }
  }

  const canDigest =
    aiEnabled && !digesting && !readingPdfs && (material.trim() !== "" || pdfs.length > 0);
  const redFlags = note?.red_flags ?? [];
  const citations = note?.citations ?? [];

  return (
    <div className="space-y-4">
      {!aiEnabled && (
        <div>
          <Badge tone="flat">
            AI is off — set ANTHROPIC_API_KEY in .env to enable the researcher
          </Badge>
        </div>
      )}

      {err && (
        <div className="rounded-lg border border-down/40 bg-down/10 px-4 py-3 text-sm text-down">
          {err}
        </div>
      )}

      {/* ---- Research note ------------------------------------------------ */}
      <Card
        title="Research note"
        subtitle="A full cited write-up from the model + everything you've fed"
        right={
          <div className="flex gap-2">
            <Button
              variant="ghost"
              onClick={onGenerateNote}
              disabled={!aiEnabled || noteLoading}
            >
              {noteLoading ? (
                <>
                  <Spinner /> Writing…
                </>
              ) : note ? (
                "Regenerate"
              ) : (
                "Generate research note"
              )}
            </Button>
            {note && (
              <Button variant="ghost" onClick={onExportMemo}>
                Export Word memo
              </Button>
            )}
          </div>
        }
      >
        {!note && !noteLoading && (
          <EmptyState
            title="No research note yet"
            hint="Feed material below (or digest filings), then generate — the analyst writes a thesis, valuation view, risks, red flags, catalysts and falsifiers, with sources."
          />
        )}

        {noteLoading && !note && (
          <div className="flex items-center justify-center gap-3 py-8 text-sm text-ink-dim">
            <Spinner /> Writing the note (this reads everything you&apos;ve fed)…
          </div>
        )}

        {note && (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-3">
              <h2 className="text-lg font-semibold text-ink">
                {note.title || "Research note"}
              </h2>
              <Badge tone={stanceTone(note.stance)}>
                {note.stance ?? "balanced"}
              </Badge>
            </div>

            {note.executive_summary && (
              <p className="text-sm text-ink-dim">{note.executive_summary}</p>
            )}

            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <NoteSection title="Thesis" items={note.thesis} />
              <NoteSection title="Key drivers" items={note.key_drivers} />
              <NoteSection title="Risks" items={note.risks} />
              {redFlags.length > 0 && (
                <NoteSection title="Red flags" items={redFlags} tone="text-down" />
              )}
              <NoteSection title="Catalysts" items={note.catalysts} />
              <NoteSection
                title="What would change my mind"
                items={note.what_would_change_my_mind}
              />
            </div>

            <div>
              <div className="text-[11px] font-semibold uppercase tracking-wider text-ink-faint">
                Valuation view
              </div>
              <p className="mt-1 text-sm text-ink-dim">
                {note.valuation_view || "—"}
              </p>
            </div>

            {citations.length > 0 && (
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-wider text-ink-faint">
                  Sources
                </div>
                <div className="mt-1.5 flex flex-wrap gap-2">
                  {citations.map((c, i) => (
                    <span key={i} title={c?.note ?? ""}>
                      <Badge tone="neutral">{c?.source ?? "—"}</Badge>
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </Card>

      {/* ---- Feed + chat --------------------------------------------------- */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* LEFT — Feed the analyst */}
        <Card
          title="Feed the analyst"
          subtitle="Paste news / notes, attach PDFs — or use the Filings tab to feed SEC filings & earnings calls directly"
        >
          <div className="space-y-3">
            <textarea
              value={material}
              disabled={digesting || readingPdfs}
              onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) =>
                setMaterial(e.target.value)
              }
              rows={6}
              placeholder="Paste an earnings release, broker note, news article…"
              className={INPUT_CLS}
            />

            <div className="space-y-2">
              <label className="block text-xs font-medium text-ink-dim">
                Attach PDFs{" "}
                <span className="font-normal text-ink-faint">
                  (up to about {fmtBytes((MAX_UPLOAD_BYTES * 3) / 4)} in total)
                </span>
                <input
                  type="file"
                  accept="application/pdf"
                  multiple
                  disabled={digesting || readingPdfs}
                  onChange={onPickPdfs}
                  className="mt-1 block w-full text-xs text-ink-dim file:mr-3 file:rounded-md file:border file:border-line file:bg-surface-raised file:px-2 file:py-1 file:text-xs file:text-ink-dim hover:file:text-ink"
                />
              </label>
              {pdfs.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {pdfs.map((p: PdfAttachment, i: number) => (
                    <Badge key={`${p.name}-${i}`} tone="neutral">
                      <span className="max-w-[12rem] truncate">{p.name}</span>
                      <button
                        type="button"
                        onClick={() => removePdf(i)}
                        disabled={digesting || readingPdfs}
                        aria-label={`Remove ${p.name}`}
                        className="ml-1.5 text-ink-faint hover:text-down"
                      >
                        ×
                      </button>
                    </Badge>
                  ))}
                </div>
              )}
            </div>

            <Button disabled={!canDigest} onClick={runDigest}>
              {digesting ? (
                <>
                  <Spinner /> Reading…
                </>
              ) : readingPdfs ? (
                <><Spinner /> Loading PDFs…</>
              ) : (
                "Digest material"
              )}
            </Button>

            {/* Digest history */}
            <div className="space-y-2 border-t border-line pt-3">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-ink-faint">
                Digests ({digests.length})
              </div>

              {digests.length === 0 && (
                <EmptyState
                  title="Nothing fed yet"
                  hint="Digest pasted material here, or use the Filings tab to read 10-Ks and earnings calls."
                />
              )}

              {[...digests]
                .map((entry: DigestEntry, i: number) => ({ entry, idx: i }))
                .reverse()
                .map(({ entry, idx }) => {
                  const d: Digest | null | undefined = entry?.digest;
                  const facts: CitedFact[] = d?.key_facts ?? [];
                  const risks: string[] = d?.risks ?? [];
                  const catalysts: string[] = d?.catalysts ?? [];
                  const suggestions: AssumptionSuggestion[] =
                    d?.suggested_assumptions ?? [];
                  const expanded = expandedDigest === idx;
                  return (
                    <div key={idx} className="rounded-lg border border-line">
                      <button
                        type="button"
                        onClick={() =>
                          setExpandedDigest(expanded ? null : idx)
                        }
                        className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left"
                      >
                        <div className="flex min-w-0 items-center gap-2">
                          <span className="truncate text-sm font-medium text-ink">
                            {entry?.source || "Material"}
                          </span>
                          <Badge tone={sentimentTone(d?.sentiment)}>
                            {d?.sentiment ?? "neutral"}
                          </Badge>
                        </div>
                        <span className="shrink-0 text-[11px] text-ink-faint">
                          {fmtDate(entry?.at)}
                        </span>
                      </button>

                      {expanded && (
                        <div className="space-y-3 border-t border-line px-3 py-3">
                          <p className="whitespace-pre-wrap text-sm text-ink">
                            {d?.summary || "—"}
                          </p>

                          <div>
                            <div className="text-[11px] font-semibold uppercase tracking-wider text-ink-faint">
                              Key facts
                            </div>
                            {facts.length > 0 ? (
                              <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-ink-dim">
                                {facts.map((f: CitedFact, fi: number) => (
                                  <li key={fi}>
                                    {f?.fact ?? "—"}{" "}
                                    {f?.source && (
                                      <Badge tone="neutral">{f.source}</Badge>
                                    )}
                                  </li>
                                ))}
                              </ul>
                            ) : (
                              <p className="mt-1 text-sm text-ink-faint">—</p>
                            )}
                          </div>

                          {risks.length > 0 && (
                            <NoteSection title="Risks" items={risks} />
                          )}
                          {catalysts.length > 0 && (
                            <NoteSection title="Catalysts" items={catalysts} />
                          )}

                          <div className="space-y-2">
                            <div className="text-[11px] font-semibold uppercase tracking-wider text-ink-faint">
                              Suggested model changes
                            </div>
                            {suggestions.length > 0 ? (
                              suggestions.map(
                                (s: AssumptionSuggestion, si: number) => (
                                  <SuggestionRow
                                    key={`${s.field}-${si}`}
                                    s={s}
                                    onApply={onApplySuggestion}
                                  />
                                )
                              )
                            ) : (
                              <p className="text-xs text-ink-faint">
                                No assumption changes warranted.
                              </p>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
            </div>
          </div>
        </Card>

        {/* RIGHT — Ask the analyst */}
        <Card
          title="Ask the analyst"
          subtitle="Grounded in your loaded model and everything you've fed."
        >
          <div className="space-y-3">
            <div
              ref={chatScrollRef}
              className="max-h-[480px] space-y-3 overflow-y-auto"
            >
              {turns.length === 0 && !chatting && (
                <p className="px-1 py-2 text-sm text-ink-faint">
                  Ask about the thesis, what&apos;s priced in, where the model is
                  most fragile…
                </p>
              )}
              {turns.map((t: ChatTurn, i: number) => (
                <div
                  key={i}
                  className={t.role === "user" ? "text-right" : "text-left"}
                >
                  <div
                    className={
                      t.role === "user"
                        ? "inline-block max-w-[85%] rounded-lg bg-surface-hi px-3 py-2 text-left text-sm text-ink"
                        : "inline-block max-w-[85%] whitespace-pre-wrap rounded-lg bg-surface-raised px-3 py-2 text-left text-sm text-ink"
                    }
                  >
                    {t.content}
                  </div>
                </div>
              ))}
              {chatting && (
                <div className="flex items-center gap-2 text-sm text-ink-dim">
                  <Spinner /> Thinking…
                </div>
              )}
            </div>

            <form
              onSubmit={(e: React.FormEvent<HTMLFormElement>) => {
                e.preventDefault();
                void send();
              }}
              className="flex items-center gap-2 border-t border-line pt-3"
            >
              <input
                type="text"
                value={chatInput}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                  setChatInput(e.target.value)
                }
                placeholder={`Ask about ${report?.summary?.ticker ?? "the company"}…`}
                className={cx(INPUT_CLS, "flex-1 py-1.5")}
              />
              <Button
                type="submit"
                disabled={!aiEnabled || chatting || !chatInput.trim()}
              >
                Send
              </Button>
            </form>
          </div>
        </Card>
      </div>

      {/* ---- Research log --------------------------------------------------- */}
      <details>
        <summary className="text-xs text-ink-faint cursor-pointer">
          Research log (context the analyst carries)
        </summary>
        <textarea
          value={researchNotes}
          onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) =>
            setResearchNotes(e.target.value)
          }
          rows={5}
          placeholder="Everything digested so far accumulates here — edit or trim freely."
          className="mt-2 w-full rounded-lg border border-line bg-surface px-3 py-2 text-xs text-ink-dim placeholder:text-ink-faint focus:border-brand focus:outline-none"
        />
      </details>
    </div>
  );
}
