"use client";

// API key settings (the alternative to editing the server's .env). The
// toggle and both fields stay available once a provider is configured, so a
// mistyped or revoked key can be replaced here: the backend checks only a
// key's form, and an AI error points the user back to this form. A field left
// blank keeps the stored key.

import React from "react";
import { Button, Spinner } from "@/components/ui";

export interface KeyStatus {
  fmp_enabled: boolean;
  anthropic_enabled: boolean;
}

export function ApiKeysToggle({
  status,
  open,
  onToggle,
}: {
  status: KeyStatus;
  open: boolean;
  onToggle: () => void;
}) {
  const all = status.fmp_enabled && status.anthropic_enabled;
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-expanded={open}
      className="rounded-md border border-brand/40 bg-brand/10 px-2 py-0.5 font-medium text-brand transition hover:bg-brand/20"
    >
      {open ? "Close" : all ? "API keys" : "Add API keys"}
    </button>
  );
}

const INPUT =
  "w-72 rounded-lg border border-line bg-surface-raised px-3 py-1.5 text-sm text-ink placeholder:text-ink-faint focus:border-brand focus:outline-none";
const REPLACE = "configured — paste to replace";

export function ApiKeysForm({
  status,
  anthropicKey,
  fmpKey,
  setAnthropicKey,
  setFmpKey,
  saving,
  onSubmit,
}: {
  status: KeyStatus;
  anthropicKey: string;
  fmpKey: string;
  setAnthropicKey: (v: string) => void;
  setFmpKey: (v: string) => void;
  saving: boolean;
  onSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
}) {
  return (
    <form
      onSubmit={onSubmit}
      className="mt-2 flex flex-wrap items-end gap-3 rounded-xl border border-line bg-surface px-4 py-3"
    >
      <label className="flex flex-col gap-1 text-[11px] text-ink-dim">
        Anthropic API key (AI researcher)
        <input
          type="password"
          value={anthropicKey}
          onChange={(e) => setAnthropicKey(e.target.value)}
          placeholder={status.anthropic_enabled ? REPLACE : "sk-ant-…"}
          className={INPUT}
        />
      </label>
      <label className="flex flex-col gap-1 text-[11px] text-ink-dim">
        FMP API key (news, transcripts, estimates)
        <input
          type="password"
          value={fmpKey}
          onChange={(e) => setFmpKey(e.target.value)}
          placeholder={status.fmp_enabled ? REPLACE : "optional"}
          className={INPUT}
        />
      </label>
      <Button type="submit" disabled={saving || (!anthropicKey && !fmpKey)}>
        {saving ? <Spinner /> : "Save keys"}
      </Button>
      <span className="text-[11px] text-ink-faint">
        Stored in the server&apos;s local .env and sent only to the selected API
        provider. A blank field keeps the stored key.
      </span>
    </form>
  );
}
