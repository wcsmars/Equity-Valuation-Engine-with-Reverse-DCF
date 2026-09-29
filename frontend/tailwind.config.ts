import type { Config } from "tailwindcss";

// Design tokens for the whole app. Panels should use these semantic colors
// (bg-surface, border-line, text-ink, text-up/down/flat) rather than ad-hoc hex.
const config: Config = {
  // lib/ too: its helpers (lib/format.ts) return tone classes.
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "#080b12",
        surface: {
          DEFAULT: "#0e1422",
          raised: "#141c2e",
          hi: "#1b2538",
        },
        line: "#243043",
        ink: {
          DEFAULT: "#e6edf6",
          dim: "#9babc2",
          faint: "#62748f",
        },
        brand: {
          DEFAULT: "#5b9dff",
          dim: "#3b6fd1",
        },
        up: "#34d399", // undervalued / positive
        down: "#fb7185", // overvalued / negative
        flat: "#fbbf24", // fairly valued / neutral
      },
      fontFamily: {
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
    },
  },
  plugins: [],
};

export default config;
