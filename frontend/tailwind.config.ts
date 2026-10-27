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
        // Text tokens meet WCAG AA (4.5:1) on every surface: ink-faint is
        // 4.7:1 on surface-hi, 5.7:1 on surface (tests/accessibility.test.mjs).
        ink: {
          DEFAULT: "#e6edf6",
          dim: "#9babc2",
          faint: "#8090aa",
        },
        // brand is the accent for text, borders and sliders; filled buttons
        // with white text use brand-dim (4.8:1) and brand-deep on hover (5.7:1).
        brand: {
          DEFAULT: "#5b9dff",
          dim: "#3b6fd1",
          deep: "#2f63c4",
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
