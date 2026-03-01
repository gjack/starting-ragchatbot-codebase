# Frontend Changes

## Dark / Light Mode Toggle Button

### Feature Summary
Added a circular icon button fixed to the top-right corner of the viewport that switches the UI between the existing dark theme and a new light theme.

---

### Files Changed

#### `frontend/index.html`
- Added a `<button id="themeToggle">` element directly inside `<body>`, before `.container`.
- The button contains two inline SVG icons:
  - **Sun** (`icon-sun`) — displayed in dark mode to invite switching to light.
  - **Moon** (`icon-moon`) — displayed in light mode to invite switching to dark.
- `aria-label` and `title` attributes provide accessible labels; icons carry `aria-hidden="true"`.

#### `frontend/style.css`
- **New component-specific tokens in `:root`** — added `--code-bg`, `--error-text/bg/border`, `--success-text/bg/border` so all previously hardcoded dark-mode colours are now variable-driven.
- **`body.light-mode` block** — full set of overrides for the light theme:

  | Token | Dark (`:root`) | Light (`body.light-mode`) | Notes |
  |---|---|---|---|
  | `--background` | `#0f172a` | `#f8fafc` | Page background |
  | `--surface` | `#1e293b` | `#ffffff` | Cards / sidebar |
  | `--surface-hover` | `#334155` | `#f1f5f9` | Hover states |
  | `--text-primary` | `#f1f5f9` | `#1e293b` | ~16:1 on bg |
  | `--text-secondary` | `#94a3b8` | `#475569` | ~7:1 on bg (upgraded) |
  | `--border-color` | `#334155` | `#cbd5e1` | Subtle visible borders |
  | `--primary-color` | `#2563eb` | `#1d4ed8` | ~5.9:1 on white — WCAG AA |
  | `--primary-hover` | `#1d4ed8` | `#1e40af` | Darker hover state |
  | `--focus-ring` | `rgba(37,99,235,0.2)` | `rgba(29,78,216,0.25)` | Matches darker primary |
  | `--assistant-message` | `#374151` | `#f1f5f9` | Bubble background |
  | `--welcome-bg` | `#1e3a5f` | `#eff6ff` | Welcome card |
  | `--welcome-border` | `#2563eb` | `#93c5fd` | Card accent |
  | `--code-bg` | `rgba(0,0,0,0.2)` | `rgba(15,23,42,0.06)` | Subtle code tint |
  | `--error-text` | `#f87171` | `#b91c1c` | ~5.6:1 on white |
  | `--error-bg` | `rgba(239,68,68,0.1)` | `rgba(220,38,38,0.08)` | |
  | `--success-text` | `#4ade80` | `#15803d` | ~5.2:1 on white |
  | `--success-bg` | `rgba(34,197,94,0.1)` | `rgba(22,163,74,0.08)` | |

- **CSS rule updates** — replaced hardcoded colour literals with variables in:
  - `.message-content code` and `.message-content pre` → `var(--code-bg)`
  - `.error-message` → `var(--error-text)`, `var(--error-bg)`, `var(--error-border)`
  - `.success-message` → `var(--success-text)`, `var(--success-bg)`, `var(--success-border)`
- **Global transition rule** — adds `transition` for `background-color`, `color`, `border-color`, and `box-shadow` on `body *` so every element animates smoothly (0.25 s ease) when the theme class toggles.
- **`.theme-toggle` styles** — 40 px circle button, fixed position `top: 1rem; right: 1rem; z-index: 1000`, inherits surface/border variables, hover scales up (`transform: scale(1.1)`) and tints with `--primary-color`, focus uses `focus-visible` with a 3 px `--focus-ring` outline.
- **Icon visibility rules** — `body:not(.light-mode) .icon-sun` / `.icon-moon` selectors swap which icon is shown using `display: block/none`.
- **SVG rotation** — `.theme-toggle:active svg` applies a 30° rotation for tactile click feedback.

#### `frontend/script.js`
- **`themeToggle` DOM reference** added to the global element list.
- **Theme restoration on load** — reads `localStorage.getItem('theme')` inside `DOMContentLoaded`; if `'light'`, adds `light-mode` to `<body>` before the first paint.
- **Click handler** in `setupEventListeners`:
  - Calls `document.body.classList.toggle('light-mode')`.
  - Saves the new preference to `localStorage` (`'light'` or `'dark'`).
  - Updates `aria-label` dynamically to reflect the current state ("Switch to dark mode" / "Switch to light mode").

---

### Accessibility
- `aria-label` on the button describes the action (not the current state).
- `aria-label` is updated after each toggle so screen readers announce the correct next action.
- `focus-visible` outline ensures keyboard users see a clear focus indicator without affecting mouse interactions.
- SVG icons carry `aria-hidden="true"` so they are not read by screen readers.

### Persistence
User preference is stored in `localStorage` under the key `theme` and restored on every page load.