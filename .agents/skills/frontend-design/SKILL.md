---
name: frontend-design
description: "Guidance for distinctive, production-grade visual UI/UX design. Based on Anthropic official frontend-design and Vercel Web Interface Guidelines: anti-AI slop, bespoke color palettes, typography, hero moments, and complete interactive states."
license: MIT
metadata:
  origin: "anthropics/skills (frontend-design) + vercel-labs/agent-skills (web-design-guidelines)"
---

# Frontend Design & UI/UX Guidelines

This skill guides coding agents to create production-grade, intentional, and visually distinct user interfaces. It actively prevents generic, predictable "AI slop" and template defaults.

---

## 1. Core Mindset: Lead Design Studio

Approach every UI task as the lead design technologist at an award-winning design studio. The client has rejected cliché, cookie-cutter templates and is paying for a distinct, memorable visual identity.

### 🚫 Anti-AI Slop Mandate
Steer clear of default AI aesthetic tropes:
- **NO Generic Glowing Cards**: Stop putting purple/blue glowing borders and arbitrary gradient backdrops on every single card.
- **NO Arbitrary Headline Highlights**: Do not randomly wrap a single word in a headline with a bright neon gradient unless specifically requested.
- **NO Uniform Pill Buttons Everywhere**: Use button shapes and sizes that match visual hierarchy (primary, secondary, subtle ghost, destructive).
- **NO Monotonous Card Grids**: Vary the layout rhythm. Combine asymmetric columns, split panels, dense tables, and focused detail views.
- **NO Default Font Fallbacks**: Avoid defaulting immediately to system sans-serif or unconfigured Inter without intentional typographic pairing.

---

## 2. Grounding in the Subject Matter & Audience

Distinctive visual choices originate from the subject's industry, domain vernacular, and target audience:
- **Financial/DevOps Analytics**: High-density typography, tabular numbers (`font-variant-numeric: tabular-nums`), restrained palettes, clear status badges, minimal ornamentation.
- **Consumer / Lifestyle**: Generous whitespace, expressive typography, rich photography, tactile micro-interactions.
- **Creative / Dev Tools**: Sharp geometric accents, mono font pairings, dark/terminal aesthetic with high contrast and fast keyboard navigation.

---

## 3. Deliberate Palettes & Typography

### 🎨 Color Palette (4–6 Concrete Hex Codes)
Define a dedicated CSS variable palette in `design.md`:
1. `--color-background`: Deep canvas base (e.g., `#0b0f19` or `#fafafa`).
2. `--color-surface`: Card/panel elevated surface (e.g., `#111827` or `#ffffff`).
3. `--color-border`: Subtle boundary definition (e.g., `#1f2937` or `#e5e7eb`).
4. `--color-primary`: Signature brand hue (deliberate, e.g., `#4f46e5` or `#0ea5e9`).
5. `--color-accent`: High-contrast secondary accent for badges and active states.
6. `--color-text-primary` & `--color-text-muted`: Ensure WCAG AA (≥4.5:1) or AAA (≥7:1) contrast against background.

### 🔤 Typography Hierarchy
- Pair an intentional **Display/Heading** font (e.g., *Outfit*, *Plus Jakarta Sans*, *Syne*, or *Playfair Display*) with a legible **Body** font (*Inter*, *Geist*, *IBM Plex Sans*).
- Establish a clear type scale:
  - Display / Hero: 32px – 48px, bold, tight letter-spacing (`-0.02em`).
  - Section Headings: 20px – 24px, semibold.
  - Body Text: 14px – 16px, line-height 1.5 – 1.6.
  - Microcopy / Metadata: 12px – 13px, muted contrast.

---

## 4. The Hero Element & Visual Rhythm

- Open with the most characteristic artifact of the domain: a live interactive preview, dynamic metrics dashboard, real-time query bar, or focused illustration.
- Use deliberate spatial composition: asymmetric two-column splits, sticky sidebars, or layered cards that break the grid intentionally.

---

## 5. Five Mandatory Interactive Component States

Every UI component in `design.md` and implementation MUST define all 5 states:
1. **Default State**: Clean, token-governed visual styling.
2. **Hover / Focus State**: Subdued, purposeful transitions (150ms–200ms ease); explicit `:focus-visible` outline for keyboard accessibility.
3. **Loading State**: Content-shaped skeleton loaders or micro-spinners — **never a blank screen or raw unformatted layout shift**.
4. **Empty State**: Domain-specific icon/illustration, empathetic explanation, and a single obvious call-to-action button.
5. **Error State**: Non-blocking inline banner or card with an actionable explanation and a retry trigger.

---

## 6. Vercel Web Interface Guidelines Compliance

- **Spring & Micro-Motions**: Transitions should feel snappy (150ms–250ms), using cubic-bezier curves (`ease-out` or spring physics); never sluggish.
- **Accessibility & Contrast**: All interactive elements must support `Tab` navigation, visible focus rings, and screen-reader labels (`aria-label`).
- **Responsive Breakpoints**:
  - Mobile (`<= 640px`): Single-column, thumb-friendly tap targets (≥ 44px).
  - Tablet (`641px - 1024px`): Adaptive 2-column grid.
  - Desktop (`> 1024px`): Full multi-column dashboard with sticky navigation.
