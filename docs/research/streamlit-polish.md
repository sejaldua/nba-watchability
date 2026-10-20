# Streamlit polish research (primary sources)

Researched 2026-10-20 for the bsg-ui restyle of NBA Watchability. Every claim cites a Streamlit doc page, a file in the `streamlit/streamlit` repo at a tagged version, or the installed package source. Non-primary material is labeled as such.

## Version scope (read this first)

- **Installed locally:** Streamlit **1.57.0** (`pip show streamlit`, Homebrew Python 3.14 at `/opt/homebrew/lib/python3.14/site-packages/streamlit`). The repo has no venv.
- **In CI:** `requirements.txt` says `streamlit>=1.32` and `.github/workflows/post_daily_slack.yml` runs `pip install -r requirements.txt` on Python 3.11. So the Action that takes the Slack screenshot installs whatever is newest on PyPI. As of today that is **1.66.0 (October 14, 2026)** ([release notes](https://docs.streamlit.io/develop/quick-reference/release-notes)).
- Findings below are scoped to **1.57.0** and cite source at that tag. Anything added in 1.58 to 1.66 is marked **[newer: 1.xx]**.

## Key takeaways for this app

1. **Pin Streamlit** (e.g. `streamlit==1.57.0`, or a newer version we test against). Local and CI run different versions nine minor releases apart, and 1.66 changed DOM tags that custom CSS may target ([1.66.0 notes](https://docs.streamlit.io/develop/quick-reference/release-notes/2026)).
2. **Load the screenshot URL with `?embed=true`.** The frontend reads this from the query string on any Streamlit server, not only Community Cloud. It hides the toolbar and header and cuts main padding from 6rem top / 10rem bottom to 2.25rem / 1rem. That is roughly 220 CSS px back in a 1200 px viewport ([utils.ts@1.57.0](https://github.com/streamlit/streamlit/blob/1.57.0/frontend/lib/src/util/utils.ts), [AppView styled-components@1.57.0](https://github.com/streamlit/streamlit/blob/1.57.0/frontend/app/src/components/AppView/styled-components.ts)).
3. **Move styling into `config.toml` before writing CSS.** In 1.57, `[theme]` already covers fonts (including self-hosted `[[theme.fontFaces]]`), radii, heading sizes and weights, the 7-color palette, chart palettes, dataframe header and border colors, and `metricValueFontSize`/`Weight` ([config.py@1.57.0](https://github.com/streamlit/streamlit/blob/1.57.0/lib/streamlit/config.py)).
4. **Self-host the bsg-ui fonts.** Put the `.woff2` files in `app/static/` (next to `app/streamlit_app.py`, which is already where static files live). Declare them with `[[theme.fontFaces]]` using `url = "app/static/..."`. `enableStaticServing = true` is already set ([fonts doc](https://docs.streamlit.io/develop/concepts/configuration/theming-customize-fonts)). This also takes the Google Fonts network fetch out of the CI screenshot path.
5. **Use `st.html("<style>...</style>")` for global CSS, not `st.markdown(unsafe_allow_html=True)`.** A style-only `st.html` call goes to the event container, takes no layout space, and is not iframed, so it applies to the whole page ([html.py@1.57.0](https://github.com/streamlit/streamlit/blob/1.57.0/lib/streamlit/elements/html.py)).
6. **`st.html` strips inline SVG.** DOMPurify runs with `USE_PROFILES: {html: true}` ("no SVG/MathML"). Network logos must be `<img>` tags (data URI or `app/static/...`), or must go through `st.markdown`, which uses rehype-raw instead of DOMPurify ([dompurifyHooks.ts@1.57.0](https://github.com/streamlit/streamlit/blob/1.57.0/frontend/lib/src/components/elements/Html/dompurifyHooks.ts)).
7. **Target `st-key-<key>` classes** from `st.container(key=...)` (and keyed widgets) instead of emotion classes or deep `data-testid` chains. It is the only element hook Streamlit documents for CSS ([layouts.py docstring@1.57.0](https://github.com/streamlit/streamlit/blob/1.57.0/lib/streamlit/elements/layouts.py)).
8. **Build the layout from native flex primitives:** `st.container(horizontal=True, gap=..., border=True)`, `st.columns(gap=..., vertical_alignment=..., border=True)`, `st.space`, and `width`/`height` on elements. Streamlit's own demo apps use exactly these plus `config.toml`, with zero custom CSS ([demo-seattle-weather](https://github.com/streamlit/demo-seattle-weather/blob/main/streamlit_app.py)).
9. **Dead selector cleanup:** `div[data-testid="collapsedControl"]` (`app/dashboard_views.py` line 244) does not exist in the 1.57 bundle. The expand button is now `stExpandSidebarButton`. `.block-container` still exists as a class alongside `data-testid="stMainBlockContainer"`.
10. **CCv2 is not needed for a static screenshot dashboard.** It earns its keep only for bidirectional interactivity (JS events back to Python). `st.html` plus theme CSS variables covers the restyle.

---

## 1. `config.toml` `[theme]` options in 1.57.0

Source of truth: `_create_theme_options(...)` calls in [lib/streamlit/config.py@1.57.0](https://github.com/streamlit/streamlit/blob/1.57.0/lib/streamlit/config.py). Docs: [config.toml reference](https://docs.streamlit.io/develop/api-reference/configuration/config.toml#theme), [theming overview](https://docs.streamlit.io/develop/concepts/configuration/theming).

**Sections.** `[theme]`, `[theme.sidebar]`, `[theme.light]`, `[theme.dark]`, `[theme.light.sidebar]`, `[theme.dark.sidebar]` (the `CustomThemeCategories` enum in config.py). Light/dark custom sections arrived in 1.51.0 ([2025 notes](https://docs.streamlit.io/develop/quick-reference/release-notes/2025)). `base` takes `"light"`, `"dark"`, a local TOML path, or a URL to a TOML theme file (config.py `base` description). Recommendation: set `base = "light"` explicitly so the headless screenshot never picks up a dark system preference.

**Options available in `[theme]` and all sub-sections:** `primaryColor`, `backgroundColor`, `secondaryBackgroundColor`, `textColor`, `linkColor`, `linkUnderline`, `codeTextColor`, `codeBackgroundColor`, `font`, `headingFont`, `headingFontSizes`, `headingFontWeights`, `codeFont`, `codeFontSize`, `codeFontWeight`, `baseRadius`, `buttonRadius`, `borderColor`, `dataframeBorderColor`, `dataframeHeaderBackgroundColor`, `showWidgetBorder`, the palette `red|orange|yellow|blue|green|violet|grayColor`, and each palette color's `...BackgroundColor` and `...TextColor` variant.

**`[theme]` only:** `base`, `fontFaces`, `baseFontSize`, `baseFontWeight`, `metricValueFontSize`, `metricValueFontWeight`, `showSidebarBorder`, `chartCategoricalColors`, `chartSequentialColors`, `chartDivergingColors`.

Details (quoted or paraphrased from config.py descriptions at 1.57.0):

- **`font` / `headingFont` / `codeFont`:** `"sans-serif"`, `"serif"`, `"monospace"`, a `family` from `[[theme.fontFaces]]`, or `"<name>:<css url>"`, as a comma-separated list for fallbacks. Names with spaces need inner quotes, e.g. `"'Space Mono':https://..."` ([fonts doc](https://docs.streamlit.io/develop/concepts/configuration/theming-customize-fonts)).
- **`[[theme.fontFaces]]`:** a table with `family`, `url`, and optional `weight`, `style`, `unicodeRange`. Repeat the table for each file. Self-hosting requires `server.enableStaticServing = true`, with URLs like `app/static/font_file.woff` (config.py; [fonts doc](https://docs.streamlit.io/develop/concepts/configuration/theming-customize-fonts)). Formats: "OTF, TTF, WOFF, and WOFF2" (fonts doc). Weight ranges such as `"200 800"` work for variable fonts (fonts doc; also [variable fonts tutorial](https://docs.streamlit.io/develop/tutorials/configuration-and-theming/variable-fonts) and [static fonts tutorial](https://docs.streamlit.io/develop/tutorials/configuration-and-theming/static-fonts)). `fontFaces` changes need a server restart ([theming overview](https://docs.streamlit.io/develop/concepts/configuration/theming)).
  ```toml
  [server]
  enableStaticServing = true

  [[theme.fontFaces]]
  family = "Action NBA Condensed"
  url = "app/static/fonts/ActionNBACondensed-Bold.woff2"
  weight = 700

  [theme]
  font = "Inter, sans-serif"            # plus a fontFaces entry for Inter
  headingFont = "'Action NBA Condensed', Inter, sans-serif"
  ```
  The static folder is "`./static/` relative to the running app file" and is served at `app/static/<file>` ([static files doc](https://docs.streamlit.io/develop/concepts/configuration/serving-static-files)). For this repo that means `app/static/`, which already exists.
  - Doc vs source discrepancy: the doc says unlisted extensions are served as `text/plain`. The 1.57.0 Starlette handler uses `mimetypes.guess_type` (`guess_content_type` in `lib/streamlit/web/server/component_file_utils.py`) with `X-Content-Type-Options: nosniff`, and has a 200 MB per-file cap (`MAX_APP_STATIC_FILE_SIZE` in `web/server/starlette/starlette_server_config.py`). Fonts are on both lists, so this does not affect fonts.
- **`baseFontSize`:** integer px, default 16. It scales rem-based spacing, gaps included (gap docstrings say "The rem unit is relative to the `theme.baseFontSize`"). **`baseFontWeight`:** multiples of 50 from 100 to 600, main area only ([fonts doc](https://docs.streamlit.io/develop/concepts/configuration/theming-customize-fonts)).
- **`headingFontSizes` / `headingFontWeights`:** an array for h1 to h6, or a single value. A shorter array leaves the remaining levels at defaults. Defaults are `2.75rem, 2.25rem, 1.75rem, 1.5rem, 1.25rem, 1rem`. Sizes are "not inherited by [theme.sidebar]". rem is recommended (config.py).
- **`baseRadius` / `buttonRadius`:** `"none" | "small" | "medium" | "large" | "full"` or a px/rem value (config.py).
- **`dataframeHeaderBackgroundColor`:** applies to "all non-interior cells" (header row, selection column, add-row row). If unset, Streamlit mixes background and secondary background. **`dataframeBorderColor`** covers dataframe and table borders (1.46.0 notes). **[newer: 1.65]** `dataframeHeaderTextColor` ([2026 notes](https://docs.streamlit.io/develop/quick-reference/release-notes/2026)).
- **Chart colors:** `chartCategoricalColors` (cycled list), `chartSequentialColors` and `chartDivergingColors` (exactly ten, or defaults are used). These "apply to Plotly, Altair, and Vega-Lite charts" (config.py). **[newer: 1.62]** chart colors can be set per light/dark/sidebar.
- **Palette:** `redColor`, etc. feed Markdown color directives, status elements, sparklines, `st.metric`, and `st.badge` (1.50.0 notes). `...BackgroundColor` defaults to the base color at 10% opacity (light) or 20% (dark) (config.py `redBackgroundColor`).
- **`metricValueFontSize` / `metricValueFontWeight`:** default 2.25rem / inherited (config.py; added 1.55.0).
- **Sidebar:** `showSidebarBorder` is `[theme]` only. Every other color, font, and radius option can be overridden in `[theme.sidebar]`.
- **[newer: 1.66]** `theme.paddingTop` / `theme.paddingBottom`, plus `[theme.sidebar]` overrides, "the gap under the header to the first widget". The 1.66 description says unset means "typically 6rem, or 8rem with top nav" on top and 10rem on the bottom ([config.py@1.66.0](https://github.com/streamlit/streamlit/blob/1.66.0/lib/streamlit/config.py); design spec [specs/2026-09-17-page-content-padding/product-spec.md](https://github.com/streamlit/streamlit/blob/develop/specs/2026-09-17-page-content-padding/product-spec.md)). That spec calls CSS on `.block-container` / `.stMainBlockContainer` a workaround that "breaks across releases".

Reference configs from Streamlit's own demos (both `.streamlit/config.toml`): [demo-stockpeers](https://github.com/streamlit/demo-stockpeers/blob/main/.streamlit/config.toml) (dark, Space Grotesk, `baseFontSize = 14`, light heading weights) and [demo-seattle-weather](https://github.com/streamlit/demo-seattle-weather/blob/main/.streamlit/config.toml) (light, Open Sans, `baseFontSize = 14`, `showWidgetBorder = true`, `headingFontSizes = ["3rem","2rem","1rem"]`).

## 2. CSS and HTML injection: `st.html` vs `st.markdown` vs `components.v1.html`

| | `st.html` (default) | `st.html(unsafe_allow_javascript=True)` | `st.markdown(unsafe_allow_html=True)` | `components.v1.html` / `st.iframe` |
|---|---|---|---|---|
| Iframed? | No | No | No | Yes |
| Sanitizer | DOMPurify, `USE_PROFILES:{html:true}`, `FORCE_BODY:true` | Same plus `ADD_TAGS:["script","style"]`; scripts are re-created so they execute | No DOMPurify; HTML is parsed by `rehype-raw` inside react-markdown | None (sandboxed frame) |
| `<style>` | Kept; applies to the whole page | Kept | Rendered; applies to the whole page | Scoped to the iframe |
| `<script>` | Stripped | Executed | Not executed (see note) | Executes inside iframe |
| Inline `<svg>` | **Stripped** (html profile excludes SVG/MathML) | Stripped | Parsed by rehype-raw | Works |
| `on*=` handlers | Stripped | Stripped | n/a | Works |

Sources:
- `st.html` docstring: "We sanitize HTML with DOMPurify", "`st.html` content is **not** iframed", "By default, JavaScript is ignored". A CSS file path is auto-wrapped in `<style>`. "When the resulting HTML content only contains style tags, Streamlit will send the content to the event container instead of the main container to avoid taking up space in the app." ([html.py@1.57.0](https://github.com/streamlit/streamlit/blob/1.57.0/lib/streamlit/elements/html.py); fix tracked as issue #9388 in the code comment.)
- Sanitizer options: `SANITIZE_HTML_BASE_OPTIONS = { USE_PROFILES: { html: true } /* no SVG/MathML */, FORCE_BODY: true }` ([dompurifyHooks.ts@1.57.0](https://github.com/streamlit/streamlit/blob/1.57.0/frontend/lib/src/components/elements/Html/dompurifyHooks.ts)). The JS path adds `ADD_TAGS: ["script","style"]` and clones `<script>` nodes to run them ([HtmlWithJs.tsx@1.57.0](https://github.com/streamlit/streamlit/blob/1.57.0/frontend/lib/src/components/elements/Html/HtmlWithJs.tsx)). DOMPurify's own `html` tag list includes `style` ([DOMPurify src/tags.ts](https://github.com/cure53/DOMPurify/blob/main/src/tags.ts)). `target="_blank"` links get `rel="noopener noreferrer"` (dompurifyHooks.ts).
- `st.markdown`: with `allowHTML`, the lazy-loaded `rehype-raw` plugin is added to react-markdown's `rehypePlugins` ([StreamlitMarkdown.tsx@1.57.0](https://github.com/streamlit/streamlit/blob/1.57.0/frontend/lib/src/components/shared/StreamlitMarkdown/StreamlitMarkdown.tsx), lines ~1088 to 1164). Docstring: with `unsafe_allow_html=True`, "any HTML expressions within `body` will be rendered" (`lib/streamlit/elements/markdown.py`). **[newer: 1.58]** blocks `javascript:` / `vbscript:` link URLs.
  - Note (inference, not a Streamlit statement): scripts don't run here because React-created `<script>` elements are inert. The usual reason "raw HTML blobs break" in `st.markdown` is Markdown parsing, not sanitizing. Under [CommonMark](https://spec.commonmark.org/0.31.2/#html-blocks), a blank line ends an HTML block, and lines indented 4+ spaces become code blocks. So dedent and strip blank lines in HTML strings passed to `st.markdown`.
- `components.v1.html` and `components.v1.iframe` are deprecated in favor of `st.iframe`, with deprecation date 2026-06-01 (`lib/streamlit/components/v1/__init__.py` at 1.57.0). `st.iframe` (added 1.56.0) takes `height="content"`, which auto-sizes HTML strings (`lib/streamlit/elements/iframe.py`). An iframe cannot style the parent page, so it is useless for global CSS.

**Recommendation:** one `st.html(Path("app/styles/bsg.css"))` call (a CSS path auto-wraps and lands in the event container). Keep `st.markdown(unsafe_allow_html=True)` only for card markup that needs inline SVG. Otherwise use `st.html`, with logos as `<img>`.

## 3. Stable selectors

- **`st-key-<key>`, documented:** every `key` docstring on containers and forms says "if `key` is provided, it will be used as a CSS class name prefixed with `st-key-`" ([layouts.py@1.57.0](https://github.com/streamlit/streamlit/blob/1.57.0/lib/streamlit/elements/layouts.py) `container`; same text in `form.py`, `arrow.py`, `plotly_chart.py`). Implementation: `convertKeyToClassName` replaces every character outside `[a-zA-Z0-9_-]` with `-` ([Block/utils.ts@1.57.0](https://github.com/streamlit/streamlit/blob/1.57.0/frontend/lib/src/components/core/Block/utils.ts)). It is applied to keyed blocks (`Block.tsx`), keyed elements and widgets (`ElementContainer.tsx`), and tabs (`Tabs.tsx`). Widget keys have appeared as `st-key-` classes since 1.39.0 ([2024 notes](https://docs.streamlit.io/develop/quick-reference/release-notes/2024)). Keys must be unique, so for a "category" style use a shared prefix and `[class*="st-key-card-"]` (limitation discussed in [issue #9666](https://github.com/streamlit/streamlit/issues/9666)).
- **`data-testid` / `stXxx` class names, used but not a contract:** the frontend sets both a class and a `data-testid` per element. Examples include `stMainBlockContainer block-container`, `stHeader`, `stToolbar`, `stSidebar`, `stMain`, `stAppDeployButton`, `stExpandSidebarButton`, `stSpace`, `stHtml` (grep of the 1.57.0 bundle in `streamlit/static/static/js/index.*.js`). Streamlit's own e2e tests drive these IDs, so they change less often than emotion classes. No doc promises they are stable, though.
  - **[newer: 1.66]** `stMain` moved from `<section>` to `<main>`, and nav containers became `<nav>`. Release notes: "Custom selectors such as `section[data-testid="stMain"]` should drop the tag." Sidebar collapse state moved from `aria-expanded` to `data-collapsed` ([2026 notes](https://docs.streamlit.io/develop/quick-reference/release-notes/2026); [PR #17418](https://github.com/streamlit/streamlit/pull/17418)). The sidebar itself is still `styled.section` in 1.66.0 (`frontend/app/src/components/Sidebar/styled-components.ts`). Rule: **write `[data-testid="stX"]`, never `tag[data-testid=...]`.**
- **Emotion classes (`st-emotion-cache-xxxx`, `e1o5kn6k0`):** hashed per build (visible as `target:"e1o5kn6k0"` in the minified bundle). Never target these.
- Staff guidance (non-doc, forum): a pinned Streamlit forum FAQ by user `dataprofessor` says custom CSS is "not officially endorsed" and that "class names and IDs ... might change in future releases" ([forum FAQ](https://discuss.streamlit.io/t/faq-how-to-customize-the-style-or-appearance-of-your-streamlit-app/63878)). The forum is Streamlit-owned, but this is not a docs page.

## 4. Layout primitives (1.57.0 signatures)

- **`st.set_page_config(layout="wide")`:** "Page elements use the entire screen width" ([doc](https://docs.streamlit.io/develop/api-reference/configuration/st.set_page_config)). In source, wide mode sets `maxWidth: initial` and 5rem side padding once the viewport passes about 864 px (`applyWideModePadding`, `wideSidePadding: "5rem"`, [sizes.ts@1.57.0](https://github.com/streamlit/streamlit/blob/1.57.0/frontend/lib/src/theme/primitives/sizes.ts)). At 1600 px that leaves about 1440 px of content width.
- **`st.columns(spec, *, gap="small", vertical_alignment="top", border=False, width="stretch")`:** gap is `"xxsmall"` (0.25rem) through `"xxlarge"` (8rem), or `None`. Docstring advises not nesting columns more than once (layouts.py). **[newer: 1.60]** integer px gap. **[newer: 1.62]** `wrap=False`. **[newer: 1.66]** vertically aligned columns fill row height.
- **`st.container(*, border=None, key=None, width="stretch", height="content", horizontal=False, horizontal_alignment="left", vertical_alignment="top", gap="small", autoscroll=None)`:** `height` takes `"content" | "stretch" | int`. An int scrolls, and the docstring warns to keep it under 500 px. `horizontal=True` gives a wrapping flex row. `horizontal_alignment="distribute"` spreads children (layouts.py). Horizontal containers arrived in 1.48.0 ([2025 notes](https://docs.streamlit.io/develop/quick-reference/release-notes/2025)). The layouts guide says: "Horizontal containers are generally preferred over `st.columns` for side-by-side layouts" ([layouts and containers](https://docs.streamlit.io/develop/concepts/design/layouts-and-containers)).
- **`st.space(size="small")`:** `"xxsmall"` 0.25rem, `"xsmall"` 0.5rem, `"small"` 0.75rem (a label height), `"medium"` 2.5rem (a button height), `"large"` 4.25rem, `"xlarge"` 6rem, `"xxlarge"` 8rem, `"stretch"`, or int px. It adds space along the parent's direction (`lib/streamlit/elements/space.py`; added 1.51.0).
- **`width` / `height` on elements:** most elements accept `width` ("stretch" | "content" | int) since 1.46.0, with charts and dataframes gaining `height` in 1.49 to 1.52 (2025 notes). Examples at 1.57: `st.html(width=...)`; `st.dataframe(height="auto"|"content"|"stretch"|int, row_height=...)`, where "auto" shows at most ten rows (`arrow.py`); `st.metric(border=..., height=..., chart_data=..., chart_type=...)` (`metric.py`).

## 5. Hiding chrome for screenshot or kiosk use

- **`?embed=true` (recommended for `capture_dashboard.py`):** handled entirely client-side. `getEmbedUrlParams` reads `window.location.search` for `embed` and `embed_options` ([utils.ts@1.57.0](https://github.com/streamlit/streamlit/blob/1.57.0/frontend/lib/src/util/utils.ts)), so it works on `localhost`. The doc lists the effects: removes the toolbar, top/bottom padding, footer, and colored line. `embed_options` can be `show_toolbar`, `show_padding`, `disable_scrolling`, `light_theme`, `dark_theme`, `hide_loading_screen` ([embed doc](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app/embed-your-app); that page is Community Cloud scoped, but the code is generic).
  - Padding math from [AppView/styled-components.ts@1.57.0](https://github.com/streamlit/streamlit/blob/1.57.0/frontend/app/src/components/AppView/styled-components.ts):
    - non-embedded: top 6rem (8rem with top nav), bottom 10rem
    - embedded without padding or toolbar: top 2.25rem (4.5rem if header or sidebar chrome remains), bottom 1rem
  - Use `?embed=true&embed_options=light_theme` to force light. Avoid `disable_scrolling`: it stops the main body from scrolling, which would also defeat the `full_page=True` capture if content overflows.
- **`[client] toolbarMode`:** `"auto"` (default) shows developer options, including the Deploy button, on localhost. `"viewer"` hides rerun, clear cache, and deploy. `"minimal"` shows only externally set or `set_page_config` items, and hides the menu if none are left (`client.toolbarMode` in config.py@1.57.0). The Deploy button renders only when `isLocalhost() && showDevelopmentOptions(...)` ([App.tsx@1.57.0](https://github.com/streamlit/streamlit/blob/1.57.0/frontend/app/src/App.tsx) `showDeployButton`). **The CI screenshot runs on localhost with `auto`, so set `toolbarMode = "minimal"` or use embed mode.**
- **`[ui] hideTopBar`:** exists but has `visibility="hidden"` (undocumented). In 1.57 the frontend skips StatusWidget, ToolbarActions and MainMenu when it is true (App.tsx ~2535 to 2570), despite the config text saying it "does *not* hide the main menu". The padding spec notes that hiding chrome via `toolbarMode` / `hideTopBar` "does not reclaim block-container inset". Prefer embed mode.
- **`st.set_page_config(initial_sidebar_state="collapsed")`** (already used) and `menu_items` (omitting a key hides that menu item) ([doc](https://docs.streamlit.io/develop/api-reference/configuration/st.set_page_config)). **[newer: 1.66]** `theme.paddingTop`/`paddingBottom` replace the `.block-container { padding-top }` CSS hack.

## 6. Fitting a dense dashboard into one 1600x1200 viewport

These are drawn from the sources above. Items marked (inference) are our own reasoning from those sources.

- **Budget:** 1200 CSS px tall. `capture_dashboard.py` uses `full_page=True` and `device_scale_factor=2`, so overflow is captured rather than clipped. "Fits in one screenshot" therefore really means "page height at most 1200" (inference).
- **Reclaim chrome:** `?embed=true` takes the header (3.75rem `headerHeight` overlay, sizes.ts) and about 13rem of top+bottom padding down to 3.25rem total (section 5).
- **Shrink the rem scale:** `baseFontSize = 14` (both official demos use 14) shrinks every rem-based gap, `st.space`, and heading size proportionally (gap docstrings, layouts.py).
- **Tighten gaps:** `gap="xsmall"` / `"xxsmall"` on containers and columns instead of the default `"small"` (1rem) (layouts.py).
- **Prefer horizontal flex rows of content-width items** over many equal `st.columns`. They pack tighter and wrap instead of squeezing ([layouts guide](https://docs.streamlit.io/develop/concepts/design/layouts-and-containers)).
- **Avoid fixed-height scrolling containers and `st.dataframe(height="auto")` for screenshot content.** Rows hidden behind a scrollbar are invisible in a static image. Use `height="content"` or render the rows as HTML (inference from the docstrings).
- **Cut heading weight in height:** set `headingFontSizes` to smaller h1/h2 values. A 2.75rem default h1 plus margins eats about 70 px.
- **Cap visible rows deliberately** (e.g. top N games) and push the rest below the fold, rather than shrinking type below legibility at the 2x Slack render (inference).
- **Verify in CI's version:** stretch-height behavior changed in 1.66 ("Stretch-height containers and forms grow to fit their content"), so pin or test on the CI version ([2026 notes](https://docs.streamlit.io/develop/quick-reference/release-notes/2026)).

## 7. Custom Components v2 (`st.components.v2`)

- Introduced in 1.51.0 "for frameless custom UI with bidirectional data flow" ([2025 notes](https://docs.streamlit.io/develop/quick-reference/release-notes/2025); [overview](https://docs.streamlit.io/develop/concepts/custom-components/components-v2)).
- Registration: `st.components.v2.component(name, *, html=None, css=None, js=None, isolate_styles=True)`. Mount: `renderer(*, key=None, data=None, default=None, width="stretch", height="content", **on_<state>_change)` returns a `ComponentResult` (`lib/streamlit/components/v2/__init__.py` and `types.py` at 1.57.0; [API ref](https://docs.streamlit.io/develop/api-reference/custom-components/st.components.v2.component)).
- Not iframed. `isolate_styles=True` mounts in a Shadow DOM. "This is style isolation only. It is not a security boundary." ([API ref](https://docs.streamlit.io/develop/api-reference/custom-components/st.components.v2.component)).
- Theme hooks: Streamlit injects `--st-*` CSS custom properties into the wrapper. Examples: `--st-primary-color`, `--st-font`, `--st-heading-font`, `--st-base-radius`, `--st-border-color`, `--st-chart-categorical-colors`, the palette `--st-red-color` etc., and per-heading `--st-heading-font-size-1..6`. They follow light/dark and sidebar automatically ([CCv2 theming](https://docs.streamlit.io/develop/concepts/custom-components/components-v2/theming)). Per-heading vars were added in 1.53.0 (2026 notes).
- **When it is worth it:**
  - You need JS events back in Python (state or triggers, see [state-and-triggers](https://docs.streamlit.io/develop/concepts/custom-components/components-v2/state-and-triggers)).
  - You need encapsulated styles for a reusable widget.
  - You want SVG plus scripts without `unsafe_allow_javascript`.
  - Not worth it for a read-only screenshot dashboard. `st.html` plus `config.toml` is simpler and has no registration lifecycle. One option: use CCv2 with `isolate_styles=False` only if a card component needs to read `--st-*` variables. Note that those variables are documented for component wrappers, not for the whole page.

## 8. Official Streamlit dashboards worth copying

Both repos are owned by the `streamlit` GitHub org.

**[streamlit/demo-seattle-weather](https://github.com/streamlit/demo-seattle-weather)** (requires `streamlit>=1.48.1`; commit `60454c7`):
- All look and feel comes from `config.toml`: light base, Open Sans via Google Fonts URL, `baseFontSize = 14`, `showWidgetBorder = true`, custom `borderColor`, and heading sizes and weights. The app has no CSS.
- KPI row: `with st.container(horizontal=True, gap="medium"):` holding `st.columns(2, gap="medium", width=300)` pairs of `st.metric(..., width="content")`. Fixed-width metric pairs flow and wrap.
- Chart grid: `with cols[0].container(border=True, height="stretch"):` makes bordered cards of equal height in each row.
- Uses `""` magic strings for small vertical spacing.

**[streamlit/demo-stockpeers](https://github.com/streamlit/demo-stockpeers)** (requires `streamlit>=1.44.2`; commit `95b7b69`):
- Dark theme via `config.toml`: slate palette, `primaryColor = "#615fff"`, Space Grotesk at `baseFontWeight = 300`, `dataframeHeaderBackgroundColor` matched to the background.
- Layout `st.columns([1, 3])`: a narrow control cell built as `st.container(border=True, height="stretch", vertical_alignment="center")`, beside a wide chart.
- Peer charts each sit in `cols[i].container(border=True)` cards.
- Title uses a Material icon in Markdown: `# :material/query_stats: Stock peer analysis`.

Techniques shared by both: theme-only styling, `layout="wide"`, bordered `st.container` as the card primitive, `height="stretch"` for equal-height rows, and Material icons. Neither uses `st.html` or `data-testid` CSS.
