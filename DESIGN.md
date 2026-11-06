# TaxAgent design context

Chinese personal tax consultation product; preserve the existing compact chat UI.
Research is an explicit sibling mode in the same page, with independent tasks.

## Canonical visual source

`web/templates/index.html` owns CSS custom properties for light/dark themes.
`web/static/research.css` consumes those properties without duplicating palette values.
Primary #4d6bfe, background #ffffff, text #1a1a1a, border #e5e5e5 are the existing light tokens.
System/PingFang/Microsoft YaHei font stack; 15px body, comfortable Chinese line height.
Research uses restrained sections and expandable source excerpts; no decorative progress.

## Interaction owners

Chat: existing conversation functions in index.html. Research: research.js and research_routes.py.
Authentication: existing get_current_user; ownership is enforced server-side.
Shared theme: existing initTheme and CSS variables. Native labeled selects support keyboard navigation.
Research requests retain input on failure, announce stages, support cancellation and refresh/recovery.
All research text uses textContent; only HTTP(S) source links are navigable.
No research content is automatically copied into chat memory or published as a case.
Confirmation applies to a specific report version and does not assert professional review.
