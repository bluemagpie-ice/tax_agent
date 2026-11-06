# TaxAgent interaction contract

Research supplements the existing Chinese chat product. Preserve chat API/interaction behavior.
The current user instruction and docs/RESEARCH.md define the Research lifecycle.

| Capability | Canonical owner | Source of truth | Allowed variants | Verification |
|---|---|---|---|---|
| Select/Listbox | Native labeled HTML select | DESIGN.md and browser semantics | Model, task, upload scope | Keyboard/browser verification pending; API ownership tests |
| Form | research.js field helper and research-form validation | research_routes.py Pydantic validation | Research and case fields | test_research.py; browser verification pending |
| Scrollbar | index.html theme tokens and research.css | DESIGN.md | Light/dark | Static CSS check; browser verification pending |
| CRUD | research.js case editor and research_cases.py | Revision checked SQLite lifecycle | Draft, consent, publish, edit, withdraw, delete | test_research_cases.py |
| Progress | Research SSE events and persisted task stages | research_agent.py | Waiting facts, running, review, finalized, failed/interrupted | test_research.py |

No native confirmation dialogs. Destructive case deletion uses an explicit in-page disclosure and confirmation action.
Rendering uses textContent. HTTP(S) source links open separately with noopener. Unknown policy fields stay unknown.
Requests preserve inputs on failure; task/report/case revisions reject stale actions. Switching identities clears private UI state
and is blocked while a Research request is running. Research and ordinary chat have independent running state.

Existing policy-management UI is outside this feature's migration scope. Static audit findings there are recorded, not waived.
Browser URL-confidence restrictions prevented live UI verification in this session; source/API tests do not replace it.
