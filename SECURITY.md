# Security Policy

## Supported versions

Only the latest `0.x` release receives security fixes until `1.0`.

| Version | Supported |
| --- | --- |
| 0.1.x   | ✅ |
| < 0.1   | ❌ |

## Reporting a vulnerability

**Do not open a public issue.** Please report privately via GitHub:

- Go to the repo's **Security** tab → **Advisories** → **Report a vulnerability**.
- Direct link: https://github.com/0x1Adi/chatlectify/security/advisories/new

This routes the report privately to the maintainer. Public issues and PRs are
not a valid channel for vulnerability disclosure.

Please include:

- A description of the issue and its impact.
- Steps to reproduce (or a proof-of-concept).
- The affected version/commit.

You'll get an acknowledgement within **72 hours** and a fix or mitigation
timeline within **7 days**. Coordinated disclosure is appreciated — credit
will be given in the release notes unless you prefer to remain anonymous.

## Scope

In scope:

- Code execution or injection via crafted export files (`conversations.json`, Gemini HTML, plaintext corpora).
- Credential exfiltration via the LLM caller or CLI fallback.
- Dependency vulnerabilities surfaced in published releases.

Out of scope:

- Misuse of a valid API key by the user who supplied it.
- Issues in third-party LLM providers or their CLIs (`claude`, `codex`).
- Denial of service on a user's own machine via huge input files.

## Hardening in this repo

- CI runs `pip-audit` on every push and PR.
- CodeQL static analysis on every push, PR, and weekly schedule.
- Dependabot keeps dependencies and GitHub Actions up to date.
- Workflow permissions are pinned to read-only by default and scoped per job.
