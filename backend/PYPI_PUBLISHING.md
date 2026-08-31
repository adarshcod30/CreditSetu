# Publishing to PyPI

Releases publish automatically via GitHub Actions using PyPI's **Trusted
Publishing** (OIDC) — no API token is stored anywhere in this repo or on
GitHub. See `.github/workflows/publish-pypi.yml`.

## One-time setup (do this once, before the first release)

Since the `creditsetu` project doesn't exist on PyPI yet, register a
*pending* trusted publisher — PyPI creates the project automatically on the
first successful publish.

1. Go to <https://pypi.org/manage/account/publishing/> (log in / create an
   account first if needed).
2. Under "Add a new pending publisher", enter exactly:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `creditsetu` |
   | Owner | `adarshcod30` |
   | Repository name | `CreditSetu` |
   | Workflow name | `publish-pypi.yml` |
   | Environment name | `pypi` |

3. Save.

That's the only manual step, ever. No token to copy, nothing to add to
GitHub secrets.

## Cutting a release (every time after)

1. Bump `version` in `backend/pyproject.toml`.
2. Commit, push to `main`.
3. On GitHub: **Releases → Draft a new release**, create a matching tag
   (e.g. `v1.0.1`), publish it.

Publishing the release triggers the workflow: it runs the test suite,
builds the sdist/wheel, validates them with `twine check`, and — only if
all of that passes — publishes to PyPI via OIDC. Nothing is uploaded if
tests or the build fail.
