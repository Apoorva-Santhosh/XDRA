# Branch Protection Setup (one-time, admin only)

This is the step that actually makes "PRs are only mergeable if they have no bugs" true. `.github/workflows/ci.yml` only *produces* pass/fail checks — nothing stops a merge until branch protection is turned on and told to require those checks. Do this once, right after the repo is created, before any pipeline work starts.

Whoever creates the GitHub repo (has admin/owner rights) should do this — it can't be done by a teammate who only has write access.

## Option A — GitHub web UI (no CLI needed)

1. Push this repo to GitHub and open it in the browser.
2. Go to **Settings → Branches**.
3. Under "Branch protection rules," click **Add branch protection rule**.
4. Branch name pattern: `main`
5. Enable these, in order:
   - ☑ **Require a pull request before merging**
     - ☑ Require approvals — set to **1** (given a 3-person team; raise later if you want)
     - ☑ Require review from Code Owners (this activates `.github/CODEOWNERS`)
   - ☑ **Require status checks to pass before merging**
     - Search for and select: `lint`, `test`, `schema-contract` (these are the three job names from `.github/workflows/ci.yml` — they'll only appear in this list after the workflow has run at least once, so push once first, then come back to this step)
     - ☑ Require branches to be up to date before merging
   - ☑ **Require conversation resolution before merging**
   - ☑ Do **not** enable "Allow force pushes" or "Allow deletions" for `main`
6. Click **Create** (or **Save changes**).

Once this is on: a PR with a failing lint/test/schema-contract check, or with zero approvals, will show a **red/disabled merge button** — GitHub itself refuses the merge, not just a social convention.

## Option B — GitHub CLI (`gh`), if you prefer scripting it

```bash
gh api repos/<org-or-username>/<repo-name>/branches/main/protection \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  -f required_status_checks[strict]=true \
  -f 'required_status_checks[contexts][]=lint' \
  -f 'required_status_checks[contexts][]=test' \
  -f 'required_status_checks[contexts][]=schema-contract' \
  -f enforce_admins=true \
  -f 'required_pull_request_reviews[required_approving_review_count]=1' \
  -f required_pull_request_reviews[require_code_owner_reviews]=true \
  -f required_conversation_resolution=true \
  -f restrictions=null
```

Replace `<org-or-username>/<repo-name>`. Run this only after the workflow in `.github/workflows/ci.yml` has run at least once (so the check names `lint`/`test`/`schema-contract` actually exist for GitHub to reference).

## Verifying it worked

1. Open a throwaway PR that deliberately breaks something (e.g. a lint violation).
2. Confirm the merge button is disabled and shows "Required status check failing."
3. Fix the issue, push again, confirm the check turns green and the merge button re-enables.
4. Delete the throwaway branch/PR.

Do this verification once, as a team, before relying on it for real work — don't assume the settings saved correctly without checking.
