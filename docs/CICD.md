# CI/CD — build & deploy

This fork ships a GitHub Actions pipeline that builds a multi-arch container
image and deploys it to a server over SSH. Workflow file:
`.github/workflows/docker-publish.yml`.

## Trigger model

| Event | `build` job | `deploy` job |
|---|---|---|
| push to `main` | runs, pushes image to ghcr | runs (if deploy secrets present) |
| push to other branch | runs, pushes image to ghcr | skipped (`if` gate) |
| pull request to `main` | runs, **no push** | skipped |

The image is published to `ghcr.io/<owner>/gemini-balance`. Branch pushes are
tagged with the branch name; tag pushes (`vX.Y.Z`) get semver tags.

## ghcr authentication

The `build` job pushes with the built-in `secrets.GITHUB_TOKEN` plus
`permissions: packages: write` on the job. **No extra PAT is required** — the
repository's own `GITHUB_TOKEN` is scoped to push packages owned by that
repository.

## Secrets vs. Variables

Sensitive values are GitHub **Secrets** (masked, never printed). Non-sensitive
operational values are repository **Variables** (plain, visible to maintainers).

| Name | Kind | Purpose |
|---|---|---|
| `DEPLOY_HOST` | secret | Deploy target hostname / IP |
| `DEPLOY_SSH_KEY` | secret | Private SSH key for the deploy account (PEM) |
| `DEPLOY_FINGERPRINT` | secret | Target host SSH key SHA256 fingerprint (host-key pinning) |
| `DEPLOY_USER` | variable | SSH login user on the deploy host |
| `DEPLOY_PORT` | variable | SSH port (e.g. `22`) |
| `DEPLOY_PATH` | variable | Directory of the `docker-compose.yml` on the deploy host |

Set them with:

```bash
gh secret   set DEPLOY_HOST        -R <owner>/gemini-balance
gh secret   set DEPLOY_SSH_KEY     -R <owner>/gemini-balance < deploy_key
gh secret   set DEPLOY_FINGERPRINT -R <owner>/gemini-balance --body 'SHA256:...'
gh variable set DEPLOY_USER        -R <owner>/gemini-balance --body '<user>'
gh variable set DEPLOY_PORT        -R <owner>/gemini-balance --body '22'
gh variable set DEPLOY_PATH        -R <owner>/gemini-balance --body '<path>'
```

## fork-safe behavior

The `deploy` job has a `Check deploy secrets present` guard step. If
`DEPLOY_HOST` is empty (typical for a downstream fork that has not configured
deploy secrets), the guard emits a notice and the SSH step is skipped — the job
finishes **green as a no-op** instead of failing. A fork therefore inherits the
pipeline without broken Actions runs.

> Note: GitHub does not expose the `secrets` context in job- or step-level `if`
> conditions, so the guard reads the secret inside a `run:` step (where the
> `secrets` context *is* available) and writes a boolean output that the deploy
> step gates on.

## Deploy mechanics

After a successful build on `main`, the `deploy` job uses
`appleboy/ssh-action` to SSH into the host and run, in `DEPLOY_PATH`:

```bash
docker image prune -f       # free dangling layers (disk-tight host)
docker compose pull
docker compose up -d
docker image prune -f       # drop the now-dangling previous image
docker compose ps
```

### appleboy/ssh-action notes (known pitfalls)

- **`request_pty: true`** is set. Older versions, on some Docker-based
  runners, swallowed stdout/stderr (logs collapsed to just the
  `======CMD======` / `======END======` markers) and could intermittently
  report a non-zero exit while the remote command had actually succeeded.
- **`script_stop` is deliberately not used.** Failure detection is delegated to
  `set -euo pipefail` inside the deploy script itself.
- **Host-key pinning:** `fingerprint` is set from `DEPLOY_FINGERPRINT` so the
  action verifies the host key instead of blind trust-on-first-use.

## Deploy key rotation

The deploy SSH key is a dedicated ed25519 keypair (not a personal key).

1. Generate a new keypair:
   `ssh-keygen -t ed25519 -f ./deploy_key -N '' -C 'gemini-balance-ci-deploy'`
2. Append `deploy_key.pub` to the deploy host's
   `~/.ssh/authorized_keys` for `DEPLOY_USER`.
3. Update the secret: `gh secret set DEPLOY_SSH_KEY -R <owner>/gemini-balance < deploy_key`
4. Verify a deploy run succeeds, then remove the **old** public key line from
   the host's `authorized_keys`.
5. Delete the local `deploy_key` / `deploy_key.pub` files.

If the host SSH key changes, refresh `DEPLOY_FINGERPRINT` with the new SHA256
fingerprint (`ssh-keygen -l -F <host>` against a trusted `known_hosts`).
