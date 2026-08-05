"""Fetch the Kubernetes docs subtree at each release branch.

``kubernetes/website`` is ~1.5 GB with full history and all assets, and we want one
subdirectory at a dozen branches. Three git features together make that cheap:

* ``--depth 1``          — no history, just the tip of each branch
* ``--filter=blob:none`` — fetch file contents lazily, only for checked-out paths
* sparse-checkout        — materialize only ``content/en/docs``

Crucially this uses **one** clone and re-checks-out each branch in place. Adjacent
release branches share the overwhelming majority of their trees and blobs, so branches
after the first cost almost nothing. Cloning twelve times separately would refetch the
same objects twelve times.

Because branches are checked out in place, callers get a generator that yields
``(version, path)`` and must finish with a tree before advancing. That keeps peak disk
usage at one checkout instead of twelve.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..config import DOCS_SUBTREE, WEBSITE_REPO, Config
from ..versions import MinorVersion

log = logging.getLogger(__name__)


class GitError(RuntimeError):
    """A git invocation failed. Carries the command and stderr for diagnosis."""


def _run_git(args: list[str], cwd: Path | None = None, timeout: int = 900) -> str:
    command = ["git", *args]
    log.debug("git %s (cwd=%s)", " ".join(args), cwd)
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - environment dependent
        raise GitError("git is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git {' '.join(args)} timed out after {timeout}s") from exc

    if result.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed with exit code {result.returncode}\n"
            f"stderr: {result.stderr.strip()}"
        )
    return result.stdout


def _is_initialized(repo_dir: Path) -> bool:
    return (repo_dir / ".git").exists()


def ensure_clone(repo_dir: Path, repo_url: str = WEBSITE_REPO, subtree: str = DOCS_SUBTREE) -> None:
    """Create the sparse, blobless, shallow clone if it does not already exist.

    Idempotent: safe to call on every run. If a previous run was interrupted midway
    the partial directory is removed and re-cloned, because a half-initialized clone
    fails in confusing ways later.
    """
    if _is_initialized(repo_dir):
        log.info("reusing existing clone at %s", repo_dir)
        return

    if repo_dir.exists():
        log.warning("removing incomplete clone at %s", repo_dir)
        shutil.rmtree(repo_dir, ignore_errors=True)

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    log.info("cloning %s (sparse: %s) -> %s", repo_url, subtree, repo_dir)

    _run_git(
        [
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "--depth",
            "1",
            "--no-single-branch",
            # Windows paths in this repo comfortably exceed 260 chars once combined
            # with a data directory prefix; without this the checkout fails partway
            # with a misleading "unable to create file" error.
            "--config",
            "core.longpaths=true",
            repo_url,
            str(repo_dir),
        ]
    )
    # --no-cone keeps the pattern a literal path prefix. Cone mode would also pull
    # every sibling directory at each level of the path.
    _run_git(["sparse-checkout", "set", "--no-cone", f"/{subtree}/*"], cwd=repo_dir)


def _fetch_branch(repo_dir: Path, branch: str) -> None:
    _run_git(
        [
            "fetch",
            "--depth",
            "1",
            "--filter=blob:none",
            "origin",
            f"+refs/heads/{branch}:refs/remotes/origin/{branch}",
        ],
        cwd=repo_dir,
    )


def _checkout_branch(repo_dir: Path, branch: str) -> None:
    _run_git(["checkout", "--force", "-B", branch, f"refs/remotes/origin/{branch}"], cwd=repo_dir)


def available_versions(repo_dir: Path) -> set[MinorVersion]:
    """Which release-1.x branches the remote actually has.

    Called before iterating so a requested-but-missing version is a clear warning
    rather than an opaque git failure mid-run.
    """
    output = _run_git(["ls-remote", "--heads", "origin", "release-*"], cwd=repo_dir)
    found: set[MinorVersion] = set()
    for line in output.splitlines():
        _, _, ref = line.partition("\t")
        name = ref.strip().removeprefix("refs/heads/")
        if name.startswith("release-"):
            version = MinorVersion.try_parse(name.removeprefix("release-"))
            if version:
                found.add(version)
    return found


@contextmanager
def _checked_out(repo_dir: Path, version: MinorVersion) -> Iterator[Path]:
    _fetch_branch(repo_dir, version.branch)
    _checkout_branch(repo_dir, version.branch)
    yield repo_dir


def iter_version_trees(
    config: Config,
    versions: list[MinorVersion] | None = None,
    repo_url: str = WEBSITE_REPO,
    subtree: str = DOCS_SUBTREE,
) -> Iterator[tuple[MinorVersion, Path]]:
    """Yield ``(version, docs_root)`` for each requested release branch, in order.

    The yielded path is only valid until the next iteration -- the branch is checked
    out in place. Consume each tree fully before advancing.

    Versions the remote does not have are skipped with a warning rather than raising,
    so requesting a window that runs past the newest release still works.
    """
    repo_dir = config.paths.raw / "website"
    ensure_clone(repo_dir, repo_url=repo_url, subtree=subtree)

    requested = versions if versions is not None else config.versions()
    present = available_versions(repo_dir)
    for version in requested:
        if version not in present:
            log.warning("branch %s not found on remote; skipping", version.branch)
            continue
        with _checked_out(repo_dir, version) as root:
            docs_root = root / subtree
            if not docs_root.is_dir():
                log.warning("%s has no %s; skipping", version.branch, subtree)
                continue
            log.info("checked out %s", version.branch)
            yield version, docs_root


def iter_markdown_files(docs_root: Path) -> Iterator[Path]:
    """Every markdown file under a docs tree, in a deterministic order.

    Sorted so that two runs over the same commit produce chunks in the same order,
    which makes diffing intermediate artifacts meaningful.
    """
    yield from sorted(
        (path for path in docs_root.rglob("*.md") if path.is_file()),
        key=lambda p: p.as_posix(),
    )
