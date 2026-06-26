param(
    [string]$SourceRoot = "E:\type10-7",
    [string]$RepoRoot = "E:\type10-7\github_publish\CVS-RFFI-repo",
    [switch]$NoCommit,
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $RepoRoot)) {
    throw "RepoRoot does not exist: $RepoRoot"
}

Set-Location -LiteralPath $RepoRoot

$pythonArgs = @(
    "scripts\sync_cvs_release_snapshot.py",
    "--source-root", $SourceRoot,
    "--repo-root", $RepoRoot
)

if (-not $NoCommit) {
    $pythonArgs += "--commit"
}
if (-not $NoPush) {
    $pythonArgs += "--push"
}

$conda = Get-Command conda -ErrorAction SilentlyContinue
if ($conda) {
    & conda run -n ssr-gpu python @pythonArgs
} else {
    & python @pythonArgs
}

if ($LASTEXITCODE -ne 0) {
    throw "CVS snapshot cycle failed with exit code $LASTEXITCODE"
}
