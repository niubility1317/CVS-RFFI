param(
    [string]$RepoPath = "C:\Users\lh594\Desktop\CVS-RFFI",
    [string]$GitPath = "C:\Program Files\Git\cmd\git.exe",
    [switch]$Push,
    [string]$MessagePrefix = "autosave"
)

$ErrorActionPreference = "Stop"
$LogDir = Join-Path $RepoPath "logs"
$LogPath = Join-Path $LogDir "git_autosave.log"

function Write-AutoSaveLog {
    param([string]$Message)
    if (-not (Test-Path $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    }
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogPath -Value "[$stamp] $Message"
}

if (-not (Test-Path $GitPath)) {
    $cmd = Get-Command git -ErrorAction SilentlyContinue
    if ($cmd) {
        $GitPath = $cmd.Source
    } else {
        throw "Git executable not found. Expected: $GitPath"
    }
}

if (-not (Test-Path $RepoPath)) {
    throw "Repo path not found: $RepoPath"
}

Push-Location $RepoPath
try {
    & $GitPath rev-parse --is-inside-work-tree *> $null
    if ($LASTEXITCODE -ne 0) {
        & $GitPath init
        if ($LASTEXITCODE -ne 0) { throw "git init failed" }
    }

    & $GitPath add -A
    if ($LASTEXITCODE -ne 0) { throw "git add failed" }

    $status = & $GitPath status --porcelain
    if (-not $status) {
        Write-AutoSaveLog "No changes to commit."
        return
    }

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    & $GitPath commit -m "${MessagePrefix}: $timestamp"
    if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
    Write-AutoSaveLog "Committed changes: $MessagePrefix $timestamp"

    $remote = & $GitPath remote
    if ($Push -and $remote) {
        $branch = (& $GitPath branch --show-current).Trim()
        if (-not $branch) { $branch = "main" }
        & $GitPath push -u origin $branch
        if ($LASTEXITCODE -eq 0) {
            Write-AutoSaveLog "Pushed branch '$branch' to origin."
        } else {
            Write-AutoSaveLog "Push failed for branch '$branch'. Check remote/auth."
        }
    } elseif ($Push) {
        Write-AutoSaveLog "Push requested but no git remote is configured."
    }
} finally {
    Pop-Location
}
