param(
    [string]$RepoPath = "C:\Users\lh594\Desktop\CVS-RFFI",
    [string]$GitPath = "C:\Program Files\Git\cmd\git.exe",
    [int]$DebounceSeconds = 20,
    [switch]$Push
)

$ErrorActionPreference = "Stop"
$ScriptPath = Join-Path $RepoPath "scripts\git_autosave.ps1"
$LogDir = Join-Path $RepoPath "logs"
$LogPath = Join-Path $LogDir "git_autosave_watcher.log"

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

function Write-WatcherLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogPath -Value "[$stamp] $Message"
}

function Invoke-AutoSave {
    try {
        $args = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $ScriptPath, "-RepoPath", $RepoPath, "-GitPath", $GitPath)
        if ($Push) { $args += "-Push" }
        & powershell.exe @args
        Write-WatcherLog "Autosave command completed."
    } catch {
        Write-WatcherLog "Autosave command failed: $($_.Exception.Message)"
    }
}

Write-WatcherLog "Watcher started for $RepoPath"
Invoke-AutoSave

$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $RepoPath
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true
$watcher.NotifyFilter = [System.IO.NotifyFilters]'FileName, DirectoryName, LastWrite, Size'

$script:IgnoredPattern = "\\(\.git|logs|__pycache__|\.pytest_cache)(\\|$)|\\(type10-4|type10-6-sat|type10-7|unkown)(\\|$)"
$script:DebounceSeconds = $DebounceSeconds
$lastRun = Get-Date "2000-01-01"

$action = {
    $path = $Event.SourceEventArgs.FullPath
    if ($path -match $script:IgnoredPattern) { return }
    $now = Get-Date
    if (($now - $script:lastRun).TotalSeconds -lt $script:DebounceSeconds) { return }
    $script:lastRun = $now
    Start-Sleep -Seconds $script:DebounceSeconds
    Invoke-AutoSave
}

Register-ObjectEvent $watcher Changed -Action $action | Out-Null
Register-ObjectEvent $watcher Created -Action $action | Out-Null
Register-ObjectEvent $watcher Deleted -Action $action | Out-Null
Register-ObjectEvent $watcher Renamed -Action $action | Out-Null

while ($true) {
    Start-Sleep -Seconds 3600
}
