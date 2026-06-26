<#
.SYNOPSIS
Read-only direct SSH preflight for N607.

.DESCRIPTION
Run this before any task that needs SSH or SCP access to N607. The script is
intentionally read-only: it verifies the local direct SSH config, the N607
identity, the project root, server time, and GPU visibility.
#>
[CmdletBinding()]
param(
    [string]$TargetAlias = "N607",
    [string]$RemoteProjectRoot = "/home/szu2070436088/2510044040/CV-SincNet",
    [string]$SshConfigPath = "",
    [string]$ExpectedLocalUser = "lh594",
    [switch]$AllowDifferentLocalUser,
    [switch]$RequireExpectedLocalUser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    Write-Host ""
    Write-Host "== $Label =="
    Write-Host ("$Command " + ($Arguments -join " "))
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

function Invoke-CapturedNativeCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & $Command @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    [pscustomobject]@{
        ExitCode = $exitCode
        Output = @($output | ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) {
                $_.Exception.Message
            }
            else {
                [string]$_
            }
        })
    }
}

Write-Host "N607 SSH preflight: read-only, short-lived, direct-only."

if ([string]::IsNullOrWhiteSpace($SshConfigPath)) {
    $SshConfigPath = Join-Path $PSScriptRoot "n607_ssh_config"
}

$resolvedSshConfigPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($SshConfigPath)
if (-not (Test-Path -LiteralPath $resolvedSshConfigPath -PathType Leaf)) {
    throw "SSH config file not found: $resolvedSshConfigPath"
}

try {
    Get-Content -LiteralPath $resolvedSshConfigPath -TotalCount 1 -ErrorAction Stop | Out-Null
}
catch {
    throw "SSH config file is not readable: $resolvedSshConfigPath. $($_.Exception.Message)"
}

$localUser = (& whoami).Trim()
Write-Host "Local user: $localUser"
Write-Host "USERPROFILE: $env:USERPROFILE"
Write-Host "SSH config: $resolvedSshConfigPath"
if (
    -not $AllowDifferentLocalUser.IsPresent -and
    -not [string]::IsNullOrWhiteSpace($ExpectedLocalUser) -and
    $localUser -notmatch "(^|\\)$([regex]::Escape($ExpectedLocalUser))$"
) {
    $localUserMessage = "Local shell is running as '$localUser', not expected user '$ExpectedLocalUser'."
    if ($RequireExpectedLocalUser.IsPresent) {
        throw "$localUserMessage Use -AllowDifferentLocalUser to override this explicit runtime-user gate."
    }
    Write-Warning "$localUserMessage Continuing because project-pinned SSH config, identity file readability, and BatchMode remote identity are the authoritative checks for automation runs."
}

$sshConfigOptions = @("-F", $resolvedSshConfigPath)

$targetConfigResult = Invoke-CapturedNativeCommand -Command "ssh" -Arguments ($sshConfigOptions + @("-T", "-G", $TargetAlias))
$targetConfig = $targetConfigResult.Output
if ($targetConfigResult.ExitCode -ne 0) {
    $targetConfig | Write-Host
    throw "ssh -F $resolvedSshConfigPath -G $TargetAlias failed"
}

$proxyJump = $targetConfig | Where-Object { $_ -match '^proxyjump\s+' } | Select-Object -First 1
if ($proxyJump -and $proxyJump -notmatch '^proxyjump\s+none\s*$') {
    throw "$TargetAlias must be a direct SSH alias, but ssh -G reported: $proxyJump"
}

$proxyCommand = $targetConfig | Where-Object { $_ -match '^proxycommand\s+' } | Select-Object -First 1
if ($proxyCommand -and $proxyCommand -notmatch '^proxycommand\s+none\s*$') {
    throw "$TargetAlias must be a direct SSH alias, but ssh -G reported: $proxyCommand"
}

$targetHost = $targetConfig | Where-Object { $_ -match '^hostname\s+' } | Select-Object -First 1
$targetUser = $targetConfig | Where-Object { $_ -match '^user\s+' } | Select-Object -First 1
Write-Host "Config OK: $TargetAlias is direct. $targetUser; $targetHost"

$identityFiles = @(
    $targetConfig |
        Where-Object { $_ -match '^identityfile\s+' } |
        ForEach-Object { ($_ -replace '^identityfile\s+', '').Trim('"') } |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and $_ -ne 'none' }
)
if ($identityFiles.Count -eq 0) {
    throw "No SSH identity file is configured for $TargetAlias in $resolvedSshConfigPath"
}
foreach ($identityFile in $identityFiles) {
    $identityPath = $identityFile
    if ($identityPath.StartsWith("~/") -or $identityPath.StartsWith("~\")) {
        $identityPath = Join-Path $env:USERPROFILE $identityPath.Substring(2)
    }
    if (-not (Test-Path -LiteralPath $identityPath -PathType Leaf)) {
        throw "SSH identity file is not present or not readable by this runtime: $identityPath"
    }
    try {
        Get-Content -LiteralPath $identityPath -TotalCount 1 -ErrorAction Stop | Out-Null
    }
    catch {
        throw "SSH identity file is not readable by this runtime: $identityPath. $($_.Exception.Message)"
    }
    Write-Host "Identity file OK: $identityPath"
}

$sshOptions = @(
    "-F", $resolvedSshConfigPath,
    "-T",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10"
)

$remoteProbe = @"
set -e
echo user=`$(whoami)
echo host=`$(hostname)
echo pwd=`$(pwd)
date
if [ -d '$RemoteProjectRoot' ]; then
  echo project_root=$RemoteProjectRoot
else
  echo project_root_missing=$RemoteProjectRoot
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
else
  echo nvidia_smi=missing
fi
"@

Invoke-LoggedCommand `
    -Label "N607 read-only probe" `
    -Command "ssh" `
    -Arguments ($sshOptions + @($TargetAlias, $remoteProbe))

Write-Host ""
Write-Host "Preflight OK: use ssh -F $resolvedSshConfigPath $TargetAlias or scp -F $resolvedSshConfigPath <local> ${TargetAlias}:<remote> for this task."
