param(
    [ValidateNotNullOrEmpty()][string]$Environment = ".venv-v2",
    [string]$Python = "python",
    [switch]$Models
)
$ErrorActionPreference = "Stop"
Write-Warning "install-v2.ps1 is deprecated; use install.ps1 for new installations."
& (Join-Path $PSScriptRoot "install.ps1") -Environment $Environment -Python $Python -Models:$Models
