param(
    [ValidateNotNullOrEmpty()][string]$Environment = ".env",
    [string]$Python = "python",
    [switch]$Models
)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$environmentPath = $Environment
if (-not [System.IO.Path]::IsPathRooted($Environment)) {
    $environmentPath = Join-Path $root $Environment
}
$environmentPath = [System.IO.Path]::GetFullPath($environmentPath)
if ($environmentPath.TrimEnd('\') -eq $root.TrimEnd('\')) {
    throw "Choose a dedicated virtual-environment folder, not the repository root."
}
$executable = Join-Path $environmentPath "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $executable)) {
    & $Python -m venv $environmentPath
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the Python environment." }
}
$manifest = "requirements.txt"
if ($Models) { $manifest = "requirements-models.txt" }
& $executable -m pip install -r (Join-Path $root $manifest)
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
if ($Models) {
    & $executable -m pip install --no-deps basic-pitch==0.4.0
    if ($LASTEXITCODE -ne 0) { throw "Basic Pitch ONNX package installation failed." }
}
Push-Location $root
try {
    & $executable -X utf8 -m audio_to_midi doctor
    if ($LASTEXITCODE -ne 0) { throw "Environment validation failed." }
} finally {
    Pop-Location
}
Write-Host "Launch: $executable -X utf8 -m audio_to_midi"
