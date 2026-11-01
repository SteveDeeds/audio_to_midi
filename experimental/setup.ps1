$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$AppPython = Join-Path $Root ".venv\Scripts\python.exe"
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $AppPython)) {
    throw "Create the app .venv first; this experiment reuses its numerical and CREPE dependencies read-only."
}
if (-not (Test-Path $Python)) {
    & $AppPython -m venv (Join-Path $PSScriptRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Could not create experimental environment." }
}
# The experiment owns its installs; shared app packages are only searched afterward.
& $Python -c "import pathlib,sysconfig,sys; pathlib.Path(sysconfig.get_path('purelib'), 'app_dependencies.pth').write_text(sys.argv[1] + '\n')" (Join-Path $Root ".venv\Lib\site-packages")
if ($LASTEXITCODE -ne 0) { throw "Could not configure read-only app dependency reuse." }
& $Python -m pip install "Cython>=3" "setuptools>=75" "pytest>=8"
if ($LASTEXITCODE -ne 0) { throw "Experimental build dependencies failed." }
& $Python -m pip install --no-build-isolation -r (Join-Path $PSScriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Experimental dependencies failed; madmom requires MSVC C++ Build Tools on Windows." }
