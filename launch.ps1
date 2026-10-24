param(
    [string]$Study = 'outputs/confirmation',
    [ValidateRange(1024,65535)][int]$Port = 8797,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'
$tetherRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $tetherRoot
$studyPath = [System.IO.Path]::GetFullPath((Join-Path $tetherRoot $Study))
if (-not (Test-Path -LiteralPath (Join-Path $studyPath 'summary.json'))) {
    throw "No saved study at $studyPath. Run: python -m tether study --output outputs/confirmation --phase confirmation --seeds 101,102,103,104,105,106,107,108"
}
if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:$Port"
}
& python -m tether serve $studyPath --port $Port
