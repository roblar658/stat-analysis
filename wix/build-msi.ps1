param(
    [switch]$SkipCargoBuild
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $SkipCargoBuild) {
    Write-Host "Bygger release-binær (stat-analysis.exe)..." -ForegroundColor Cyan
    cargo build --release
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

New-Item -ItemType Directory -Force -Path "$root\dist" | Out-Null
$outMsi = "$root\dist\stat-analysis-0.1.0-x64.msi"

Write-Host "Bygger WiX MSI-installasjonsfil: $outMsi" -ForegroundColor Cyan
wix build "$root\wix\Package.wxs" `
    -arch x64 `
    -ext WixToolset.UI.wixext `
    -ext WixToolset.Util.wixext `
    -o $outMsi

if ($LASTEXITCODE -eq 0) {
    $info = Get-Item $outMsi
    Write-Host "Ferdig! MSI opprettet: $($info.FullName) ($([math]::Round($info.Length/1MB, 2)) MB)" -ForegroundColor Green
} else {
    exit $LASTEXITCODE
}
