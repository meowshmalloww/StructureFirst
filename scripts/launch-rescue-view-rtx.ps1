param(
  [string]$Url = "http://127.0.0.1:5173",
  [switch]$ConfigureOnly
)

$ErrorActionPreference = "Stop"
$chromeCandidates = @(
  "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
  "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe"
)
$edgeCandidates = @(
  "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
  "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
)
$chromePath = $chromeCandidates |
  Where-Object { $_ -and (Test-Path -LiteralPath $_) } |
  Select-Object -First 1

if (-not $chromePath) {
  throw "Google Chrome was not found in either Program Files directory."
}

$preferencesPath = "HKCU:\Software\Microsoft\DirectX\UserGpuPreferences"
New-Item -Path $preferencesPath -Force | Out-Null
foreach ($browserPath in @($chromePath) + $edgeCandidates) {
  if ($browserPath -and (Test-Path -LiteralPath $browserPath)) {
    New-ItemProperty `
      -Path $preferencesPath `
      -Name $browserPath `
      -PropertyType String `
      -Value "GpuPreference=2;" `
      -Force | Out-Null
  }
}

Write-Host "Windows high-performance GPU preference is set for Chrome and Edge."
if ($ConfigureOnly) {
  exit 0
}

$profilePath = Join-Path $PSScriptRoot "..\data\chrome-rtx-profile"
$profilePath = [System.IO.Path]::GetFullPath($profilePath)
New-Item -ItemType Directory -Path $profilePath -Force | Out-Null

$arguments = @(
  "--force_high_performance_gpu",
  "--user-data-dir=$profilePath",
  "--new-window",
  "--app=$Url"
)
Start-Process -FilePath $chromePath -ArgumentList $arguments
Write-Host "Opened Rescue View in an isolated high-performance Chrome process."
