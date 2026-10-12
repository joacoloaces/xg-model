$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

# Preferimos Python 3.12 por compatibilidad; aceptamos otras versiones modernas
# si ya están instaladas, evitando obligar al usuario a instalar otra versión.
$pythonExe = $null
$pythonVersion = $null
$launcher = Get-Command py -ErrorAction SilentlyContinue

if ($launcher) {
    foreach ($version in @('3.12', '3.13', '3.14', '3.11')) {
        $candidate = & $launcher.Source "-$version" -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $candidate) {
            $pythonExe = ($candidate | Select-Object -Last 1).Trim()
            $pythonVersion = $version
            break
        }
    }
}

# Alternativa para instalaciones de Python que no tienen el lanzador `py`.
if (-not $pythonExe) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $candidate = @(& $pythonCommand.Source -c "import sys; print(sys.version_info >= (3,11)); print(sys.executable)" 2>$null)
        if ($LASTEXITCODE -eq 0 -and $candidate -and $candidate.Count -ge 2 -and $candidate[0].Trim() -eq 'True') {
            $pythonExe = $candidate[-1].Trim()
            $pythonVersion = (& $pythonExe -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" | Select-Object -Last 1).Trim()
        }
    }
}

if (-not $pythonExe) {
    Write-Host ''
    Write-Host 'No encontre una instalacion de Python 3.11, 3.12, 3.13 o 3.14.' -ForegroundColor Red
    Write-Host 'Instala Python 3.12 (recomendado) desde https://www.python.org/downloads/release/python-31210/'
    Write-Host 'Durante la instalacion, marca Add python.exe to PATH. Despues, cerra y volve a abrir PowerShell.'
    Write-Host 'Tambien podes comprobar las versiones detectadas con: py -0p'
    throw 'Falta una version compatible de Python.'
}

Write-Host "Python detectado: $pythonVersion ($pythonExe)" -ForegroundColor Cyan

# Si ya existe un entorno virtual creado con otra version, recrearlo.
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$recreateVenv = $false
if (Test-Path $venvPython) {
    $existingExe = (& $venvPython -c "import sys; print(sys.executable)" 2>$null | Select-Object -Last 1)
    $existingVersion = (& $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null | Select-Object -Last 1)
    $chosenMinor = (& $pythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
    if ($LASTEXITCODE -ne 0 -or $existingVersion.Trim() -ne $chosenMinor) { $recreateVenv = $true }
}
if (-not (Test-Path $venvPython) -or $recreateVenv) {
    if (Test-Path '.venv') { Remove-Item '.venv' -Recurse -Force }
    & $pythonExe -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo crear el entorno virtual .venv.' }
}

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$uvicorn = Join-Path $PSScriptRoot '.venv\Scripts\uvicorn.exe'

if (-not (Test-Path '.venv\api-deps-installed.txt')) {
    & $python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo actualizar pip.' }
    & $python -m pip install -r requirements-api.txt
    if ($LASTEXITCODE -ne 0) { throw 'Fallo la instalacion de dependencias. Revisa la conexion y los mensajes anteriores.' }
    New-Item -ItemType File '.venv\api-deps-installed.txt' -Force | Out-Null
}

Write-Host ''
Write-Host 'Iniciando la API xG en http://localhost:8000' -ForegroundColor Green
Write-Host 'Comprobacion de salud: http://localhost:8000/health'
Write-Host 'Deja esta ventana abierta mientras utilizas el frontend.'
Write-Host ''
& $uvicorn api.server:app --reload --host 127.0.0.1 --port 8000
