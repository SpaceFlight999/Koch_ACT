$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.conda-koch-replay\python.exe'
$knownPython = 'D:\miniforge3\envs\lerobot\python.exe'
$python = if ($env:KOCH_PYTHON) {
    $env:KOCH_PYTHON
} elseif (Test-Path -LiteralPath $projectPython -PathType Leaf) {
    $projectPython
} elseif (Test-Path -LiteralPath $knownPython -PathType Leaf) {
    $knownPython
} elseif ($env:CONDA_PREFIX) {
    Join-Path $env:CONDA_PREFIX 'python.exe'
} else {
    $command = Get-Command python.exe -CommandType Application -ErrorAction SilentlyContinue |
        Where-Object { Test-Path -LiteralPath $_.Source -PathType Leaf } |
        Select-Object -First 1
    if ($command) { $command.Source }
}
if (-not $python -or -not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment missing: $python. Set KOCH_PYTHON to the full path of a working conda environment's python.exe."
}
$python = (Resolve-Path -LiteralPath $python).Path
$module = 'koch_autonomy'
$arguments = $args
if ($args.Count -gt 0 -and $args[0] -in @('vision-ui', 'color-ui')) {
    $module = if ($args[0] -eq 'vision-ui') { 'koch_vision.grid_reader.app' } else { 'koch_vision.color_app' }
    $arguments = @($args | Select-Object -Skip 1)
}
$oldPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = if ($oldPythonPath) { "$PSScriptRoot;$oldPythonPath" } else { $PSScriptRoot }
    & $python -P -s -X utf8 -B -m $module @arguments
    $result = $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $oldPythonPath
}
exit $result
