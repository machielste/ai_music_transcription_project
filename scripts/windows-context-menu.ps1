param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('install', 'uninstall')]
    [string]$Action
)

# A temporary interactive-user task avoids registry isolation inherited from
# desktop automation hosts. It has no trigger and is removed after this run.
$ErrorActionPreference = 'Stop'
$repository = Split-Path $PSScriptRoot -Parent
$pythonw = Join-Path $repository '.venv\Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $pythonw -PathType Leaf)) {
    throw 'Run uv sync in the repository before installing the Explorer menu.'
}
$taskName = 'BassTranscriberMenu-' + [guid]::NewGuid().ToString('N')
$workFolder = Join-Path ([System.IO.Path]::GetTempPath()) $taskName
$null = New-Item -ItemType Directory -Path $workFolder
$helperPath = Join-Path $workFolder 'register.py'
$reportPath = Join-Path $workFolder 'result.txt'
@'
import sys
import traceback
from pathlib import Path
from bass_transcriber.context_menu import main
action, report = sys.argv[1:]
try:
    sys.argv = ["context_menu", action]
    code = main()
    Path(report).write_text("OK" if code == 0 else f"Installer exited with {code}", encoding="utf-8")
except BaseException:
    Path(report).write_text(traceback.format_exc(), encoding="utf-8")
'@ | Set-Content -LiteralPath $helperPath -Encoding utf8

$service = New-Object -ComObject Schedule.Service
$service.Connect()
$folder = $service.GetFolder('\')
$registered = $null
try {
    $definition = $service.NewTask(0)
    $definition.Settings.Enabled = $true
    $definition.Settings.ExecutionTimeLimit = 'PT1M'
    $definition.Principal.LogonType = 3 # The signed-in user, without elevation.
    $definition.Principal.RunLevel = 0
    $execute = $definition.Actions.Create(0)
    $execute.Path = $pythonw
    $execute.Arguments = '"' + $helperPath + '" ' + $Action + ' "' + $reportPath + '"'
    $execute.WorkingDirectory = $repository
    $registered = $folder.RegisterTaskDefinition($taskName, $definition, 6, $null, $null, 3, $null)
    $null = $registered.Run($null)
    $deadline = [datetime]::UtcNow.AddSeconds(30)
    while (-not (Test-Path -LiteralPath $reportPath)) {
        if ([datetime]::UtcNow -ge $deadline) {
            throw 'The Explorer menu installer did not finish within 30 seconds.'
        }
        Start-Sleep -Milliseconds 100
    }
    $result = Get-Content -LiteralPath $reportPath -Raw
    if ($result -ne 'OK') { throw $result }
    Write-Output "Explorer menu $Action completed for the signed-in Windows user."
}
finally {
    if ($null -ne $registered) {
        if ($registered.State -eq 4) { $registered.Stop(0) }
        $folder.DeleteTask($taskName, 0)
    }
    # Remove only the exact temporary files and empty folder created above.
    if (Test-Path -LiteralPath $reportPath) { Remove-Item -LiteralPath $reportPath }
    Remove-Item -LiteralPath $helperPath
    Remove-Item -LiteralPath $workFolder
}
