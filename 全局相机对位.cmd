@echo off
setlocal
set "site=%~dp0replay-auto.site.json"
if not exist "%site%" if exist "%~dp0..\replay-auto.site.json" set "site=%~dp0..\replay-auto.site.json"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Koch-Replay.ps1" align-camera --site "%site%" %*
exit /b %errorlevel%
