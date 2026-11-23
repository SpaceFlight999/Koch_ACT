@echo off
setlocal
set "profile=%~dp0configs\replay_camera_alignment.json"
if not exist "%profile%" if exist "%~dp0..\configs\replay_camera_alignment.json" set "profile=%~dp0..\configs\replay_camera_alignment.json"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Koch-Replay.ps1" align-camera --profile "%profile%" %*
exit /b %errorlevel%
