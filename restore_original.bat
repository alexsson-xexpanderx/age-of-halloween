@echo off
rem Age of Halloween - undoes build_mod_direct.bat: puts your original files back.
py -3 "%~dp0tools\build_mod.py" --restore %*
pause
