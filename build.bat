@echo off
REM Build Seed VC PyQt6 GUI as a standalone executable

REM Change to the script directory
cd /d "%~dp0"

REM Run PyInstaller with all required data folders
pyinstaller --noconfirm --onefile --windowed ^
  --add-data "configs;configs" ^
  --add-data "checkpoints;checkpoints" ^
  --add-data "examples;examples" ^
  --add-data "modules;modules" ^
  --exclude PyQt5 ^
  --exclude PySide6 ^
  app_vc_v2_pyqt.py

echo.
echo Build complete! The executable is in the "dist" folder.
pause