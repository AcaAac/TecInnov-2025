@echo off
:: Navigate to project root (parent of scripts folder)
cd /d "%~dp0.."

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Running Full Pipeline (Collection - BC - RL - Eval)...
echo Output will be saved to data/evasion_experiment/

if not exist "data\evasion_experiment" mkdir "data\evasion_experiment"

:: Navigate to source directory
cd src
python train_blue.py --all --output_dir ../data/evasion_experiment

echo.
echo Pipeline finished successfully! Check data/evasion_experiment/ for results.
pause
