@echo off
REM ---------------------------------------------------------------
REM  Build script: gui.py + run.py  ->  fast-opening standalone .exe
REM  Run this from inside the folder that contains gui.py and run.py.
REM  Auto-detects and activates a local venv (venv\ or .venv\) if present.
REM ---------------------------------------------------------------

setlocal

if exist "venv\Scripts\activate.bat" (
    echo Activation du venv : venv\
    call "venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    echo Activation du venv : .venv\
    call ".venv\Scripts\activate.bat"
) else (
    echo Aucun venv detecte ^(venv\ ou .venv\^) - utilisation du Python systeme.
    echo Si tu as un venv ailleurs, active-le AVANT de lancer ce script,
    echo puis relance build.bat depuis ce meme terminal.
)

echo.
echo Python utilise :
where python
python --version

pip install --upgrade pyinstaller

pyinstaller --noconfirm --clean --onedir --windowed --noupx ^
  --name "AI_Trading_Bot" ^
  --collect-all sklearn ^
  --collect-all pandas_ta ^
  --collect-all MetaTrader5 ^
  --collect-submodules sklearn.ensemble ^
  --collect-submodules sklearn.tree ^
  --collect-submodules sklearn.neighbors ^
  --hidden-import sklearn.utils._typedefs ^
  --hidden-import sklearn.utils._heap ^
  --hidden-import sklearn.utils._sorting ^
  --hidden-import sklearn.utils._vector_sentinel ^
  --hidden-import sklearn.neighbors._partition_nodes ^
  --exclude-module matplotlib ^
  --exclude-module scipy.spatial.cKDTree ^
  --exclude-module test ^
  --exclude-module tkinter.test ^
  gui.py

echo.
echo Build termine. L'exe se trouve dans dist\AI_Trading_Bot\AI_Trading_Bot.exe
echo Copie config.json ^(si tu en as un^) a cote de lui dans ce dossier.
pause