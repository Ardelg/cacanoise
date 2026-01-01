@echo off
setlocal ENABLEDELAYEDEXPANSION

REM === Ir al directorio del script ===
pushd "%~dp0"

set "VENV_DIR=.venv"

REM === Crear venv si no existe ===
if not exist "%VENV_DIR%" (
  echo Creando entorno virtual para el ayudante...
  py -m venv "%VENV_DIR%"
)
call "%VENV_DIR%\Scripts\activate"

REM === Verificación Inteligente de Dependencias ===
REM Si el usuario pide reinstalar explícitamente
if /I "%~1"=="--reinstall" goto :install_deps

REM Intentamos importar las librerías críticas. Si falla, instalamos.
echo Verificando dependencias...
python -c "import deep_translator; import faster_whisper; import openai; import PySide6; import google.generativeai" >nul 2>&1
if errorlevel 1 (
    echo [!] Se detectaron librerias faltantes o corruptas.
    echo Iniciando instalacion de dependencias...
    goto :install_deps
)
goto :run_app

:install_deps
python -m pip install --upgrade pip --disable-pip-version-check
if errorlevel 1 goto :pip_error
pip install -r requirements.txt
if errorlevel 1 goto :pip_error
echo.
echo [OK] Dependencias instaladas correctamente.
echo.

:run_app

REM === Ejecutar la app ===
echo Iniciando Cacatua Noise...
python cacatuanoise.py
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
  echo.
  echo [ERROR] Ocurrio un error al ejecutar la aplicacion. Codigo: %EXITCODE%
  pause
)

REM === Salir ===
popd
exit /b %EXITCODE%

:pip_error
echo.
echo [FATAL] Hubo un problema instalando dependencias.
echo Revisa tu conexion a internet o el archivo requirements.txt.
pause
popd
exit /b 1