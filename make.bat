@echo off
rem Equivalente del Makefile para Windows (cmd o PowerShell): .\make.bat setup
rem Requiere uv: https://docs.astral.sh/uv/
setlocal
set TARGET=%~1
if "%TARGET%"=="setup" goto setup
if "%TARGET%"=="lint" goto lint
if "%TARGET%"=="test" goto test
if "%TARGET%"=="test-all" goto test_all
if "%TARGET%"=="eda" goto eda
if "%TARGET%"=="forecast" goto forecast
if "%TARGET%"=="requirements" goto requirements
echo Uso: .\make.bat [setup ^| test ^| test-all ^| forecast ^| eda ^| lint ^| requirements]
exit /b 1

:setup
rem Entorno con las versiones fijadas en uv.lock, kernel de Jupyter y hooks de pre-commit
uv sync --all-groups || exit /b 1
uv run python -m ipykernel install --sys-prefix --name retail-analytics --display-name "retail-analytics (.venv)" || exit /b 1
uv run pre-commit install || exit /b 1
exit /b 0

:lint
uv run ruff check src tests notebooks || exit /b 1
uv run ruff format --check src tests || exit /b 1
exit /b 0

:test
rem Pruebas rapidas (datos sinteticos)
uv run pytest -m "not data" || exit /b 1
exit /b 0

:test_all
rem Incluye las pruebas que leen los CSV reales
uv run pytest || exit /b 1
exit /b 0

:eda
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb || exit /b 1
exit /b 0

:forecast
rem Seleccion, ajuste y comparacion de modelos (aprox. 30 min); resultados en data\processed
uv run python -m retail_analytics.pipeline || exit /b 1
exit /b 0

:requirements
uv export --format requirements-txt --no-hashes --all-groups --no-emit-project --quiet -o requirements.txt || exit /b 1
exit /b 0
