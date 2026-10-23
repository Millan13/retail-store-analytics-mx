.PHONY: setup lint test test-all eda forecast requirements

setup:  ## Crea el entorno con las versiones fijadas en uv.lock
	uv sync --all-groups
	uv run python -m ipykernel install --sys-prefix --name retail-analytics --display-name "retail-analytics (.venv)"
	uv run pre-commit install

lint:
	uv run ruff check src tests notebooks
	uv run ruff format --check src tests

test:  ## Tests rápidos (datos sintéticos)
	uv run pytest -m "not data"

test-all:  ## Incluye los tests que leen los CSV reales
	uv run pytest

eda:  ## Re-ejecuta el notebook de EDA en sitio
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

forecast:  ## Selección, ajuste y comparación de modelos (≈30 min); resultados en data/processed
	uv run python -m retail_analytics.pipeline

requirements:  ## Regenera requirements.txt (instalación con pip) desde uv.lock
	uv export --format requirements-txt --no-hashes --all-groups --no-emit-project --quiet -o requirements.txt
