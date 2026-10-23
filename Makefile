.PHONY: setup lint test test-all eda

setup:  ## Crea el entorno con las versiones fijadas en uv.lock
	uv sync --all-groups
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
