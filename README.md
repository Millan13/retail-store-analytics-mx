# retail-store-analytics-mx

Análisis de operaciones de una cadena de retail con 80 tiendas en 5 regiones de México
(enero 2023 – febrero 2024). El proyecto responde tres preguntas de negocio:

1. **¿Cuánto voy a vender la próxima semana y cuánto reponer?** Pronóstico de unidades por
   tienda × categoría a 7 días, comparado contra la señal de reposición actual.
2. **¿Qué tiendas venden menos de lo que deberían?** Potencial frente a tiendas comparables
   (escenarios P50/P75), con vista regional.
3. **¿Las promociones funcionan?** Evaluación de las promociones como experimento.

> En construcción. Estado actual: estructura del repo + análisis exploratorio
> ([notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)).

## Cómo correrlo

Requisitos: [uv](https://docs.astral.sh/uv/) (instala Python 3.12 por su cuenta).

```bash
make setup      # entorno con versiones fijadas (uv.lock) + hooks de pre-commit
make test       # tests rápidos
make test-all   # incluye validación de los CSV reales
make eda        # re-ejecuta el notebook de EDA
```

## Estructura

```
configs/            configuración central (rutas, columnas prohibidas para modelar)
data/raw/           datos originales, inmutables, con checksum (MANIFEST.sha256)
docs/               diccionario de datos y bitácora de decisiones
notebooks/          exploración y lectura de resultados (la lógica vive en src/)
src/retail_analytics/
  io.py             carga de datos (local o gs://)
  contracts.py      contratos de datos (pandera)
  quality.py        diagnósticos de calidad
  cleaning.py       limpieza y reconstrucción de campos
  viz.py            estilo y paleta de gráficas
tests/              pruebas unitarias e integridad de datos
```
