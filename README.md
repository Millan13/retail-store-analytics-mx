# retail-store-analytics-mx

Análisis de operaciones de una cadena de retail con **80 tiendas en 5 regiones de México**
(enero 2023 a febrero 2024, 6 categorías, ~204 mil registros diarios). El proyecto responde tres
preguntas de negocio, cada una conectada a una palanca distinta:

| Pregunta | Palanca | Resultado principal |
|---|---|---|
| **¿Cuánto voy a vender la próxima semana y cuánto reponer?** | Operar mejor | Un pronóstico LightGBM reduce **23% el error** frente al sistema de reposición actual en 18 semanas que ningún modelo vio, y en la simulación de pedido evita **711 M MXN** entre venta perdida y sobreinventario |
| **¿Qué tiendas venden menos de lo que deberían?** | Crecer | Llevar cada tienda a la mediana de sus pares vale **~1,700 M MXN al año** (+5.7%); 27 tiendas tienen capacidad ociosa y la palanca es comercial |
| **¿Las promociones funcionan?** | Dejar de gastar | **No mueven la venta**: efecto de −0.2% (IC 95%: −0.4% a 0.0%) en 6,953 M MXN de venta promocionada, sin descuento medible |

Antes de modelar, el análisis encontró **dos campos del sistema que llevan a conclusiones falsas**:
la señal de reposición usa la venta del día siguiente, y el conteo de transacciones viene inflado
~22% en días de promoción. Ambos se corrigen o excluyen en todo el proyecto.

## Cómo navegar el repositorio

| Si quieres... | Ve a |
|---|---|
| Entender los datos y cómo se definió el problema | [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) |
| Ver el pronóstico, la comparación de modelos y el impacto en MXN | [notebooks/02_pronostico.ipynb](notebooks/02_pronostico.ipynb) |
| Ver la evaluación de promociones | [notebooks/03_promociones.ipynb](notebooks/03_promociones.ipynb) |
| Ver el potencial de tiendas, el mapa y la matriz | [notebooks/04_potencial_tiendas.ipynb](notebooks/04_potencial_tiendas.ipynb) · [mapa interactivo](reports/figures/mapa_potencial_regional.html) |
| Revisar por qué se tomó cada decisión | [docs/decisiones.md](docs/decisiones.md) |
| Leer la ficha del modelo | [MODEL_CARD.md](MODEL_CARD.md) |
| Saber cómo se trabajó y cómo se usó IA | [PROCESS.md](PROCESS.md) |

Los notebooks se leen de arriba abajo: cada sección abre con una pregunta y cierra con una
decisión o conclusión. La lógica vive en `src/retail_analytics/` y tiene pruebas automáticas.

## Cómo correrlo

Requisitos: [uv](https://docs.astral.sh/uv/) (instala Python 3.12 por su cuenta) y `make`.

```bash
make setup      # entorno con versiones fijadas (uv.lock), kernel de Jupyter y hooks de pre-commit
make test-all   # 41 pruebas, incluida la validación de los CSV reales (checksum + contratos)
make forecast   # selección de variables, ajuste y comparación de modelos (~30 min); escribe data/processed/
make eda        # re-ejecuta el notebook de EDA
```

Después de `make forecast`, los notebooks 02 a 04 se pueden ejecutar en Jupyter o VS Code con el
kernel **retail-analytics (.venv)**. Cada corrida del pronóstico queda registrada en MLflow:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**Datos en la nube.** Por defecto se leen de `data/raw/`. Para leerlos de Google Cloud Storage,
copia `.env.example` a `.env`, define `RAW_DATA_URI=gs://...` e instala el extra
`uv sync --extra gcs`. Las credenciales se pasan solo por variables de entorno.

## Estructura

```
configs/config.yaml     rutas, horizonte del pronóstico, semanas de validación y prueba, columnas prohibidas
data/raw/               datos originales, inmutables, con checksum (MANIFEST.sha256)
data/external/          polígonos de estados (solo para el mapa; fuente en su README)
data/processed/         resultados regenerables (no se versionan)
docs/                   diccionario de datos y bitácora de decisiones
notebooks/              01 EDA · 02 pronóstico · 03 promociones · 04 potencial
reports/figures/        mapa interactivo
src/retail_analytics/
  io.py, config.py      carga de datos (local o gs://) y configuración
  contracts.py          contratos de datos (pandera)
  quality.py            diagnósticos de calidad
  cleaning.py           limpieza y reconstrucción de campos
  features.py           variables del pronóstico (todas con rezago ≥ 7 días)
  forecast.py           modelos y backtest con origen móvil
  evaluate.py           métricas, impacto en MXN y regla de elección de modelo
  pipeline.py           selección, ajuste y comparación (make forecast) + MLflow
  promotions.py         promociones como experimento (efectos fijos, potencia, segmentos)
  potential.py          potencial de tiendas (consulta SQL en DuckDB + regresión)
  viz.py                estilo de gráficas (paleta validada para daltonismo)
tests/                  pruebas unitarias, de fuga de información y de integridad de datos
```

## Limitaciones principales

- Los datos empiezan en enero de 2023: el Buen Fin y la Navidad de 2023 son los primeros, y
  ningún modelo pudo aprenderlos.
- El impacto en MXN del pronóstico es una simulación (no hay datos de inventario, agotados ni
  costos); se propone confirmarlo con un piloto con grupo control.
- Las tiendas no tienen ubicación: el mapa es por región, con una asignación de estados que es un
  supuesto documentado y no cambia ninguna cifra.
