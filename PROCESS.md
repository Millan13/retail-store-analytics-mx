# Proceso de trabajo

Este documento describe cómo construí la solución, qué herramientas usé (incluida la IA) y cómo
verifiqué los resultados. El detalle de cada decisión está en [docs/decisiones.md](docs/decisiones.md).

## 1. Cómo llegué al problema de negocio

El caso no definía el problema. Antes de elegir, exploré los datos completos
([notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)) buscando dónde había señal real:

- **Descarté** lo que salió plano en los datos: interacciones entre categorías, diferencias de
  mezcla por tienda, efecto de la antigüedad de la tienda y uso de efectivo por nivel
  socioeconómico. Probarlas y descartarlas con evidencia es parte del resultado.
- **Me quedé** con tres preguntas donde los datos sí tienen algo que decir, conectadas a palancas
  de negocio distintas:
  1. **Pronóstico de demanda** (operar mejor): la señal de reposición actual es reactiva y llega
     tarde a los picos.
  2. **Potencial de tiendas** (crecer): el desempeño de cada tienda frente a sus pares es estable
     y medible.
  3. **Promociones** (dejar de gastar en lo que no funciona): la promoción se asignó como al
     azar, lo que permite medirla como experimento.

La elección viene de mi experiencia en consultoría: pronóstico para reposición
(prevención de agotados), potencial de mercado con metas P50/P75 frente a pares, y
matrices de priorización para decidir dónde actuar.

## 2. Etapas y entregables

| Etapa | Entregable | Validación |
|---|---|---|
| Estructura y datos | `src/retail_analytics/io.py`, `contracts.py`, checksum de los CSV | Contratos pandera + `tests/test_raw_data.py` |
| Análisis exploratorio y limpieza | `notebooks/01_eda.ipynb`, `cleaning.py`, `quality.py` | Identidades contables, reconstrucción verificada contra valores reales |
| Pronóstico | `notebooks/02_pronostico.ipynb`, `features.py`, `forecast.py`, `evaluate.py`, `pipeline.py` | Test de fuga, auditoría independiente, backtest con origen móvil |
| Promociones | `notebooks/03_promociones.ipynb`, `promotions.py` | Prueba A/A, test que recupera un efecto conocido |
| Potencial | `notebooks/04_potencial_tiendas.ipynb`, `potential.py` | Estabilidad del índice entre periodos, sensibilidad al método |
| Comunicación | `README.md`, `MODEL_CARD.md` (incluye cómo se operaría) y `reports/presentacion_ejecutiva.pdf` | Cifras de la presentación generadas desde los resultados del repositorio |

Cada etapa se trabajó en su propia rama de git y se integró a `main` al cerrarse.

## 3. Herramientas

| Necesidad | Herramienta |
|---|---|
| Entorno reproducible | `uv` (Python 3.12, versiones fijadas en `uv.lock`); `requirements.txt` para pip; `Makefile` y `make.bat` con los mismos comandos en Mac/Linux y Windows |
| Datos y transformaciones | pandas, DuckDB, pyarrow |
| Contratos de datos | pandera |
| Estadística | statsmodels (OLS, efectos fijos, errores agrupados), SciPy |
| Modelos | scikit-learn (regresión Poisson), LightGBM, XGBoost, statsmodels (Holt-Winters) |
| Seguimiento de experimentos | MLflow (backend SQLite local, `mlflow.db`) |
| Visualización | seaborn / matplotlib; geopandas y folium para mapas |
| Calidad de código | pytest, ruff, pre-commit, GitHub Actions (Linux y Windows) |
| Datos en la nube | lectura `gs://` vía `fsspec` (variable `RAW_DATA_URI`); credenciales solo por variables de entorno |

## 4. Uso de inteligencia artificial

Usé **Claude Code** (Anthropic) como mi brazo derecho para la ejecución: escribir y probar
código, construir notebooks, revisar documentación técnica y redactar. **Las decisiones de
alcance, enfoque y criterio fueron mías**: qué problema resolver, qué preguntas responder, el
estilo de la entrega y qué se presenta al negocio. Le pedí explícitamente que sustentara cada
decisión de método en práctica establecida y que me contradijera cuando algo que yo propusiera
no fuera correcto.

### Cómo verifiqué lo que produjo la IA

No di por bueno ningún resultado sin verificación. Los controles fueron:

1. **Tests automáticos** en cada etapa (42 tests): contratos de datos, identidades contables,
   un test de fuga que altera las ventas futuras y comprueba que ninguna variable del pasado
   cambie, y tests que verifican que los estimadores recuperan efectos conocidos.
2. **Auditoría independiente**: un segundo agente de IA, sin acceso al razonamiento del primero,
   revisó el código de pronóstico buscando fugas y errores de evaluación. No encontró fugas, pero
   sí **un error real** (Holt-Winters colocaba su pronóstico corrido 1–2 días en ~15 de 480
   series por semana) y tests que daban falsa confianza. Todo se corrigió antes de correr los
   resultados finales.
3. **Contraste de cada afirmación del texto contra la salida real** de los notebooks.

### Errores que la verificación detectó (y se corrigieron)

Los dejo documentados porque muestran por qué la verificación importa:

| Qué se afirmó primero | Qué decían realmente los datos | Cómo se detectó |
|---|---|---|
| "Supercenter cae menos en Navidad" | Supercenter **sube más** (+70% vs ~+50%) | Revisión de la tabla contra el texto |
| Promociones: +0.6% en monto y ~1% de descuento, con un costo estimado de ~60 M MXN | −0.2% en monto (IC −0.4% a 0.0%) y sin descuento medible; la cifra de costo no se sostiene | El estimador preliminar (promedio de cocientes) estaba sesgado; el modelo de efectos fijos lo corrigió |
| Identidad de montos al 94% | 100%: el chequeo contaba filas sin dato como incumplimiento | Revisión de la salida |
| Holt-Winters con pronósticos desalineados | Corregido; test agregado | Auditoría independiente |
| Selección de variables podía quitar grupos redundantes juntos | Se agregó una verificación contra el conjunto completo; terminó conservando todos los grupos | Auditoría independiente |

### Skills y guías usadas

- **dataviz**: paleta validada para daltonismo y visión normal, forma de cada gráfica y
  catálogo de errores. Con ella se detectó que los dos verdes del kit visual no se distinguen
  entre sí.
- **pptx**: lectura del kit visual y de presentaciones de referencia, y construcción de la
  presentación ejecutiva.

## 5. Cómo reproducir

Ver [README.md](README.md#cómo-correrlo). En resumen: instalar uv, `make setup` y `make test-all`
(en Windows, `.\make.bat setup` y `.\make.bat test-all`) y abrir los notebooks. Los resultados del
pronóstico ya vienen en `data/processed/`; `make forecast` los regenera (≈30 min).
