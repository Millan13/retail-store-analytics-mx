# Bitácora de decisiones

Cada decisión con su alternativa, la razón y dónde está la evidencia. Ordenadas por etapa.

## Datos y limpieza

| # | Decisión | Alternativa descartada | Por qué | Evidencia |
|---|---|---|---|---|
| D1 | `replenishment_signal` no se usa como variable de modelo | Usarla como variable "porque viene del sistema" | Es el promedio de `total_transactions` de t−5 a t+1: incluye el día siguiente (coincide exacto en el 100% de las filas con señal). Usarla es fuga de información | EDA §5; `quality.replenishment_window_check`; `tests/test_quality.py` |
| D2 | El conteo de transacciones válido es `transactions_clean = efectivo + tarjeta` | Usar `total_transactions` | En días con promoción `total_transactions` viene inflado 8–35% (100% de esas filas) y el ticket del sistema se calcula sobre efectivo + tarjeta | EDA §4; `quality.accounting_identities` |
| D3 | Las 42 filas faltantes se agregan marcadas y sin rellenar | Rellenar con ceros o interpolar | No sabemos si la tienda abrió; un cero inventa una caída de demanda | EDA §2; `cleaning.complete_grid` |
| D4 | Efectivo, monto en efectivo y ticket se reconstruyen por identidad contable | Imputar con medianas o modelos | Las identidades se cumplen en el 100% de las filas comprobables; la reconstrucción se validó borrando y recuperando 5,000 valores (100% exacto) | EDA §6 |
| D5 | `units_sold` no se imputa en la limpieza | Imputar en la limpieza | Cualquier imputación que aprenda de los datos filtraría información del periodo de prueba; LightGBM y XGBoost manejan nulos de forma nativa y el GLM imputa dentro de su pipeline con datos de entrenamiento | `cleaning.py` (docstring); `forecast.poisson_glm` |
| D6 | Los picos de eventos no se tratan como outliers | Recortarlos o eliminarlos | El 99% de los valores extremos cae en fechas de evento (Buen Fin, Navidad, Año Nuevo): son demanda real | EDA §12 |
| D7 | No se construye un análisis de interacciones entre categorías | Analizar complementos y sustitutos | Tienda × categoría explica 0.02% de la varianza; sin el efecto común de afluencia, la correlación entre categorías es la de ruido puro | EDA §10 |

## Pronóstico

| # | Decisión | Alternativa descartada | Por qué | Evidencia |
|---|---|---|---|---|
| F1 | Pronosticar `units_sold` por tienda × categoría × día, 7 días adelante | Pronosticar monto o transacciones | La reposición se pide en unidades; las transacciones vienen infladas en promoción | Notebook 02 §1 |
| F2 | Toda variable de ventas usa datos de 7 o más días antes (enfoque directo) | Rezagos de 1 día con pronóstico recursivo | Un solo modelo sirve para los 7 días, sin acumular errores ni riesgo de fuga. Es el enfoque directo usado en los primeros lugares de la competencia M5 | `features.py`; test de fuga sobre `build_features` |
| F3 | Validación con origen móvil semanal | Partición aleatoria o un solo corte | Es el estándar para series de tiempo (Hyndman, FPP3 §5.10): se entrena solo con el pasado de cada origen | `forecast.backtest` |
| F4 | Validación (sep–oct) y prueba (nov–feb) separadas; la prueba se usa una sola vez | Elegir variables o parámetros mirando la prueba | Ajustar con la prueba da resultados optimistas e indefendibles | `pipeline.py`; `configs/config.yaml` |
| F5 | Seis modelos en escalera de complejidad | Ir directo a LightGBM | El modelo debe ganarse su lugar; si un modelo simple empata, gana el simple | Notebook 02 §3 |
| F6 | Selección de variables por grupos (quitar un grupo a la vez) y verificación del conjunto final contra el completo | Selección variable por variable | Las variables de un grupo están correlacionadas; quitarlas una a una subestima su aporte. La verificación final evita quitar juntos grupos redundantes | `pipeline.select_feature_groups`; `validation_selection_check.csv` |
| F7 | Se conservan todos los grupos de variables | Quedarse con el conjunto reducido (calendario, promedios, categoría) | El conjunto reducido empeoró el WAPE de validación en 0.13 puntos (más que el umbral de 0.1): los grupos que individualmente "no aportaban" eran redundantes entre sí, no inútiles | `validation_selection_check.csv` |
| F8 | Objetivo Poisson o Tweedie lo decide la validación | Fijarlo a priori | Son conteos sin ceros (mínimo 22 unidades); ambos son razonables según la documentación de LightGBM | `validation_tuning.csv` |
| F9 | Métrica principal WAPE + sesgo | MAPE, RMSE | WAPE permite sumar series de distinto tamaño y no explota con ventas pequeñas; el sesgo distingue sobreinventario de venta perdida | `evaluate.py` |
| F10 | Impacto de negocio con una política de pedido semanal idéntica para todos los modelos | Comparar solo métricas estadísticas | El negocio decide en pesos; la misma política para todos hace justa la comparación | `evaluate.inventory_impact` |
| F11 | `has_promotion` se trata como conocida de antemano | Excluirla | Las promociones se planean; además la selección muestra que no aporta al pronóstico, así que el supuesto no cambia resultados | Notebook 02 §5 |

## Promociones

| # | Decisión | Alternativa descartada | Por qué | Evidencia |
|---|---|---|---|---|
| P1 | Tratar la promoción como experimento natural | Modelo observacional con controles | La asignación es pareja en todos los cortes y la prueba A/A (venta de la semana previa) sale limpia (p = 0.95) | Notebook 03 §2 |
| P2 | Estimador de efectos fijos (tienda × categoría × mes, día, festivo) con errores agrupados por tienda | Promedio de cocientes por celda (usado en la vista preliminar del EDA) | El promedio de cocientes con pocas promociones por celda se sesga hacia arriba (desigualdad de Jensen): daba +0.6% en monto; el estimador correcto da −0.2% (IC −0.4% a 0.0%) | Notebook 03 §3; `tests/test_promotions.py` recupera un efecto conocido de +10% |
| P3 | Reportar el efecto mínimo detectable | Solo decir "no es significativo" | Un resultado nulo solo es informativo si el análisis tenía potencia: aquí detectaría efectos desde 0.3% | Notebook 03 §4 |
| P4 | Corrección Benjamini-Hochberg en el análisis por segmento | Reportar p-valores sin corregir | Con 24 segmentos, uno o dos salen significativos por azar (p. ej. A/B con p = 0.017, ajustado 0.19) | Notebook 03 §5 |

## Potencial de tiendas

| # | Decisión | Alternativa descartada | Por qué | Evidencia |
|---|---|---|---|---|
| T1 | Venta esperada con formato, región, nivel socioeconómico y tamaño | Comparar solo contra el formato | Atribuirle a la tienda diferencias de mercado que no controla infla la oportunidad (de 1,707 a 2,317 M MXN en P50) | Notebook 04 §4 |
| T2 | Escenarios P50 (conservador) y P75 (ambicioso) dentro del formato | Una sola meta | Mismo par de escenarios usado en trabajos de potencial de mercado; da un rango en lugar de un número único | Notebook 04 §3 |
| T3 | Mapa por región con asignación de estados documentada como supuesto | Inventar coordenadas de tiendas | Los IDs son sintéticos; inventar ubicaciones sería fabricar datos | `potential.REGION_STATES`; `data/external/README.md` |
| T4 | Matriz desempeño × carga de cajas, ambos contra la mediana del formato | Solo ranking por brecha | Distingue si la palanca es comercial (capacidad ociosa) u operativa (cajas saturadas) | Notebook 04 §6 |

## Estilo y entregables

| # | Decisión | Por qué |
|---|---|---|
| E1 | Paleta del kit visual: fondo blanco, texto Bentonville, True Blue principal, amarillo solo de acento | Reglas explícitas del kit |
| E2 | Bodega usa una tinta oscura de su amarillo (`#E0A100`) en lugar del verde del kit | Los dos verdes del kit (Bodega y Express) no pasan el piso de distinción con visión normal (ΔE 11.2 < 15) en el validador de dataviz; la paleta final pasa en todos los pares |
| E3 | Gráficas en seaborn (estáticas) en los notebooks; HTML interactivo solo para el mapa | GitHub no muestra gráficas interactivas dentro de un notebook |
| E4 | Código, docstrings y comentarios en inglés; notebooks y documentos en español | Estándar del repo: el código usa nombres en inglés; la audiencia de negocio lee español |
