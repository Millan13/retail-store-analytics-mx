# Ficha del modelo: pronóstico de demanda a 7 días

## Objetivo

Pronosticar las **unidades vendidas por tienda × categoría para cada uno de los próximos 7 días**,
para que la reposición semanal se base en la demanda esperada y no en el promedio de los días
pasados. El sistema de reposición actual (`replenishment_signal`) es un promedio móvil de 7 días
que reacciona tarde a los picos y además usa información del día siguiente.

**Usuario:** planeación de inventarios / reposición de tienda.
**Decisión que apoya:** cuánto pedir por tienda y categoría para la semana siguiente.

## Datos

| Elemento | Detalle |
|---|---|
| Fuente | `transactions.csv`, `stores.csv`, `calendar.csv` (80 tiendas, 6 categorías, 2023-01-01 a 2024-02-29) |
| Unidad | Tienda × categoría × día (204,000 filas tras completar la malla) |
| Objetivo | `units_sold` (3% nulo: no se imputa; esas filas no se usan para entrenar ni para medir) |
| Limpieza | Ver EDA §6 y `docs/decisiones.md` (D1–D7) |
| Excluidas como variables | `replenishment_signal` (incluye t+1), `total_transactions` (inflada en promoción) y cualquier medida de venta del mismo día |

## Variables

31 variables en 7 grupos (`src/retail_analytics/features.py`). Toda variable construida con
ventas usa información de **7 o más días antes** de la fecha pronosticada. Un test lo verifica:
alterar las ventas futuras no cambia ninguna variable del pasado.

| Grupo | Contenido |
|---|---|
| Calendario | Día de la semana y del mes, mes, fin de semana, quincena, festivo y su nombre, Buen Fin, temporada navideña, Semana Santa |
| Rezagos | Unidades de hace 7, 14, 21 y 28 días; si hace 7 días fue festivo o quincena |
| Promedios móviles | Media de 7 y 28 días, mediana y desviación de 28 días (terminan 7 días antes), media del mismo día en 4 semanas |
| Tráfico de tienda | Transacciones diarias de la tienda (todas las categorías), promedio de 7 días que termina 7 días antes |
| Tienda | Formato, región, nivel socioeconómico, m², cajas, farmacia, gasolinera |
| Categoría | Categoría |
| Promoción | Promoción planeada (supuesto: se conoce de antemano) |

## Metodología

- **Validación con origen móvil semanal** (Hyndman y Athanasopoulos, FPP3 §5.10): cada domingo se
  entrena con todo lo anterior y se pronostica la semana siguiente.
- **Validación** (9 semanas, 28-ago a 29-oct-2023): selección de variables por grupos y ajuste
  de hiperparámetros.
- **Prueba** (18 semanas, 30-oct-2023 a 29-feb-2024): usada una sola vez para comparar modelos.
- **Modelos comparados:** sistema actual corregido (promedio de 7 días sin ver el futuro), naive
  estacional, Holt-Winters por serie, regresión Poisson, LightGBM y XGBoost.
- **Regla de elección** (fijada antes de la prueba): menor WAPE; si un modelo más simple empata
  (bootstrap por tienda de la diferencia de WAPE con IC 95% que incluye cero), gana el simple.
- **Auditoría independiente** del código de variables y evaluación antes de la corrida final.

## Resultados

**Modelo elegido: LightGBM** (objetivo Tweedie, 31 hojas, 600 árboles, tasa de aprendizaje 0.05;
hiperparámetros elegidos en validación). Prueba: 18 semanas que ningún modelo usó para entrenar
ni para ajustarse.

| Modelo | WAPE | Sesgo | WAPE días normales | Diferencia vs LightGBM (IC 95%, bootstrap por tienda) |
|---|---|---|---|---|
| **LightGBM** | **34.9%** | −3.8% | **27.6%** | — |
| XGBoost | 35.7% | −1.1% | 29.7% | +0.75 pts (+0.53 a +0.96) |
| Regresión Poisson | 37.2% | −4.6% | 30.2% | +2.21 pts (+1.88 a +2.52) |
| Holt-Winters | 42.8% | +1.0% | 37.0% | +7.83 pts (+7.13 a +8.58) |
| Sistema actual (promedio 7 días, sin ver el futuro) | 45.2% | +0.6% | 37.1% | +10.29 pts (+9.87 a +10.68) |
| Naive estacional | 47.1% | +0.2% | 44.4% | +12.17 pts (+11.74 a +12.57) |

- **23% menos error** que el sistema actual (26% en días normales, 27% en temporada navideña).
- **Buen Fin:** 63.9% vs 64.6%; sin mejora porque es el primero en la historia de datos.
- **Por segmento:** mejora entre 19% y 25% en cada formato, región, nivel socioeconómico y
  categoría; no empeora en ninguno.
- **Piso de error:** en días normales el modelo (27.6%) está al nivel de un oráculo que conoce de
  antemano el perfil real de cada serie (27.3%).
- **Impacto simulado (pedido semanal):** −104 M MXN de venta perdida (−6%) y −607 M MXN de
  sobreinventario (−34%) en 18 semanas; 711 M MXN en total sobre una venta de 12,076 M MXN.
  Anualizado en proporción a la venta: ~1,760 M MXN (de ellos ~260 M MXN de venta recuperada).

Detalle completo en [notebooks/02_pronostico.ipynb](notebooks/02_pronostico.ipynb).

## Limitaciones y consideraciones de uso

- **Eventos sin historia.** Los datos empiezan en enero de 2023: el Buen Fin y la Navidad de 2023
  son los primeros. Ningún modelo pudo aprenderlos. Con un año más de datos, un rezago de 364 días
  los capturaría; mientras tanto, esas semanas requieren un ajuste manual del planeador.
- **Ruido diario irreducible.** Aun conociendo el patrón real de cada serie, el error en días
  normales sería ~18–21% (notebook 02 §8). El modelo debe usarse junto con inventario de
  seguridad, no como cifra exacta.
- **Impacto en pesos simulado.** No hay datos de inventario, agotados ni costos: la venta perdida
  y el sobreinventario salen de una política de pedido simplificada, igual para todos los modelos.
- **Promoción como dato conocido.** Se asume que la promoción está planeada; la selección de
  variables muestra que no aporta al pronóstico, así que el supuesto no cambia los resultados.
- **Monitoreo sugerido.** WAPE y sesgo semanales por formato y región; alerta si el WAPE de dos
  semanas seguidas supera al del sistema actual o si el sesgo pasa de ±5%.

## Reproducibilidad

`make setup && make forecast` regenera todos los resultados (versiones fijadas en `uv.lock`,
semilla fija, checksum de los datos crudos). Cada corrida queda registrada en MLflow
(`mlflow.db`; ver con `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`).
