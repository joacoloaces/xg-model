# xG model — StatsBomb Open Data

Modelo de expected goals entrenado con tiros de todas las competiciones abiertas de StatsBomb
(o el subconjunto que se configure).

## Estructura

```
main.py               orquestador (download -> features -> train -> evaluate)
config.yaml           competiciones, split, modelos, rutas
src/data_loader.py    selección de competiciones y descarga paralela con caché por partido
src/preprocessing.py  limpieza, booleanos, target, filtros (tandas de penales, penales)
src/features.py       geometría, interacciones y features del freeze frame
src/split.py          split por partido (group) o temporal
src/models.py         modelos y espacios de hiperparámetros
src/train.py          RandomizedSearchCV + StratifiedGroupKFold, selección por log loss
src/evaluate.py       métricas en test, benchmark vs StatsBomb xG, calibración, desglose
```

## Uso

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Prueba rápida:

```
python main.py --competitions "FIFA World Cup" --max-matches 40 --n-iter 3
```

Re-entrenar sin volver a descargar:

```
python main.py --steps features train evaluate
```

## Salidas (`outputs/`)

- `models/*.joblib`, `models/best_model.joblib` — pipelines completos (preprocesado + modelo)
- `reports/cv_leaderboard.csv`, `reports/test_metrics.csv`, `reports/test_por_competicion.csv`
- `reports/calibracion.png`, `reports/scatter_<modelo>_vs_sb.png`, `reports/feature_importance_<modelo>.csv`

## Usar el modelo

```python
import joblib
model = joblib.load("outputs/models/best_model.joblib")
xg = model.predict_proba(df[NUM_FEATURES + CAT_FEATURES])[:, 1]
```

## Simulador xG (Interfaz Gráfica)

La aplicación incluye un simulador visual con React y FastAPI. Para iniciarlo, abrí **dos terminales**:

**Terminal 1 (Backend - FastAPI):**
Desde la carpeta raíz del proyecto, ejecutá:
```bash
uvicorn api.server:app --reload
```
*Esto levantará el servidor en `http://localhost:8000` exponiendo el modelo entrenado.*

**Terminal 2 (Frontend - React/Vite):**
Desde la carpeta raíz, entrá al frontend e iniciá la app:
```bash
cd frontend
npm install  # (Solo la primera vez)
npm run dev
```
*La interfaz estará disponible en `http://localhost:5173`.*

## Ejecutar la aplicación en Windows

1. Instala Python 3.12 y Node.js.
2. Desde la carpeta raíz, ejecuta `iniciar_backend.ps1` en PowerShell. Si Windows bloquea scripts locales, abre PowerShell y ejecuta `Set-ExecutionPolicy -Scope Process Bypass` antes de volver a intentarlo. La primera ejecución instala las dependencias.
3. En otra terminal ejecuta `cd frontend`, `npm ci` y `npm run dev`.
4. Abre la URL que muestre Vite. La API debe responder en `http://localhost:8000/health`.

El campo se muestra en orientación vertical, con la portería rival arriba; los remates solo se pueden simular en la mitad ofensiva.
