# Touchline Lab — frontend actualizado

Se rediseñó el frontend del simulador xG con una interfaz de analítica deportiva más cercana a una herramienta de scouting y rendimiento:

- Dashboard responsive con indicadores de sesión: remates, xG total, xG medio y ocasiones con xG >= 0,30.
- Campo interactivo con más espacio visual, mapa de remates codificado por xG y selección de ubicación para simular.
- Panel de contexto del tiro reutilizando los controles y el endpoint del modelo existentes.
- Vista de análisis con distribución de calidad de ocasión, ranking de remates y tabla detallada.
- Exportación CSV de las simulaciones.
- Estado visible de conexión con la API y mensajes de error.

## Ejecutar en local

Desde la raíz del proyecto, terminal 1:

```bash
python -m pip install -r requirements-api.txt
uvicorn api.server:app --reload
```

Terminal 2:

```bash
cd frontend
npm install
npm run dev
```

Abrí la URL local que indique Vite (normalmente `http://localhost:5173`). La API debe estar disponible en `http://localhost:8000`.

## Importante sobre la interpretación

El frontend reutiliza el modelo y los parámetros de simulación que ya existían. Los gráficos resumen las simulaciones de la sesión actual; no representan por sí mismos estadísticas reales de un club ni una validación profesional del modelo. En particular, las variables del freeze frame y del contexto defensivo son parámetros hipotéticos, según la implementación actual del backend.
