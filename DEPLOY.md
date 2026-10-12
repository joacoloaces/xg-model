# Arranque local en Windows

1. Instala Node.js LTS desde https://nodejs.org/.
2. Instala Python 3.12 (recomendado). El script tambien detecta Python 3.13, 3.14 o 3.11 si ya esta instalado.
3. Abre PowerShell en la carpeta raiz del proyecto y ejecuta `./iniciar_backend.ps1`.
4. Espera a que termine la instalacion de dependencias. Deja esa terminal abierta.
5. En una segunda terminal ejecuta `cd frontend`, `npm ci` y `npm run dev`.
6. Abre la URL local que muestre Vite, normalmente http://localhost:5173.
7. Comprueba la API en http://localhost:8000/health.

Si PowerShell bloquea la ejecucion del script, ejecuta para esa sesion:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

El campo se muestra en orientacion vertical. El backend necesita conectividad a Internet la primera vez para instalar sus dependencias de Python.
