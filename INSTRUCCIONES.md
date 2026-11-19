# BRENT. — versión Figma Make conectada al proyecto

Esta es la interfaz **original de Figma Make** (React, TypeScript, Recharts y Vite), con una pestaña nueva «Prueba a ciegas». No se sustituye por la web HTML alternativa.

## Instalar en `caso02_petroleo`

1. Copia la carpeta `app/` de este ZIP a la raíz de tu repositorio. Si ya existe una carpeta `app` de la web HTML anterior, renómbrala como `app_anterior` antes de copiar.
2. Copia `export_web_data_figma.py` a la raíz del repositorio (junto a `data/`, `resultados/`, `src/`).
3. No sustituyas tus carpetas `data/`, `notebook/`, `resultados/`, `scripts/` o `src/`.
4. En la raíz del repo, ejecuta `python export_web_data_figma.py`. Esto genera `app/public/datos/brent.json` con histórico, resultados y prueba a ciegas, sin entrenar.
5. Ejecuta `cd app && npm install && npm run dev` (también puedes usar `pnpm install` y `pnpm run dev`).
6. Abre el enlace que imprima Vite; si estás conectada a la GPU por SSH, reenvía ese puerto desde VS Code.

**Importante:** la web React **no** funciona correctamente con `python -m http.server --directory app`: necesita Vite para desarrollo o el compilado `dist/` para producción.

## Observaciones reales

El exportador solo muestra precios observados si existe `resultados/prueba_ciega_2026-11-19/observaciones.csv`, con estas columnas:

```csv
fecha,precio,tipo,fuente
```

Añade únicamente valores cotejados con su fuente, indicando `intradía` o `cierre comparable`. No se confunden cotizaciones intradía de futuros con el precio diario al contado del dataset. El exportador nunca usa estas observaciones para entrenamiento.

## Compilar y desplegar

Desde `app/`: `npm run build` genera `dist/`. Para publicar con Vercel, configura Root Directory = `app` y Build Command = `npm run build`, Output Directory = `dist`. El `vercel.json` original puede usar pnpm; si usas npm puedes ajustar ese archivo antes de desplegar.

Para GitHub Pages, ejecuta `GITHUB_PAGES=true npm run build`; publica el contenido de `app/dist/` mediante GitHub Actions (ruta base `/caso02_petroleo/`).

## Datos

No hay datos falsos de las redes en la nueva pestaña. Si `predicciones.csv` no está publicado, la web indica que está pendiente. Los archivos `app/public/datos/brent.json` incluidos vienen del ZIP original de Figma y pueden estar desactualizados; ejecútalo con el CSV y los resultados de tu servidor antes de presentar la versión definitiva.
