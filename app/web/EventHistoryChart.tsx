import { useMemo } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import type { Price } from './types';

type Props = {
  history: Price[];
  title: string;
  eventDate: string;
  eventYear: string;
};

const fechaES = (fecha: string) =>
  new Date(`${fecha}T12:00:00Z`).toLocaleDateString(
    'es-ES',
    { month: 'short', year: 'numeric', timeZone: 'UTC' }
  );

const fechaCompleta = (fecha: string) =>
  new Date(`${fecha}T12:00:00Z`).toLocaleDateString(
    'es-ES',
    { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' }
  );

export default function EventHistoryChart({
  history,
  title,
  eventDate,
  eventYear,
}: Props) {

  const datos = useMemo(() => {
    const centro = Date.parse(`${eventDate}T12:00:00Z`);
    const margen = 365 * 86400000;

    return history
      .filter((p) => {
        const fecha = Date.parse(`${p.fecha}T12:00:00Z`);
        return (
          Number.isFinite(fecha) &&
          Math.abs(fecha - centro) <= margen &&
          Number.isFinite(p.precio) &&
          p.precio > 0
        );
      })
      .sort((a, b) => a.fecha.localeCompare(b.fecha));

  }, [history, eventDate]);

  if (datos.length < 10) {
    return (
      <section className="panel">
        <h2>{title}</h2>
        <p className="source-note">
          No hay suficientes precios alrededor del acontecimiento.
        </p>
      </section>
    );
  }

  const precios = datos.map(p => p.precio);
  const minimo = Math.min(...precios);
  const maximo = Math.max(...precios);

  const margen = Math.max((maximo - minimo) * 0.10, 2);

  // Seleccionar el día bursátil más cercano al evento.
  const eventoMs = Date.parse(`${eventDate}T12:00:00Z`);

  const fechaMarcador = datos.reduce((mejor, p) => {
    const distancia = Math.abs(
      Date.parse(`${p.fecha}T12:00:00Z`) - eventoMs
    );

    const distanciaMejor = Math.abs(
      Date.parse(`${mejor.fecha}T12:00:00Z`) - eventoMs
    );

    return distancia < distanciaMejor ? p : mejor;
  }, datos[0]).fecha;

  const primero = datos[0];
  const ultimo = datos[datos.length - 1];

  return (
    <section className="panel history-panel">
      <div className="panel-heading">
        <div>
          <h2>
            {title}
            <span className="small-tag" style={{ marginLeft: 12 }}>
              BRENT
            </span>
          </h2>
          <p>Precio al contado · USD por barril</p>
        </div>
        <span className="mini-label">
          {eventYear}
        </span>
      </div>

      <div
        className="legend"
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 24,
          margin: '22px 0 16px',
        }}
      >
        <span style={{ display: 'inline-flex', gap: 9, alignItems: 'center' }}>
          <i
            style={{
              width: 22,
              height: 3,
              background: '#c3a777',
              display: 'inline-block',
            }}
          />
          Brent · FRED / EIA
        </span>

        <span style={{ display: 'inline-flex', gap: 9, alignItems: 'center' }}>
          <i
            style={{
              width: 3,
              height: 18,
              background: '#77a89c',
              display: 'inline-block',
            }}
          />
          Acontecimiento · {fechaCompleta(eventDate)}
        </span>
      </div>

      <div style={{ width: '100%', height: 420, minWidth: 0 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={datos}
            margin={{
              top: 26,
              right: 35,
              bottom: 15,
              left: 5,
            }}
          >
            <CartesianGrid
              vertical={false}
              stroke="#303638"
              strokeDasharray="3 5"
            />

            <XAxis
              dataKey="fecha"
              tickFormatter={fechaES}
              tick={{ fill: '#919a9c', fontSize: 12 }}
              tickLine={false}
              axisLine={false}
              minTickGap={75}
            />

            <YAxis
              domain={[
                Math.floor(minimo - margen),
                Math.ceil(maximo + margen),
              ]}
              tick={{ fill: '#919a9c', fontSize: 12 }}
              tickLine={false}
              axisLine={false}
              width={52}
              tickFormatter={(valor: number) => `$${valor}`}
            />

            <Tooltip
              labelFormatter={(fecha) =>
                fechaCompleta(String(fecha))
              }
              formatter={(valor) => [
                `${Number(valor).toFixed(2)} USD`,
                'Brent',
              ]}
              contentStyle={{
                background: '#171c1e',
                border: '1px solid #3a4244',
                borderRadius: 9,
                color: '#f1f1eb',
              }}
            />

            <Line
              dataKey="precio"
              type="monotone"
              stroke="#c3a777"
              strokeWidth={2.3}
              dot={false}
              activeDot={{ r: 5 }}
              isAnimationActive={false}
            />

            <ReferenceLine
              x={fechaMarcador}
              stroke="#77a89c"
              strokeWidth={2}
              strokeDasharray="6 5"
              label={{
                value: `Evento ${eventYear}`,
                position: 'insideTopRight',
                fill: '#9fc6bc',
                fontSize: 12,
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div
        className="chart-footer"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          gap: 16,
          flexWrap: 'wrap',
        }}
      >
        <span>
          <i className="status-dot muted-dot" />
          {fechaCompleta(primero.fecha)}
          {' — '}
          {fechaCompleta(ultimo.fecha)}
        </span>

        <span>
          {datos.length} observaciones
          {' · '}
          Rango: ${minimo.toFixed(2)}–${maximo.toFixed(2)}
        </span>
      </div>
    </section>
  );
}
