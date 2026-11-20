import { useState } from 'react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { CalendarDays, ShieldCheck } from 'lucide-react';
import type { Data } from './types';
import { dateLabel, number } from './Charts';

type Point = {
  fecha: string;
  historico?: number;
  prediccion?: number;
  observado?: number;
  banda?: [number, number];
};

export function BlindTest({ data }: { data: Data }) {
  const [horizon, setHorizon] =
    useState<'todos' | '1' | '2'>('todos');
  const [showBand, setShowBand] = useState(true);
  const [showObserved, setShowObserved] = useState(true);

  const test = data.pruebaCiega;

  if (!test?.disponible || !test.predicciones.length) {
    return (
      <section className="panel blind-panel">
        <h2>La prueba a ciegas.</h2>
        <p className="subtle">
          No hay predicciones publicadas.
        </p>
      </section>
    );
  }

  const rows = test.predicciones
    .filter(
      p =>
        p.modelo === 'Comb. redes' &&
        (horizon === 'todos' ||
          String(p.horizonte) === horizon)
    )
    .sort((a, b) => a.horizonte - b.horizonte);

  const latest = data.historico
    .filter(p => p.fecha <= test.fechaCorte)
    .slice(-10);

  const origin = latest.at(-1);
  const points = new Map<string, Point>();

  latest.forEach(p =>
    points.set(p.fecha, {
      fecha: p.fecha,
      historico: p.precio,
    })
  );

  // Conectar la predicción al último precio histórico.
  if (origin) {
    points.set(origin.fecha, {
      ...points.get(origin.fecha),
      fecha: origin.fecha,
      prediccion: origin.precio,
    });
  }

  rows.forEach(p => {
    points.set(p.fecha, {
      ...points.get(p.fecha),
      fecha: p.fecha,
      prediccion: p.precio,
      ...(p.inferior !== null &&
      p.superior !== null
        ? {
            banda: [
              p.inferior,
              p.superior,
            ] as [number, number],
          }
        : {}),
    });
  });

  if (showObserved) {
    test.observaciones.forEach(o => {
      if (rows.some(r => r.fecha === o.fecha)) {
        points.set(o.fecha, {
          ...points.get(o.fecha),
          fecha: o.fecha,
          observado: o.precio,
        });
      }
    });
  }

  const plot = [...points.values()].sort(
    (a, b) => a.fecha.localeCompare(b.fecha)
  );

  const getObserved = (fecha: string) =>
    test.observaciones.find(o => o.fecha === fecha);

  return (
    <div className="page-stack">
      <section className="panel blind-panel">
        <div className="panel-heading">
          <div>
            <div className="eyebrow">
              <CalendarDays size={13} />
              EXPERIMENTO FUERA DE MUESTRA
            </div>

            <h2>Antes de saber qué ocurrirá.</h2>

            <p>
              Histórico congelado el 18 de noviembre de 2026.
              Predicciones de 1 y 2 días.
            </p>
          </div>

          <span className="mini-label">
            CORTE CONGELADO
          </span>
        </div>

        <div className="blind-controls">
          <div className="periods">
            {(['todos', '1', '2'] as const).map(h => (
              <button
                key={h}
                className={horizon === h ? 'selected' : ''}
                aria-pressed={horizon === h}
                onClick={() => setHorizon(h)}
              >
                {h === 'todos' ? 'Ambos días' : `Día ${h}`}
              </button>
            ))}
          </div>

          <label>
            <input
              type="checkbox"
              checked={showBand}
              onChange={e =>
                setShowBand(e.target.checked)
              }
            />
            Intervalo 80 %
          </label>

          <label>
            <input
              type="checkbox"
              checked={showObserved}
              onChange={e =>
                setShowObserved(e.target.checked)
              }
            />
            Observaciones
          </label>
        </div>

        <div className="blind-chart">
          <ResponsiveContainer
            width="100%"
            height="100%"
          >
            <ComposedChart
              data={plot}
              margin={{
                top: 20,
                right: 20,
                bottom: 12,
                left: 0,
              }}
            >
              <CartesianGrid
                stroke="#27302e"
                strokeDasharray="3 5"
                vertical={false}
              />

              <XAxis
                dataKey="fecha"
                tick={{
                  fill: '#87928e',
                  fontSize: 11,
                }}
                tickFormatter={v => dateLabel(v)}
                minTickGap={26}
                tickLine={false}
                axisLine={false}
              />

              <YAxis
                domain={['auto', 'auto']}
                tick={{
                  fill: '#87928e',
                  fontSize: 11,
                }}
                tickFormatter={v => number(v, 0)}
                tickLine={false}
                axisLine={false}
              />

              <Tooltip
                contentStyle={{
                  background: '#161d20',
                  border: '1px solid #3b4844',
                  borderRadius: 8,
                  color: '#e8e4da',
                }}
                labelFormatter={v =>
                  dateLabel(String(v), true)
                }
                formatter={(value, name) => [
                  Array.isArray(value)
                    ? value
                        .map(v => number(Number(v)))
                        .join('–')
                    : `${number(Number(value))} USD`,
                  String(name),
                ]}
              />

              {showBand && (
                <Area
                  name="Intervalo nominal 80 %"
                  type="linear"
                  dataKey="banda"
                  stroke="none"
                  fill="#d6a465"
                  fillOpacity={0.2}
                  connectNulls={false}
                  isAnimationActive={false}
                />
              )}

              <Line
                name="Histórico del Brent"
                dataKey="historico"
                stroke="#78a6dc"
                strokeWidth={2.4}
                dot={{ r: 2 }}
                isAnimationActive={false}
              />

              <Line
                name="Predicción de redes"
                dataKey="prediccion"
                stroke="#d5a66e"
                strokeWidth={2.5}
                strokeDasharray="5 4"
                dot={{
                  r: 5,
                  stroke: '#171b1d',
                  strokeWidth: 2,
                }}
                connectNulls
                isAnimationActive={false}
              />

              {showObserved && (
                <Scatter
                  name="Observado / referencia"
                  dataKey="observado"
                  fill="#5bc3a6"
                  isAnimationActive={false}
                />
              )}

              {/* Barras de los intervalos por fecha */}
              {showBand &&
                rows.map(r =>
                  r.inferior !== null &&
                  r.superior !== null ? (
                    <ReferenceLine
                      key={`banda-${r.fecha}`}
                      segment={[
                        {
                          x: r.fecha,
                          y: r.inferior,
                        },
                        {
                          x: r.fecha,
                          y: r.superior,
                        },
                      ]}
                      stroke="#b48f63"
                      strokeWidth={2}
                      strokeOpacity={0.8}
                    />
                  ) : null
                )}

              <ReferenceLine
                x={test.fechaCorte}
                stroke="#7e837d"
                strokeDasharray="4 5"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div className="blind-legend">
          <span>
            <i style={{ background: '#78a6dc' }} />
            Histórico
          </span>
          <span>
            <i style={{ background: '#d5a66e' }} />
            Predicción de redes
          </span>
          <span>
            <i style={{ background: '#5bc3a6' }} />
            Observado / referencia
          </span>
          <span>
            <i style={{ background: '#b48f63' }} />
            Intervalo 80 %
          </span>
        </div>

        <div className="blind-grid">
          {rows.map(r => {
            const observed = getObserved(r.fecha);
            const error = observed
              ? Math.abs(r.precio - observed.precio)
              : null;

            const errorPct =
              observed && observed.precio !== 0
                ? (error! / Math.abs(observed.precio)) * 100
                : null;

            const dentro =
              observed &&
              r.inferior !== null &&
              r.superior !== null
                ? observed.precio >= r.inferior &&
                  observed.precio <= r.superior
                : null;

            return (
              <article
                key={r.horizonte}
                className="blind-stat"
              >
                <small>
                  HORIZONTE {r.horizonte} DÍA
                  {r.horizonte === 1 ? '' : 'S'}
                  {' · '}
                  {dateLabel(r.fecha)}
                </small>

                <strong>
                  {number(r.precio)}
                  <span> USD</span>
                </strong>

                <p>
                  Referencia observada:{' '}
                  {observed
                    ? `${number(observed.precio)} USD`
                    : 'Pendiente'}
                </p>

                {observed && (
                  <>
                    <p className="gold-text">
                      Diferencia absoluta:{' '}
                      {number(error!)} USD
                    </p>

                    <p className="gold-text">
                      Diferencia porcentual:{' '}
                      {number(errorPct!)} %
                    </p>

                    <p>
                      Dentro del intervalo 80 %:{' '}
                      {dentro === null
                        ? 'No disponible'
                        : dentro
                          ? 'Sí'
                          : 'No'}
                    </p>

                    <p className="source-note">
                      {observed.tipo} · {observed.fuente}
                    </p>
                  </>
                )}

                <p>
                  Intervalo nominal 80 %:{' '}
                  {r.inferior === null
                    ? 'No disponible'
                    : `${number(r.inferior)}–${number(
                        r.superior!
                      )} USD`}
                </p>
              </article>
            );
          })}
        </div>

        <p className="source-note">
          Predicciones congeladas al 18/11/2026.
          Los precios de futuros se muestran como
          referencia descriptiva, no como una
          evaluación definitiva contra el Brent spot.
          No se modifican las predicciones originales.
        </p>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <h2>Comparación con los valores observados</h2>
          <ShieldCheck size={19} />
        </div>

        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Fecha</th>
                <th>Predicción</th>
                <th>Referencia</th>
                <th>Diferencia absoluta</th>
                <th>Diferencia %</th>
                <th>Fuente / estado</th>
              </tr>
            </thead>

            <tbody>
              {rows.map(r => {
                const o = getObserved(r.fecha);
                const err = o
                  ? Math.abs(r.precio - o.precio)
                  : null;
                const pct =
                  o && o.precio !== 0
                    ? (err! / Math.abs(o.precio)) * 100
                    : null;

                return (
                  <tr key={r.horizonte}>
                    <td>{dateLabel(r.fecha, true)}</td>
                    <td>{number(r.precio)} USD</td>
                    <td>
                      {o
                        ? `${number(o.precio)} USD`
                        : 'Pendiente'}
                    </td>
                    <td>
                      {err !== null
                        ? `${number(err)} USD`
                        : '—'}
                    </td>
                    <td>
                      {pct !== null
                        ? `${number(pct)} %`
                        : '—'}
                    </td>
                    <td>
                      {o
                        ? `${o.tipo} · ${o.fuente}`
                        : 'Pendiente de precio comparable'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
