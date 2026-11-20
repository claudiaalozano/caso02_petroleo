import { useEffect, useMemo, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

type Modelo = 'RNN' | 'LSTM' | 'GRU' | 'Ingenuo';

type Fila = {
  fecha: string;
  numero: number;
  RNN: number;
  LSTM: number;
  GRU: number;
  Ingenuo: number;
};

type Datos = {
  objetivo: string;
  horizonte: number;
  observaciones: number;
  fechaInicio: string;
  fechaFin: string;
  metricas: {
    modelo: Modelo;
    mae: number;
    total: number;
  }[];
  filas: Fila[];
};

const colores: Record<Modelo, string> = {
  RNN: '#a484cc',
  LSTM: '#4c9db2',
  GRU: '#c3a777',
  Ingenuo: '#8b9496',
};

const formato = (n: number) =>
  n.toLocaleString('es-ES', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });

const fecha = (s: string) =>
  new Date(s + 'T12:00:00').toLocaleDateString('es-ES', {
    month: 'short',
    year: '2-digit',
  });

export default function CumulativeErrorChart() {
  const [datos, setDatos] = useState<Datos | null>(null);
  const [error, setError] = useState('');
  const [modelos, setModelos] = useState<Modelo[]>([
    'RNN', 'LSTM', 'GRU', 'Ingenuo',
  ]);

  useEffect(() => {
    fetch('/datos/error_acumulado.json')
      .then(r => {
        if (!r.ok) throw new Error('Archivo de datos no disponible');
        return r.json();
      })
      .then((json: Datos) => setDatos(json))
      .catch(e => setError(String(e)));
  }, []);

  const grafica = useMemo(() => {
    if (!datos) return [];

    // Para visualizar menos puntos sin alterar los
    // errores, que se calcularon sobre todas las fechas.
    const paso = Math.max(
      1, Math.ceil(datos.filas.length / 350)
    );

    return datos.filas.filter(
      (_, i) => i % paso === 0 ||
        i === datos.filas.length - 1
    );
  }, [datos]);

  const alternar = (modelo: Modelo) => {
    setModelos(actual =>
      actual.includes(modelo)
        ? actual.filter(m => m !== modelo)
        : [...actual, modelo]
    );
  };

  return (
    <section className="panel" style={{ minWidth: 0 }}>
      <div className="panel-heading">
        <div>
          <h2>Error absoluto acumulado</h2>
          <p>
            Test final · NIVEL · Horizonte 5 días
          </p>
        </div>
      </div>

      {error && (
        <p className="source-note">
          No se pudieron cargar los resultados: {error}
        </p>
      )}

      {!datos && !error && (
        <p className="source-note">Cargando resultados...</p>
      )}

      {datos && (
        <>
          <div
            style={{
              display: 'flex',
              gap: 12,
              flexWrap: 'wrap',
              marginBottom: 18,
            }}
          >
            {(['RNN', 'LSTM', 'GRU', 'Ingenuo'] as Modelo[])
              .map(modelo => (
                <button
                  key={modelo}
                  type="button"
                  onClick={() => alternar(modelo)}
                  aria-pressed={modelos.includes(modelo)}
                  style={{
                    padding: '8px 12px',
                    borderRadius: 8,
                    cursor: 'pointer',
                    background: modelos.includes(modelo)
                      ? '#242b2d' : 'transparent',
                    border: '1px solid #3a4244',
                    color: modelos.includes(modelo)
                      ? '#f2f2ee' : '#899396',
                  }}
                >
                  <span
                    style={{
                      display: 'inline-block',
                      width: 9,
                      height: 9,
                      borderRadius: '50%',
                      background: colores[modelo],
                      marginRight: 8,
                    }}
                  />
                  {modelo}
                </button>
              ))}
          </div>

          <div style={{ width: '100%', height: 350 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={grafica}
                margin={{
                  top: 15,
                  right: 18,
                  bottom: 10,
                  left: 3,
                }}
              >
                <CartesianGrid
                  stroke="#303638"
                  strokeDasharray="3 5"
                  vertical={false}
                />
                <XAxis
                  dataKey="fecha"
                  tickFormatter={fecha}
                  tick={{ fill: '#879093', fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  minTickGap={48}
                />
                <YAxis
                  tick={{ fill: '#879093', fontSize: 11 }}
                  tickFormatter={(n: number) =>
                    n.toLocaleString('es-ES', {
                      maximumFractionDigits: 0,
                    })
                  }
                  tickLine={false}
                  axisLine={false}
                  width={58}
                />
                <Tooltip
                  labelFormatter={value =>
                    String(value).slice(0, 10)
                  }
                  formatter={value => [
                    `${formato(Number(value))} USD`,
                    'Error acumulado',
                  ]}
                  contentStyle={{
                    background: '#171c1e',
                    border: '1px solid #3a4244',
                    borderRadius: 8,
                    color: '#f4f4f0',
                  }}
                />

                {modelos.map(modelo => (
                  <Line
                    key={modelo}
                    name={modelo}
                    dataKey={modelo}
                    type="monotone"
                    stroke={colores[modelo]}
                    strokeWidth={modelo === 'GRU' ? 2.6 : 1.8}
                    dot={false}
                    activeDot={{ r: 5 }}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Modelo</th>
                  <th>MAE (USD)</th>
                  <th>Error total (USD)</th>
                </tr>
              </thead>
              <tbody>
                {datos.metricas.map((m, i) => (
                  <tr key={m.modelo}>
                    <td>
                      {m.modelo}
                      {i === 0 && (
                        <span className="best-tag">
                          Menor MAE
                        </span>
                      )}
                    </td>
                    <td>{formato(m.mae)}</td>
                    <td>{formato(m.total)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="source-note">
            {datos.observaciones} fechas evaluadas.
            Periodo: {datos.fechaInicio} a {datos.fechaFin}.
            Cada curva suma los errores absolutos de sus
            predicciones. Se utilizan precios reales del
            dataset congelado y resultados del test final.
            El menor error acumulado indica un mejor ajuste
            en este periodo; no garantiza mejores
            predicciones futuras.
          </p>
        </>
      )}
    </section>
  );
}
