import { useEffect, useMemo, useState } from "react";
import "./RecentBrentComparison.css";

type Modelo = "RNN" | "LSTM" | "GRU" | "Combinación";

type Fila = {
  fecha: string;
  real: number;
  predicciones: Record<Modelo, number>;
};

type Datos = {
  titulo: string;
  horizonte: number;
  mejorModelo: Modelo;
  ranking: { modelo: Modelo; mae: number; rmse: number }[];
  historico: { fecha: string; precio: number }[];
  filas: Fila[];
  nota: string;
};

const modelos: Modelo[] = ["RNN", "LSTM", "GRU", "Combinación"];

const formato = (n: number) =>
  n.toLocaleString("es-ES", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });

const fechaCorta = (s: string) =>
  s.slice(8, 10) + "/" + s.slice(5, 7);

export default function RecentBrentComparison() {
  const [datos, setDatos] = useState<Datos | null>(null);
  const [modelo, setModelo] = useState<Modelo>("GRU");
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/datos/comparacion_reciente.json")
      .then((r) => {
        if (!r.ok) throw new Error("No se pudo cargar el JSON");
        return r.json();
      })
      .then((d: Datos) => {
        setDatos(d);
        setModelo(d.mejorModelo);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const filas = useMemo(() => {
    if (!datos) return [];
    return datos.filas.map((f) => {
      const pred = f.predicciones[modelo];
      const diferencia = Math.abs(pred - f.real);
      return {
        ...f,
        pred,
        diferencia,
        porcentaje: (diferencia / Math.abs(f.real)) * 100,
      };
    });
  }, [datos, modelo]);

  if (error) {
    return <p>No se pudo cargar la comparación: {error}</p>;
  }
  if (!datos) return <p>Cargando comparación reciente...</p>;

  const mae = filas.reduce((s, f) => s + f.diferencia, 0) / filas.length;
  const rmse = Math.sqrt(
    filas.reduce((s, f) => s + (f.pred - f.real) ** 2, 0) /
      filas.length
  );

  // Gráfico SVG adaptable al ancho disponible.
  const ancho = 960;
  const alto = 340;
  const izq = 62;
  const der = 28;
  const arriba = 20;
  const abajo = 49;

  const puntosHistoricos = datos.historico.map((h) => ({
    fecha: h.fecha,
    valor: h.precio,
  }));

  const n = puntosHistoricos.length + filas.length;

  const valores = [
    ...puntosHistoricos.map((p) => p.valor),
    ...filas.map((f) => f.real),
    ...filas.map((f) => f.pred),
  ];

  const minimo = Math.min(...valores);
  const maximo = Math.max(...valores);
  const margen = Math.max((maximo - minimo) * 0.13, 0.8);
  const yMin = minimo - margen;
  const yMax = maximo + margen;

  const x = (i: number) =>
    izq + (i * (ancho - izq - der)) / Math.max(n - 1, 1);

  const y = (valor: number) =>
    arriba +
    ((yMax - valor) / (yMax - yMin)) *
      (alto - arriba - abajo);

  const path = (puntos: { x: number; y: number }[]) =>
    puntos
      .map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`)
      .join(" ");

  const historicoXY = puntosHistoricos.map((p, i) => ({
    x: x(i),
    y: y(p.valor),
  }));

  const realesXY = filas.map((f, i) => ({
    x: x(i + puntosHistoricos.length),
    y: y(f.real),
  }));

  const predXY = filas.map((f, i) => ({
    x: x(i + puntosHistoricos.length),
    y: y(f.pred),
  }));

  const ticks = Array.from(
    { length: 5 },
    (_, i) => yMin + ((yMax - yMin) * i) / 4
  );

  const fechas = [
    ...datos.historico.map((h) => h.fecha),
    ...filas.map((f) => f.fecha),
  ];

  const corteX =
    (x(puntosHistoricos.length - 1) +
      x(puntosHistoricos.length)) /
    2;

  return (
    <section className="recent-brent">
      <div className="recent-brent-header">
        <div>
          <p className="recent-brent-eyebrow">
            EVALUACIÓN HISTÓRICA · HORIZONTE {datos.horizonte} DÍAS
          </p>
          <h2>Comparación reciente con el Brent real</h2>
          <p className="recent-brent-description">
            Predicciones frente a los precios del 11, 12, 13 y 16 de noviembre de 2026.
          </p>
        </div>

        <div className="recent-brent-selector">
          <label htmlFor="recent-model">Modelo neuronal</label>
          <select
            id="recent-model"
            value={modelo}
            onChange={(e) => setModelo(e.target.value as Modelo)}
          >
            {modelos.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="recent-brent-metrics">
        <div>
          <span>Modelo seleccionado</span>
          <strong>{modelo}</strong>
        </div>
        <div>
          <span>MAE · 4 fechas</span>
          <strong>{formato(mae)} USD</strong>
        </div>
        <div>
          <span>RMSE · 4 fechas</span>
          <strong>{formato(rmse)} USD</strong>
        </div>
        <div>
          <span>Menor MAE retrospectivo</span>
          <strong>{datos.mejorModelo}</strong>
        </div>
      </div>

      <div className="recent-brent-legend">
        <span><i className="recent-dot blue" /> Brent histórico</span>
        <span><i className="recent-dot green" /> Brent real</span>
        <span><i className="recent-dot orange" /> Predicción {modelo}</span>
      </div>

      <div className="recent-brent-chart">
        <svg
          viewBox={`0 0 ${ancho} ${alto}`}
          role="img"
          aria-label={`Comparación del Brent real frente a ${modelo}`}
        >
          {ticks.map((t, i) => (
            <g key={i}>
              <line
                x1={izq}
                x2={ancho - der}
                y1={y(t)}
                y2={y(t)}
                stroke="#30383e"
                strokeWidth="1"
              />
              <text
                x={izq - 12}
                y={y(t) + 5}
                textAnchor="end"
                fill="#a9b1b8"
                fontSize="13"
              >
                {formato(t)}
              </text>
            </g>
          ))}

          <line
            x1={corteX}
            x2={corteX}
            y1={arriba}
            y2={alto - abajo}
            stroke="#7c858c"
            strokeDasharray="4 5"
          />

          <path
            d={path(historicoXY)}
            fill="none"
            stroke="#428de2"
            strokeWidth="3"
            strokeLinejoin="round"
          />

          <path
            d={path(realesXY)}
            fill="none"
            stroke="#20a579"
            strokeWidth="3"
            strokeLinejoin="round"
          />

          <path
            d={path(predXY)}
            fill="none"
            stroke="#f47b43"
            strokeWidth="3"
            strokeDasharray="8 6"
            strokeLinejoin="round"
          />

          {historicoXY.map((p, i) => (
            <circle
              key={`h${i}`}
              cx={p.x}
              cy={p.y}
              r="4.5"
              fill="#428de2"
            />
          ))}

          {realesXY.map((p, i) => (
            <g key={`r${i}`}>
              <circle cx={p.x} cy={p.y} r="6" fill="#20a579" />
              <title>
                {`${fechaCorta(filas[i].fecha)}: ${formato(filas[i].real)} USD`}
              </title>
            </g>
          ))}

          {predXY.map((p, i) => (
            <g key={`p${i}`}>
              <circle cx={p.x} cy={p.y} r="6" fill="#f47b43" />
              <title>
                {`${fechaCorta(filas[i].fecha)}: ${formato(filas[i].pred)} USD`}
              </title>
            </g>
          ))}

          {fechas.map((f, i) => (
            <text
              key={f}
              x={x(i)}
              y={alto - 17}
              fontSize="12"
              fill="#a9b1b8"
              textAnchor="middle"
            >
              {fechaCorta(f)}
            </text>
          ))}
        </svg>
      </div>

      <div className="recent-brent-table-wrap">
        <table className="recent-brent-table">
          <thead>
            <tr>
              <th>Fecha</th>
              <th>Predicción {modelo}</th>
              <th>Brent real</th>
              <th>Error absoluto</th>
              <th>Error %</th>
            </tr>
          </thead>
          <tbody>
            {filas.map((f) => (
              <tr key={f.fecha}>
                <td>{fechaCorta(f.fecha)}/2026</td>
                <td>{formato(f.pred)} USD</td>
                <td>{formato(f.real)} USD</td>
                <td>{formato(f.diferencia)} USD</td>
                <td>{formato(f.porcentaje)} %</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="recent-brent-note">{datos.nota}</p>
    </section>
  );
}
