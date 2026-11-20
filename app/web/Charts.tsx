import { useEffect, useState } from 'react';
import { Area, AreaChart, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Maximize2, Minimize2 } from 'lucide-react';
import type { Price, Scenario } from './types';

export const number = (v: number, digits = 2) => new Intl.NumberFormat('es-ES', { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(v);
export const dateLabel = (s: string, full = false) => new Date(s + 'T12:00:00').toLocaleDateString('es-ES', full ? { day: 'numeric', month: 'long', year: 'numeric' } : { day: 'numeric', month: 'short' });
const axis = { fill: '#7f8789', fontSize: 10, fontFamily: 'DM Sans' };
const names: Record<string, string> = { precio: 'Brent', central: 'Modelo ingenuo', banda80: 'Intervalo nominal del 80 %', banda50: 'Banda central del 50 %', sim1: 'Escenario simulado 1', sim2: 'Escenario simulado 2', sim3: 'Escenario simulado 3', train: 'Entrenamiento', val: 'Validación', ingenuo: 'Modelo ingenuo' };

function ChartTooltip({ active, payload, label }: { active?: boolean; payload?: readonly { dataKey?: string | number; value?: number | number[]; color?: string }[]; label?: string | number }) {
  if (!active || !payload?.length) return null;
  return <div className="chart-tooltip"><span>{typeof label === 'string' && /^\d{4}-/.test(label) ? dateLabel(label, true) : `Época ${label}`}</span>{payload.map((p, i) => <div key={i}><i style={{ background: p.color }} />{names[String(p.dataKey)] || p.dataKey}<strong>{Array.isArray(p.value) ? p.value.map(v => number(v)).join(' – ') : number(Number(p.value), String(p.dataKey) === 'train' || p.dataKey === 'val' ? 4 : 2)}</strong></div>)}</div>;
}

export function HistoryChart({ history, title = 'Evolución del Brent', eventDate, eventTitle, initialPeriod = '1A' }: { history: Price[]; title?: string; eventDate?: string; eventTitle?: string; initialPeriod?: string }) {
  const [period, setPeriod] = useState(initialPeriod);
  const [expanded, setExpanded] = useState(false);
  const [showAverage, setShowAverage] = useState(false);
  useEffect(() => {
    if (!expanded) return;
    const close = (e: KeyboardEvent) => { if (e.key === 'Escape') setExpanded(false); };
    window.addEventListener('keydown', close);
    return () => window.removeEventListener('keydown', close);
  }, [expanded]);
  const cutoff = new Date(history.at(-1)!.fecha);
  cutoff.setUTCMonth(cutoff.getUTCMonth() - ({ '1M': 1, '6M': 6, '1A': 12, '5A': 60, 'Todo': 0 } as Record<string, number>)[period]);
  const visible = eventDate ? history.filter(p => Math.abs(new Date(p.fecha).getTime() - new Date(eventDate).getTime()) < 370 * 86400000) : period === 'Todo' ? history : history.filter(p => p.fecha >= cutoff.toISOString().slice(0, 10));
  const sampled = visible.filter((_, i) => i % Math.max(1, Math.floor(visible.length / 600)) === 0 || i === visible.length - 1 || visible[i].fecha === eventDate);
  const avg = visible.reduce((a, b) => a + b.precio, 0) / visible.length;
  const last = visible.at(-1)?.precio;
  return <section className={`panel history-panel ${expanded ? 'expanded-chart' : ''}`}>
    <div className="panel-heading"><div><h2>{title} <span className="small-tag">BRENT</span></h2><p>Precio al contado · USD por barril</p></div><button className="icon-button" aria-label={expanded ? 'Reducir gráfico' : 'Ampliar gráfico'} onClick={() => setExpanded(!expanded)}>{expanded ? <Minimize2 size={15} /> : <Maximize2 size={15} />}</button></div>
    <div className="chart-toolbar"><div className="legend"><span className="legend-line" />Brent <span className="source-tag">FRED / EIA</span></div>{!eventDate && <div className="periods" aria-label="Periodo del gráfico">{['1M', '6M', '1A', '5A', 'Todo'].map(p => <button key={p} title={({ '1M': '1 mes', '6M': '6 meses', '1A': '1 año', '5A': '5 años', 'Todo': 'Histórico completo' })[p]} aria-pressed={period === p} className={period === p ? 'selected' : ''} onClick={() => setPeriod(p)}>{p}</button>)}</div>}</div>
    <div className="history-chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={sampled} margin={{ top: 15, right: 6, left: -20, bottom: 2 }}>
      <defs><linearGradient id="gold-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#c3a777" stopOpacity={.20} /><stop offset="100%" stopColor="#c3a777" stopOpacity={0} /></linearGradient></defs>
      <CartesianGrid vertical={false} stroke="#252a2c" strokeDasharray="3 5" />
      <XAxis dataKey="fecha" tickFormatter={s => new Date(s + 'T12:00:00').toLocaleDateString('es-ES', period === 'Todo' || period === '5A' ? { year: 'numeric' } : { month: 'short', ...(period === '1M' ? { day: 'numeric' as const } : {}) })} minTickGap={65} tick={axis} axisLine={false} tickLine={false} dy={10} />
      <YAxis domain={['auto', 'auto']} tick={axis} axisLine={false} tickLine={false} tickFormatter={v => `$${number(v, 0)}`} tickCount={5} />
      <Tooltip content={<ChartTooltip />} cursor={{ stroke: '#79715f', strokeDasharray: '3 4' }} />
      <Area dataKey="precio" stroke="#c3a777" strokeWidth={1.8} fill="url(#gold-fill)" isAnimationActive={false} activeDot={{ r: 4, stroke: '#090b0d', strokeWidth: 2 }} />
      {last && !eventDate && <ReferenceLine y={last} stroke="#c3a777" strokeOpacity={.4} strokeDasharray="4 5" label={{ value: number(last), position: 'insideRight', fill: '#c3a777', fontSize: 10 }} />}
      {showAverage && <ReferenceLine y={avg} stroke="#7b9f9b" strokeDasharray="5 5" label={{ value: `Media ${number(avg)}`, fill: '#7b9f9b', fontSize: 10 }} />}
      {eventDate && <ReferenceLine x={eventDate} stroke="#c3a777" label={{ value: eventTitle, fill: '#c3a777', fontSize: 11, position: 'insideTopRight' }} />}
    </AreaChart></ResponsiveContainer></div>
    <div className="chart-footer"><span><i className="status-dot muted-dot" />{dateLabel(visible[0]?.fecha || history[0].fecha)} — {dateLabel(visible.at(-1)?.fecha || history.at(-1)!.fecha, true)}</span><button className={showAverage ? 'gold-text' : ''} onClick={() => setShowAverage(!showAverage)} aria-pressed={showAverage}>{showAverage ? 'Ocultar media' : 'Mostrar media del periodo'}</button></div>
  </section>;
}

export function ScenarioChart({ history, scenarios, compact = false, simulations = false }: { history: Price[]; scenarios: Scenario[]; compact?: boolean; simulations?: boolean }) {
  const past = history.slice(-Math.max(22, Math.min(66, scenarios.length))).map(p => ({ fecha: p.fecha, precio: p.precio }));
  const data = [...past.slice(0, -1), { ...past.at(-1), ...scenarios[0] }, ...scenarios.slice(1)];
  return <div style={{ height: compact ? 120 : 350 }}><ResponsiveContainer width="100%" height="100%"><ComposedChart data={data} margin={{ top: 10, left: compact ? 0 : -15, right: 5, bottom: 0 }}>
    {!compact && <CartesianGrid vertical={false} stroke="#252a2c" strokeDasharray="3 5" />}
    <XAxis dataKey="fecha" hide={compact} tickFormatter={s => dateLabel(s)} minTickGap={60} tick={axis} tickLine={false} axisLine={false} />
    <YAxis hide={compact} domain={['auto', 'auto']} tick={axis} tickFormatter={v => `$${number(v, 0)}`} axisLine={false} tickLine={false} />
    <Tooltip content={<ChartTooltip />} />
    <Area type="monotone" dataKey="banda80" fill="#c3a777" fillOpacity={.09} stroke="none" isAnimationActive={false} />
    <Area type="monotone" dataKey="banda50" fill="#c3a777" fillOpacity={.15} stroke="none" isAnimationActive={false} />
    <Line dataKey="precio" dot={false} stroke="#8b9496" strokeWidth={1.4} isAnimationActive={false} />
    <Line dataKey="central" dot={false} stroke="#c3a777" strokeDasharray="4 3" strokeWidth={1.6} isAnimationActive={false} />
    {simulations && ['sim1', 'sim2', 'sim3'].map((key, i) => <Line key={key} dataKey={key} dot={false} stroke={['#4bae91', '#7e939f', '#c47769'][i]} strokeOpacity={.55} strokeWidth={1} isAnimationActive={false} />)}
    {!compact && <ReferenceLine x={history.at(-1)!.fecha} stroke="#686c68" strokeDasharray="3 4" label={{ value: 'Último dato', fill: '#8e9698', position: 'insideTopLeft', fontSize: 10 }} />}
  </ComposedChart></ResponsiveContainer></div>;
}

export function LearningChart({ train, val, naive }: { train: number[]; val: number[]; naive: number }) {
  const data = train.map((v, i) => ({ epoca: i + 1, train: v, val: val[i], ingenuo: naive }));
  return <div style={{ height: 310 }}><ResponsiveContainer width="100%" height="100%"><ComposedChart data={data} margin={{ top: 20, left: -10, right: 10 }}>
    <CartesianGrid vertical={false} stroke="#252a2c" strokeDasharray="3 5" />
    <XAxis dataKey="epoca" tick={axis} axisLine={false} tickLine={false} /><YAxis tick={axis} axisLine={false} tickLine={false} tickFormatter={v => number(v, 2)} />
    <Tooltip content={<ChartTooltip />} />
    <Line dataKey="train" stroke="#7d9b9a" dot={false} strokeWidth={1.8} />
    <Line dataKey="val" stroke="#c3a777" dot={false} strokeWidth={1.8} />
    <Line dataKey="ingenuo" stroke="#7f8582" dot={false} strokeDasharray="5 5" />
  </ComposedChart></ResponsiveContainer></div>;
}
