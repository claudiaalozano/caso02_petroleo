import { useState } from 'react';
import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from 'recharts';
import { CalendarDays, ShieldCheck } from 'lucide-react';
import type { Data } from './types';
import { dateLabel, number } from './Charts';

type Point = { fecha: string; historico?: number; prediccion?: number; observado?: number; banda?: [number, number] };

export function BlindTest({ data }: { data: Data }) {
  const [horizon, setHorizon] = useState<'todos'|'1'|'2'>('todos');
  const [showBand, setShowBand] = useState(true);
  const [showObserved, setShowObserved] = useState(true);
  const test = data.pruebaCiega;
  if (!test?.disponible || !test.predicciones.length) {
    return <section className="panel blind-panel"><div className="eyebrow">VALIDACIÓN PROSPECTIVA</div><h2>La prueba a ciegas.</h2><p className="subtle">Aún no se ha publicado <code>resultados/prueba_ciega_2026-11-19/redes_h1_h2/predicciones.csv</code>. Ejecuta el exportador Python para incorporar los resultados reales; no se generan cifras de demostración.</p></section>;
  }
  const rows = test.predicciones.filter(p => p.modelo === 'Comb. redes' && (horizon === 'todos' || String(p.horizonte) === horizon)).sort((a,b)=>a.horizonte-b.horizonte);
  const latest = data.historico.filter(p => p.fecha <= test.fechaCorte).slice(-10);
  const points = new Map<string, Point>();
  latest.forEach(p => points.set(p.fecha, { fecha: p.fecha, historico: p.precio }));
  rows.forEach(p => points.set(p.fecha, { ...points.get(p.fecha), fecha: p.fecha, prediccion: p.precio,
    ...((p.inferior !== null && p.superior !== null) ? { banda: [p.inferior, p.superior] as [number, number] } : {}) }));
  if (showObserved) test.observaciones.forEach(p => {
    if (rows.some(r => r.fecha === p.fecha)) points.set(p.fecha, { ...points.get(p.fecha), fecha: p.fecha, observado: p.precio });
  });
  const plot = [...points.values()].sort((a,b) => a.fecha.localeCompare(b.fecha));
  const origin = latest.at(-1);
  const availableObs = test.observaciones.filter(p => rows.some(r => r.fecha === p.fecha));
  return <div className="page-stack">
    <section className="panel blind-panel">
      <div className="panel-heading"><div><div className="eyebrow"><CalendarDays size={13}/>EXPERIMENTO FUERA DE MUESTRA</div><h2>Antes de saber qué ocurrirá.</h2><p>Histórico de 10 observaciones y predicción desde el 18 de noviembre de 2026</p></div><span className="mini-label">CORTE CONGELADO</span></div>
      <div className="blind-controls"><div className="periods">{(['todos','1','2'] as const).map(h => <button key={h} className={horizon===h?'selected':''} aria-pressed={horizon===h} onClick={()=>setHorizon(h)}>{h==='todos'?'Ambos días':`Día ${h}`}</button>)}</div><label><input type="checkbox" checked={showBand} onChange={e=>setShowBand(e.target.checked)}/> Intervalo 80 %</label><label><input type="checkbox" checked={showObserved} onChange={e=>setShowObserved(e.target.checked)}/> Observaciones</label></div>
      <div className="blind-chart"><ResponsiveContainer width="100%" height="100%"><ComposedChart data={plot} margin={{top:20,right:14,bottom:12,left:0}}>
        <CartesianGrid stroke="#27302e" strokeDasharray="3 5" vertical={false}/>
        <XAxis dataKey="fecha" tick={{fill:'#87928e',fontSize:11}} tickFormatter={v=>dateLabel(v)} minTickGap={26} tickLine={false} axisLine={false}/>
        <YAxis domain={['auto','auto']} tick={{fill:'#87928e',fontSize:11}} tickFormatter={v=>number(v,0)} tickLine={false} axisLine={false}/>
        <Tooltip contentStyle={{background:'#161d20',border:'1px solid #3b4844',borderRadius:8,color:'#e8e4da'}} labelFormatter={v=>dateLabel(String(v),true)} formatter={(value,name)=>[Array.isArray(value)?value.map(v=>number(Number(v))).join('–'):number(Number(value)), String(name)]}/>
        {showBand && <Area name="Intervalo 80 %" type="linear" dataKey="banda" stroke="none" fill="#d6a465" fillOpacity={0.18} connectNulls={false} isAnimationActive={false}/>}
        <Line name="Histórico del Brent" dataKey="historico" stroke="#78a6dc" strokeWidth={2.4} dot={{r:2}} isAnimationActive={false}/>
        <Line name="Predicción de redes" dataKey="prediccion" stroke="#d5a66e" strokeWidth={2} strokeDasharray="5 4" dot={{r:5,stroke:'#171b1d',strokeWidth:2}} isAnimationActive={false}/>
        {showObserved && <Scatter name="Observado" dataKey="observado" fill="#5bc3a6" isAnimationActive={false}/>}
        <ReferenceLine x={test.fechaCorte} stroke="#7e837d" strokeDasharray="4 5"/>
      </ComposedChart></ResponsiveContainer></div>
      <div className="blind-legend"><span><i style={{background:'#78a6dc'}}/> Histórico</span><span><i style={{background:'#d5a66e'}}/> Predicción de redes</span><span><i style={{background:'#5bc3a6'}}/> Observado</span><span><i style={{background:'#b48f63'}}/> Banda 80 %</span></div>
      <div className="blind-grid">{rows.map(r => {
        const observed = test.observaciones.find(o=>o.fecha===r.fecha);
        return <article key={r.horizonte} className="blind-stat"><small>HORIZONTE {r.horizonte} DÍA{r.horizonte===1?'':'S'} · {dateLabel(r.fecha)}</small><strong>{number(r.precio)} <span>USD</span></strong><p>Observado: {observed ? `${number(observed.precio)} USD (${observed.tipo})` : 'Pendiente'}</p><p>Intervalo nominal 80 %: {r.inferior === null ? 'No disponible' : `${number(r.inferior)}–${number(r.superior!)} USD`}</p>{observed && <p className="gold-text">Diferencia descriptiva: {number(Math.abs(r.precio-observed.precio))} USD</p>}</article>;
      })}</div>
      <p className="source-note">La prueba utiliza predicciones guardadas con fecha de origen 18/11/2026. {origin ? `Último valor de referencia: ${number(origin.precio)} USD.` : ''} Las observaciones intradía no son cierres comparables. El intervalo nominal no garantiza cobertura.</p>
    </section>
    <section className="panel"><div className="panel-heading"><h2>Transparencia de la comparación</h2><ShieldCheck size={19}/></div><p className="subtle">Las RNN, LSTM y GRU se reentrenaron para horizontes de 1 y 2 días; no son modelos a 5 días extrapolados. Se respeta el conjunto congelado al 18/11. Las observaciones incorporadas después son exclusivamente para evaluar o visualizar, nunca para modificar las predicciones guardadas.</p><div className="table-scroll"><table><thead><tr><th>Fecha objetivo</th><th>Predicción</th><th>Observado</th><th>Fuente / estado</th></tr></thead><tbody>{rows.map(r=>{const o=availableObs.find(v=>v.fecha===r.fecha);return <tr key={r.horizonte}><td>{dateLabel(r.fecha,true)}</td><td>{number(r.precio)} USD</td><td>{o?`${number(o.precio)} USD`:'Pendiente'}</td><td>{o?`${o.tipo} · ${o.fuente}`:'Sin cotización comparable registrada'}</td></tr>;})}</tbody></table></div></section>
  </div>;
}
