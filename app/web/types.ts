export type Row = Record<string, string | number | null>;
export type Price = { fecha: string; precio: number };
export type Scenario = { fecha: string; central: number; banda80: [number, number]; banda50: [number, number]; sim1: number; sim2: number; sim3: number };
export type Data = {
  historico: Price[];
  pruebaCiega?: {
    fechaCorte: string;
    predicciones: { fecha: string; horizonte: number; modelo: string; precio: number; inferior: number | null; superior: number | null }[];
    observaciones: { fecha: string; precio: number; tipo: string; fuente: string }[];
    disponible: boolean;
  };
  volatilidad: { anual: number; percentil: number; nivel: string };
  escenarios: Record<string, Scenario[]>;
  metricas: Row[];
  validacion: Row[];
  calibracion: Row[];
  curvas: Record<string, Record<string, Record<string, { train: number[]; val: number[] }> & { ingenuo_val: number }>>;
  predicciones: Row[];
  reentrenamiento: string | null;
  fuente: string;
  fuenteMetricas: string;
  horaActualizacion: string | null;
};
