import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import App from './App';
import fixture from '../public/datos/brent.json';

vi.mock('./Charts', async () => {
  const actual = await vi.importActual<typeof import('./Charts')>('./Charts');
  return {
    ...actual,
    ScenarioChart: () => <div>Gráfico de escenarios</div>,
    LearningChart: () => <div>Gráfico de aprendizaje</div>,
  };
});

beforeEach(() => {
  window.location.hash = '';
  window.scrollTo = vi.fn();
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} });
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({ width: 600, height: 280, top: 0, left: 0, bottom: 280, right: 600, x: 0, y: 0, toJSON: () => ({}) });
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => fixture }));
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

async function navigate(page: string) {
  window.location.hash = page;
  fireEvent(window, new HashChangeEvent('hashchange'));
  await waitFor(() => expect(document.querySelector(`nav a[href="#${page}"]`)?.getAttribute('aria-current')).toBe('page'));
}

describe('BRENT. — flujos de análisis', () => {
  it('carga el precio real y permite cambiar el periodo', async () => {
    render(<App />);
    await screen.findByText('Petróleo Brent');
    expect(screen.getAllByText('92,94').length).toBeGreaterThan(0);
    const period = screen.getByRole('button', { name: '5A' });
    fireEvent.click(period);
    expect(period.getAttribute('aria-pressed')).toBe('true');
    fireEvent.click(screen.getByRole('button', { name: 'Mostrar media del periodo' }));
    expect(screen.getByRole('button', { name: 'Ocultar media' })).toBeTruthy();
  });

  it('no inventa una predicción neuronal al cambiar de modelo', async () => {
    render(<App />);
    await screen.findByText('Petróleo Brent');
    await navigate('predicciones');
    const select = screen.getByLabelText('Modelo predictivo');
    fireEvent.change(select, { target: { value: 'LSTM' } });
    expect(await screen.findByRole('heading', { name: 'Pendiente de actualización' })).toBeTruthy();
    fireEvent.change(select, { target: { value: 'Ingenuo' } });
    expect(screen.queryByText('Pendiente de actualización')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: /252 días/ }));
    expect(screen.getByRole('button', { name: /252 días/ }).getAttribute('aria-pressed')).toBe('true');
  });

  it('filtra las métricas reales por horizonte y objetivo', async () => {
    render(<App />);
    await screen.findByText('Petróleo Brent');
    await navigate('modelos');
    fireEvent.change(screen.getByLabelText('Horizonte'), { target: { value: '63' } });
    expect(screen.getAllByText('11,29').length).toBeGreaterThan(0);
    fireEvent.change(screen.getByLabelText('Objetivo'), { target: { value: 'nivel' } });
    expect(screen.getAllByText('14,98').length).toBeGreaterThan(0);
    fireEvent.change(screen.getByLabelText('Arquitectura'), { target: { value: 'LSTM' } });
    expect(screen.getAllByRole('row').length).toBe(2);
  });

  it('abre las fuentes y permite cerrar con Escape', async () => {
    render(<App />);
    await screen.findByText('Petróleo Brent');
    fireEvent.click(screen.getByRole('button', { name: /Fuentes y datos/ }));
    expect(screen.getByRole('dialog')).toBeTruthy();
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('muestra error de carga y ofrece reintentar', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('sin conexión')));
    render(<App />);
    expect(await screen.findByRole('button', { name: /Volver a intentar/ })).toBeTruthy();
  });
});
