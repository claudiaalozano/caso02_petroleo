### Resultados del test — 252 días

| Objetivo | Modelo | MAE (USD/barril) | Skill | Sesgo (USD) | Acierto dir. | DM p-valor |
| --- | --- | --- | --- | --- | --- | --- |
| — | Ingenuo | 16,455 | — | +1,96 | — | — |
| nivel | RNN | 16,967 | -3,1 % | +12,38 | 57,7 % | 0,909 |
| retorno | RNN | 22,857 | -38,9 % | +11,14 | 58,2 % | 0,221 |
| nivel | LSTM | 25,736 | -56,4 % | +21,15 | 48,4 % | 0,320 |
| retorno | LSTM | 19,908 | -21,0 % | +7,57 | 56,9 % | 0,522 |
| nivel | GRU | 26,937 | -63,7 % | +24,53 | 46,9 % | 0,222 |
| retorno | GRU | 21,617 | -31,4 % | +8,43 | 54,4 % | 0,337 |