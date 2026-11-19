### Resultados del test — 5 días

| Objetivo | Modelo | MAE (USD/barril) | Skill | Sesgo (USD) | Acierto dir. | DM p-valor |
| --- | --- | --- | --- | --- | --- | --- |
| — | Ingenuo | 3,646 | — | -0,07 | — | — |
| nivel | RNN | 5,087 | -39,5 % | +2,62 | 47,5 % | < 0,001 |
| retorno | RNN | 3,669 | -0,6 % | +0,23 | 49,5 % | 0,275 |
| nivel | LSTM | 5,404 | -48,2 % | -3,47 | 48,3 % | < 0,001 |
| retorno | LSTM | 3,682 | -1,0 % | +0,26 | 49,4 % | 0,146 |
| nivel | GRU | 4,450 | -22,0 % | +0,18 | 49,4 % | < 0,001 |
| retorno | GRU | 3,704 | -1,6 % | +0,13 | 48,0 % | 0,103 |