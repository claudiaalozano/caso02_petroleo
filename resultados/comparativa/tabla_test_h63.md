### Resultados del test — 63 días

| Objetivo | Modelo | MAE (USD/barril) | Skill | Sesgo (USD) | Acierto dir. | DM p-valor |
| --- | --- | --- | --- | --- | --- | --- |
| — | Ingenuo | 11,288 | — | -1,31 | — | — |
| nivel | RNN | 14,978 | -32,7 % | +10,18 | 56,6 % | 0,411 |
| retorno | RNN | 12,735 | -12,8 % | +1,74 | 55,9 % | 0,086 |
| nivel | LSTM | 18,342 | -62,5 % | +14,92 | 50,8 % | 0,131 |
| retorno | LSTM | 13,336 | -18,1 % | +3,18 | 52,8 % | 0,072 |
| nivel | GRU | 17,059 | -51,1 % | +14,31 | 49,0 % | 0,183 |
| retorno | GRU | 12,957 | -14,8 % | +3,03 | 58,2 % | 0,220 |