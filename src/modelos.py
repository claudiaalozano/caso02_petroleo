
import torch
import torch.nn as nn


class ModeloRecurrente(nn.Module):

    def __init__(
        self,
        tipo="RNN",
        input_size=11,
        hidden_size_1=64,
        hidden_size_2=32,
        dropout=0.2,
        output_size=1,
        residual=False
    ):
        super().__init__()

        tipo = tipo.upper()

        capas = {
            "RNN": nn.RNN,
            "LSTM": nn.LSTM,
            "GRU": nn.GRU
        }

        if tipo not in capas:
            raise ValueError("El tipo debe ser RNN, LSTM o GRU")

        self.tipo = tipo
        self.residual = residual

        capa_recurrente = capas[tipo]

        # Primera capa recurrente
        self.recurrente1 = capa_recurrente(
            input_size=input_size,
            hidden_size=hidden_size_1,
            batch_first=True
        )

        self.dropout1 = nn.Dropout(dropout)

        # Segunda capa recurrente
        self.recurrente2 = capa_recurrente(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            batch_first=True
        )

        self.dropout2 = nn.Dropout(dropout)

        # Capas densas
        self.fc1 = nn.Linear(hidden_size_2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

        # Inicialización de la salida residual:
        # la predicción empieza siendo el último Brent
        if self.residual:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # Primera capa recurrente
        out, _ = self.recurrente1(x)
        out = self.dropout1(out)

    # Segunda capa recurrente
        out, _ = self.recurrente2(out)

    # Último instante temporal
        out = out[:, -1, :]

    # Capas densas
        out = self.dropout2(out)
        out = self.relu(self.fc1(out))

    # Predicción directa del precio futuro
        out = self.fc2(out)

        return out


def contar_parametros(modelo):

    return sum(
        p.numel()
        for p in modelo.parameters()
        if p.requires_grad
    )


def crear_modelo(
    tipo,
    input_size=11,
    hidden_size_1=64,
    hidden_size_2=32,
    dropout=0.2,
    residual=False
):

    modelo = ModeloRecurrente(
        tipo=tipo,
        input_size=input_size,
        hidden_size_1=hidden_size_1,
        hidden_size_2=hidden_size_2,
        dropout=dropout,
        residual=residual
    )

    print("=" * 45)
    print(f"MODELO {tipo.upper()}")
    print("=" * 45)

    print("Capas recurrentes:", hidden_size_1, "→", hidden_size_2)
    print("Dropout:", dropout)
    print("Residual:", residual)
    print("Parámetros:", contar_parametros(modelo))

    return modelo
