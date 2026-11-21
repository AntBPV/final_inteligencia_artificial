import random
from dataclasses import dataclass

import torch
from torch import Tensor, nn


INTENTS = (
    "occupancy_now",
    "entries_today",
    "exits_today",
    "maximum_occupancy",
    "last_hour",
    "daily_summary",
)


INTENT_EXAMPLES = {
    "occupancy_now": (
        "cuantas personas hay ahora",
        "cuantas personas hay",
        "cual es la ocupacion actual",
        "dime la ocupacion",
        "personas presentes",
        "cuantos hay dentro",
        "hay gente ahora",
    ),
    "entries_today": (
        "cuantas entradas hubo hoy",
        "total de entradas",
        "cuantas personas entraron",
        "entradas del dia",
        "cuantas entradas se registraron",
    ),
    "exits_today": (
        "cuantas salidas hubo hoy",
        "total de salidas",
        "cuantas personas salieron",
        "salidas del dia",
        "cuantas salidas se registraron",
    ),
    "maximum_occupancy": (
        "cual fue la ocupacion maxima",
        "dime el pico de ocupacion",
        "ocupacion maxima historica",
        "cuando hubo mas personas",
        "maximo de personas",
    ),
    "last_hour": (
        "resumen de la ultima hora",
        "que paso en los ultimos 60 minutos",
        "estadisticas de la ultima hora",
        "entradas y salidas recientes",
        "movimiento de la ultima hora",
    ),
    "daily_summary": (
        "resumen de hoy",
        "dame un resumen",
        "estadisticas del dia",
        "como estuvo hoy",
        "totales de hoy",
    ),
}


@dataclass(frozen=True)
class IntentResult:
    intent: str
    confidence: float


class IntentTransformer:
    def __init__(self) -> None:
        random.seed(7)
        torch.manual_seed(7)

        corpus = "abcdefghijklmnopqrstuvwxyz0123456789áéíóúñ "
        self.characters = ["<pad>", "<unk>", *sorted(set(corpus))]
        self.character_to_id = {
            character: index
            for index, character in enumerate(self.characters)
        }
        self.pad_id = 0
        self.unknown_id = 1

        self.embedding = nn.Embedding(
            num_embeddings=len(self.characters),
            embedding_dim=32,
            padding_idx=self.pad_id,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            dropout=0.05,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
        )
        self.classifier = nn.Linear(32, len(INTENTS))
        self.intent_to_id = {
            intent: index
            for index, intent in enumerate(INTENTS)
        }
        self._train()

    def _encode(self, text: str, max_length: int = 48) -> Tensor:
        normalized = text.lower().strip()[:max_length]
        ids = [
            self.character_to_id.get(
                character,
                self.unknown_id,
            )
            for character in normalized
        ]
        if not ids:
            ids = [self.unknown_id]
        if len(ids) < max_length:
            ids += [self.pad_id] * (max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def _augment(self, text: str) -> str:
        variants = (
            text,
            f"por favor {text}",
            f"{text} por favor",
            f"dime {text}",
            f"quiero saber {text}",
        )
        return random.choice(variants)

    def _train(self) -> None:
        texts: list[str] = []
        labels: list[int] = []
        for intent in INTENTS:
            examples = INTENT_EXAMPLES[intent]
            for example in examples:
                for _ in range(3):
                    texts.append(self._augment(example))
                    labels.append(self.intent_to_id[intent])

        tensors = torch.stack([self._encode(text) for text in texts])
        targets = torch.tensor(labels, dtype=torch.long)

        parameters = list(self.embedding.parameters()) + list(
            self.encoder.parameters()
        ) + list(self.classifier.parameters())
        optimizer = torch.optim.Adam(parameters, lr=0.008)
        criterion = nn.CrossEntropyLoss()

        self.embedding.train()
        self.encoder.train()
        self.classifier.train()

        batch_size = 24
        for _ in range(35):
            permutation = torch.randperm(len(texts))
            for start in range(0, len(texts), batch_size):
                batch_ids = permutation[start:start + batch_size]
                batch_tokens = tensors[batch_ids]
                batch_targets = targets[batch_ids]
                padding_mask = batch_tokens.eq(self.pad_id)

                embedded = self.embedding(batch_tokens)
                encoded = self.encoder(
                    embedded,
                    src_key_padding_mask=padding_mask,
                )
                pooled = encoded.masked_fill(
                    padding_mask.unsqueeze(-1),
                    0.0,
                ).sum(dim=1)
                lengths = (
                    (~padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
                )
                logits = self.classifier(pooled / lengths)

                loss = criterion(logits, batch_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.embedding.eval()
        self.encoder.eval()
        self.classifier.eval()

    def predict(self, text: str) -> IntentResult:
        tokens = self._encode(text).unsqueeze(0)
        padding_mask = tokens.eq(self.pad_id)

        with torch.no_grad():
            embedded = self.embedding(tokens)
            encoded = self.encoder(
                embedded,
                src_key_padding_mask=padding_mask,
            )
            pooled = encoded.masked_fill(
                padding_mask.unsqueeze(-1),
                0.0,
            ).sum(dim=1)
            lengths = (~padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
            logits = self.classifier(pooled / lengths)
            probabilities = torch.softmax(logits, dim=1)[0]

        intent_id = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[intent_id].item())
        return IntentResult(
            intent=INTENTS[intent_id],
            confidence=confidence,
        )
