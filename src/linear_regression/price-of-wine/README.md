# Bordeaux Wine Price Prediction

Vorhersage der Qualität von Bordeaux-Weinjahrgängen anhand von Wetterdaten, basierend auf dem Modell von Orley Ashenfelter (Princeton, 1995).

## Hintergrund

In den 1990ern zeigte der Ökonom Orley Ashenfelter, dass sich die Qualität eines Bordeaux-Jahrgangs — gemessen am Auktionspreis — zu ca. 80% allein durch Wetterdaten vorhersagen lässt. Das Modell wurde bekannt als die "Bordeaux-Gleichung".

Die Kernaussage: **Heißer Sommer + trockene Ernte + nasser Winter = guter Jahrgang.**

Quelle: Ashenfelter, O., Ashmore, D. & Lalonde, R. (1995). *Bordeaux Wine Vintage Quality and the Weather.* Chance, 8(4), 7-14.

## Features

| Feature | Beschreibung | Einheit | Einfluss |
|---|---|---|---|
| AGST | Durchschnittstemperatur April–September | °C | Stark positiv |
| HarvestRain | Niederschlag August–September | ml | Negativ |
| WinterRain | Niederschlag Oktober–März | ml | Leicht positiv |
| Age | Alter des Weins (Lagerzeit) | Jahre | Positiv (Knappheit) |

**Zielvariable:** Logarithmus des relativen Marktpreises (Auktionspreise, relativ zum 1961er Jahrgang).

## Datensatz

27 Bordeaux-Jahrgänge (1952–1980) aus Ashenfelters Originaldaten. Wetterdaten stammen von der Wetterstation Bordeaux-Mérignac, Preise aus Londoner Weinauktionen (1990–1991).

- Training: 22 Jahrgänge (1952–1975)
- Test: 5 Jahrgänge (1976–1980)

Quelle: https://raw.githubusercontent.com/egarpor/handy/master/datasets/wine.csv

## Modell

```
Input [4] → Dense(16, ReLU) → Dense(1, linear)
```

- Optimizer: Adam
- Loss: Mean Squared Error
- Normalisierung: Min-Max-Scaling auf [0, 1]
- Epochs: 200, Batch Size: 8

## Evaluation

- MSE / RMSE auf dem Test-Set
- R² (Bestimmtheitsmaß)
- Demo-Prediction für den 1976er Jahrgang

## Bekannte Einschränkungen

- **Nur 27 Datenpunkte** — zu wenig für ein neuronales Netz, anfällig für Overfitting. Eine reine lineare Regression (ohne Hidden Layer) wäre für diese Datenmenge angemessener.
- **Keine erweiterten Daten verfügbar** — Wetterdaten nach 1980 existieren (ERA5), aber die zugehörigen Auktionspreise im selben Format sind nicht frei zugänglich.
- **Age ist kein Qualitätsindikator** — das Feature bildet Knappheit ab, nicht Geschmack. Für eine reine Qualitätsvorhersage direkt nach der Ernte reichen die drei Wetter-Features.

## Dateien

| Datei | Inhalt |
|---|---|
| `data.ts` | Datensatz, Normalisierung, Train/Test-Split |
| `model.ts` | Modellarchitektur und Hyperparameter |
| `index.ts` | Orchestrierung: Training, Evaluation, Prediction |
