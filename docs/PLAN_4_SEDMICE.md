# Plan rada — FinGraph AI

## Opseg

- **Obavezno:** audit Elliptic++, XGBoost, GraphSAGE, kronološki test,
  PR-AUC/F1/Recall, pregled nekoliko visokorizičnih susjedstava, README.
- **Ako ostane vremena:** GraphSAGE + dodatne grafovske karakteristike,
  ablacijska studija, simulirane tipologije iz BiH.
- **Ne raditi u prvoj verziji:** Elliptic2 sa 49M čvorova, web aplikacija,
  drugi puni GNN trening na IBM-ovom velikom skupu, kompleksni deployment.

## Istraživačka pitanja

1. Da li GraphSAGE postiže veći PR-AUC od XGBoost modela na istom vremenskom
   testnom skupu?
2. Koji je kompromis između recall i broja lažnih upozorenja?
3. Kakve povezane strukture se pojavljuju oko transakcija s najvećim skorom?

## Sedmica 1 — podaci i baseline (ključni prolaz)

- Preuzeti `txs_features.csv`, `txs_classes.csv`, `txs_edgelist.csv`.
- Pokrenuti `python -m fingraph.audit`.
- Potvrditi broj čvorova, veza, raspored klasa i pokrivenost po koracima.
- Provjeriti numeričke kolone, NaN i ID poravnanje.
- Zaključati train/val/test protokol prije evaluacije.
- Pokrenuti XGBoost i sačuvati prvi `metrics.json`.

**Izlaz:** dokaz da dataset radi i reproduktibilan baseline.

## Sedmica 2 — duboko učenje

- Pokrenuti GraphSAGE na Colab GPU-u.
- Provjeriti dimenzije, memoriju i vrijeme izvođenja.
- Podesiti samo po validacijskom PR-AUC (npr. hidden dim, lr, dropout).
- Ponoviti za 3 različita seeda ako resursi dozvole.
- Uporediti metrike i zabilježiti ograničenja.

**Izlaz:** završena glavna naučna usporedba.

## Sedmica 3 — mreže i originalnost

- Prikazati 2–3 povezana visokorizična podgrafa s objašnjenjem.
- Izračunati jednostavne mrežne karakteristike i napraviti ablation.
- Iz SIPA tipologija odabrati 1–2 obrasca pogodna za sintetički prikaz.
- Ne prikazivati stvarne osobe kao počinioce; ne miješati sim i real podatke.
- Pripremiti PR krivulje, matrice zabune i tabelu poređenja.

**Izlaz:** interpretabilni nalazi i uvjerljiv dio originalnosti.

## Sedmica 4 — konačna predaja

- Očistiti kod i ponoviti eksperimente.
- Pripremiti poster, zaključke i sekciju ograničenja.
- U README dokumentovati setup, arhitekturu, rezultate i seedove.
- Uvježbati odbranu: zašto graf, zašto PR-AUC, kako nema leakagea,
  šta unknown znači i šta model ne može tvrditi.

**Izlaz:** repo, rezultati, poster i izlaganje.

## Bodovanje — 60 bodova

| Element | Max | Prioritet |
|---|---:|---|
| Problem i literatura (10%) | 6 | Na početku i pri zaključku |
| Kreativnost i originalnost (20%) | 12 | Mrežne karakteristike i BiH scenariji |
| Arhitektura / optimizacija / metrike (20%) | 12 | Glavni eksperimenti |
| Podaci (10%) | 6 | Integritet i hronološki protokol |
| Kod / dokumentacija (15%) | 9 | Reproducibilnost i moduli |
| Vizualizacije (10%) | 6 | PR, F1, grafovi |
| Poster (5%) | 3 | Jasnoća |
| Zaključci (10%) | 6 | Mjerljivi, pošteno interpretirani |

Ako nešto kasni, **prvo završiti glavnu usporedbu i evaluaciju**, a tek
onda sintetičke BiH scenarije i dodatne modele.
