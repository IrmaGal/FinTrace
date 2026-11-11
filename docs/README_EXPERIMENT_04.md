
# FINTRACE — Četvrti eksperiment
## Dijagnostička analiza vremenske generalizacije

### Cilj eksperimenta

Četvrti eksperiment nastavlja analizu rezultata XGBoost, GraphSAGE i MLP modela na Elliptic++ Transactional datasetu.

Prethodni eksperimenti pokazali su da modeli najveći dio nezakonitih transakcija uspješno otkrivaju u vremenskom koraku 42, dok u koracima 43–49 ostvaruju veoma slab recall.

Cilj ovog eksperimenta bio je detaljnije istražiti moguće uzroke takvog ponašanja, uključujući neuravnoteženost klasa, promjene distribucija karakteristika, klasifikacijske pragove i ograničenja grafovske strukture.

### Implementacija

Eksperiment je implementiran u notebooku:

`notebooks/03_phase04_diagnostics.ipynb`

Notebook koristi postojeće Python module i ranije sačuvane predikcije modela.

### Provedene analize

1. Analiza raspodjele zakonitih, nezakonitih i neoznačenih transakcija kroz vrijeme.
2. Ponovna evaluacija prethodno sačuvanih predikcija.
3. PR-AUC, AP lift, precision, recall i F1 po vremenskim koracima.
4. Analiza predikcijskih skorova i različitih klasifikacijskih pragova.
5. Precision@K i Recall@K.
6. Analiza kalibracije i Brier score.
7. Analiza vremenskih promjena karakteristika primjenom PSI, KS i Wassersteinove udaljenosti.
8. Analiza grafovske strukture, komponenti i neoznačenih susjeda.
9. Izdvajanje tri visokorizična lokalna podgrafa.
10. Ponovno treniranje sva tri modela na dodatnim hronološkim podjelama.

### Hronološke podjele

| Podjela | Trening | Validacija |
|---|---|---|
| A | 1–20 | 21–27 |
| B | 1–27 | 28–34 |
| C | 1–34 | 35–41 |

Svi modeli ponovo su trenirani na svakoj od tri podjele uz zadržavanje početnih konfiguracija.

Izvršeno je ukupno devet dodatnih treniranja.

### Rezultati ponovljenih treniranja

| Model | PR-AUC A | PR-AUC B | PR-AUC C |
|---|---:|---:|---:|
| XGBoost | 0.8617 | 0.9806 | 0.9341 |
| GraphSAGE | 0.6321 | 0.8315 | 0.5384 |
| MLP | 0.6648 | 0.6938 | 0.7294 |

XGBoost ostvaruje najbolje rezultate u sve tri podjele, dok GraphSAGE pokazuje najveće oscilacije između validacijskih perioda.

Rezultati ne pokazuju jednostavan trend kontinuiranog pogoršanja kroz vrijeme, ali pojedini vremenski koraci znatno su zahtjevniji za klasifikaciju.

### Najvažniji nalazi

**1. Vremenska nestabilnost**

Na originalnom testnom periodu 42–49 većina uspješnih detekcija ostvarena je u koraku 42.

Modeli ostvaruju veoma nizak recall u koracima 43–49.

**2. Predikcijski skorovi**

Analiza je pokazala da modeli u kasnijim periodima nezakonitim transakcijama često dodjeljuju niske procjene rizika.

Promjena klasifikacijskog praga može povećati recall, ali uz značajan rast broja lažno pozitivnih predikcija.

**3. Distribution shift**

Utvrđene su značajne promjene distribucija pojedinih ulaznih karakteristika.

Posebno su izražene promjene među označenim nezakonitim transakcijama.

Od 182 karakteristike, 150 ima KS statistiku veću od 0.2 pri poređenju nezakonitih transakcija iz trening perioda i kasnijeg testnog perioda.

Ovo predstavlja moguće objašnjenje slabije generalizacije, ali ne dokazuje uzročnu vezu.

**4. Grafovska struktura**

Svih 234.355 usmjerenih veza povezuje transakcije unutar istog vremenskog koraka.

U originalnom transakcijskom grafu ne postoje veze između različitih vremenskih koraka.

Zbog toga GraphSAGE ne može direktno prenositi informacije između vremenskih perioda korištenjem postojećih veza.

**5. Dodatna treniranja**

MLP pokazuje znakove preprilagođavanja u pojedinim hronološkim podjelama.

GraphSAGE u podjelama B i C postiže najbolje rezultate kasno tokom treniranja, što opravdava dodatno ispitivanje trajanja treniranja.

### Izlazne datoteke

Rezultati se spremaju u:

`outputs/phase04_diagnostics/`

Struktura:

```text
outputs/phase04_diagnostics/
├── phase04_manifest.json
├── tables/
│   └── phase04_*.csv
├── figures/
│   └── phase04_*.png
├── subgraphs/
│   └── phase04_*.png / *.csv
└── rolling_runs/
    ├── phase04_fold_A_xgboost/
    ├── phase04_fold_B_xgboost/
    ├── phase04_fold_C_xgboost/
    ├── phase04_fold_A_mlp/
    ├── phase04_fold_B_mlp/
    ├── phase04_fold_C_mlp/
    ├── phase04_fold_A_graphsage/
    ├── phase04_fold_B_graphsage/
    └── phase04_fold_C_graphsage/
```

Svaki završeni trening čuva validacijske metrike i predikcije, dok neuronske mreže dodatno čuvaju historiju treniranja.

### Pokretanje

Za reprodukciju su potrebni:

- Python moduli iz postojećeg FINTRACE projekta.
- Elliptic++ CSV podaci u `data/raw/`.
- Prethodne predikcije XGBoost, GraphSAGE i MLP modela.
- Google Colab ili lokalno Python okruženje s odgovarajućim bibliotekama.

Otvoriti:

`notebooks/03_phase04_diagnostics.ipynb`

Postaviti odgovarajući `PROJECT_DIR` i izvršiti ćelije redom.

Za dodatne hronološke treninge koriste se postavke:

```python
RUN_ROLLING_XGBOOST = True
RUN_ROLLING_MLP = True
RUN_ROLLING_GRAPHSAGE = True
```

### Metodološka ograničenja

Testni period 42–49 prethodno je analiziran i ne smije se koristiti za izbor najboljih konfiguracija u narednom eksperimentu.

Klasifikacijski pragovi određeni na istim validacijskim podjelama koriste se za razvojno poređenje i nisu nezavisna procjena buduće uspješnosti.

Neoznačene transakcije nisu automatski zakonite.

Prikaz visokorizičnog podgrafa ne predstavlja dokaz nezakonite finansijske aktivnosti.
