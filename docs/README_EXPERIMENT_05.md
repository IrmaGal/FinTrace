# FINTRACE — Peti eksperiment
## Ablacija karakteristika, preprocessing i tuning modela

### Cilj

Peti eksperiment nastavlja istraživanje XGBoost, GraphSAGE i MLP modela na Elliptic++ Transactional datasetu.

Eksperiment obuhvata:

1. Analizu utjecaja različitih grupa ulaznih karakteristika.
2. Ispitivanje alternativnih postupaka predobrade podataka.
3. Kombinovanje karakteristika i postupaka predobrade.
4. Provjeru stabilnosti modela kroz različite slučajne inicijalizacije.
5. Podešavanje hiperparametara i završnu evaluaciju na kasnijim vremenskim koracima.

### Nove datoteke

```text
FinGraph_AI_MPVI_Starter/
├── notebooks/
│   └── 04_phase05_ablation_preprocessing_tuning.ipynb
├── src/
│   └── fingraph/
│       ├── phase05.py
│       └── phase05_stage_C.py
├── tests/
│   └── test_phase05.py
├── docs/
│   ├── README_EXPERIMENT_04.md
│   └── README_EXPERIMENT_05.md
└── outputs/
    └── phase05/
```

Ne mijenjati postojeće module ili prethodne notebookove.

### Eksperimentalni protokol

Koriste se tri hronološke podjele:

| Podjela | Trening | Validacija |
|---|---|---|
| A | 1–20 | 21–27 |
| B | 1–27 | 28–34 |
| C | 1–34 | 35–41 |

Svi postupci predobrade prilagođavaju se isključivo podacima odgovarajućeg trening perioda.

Testni koraci 42–49 ne koriste se za izbor konfiguracija tokom razvojnih faza.

Neoznačeni čvorovi mogu se koristiti za nenadziranu procjenu distribucija karakteristika i kao grafovski kontekst, ali se ne tretiraju kao negativna klasa.

### Faza A — Ablacija karakteristika

Ispituje se šest varijanti ulaznih karakteristika:

| Varijanta | Opis |
|---|---|
| all | Svih 182 karakteristike |
| without_aggregate | 110 karakteristika: 93 lokalne i 17 dodatnih imenovanih |
| local_only | Samo 93 lokalne karakteristike |
| aggregate_only | Samo 72 agregirane karakteristike |
| named_only | Samo 17 dodatnih imenovanih karakteristika |
| all_pca95 | PCA nad svim karakteristikama uz zadržavanje najmanje 95% varijanse |

PCA se prilagođava isključivo na trening podacima. Za svaku kombinaciju modela, skupa karakteristika i hronološke podjele provodi se zasebno treniranje.

Koriste se početne konfiguracije XGBoost, GraphSAGE i MLP modela.

### Faza B — Alternativni preprocessing

Ispituje se pet postupaka predobrade nad kompletnim skupom karakteristika:

- **median_standard** – medijanska imputacija i StandardScaler.
- **median_robust** – medijanska imputacija i RobustScaler.
- **median_log_robust** – selektivna log1p transformacija i RobustScaler.
- **median_quantile** – medijanska imputacija i QuantileTransformer.
- **median_power** – medijanska imputacija i PowerTransformer (Yeo–Johnson).

Sve transformacije prilagođavaju se isključivo na trening podacima. Za sve postupke koriste se iste početne konfiguracije modela, a evaluacija se provodi na tri hronološke podjele.

### Faza C — Kombinovanje konfiguracija i optimizacija modela

Faza C podijeljena je na tri dijela.

**C.1 — Kombinovanje karakteristika i predobrade**

Ispituju se tri konfiguracije za svaki model:

- **XGBoost:** sve karakteristike uz standardnu i Yeo–Johnson predobradu te samo lokalne karakteristike uz standardnu predobradu.
- **MLP:** sve karakteristike, bez agregiranih i samo lokalne karakteristike uz kvantilsku transformaciju.
- **GraphSAGE:** iste tri varijante karakteristika uz kvantilsku transformaciju.

Evaluacija se provodi na tri hronološke podjele, uz osnovni seed 42.

**C.2 — Provjera stabilnosti**

Odabrane konfiguracije dodatno se evaluiraju kroz tri seed vrijednosti: 42, 123 i 2026.

Na ovaj način ispituje se osjetljivost modela na slučajnu inicijalizaciju, uz zadržavanje istih kombinacija karakteristika, predobrade i hronoloških podjela.

**C.3 — Ciljano podešavanje hiperparametara**

Za svaki model ispituju se četiri dodatne konfiguracije.

Kod XGBoost modela mijenjaju se dubina stabala, regularizacija, stopa učenja, broj stabala i ponderisanje pozitivne klase.

Kod MLP modela ispituju se duže treniranje, veličina mreže, regularizacija, stopa učenja i ponderisanje klasa.

Kod GraphSAGE modela ispituju se duže treniranje, broj grafovskih slojeva i regularizacija.

Svaka konfiguracija evaluira se na tri hronološke podjele i tri seed vrijednosti (42, 123 i 2026). Poređenje se vrši u odnosu na odgovarajuće početne konfiguracije.

### Završna evaluacija — Testni skup

Završna evaluacija provodi se na testnim koracima 42–49, nakon odabira konfiguracija u prethodnim fazama.

Koriste se četiri konfiguracije:

- **XGBoost:** svih 182 karakteristike, standardna predobrada i početni hiperparametri.
- **MLP:** sve karakteristike, kvantilska transformacija, prilagođena težina pozitivne klase i duže treniranje.
- **GraphSAGE (2 sloja):** sve karakteristike, kvantilska transformacija i duže treniranje.
- **GraphSAGE (3 sloja):** alternativna konfiguracija za poređenje arhitektura.

Modeli se treniraju na koracima 1–34, validacija se provodi na koracima 35–41, a testiranje na koracima 42–49.

Za svaku konfiguraciju koriste se tri seed vrijednosti (42, 123 i 2026). Predobrada se prilagođava isključivo na trening podacima.

Kod GraphSAGE modela koriste se grafovske veze dostupne do kraja testnog perioda, što ne predstavlja strogu online predikciju pojedinačnih transakcija.

Evaluacija obuhvata PR-AUC, precision, recall, F1, Precision@100 i analizu po pojedinačnim vremenskim koracima. Klasifikacijski prag određuje se na validacijskom skupu.

### Kriteriji poređenja

Primarni kriterij tokom razvojnih faza je validacijski PR-AUC.

Dodatno se posmatraju:

- AP lift.
- Najlošiji PR-AUC među hronološkim podjelama.
- Standardna devijacija rezultata između seed vrijednosti.
- Precision.
- Recall.
- F1.
- Historija treniranja.
- Vrijeme izvršavanja.

Klasifikacijski prag bira se na validaciji, pa precision, recall i F1 predstavljaju metrike zavisne od odabranog praga.

### Pokretanje

Otvoriti:

`notebooks/04_phase05_ablation_preprocessing_tuning.ipynb`

Faze A i B kontrolišu se varijablama:

```python
RUN_ABLATION = False
RUN_PREPROCESSING = False

MODELS_TO_RUN = [
    "XGBoost",
    "MLP",
    "GraphSAGE",
]

FOLDS_TO_RUN = ["A", "B", "C"]
SKIP_EXISTING = True
```

Za pokretanje faze A postaviti `RUN_ABLATION = True`, a za fazu B `RUN_PREPROCESSING = True`.

Faze C.1, C.2 i C.3 pokreću se zasebnim prekidačima u odgovarajućim ćelijama notebooka:

```python
RUN_C1_COMBINATIONS = True
RUN_C2_SEED_STABILITY = False
RUN_C3_TUNING = False
```

Faza C.3 koristi funkciju `run_tuning()` iz modula `phase05_stage_C.py`.

Završna evaluacija C.4 pokreće se zasebno:

```python
RUN_C4_FINAL_TEST = True
SKIP_EXISTING_C4 = True
```

`RUN_TUNING` je starija kontrolna varijabla i ne pokreće stvarnu fazu C.3.

### Izlazne datoteke

```text
outputs/phase05/
├── runs/
│   ├── ablation/
│   ├── preprocessing/
│   └── tuning/
├── tables/
├── figures/
└── phaseC_final_test/
    ├── runs/
    └── tables/
        ├── phase05_c4_all_test_metrics.csv
        ├── phase05_c4_run_metadata.csv
        ├── phase05_c4_summary.csv
        ├── phase05_c4_per_step.csv
        ├── phase05_c4_original_models_detailed.csv
        └── phase05_c4_vs_original_models.csv
```

Pojedinačna treniranja spremaju metrike, predikcije i metapodatke. Za MLP i GraphSAGE čuva se i historija treniranja, a za XGBoost važnost karakteristika.

### Testiranje

Iz korijena projekta:

```bash
python -m unittest discover -s tests -v
```

Testovi provjeravaju izbor grupa karakteristika, vremenske maske i ispravnost predobrade.

### Metodološka ograničenja

Testni skup 42–49 već je analiziran u prethodnim eksperimentima. Zato se ne koristi za izbor konfiguracija tokom razvojnih faza.

Kod GraphSAGE modela korištenje grafovskih veza do kraja testnog perioda znači da evaluacija nije strogo online.

Za nezavisnu procjenu vremenske generalizacije potrebno je koristiti novi, prethodno neanalizirani vremenski skup.

