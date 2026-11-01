# FinGraph AI — Faza 02: vremenska evaluacija i MLP

## Cilj

Ova faza **ne mijenja postojeće XGBoost i GraphSAGE modele**. Učitava njihove spremljene predikcije za testni period 42–49, računa metrike po periodima i uvodi treći, kontrolni model (MLP) bez eksplicitnog message-passinga po grafu.

## Novi fajlovi

| Fajl | Svrha |
|---|---|
| `src/fingraph/temporal.py` | Validacija testnih predikcija, poravnanje po `txId`, vremenske metrike |
| `src/fingraph/mlp.py` | Treniranje MLP kontrolnog modela, spremanje modela i predikcija |
| `notebooks/02_temporal_evaluation_and_baselines.ipynb` | Korak-po-korak analiza i prikaz rezultata u Colabu |
| `tests/test_phase02.py` | Testovi na sintetičkom datasetu |
| `docs/FAZA_02_UPUTSTVO.md` | Ovo uputstvo |

Sadržaj ZIP-a kopirati **u korijen postojećeg projekta**, tako da se putanje poklope. Ovo su isključivo nove datoteke. Ne brisati i ne prepisivati postojeće `data.py`, `gnn.py`, `baseline.py`, `01_colab_quickstart.ipynb`, `01_eda_and_results_analysis.ipynb`.

## Ulazni podaci koji moraju biti u Google Driveu

```text
data/raw/txs_features.csv
data/raw/txs_classes.csv
data/raw/txs_edgelist.csv
outputs/xgboost/metrics.json
outputs/xgboost/test_predictions.csv
outputs/graphsage/metrics.json
outputs/graphsage/test_predictions.csv
```

Ovi ulazni fajlovi **nisu uključeni u paket** i nalaze se u tvojoj postojećoj Drive kopiji projekta. `outputs/` je isključen iz Git-a i zato stare rezultate ne treba očekivati u GitHub klonu. Ako ih nema u Driveu, potrebno ih je obnoviti iz prve uspješne sesije.

## Pokretanje

1. Raspakuj paket na lokalni disk i kopiraj folder `src`, `notebooks`, `tests` i `docs` **u postojeći projekt**, spajajući sadržaj foldera.
2. Prenesi iste nove fajlove u postojeću **Google Drive** kopiju projekta (`MyDrive/fingraph-ai/FinGraph_AI_MPVI_Starter`).
3. Otvori `notebooks/02_temporal_evaluation_and_baselines.ipynb` u Colabu.
4. Pokreni ćelije redom. Početna ćelija za provjeru ispisaće šta nedostaje, ako nije preneseno.
5. Preporučena Colab runtime konfiguracija je **T4 GPU**, ali MLP može raditi i na CPU-u uz duže izvršavanje.
6. Ne mijenjaj parametre modela na osnovu testnih rezultata.

## Rezultati

- `outputs/phase02/tables/temporal_metrics_initial_models.csv` — temporalne metrike XGBoost i GraphSAGE
- `outputs/phase02/tables/temporal_metrics_all_models.csv` — temporalne metrike svih triju modela
- `outputs/phase02/tables/baseline_comparison_three_models.csv` — sveukupne metrike
- `outputs/phase02/figures/` — grafikoni
- `outputs/phase02/phase02_summary.json` — sažetak
- `outputs/mlp_v1/metrics.json` — MLP rezultati
- `outputs/mlp_v1/test_predictions.csv` i `validation_predictions.csv` — skorovi MLP-a
- `outputs/mlp_v1/training_history.csv` — loss i validacijski PR-AUC po epohi
- `outputs/mlp_v1/mlp_checkpoint.pt` — najbolje težine i konfiguracija
- `outputs/mlp_v1/preprocessing.joblib` — imputacija i skaliranje

**Ne uključivati sirove CSV datoteke ili `outputs/` u GitHub commit.** `*.joblib` i `.pt` treba učitavati samo ako potiču iz pouzdanog izvora.

## Testiranje

U lokalnom terminalu, nakon instalacije projekta, izvrši:

```bash
python -m unittest discover -s tests -v
```

`tests/test_phase02.py` testira kvalitet poravnanja predikcija, odbacivanje neispravnih podataka, metrike kroz vrijeme i kratko MLP treniranje na malim sintetičkim podacima. Originalni Elliptic++ dataset nije potreban za testove.

## Git workflow

Nakon uspješnog pokretanja i provjere lokalnih fajlova:

```bash
git switch main
git status
git add src/fingraph/mlp.py src/fingraph/temporal.py notebooks/02_temporal_evaluation_and_baselines.ipynb tests/test_phase02.py docs/FAZA_02_UPUTSTVO.md
git commit -m "Add temporal evaluation and MLP baseline"
git push origin main
```

Sačuvaj stabilnu verziju `v3` **tek nakon što potvrdiš da faza radi**, ako želiš historijsku verziju. Za sljedeće promjene nastavi raditi u `main`.

## Metodološko upozorenje

- Udio pozitivne klase varira po vremenskim koracima, a neki koraci imaju vrlo malo pozitivnih primjera.
- PR-AUC po koracima nije direktno uporediv bez pregleda prevalencije i broja primjera.
- MLP s 182 karakteristike **nije** strogo „bez mrežnih informacija“ jer Elliptic++ uključuje izvorno agregirane karakteristike. MLP je model **bez eksplicitnog korištenja `edge_index`**.
- Postojeći GraphSAGE koristi dvosmjerne veze pri message-passingu.
- Rezultati testa iz početne faze su već pregledani; hiperparametre u narednim fazama treba birati samo na validaciji, a ovu činjenicu dokumentovati.
- Ovo je klasifikacija transakcija, a ne validirana klasifikacija kriminalnih grupa/podgrafova.
