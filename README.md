# FinGraph AI — MPVI

**Tema:** Duboko učenje za otkrivanje složenih mreža finansijskih anomalija  
**Rok:** 3–4 sedmice · **Rad:** samostalni · **Resursi:** Google Colab / cloud

Ovo je **početna, reproducibilna tehnička osnova**: loader, provjera dataseta,
vremenska podjela, XGBoost baseline i dvostepeni GraphSAGE. Nije gotov rezultat
istraživanja; trening na kompletnom Elliptic++ skupu tek treba pokrenuti.

## 1. Glavni cilj

Na istoj kronološkoj podjeli Elliptic++ transakcija provjeriti daje li
GraphSAGE bolje rezultate od XGBoost modela. Primarna metrika je PR-AUC;
dodatno pratimo recall, precision, F1, ROC-AUC i Precision/Recall@100.

Oznake: `1` = illicit, `2` = licit, `3` = unknown. U našem kodu su
`1` = illicit, `0` = licit i `-1` = unknown; nepoznate oznake nisu negativna klasa.

## 2. Preuzimanje podataka

Preuzmi **originalni Elliptic++ Transactions Dataset** iz javnog foldera autora:

https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l?usp=sharing

Unutar `data/raw/` stavi tačno:

```
data/raw/
  txs_features.csv
  txs_classes.csv
  txs_edgelist.csv
```

**Važno:** fajlovi istih imena vidljivi na GitHubu su mali Git LFS pokazivači,
ne stvarni CSV podaci. Preuzmi stvarne datoteke s Drivea autora.

Originalni rad i dokumentacija:

- https://github.com/git-disl/EllipticPlusPlus
- https://doi.org/10.1145/3580305.3599803

Ne dodavati sirove podatke u GitHub repozitorij. Provjeriti pravila
redistribucije prije eventualnog dijeljenja dataseta (repo nema eksplicitno
navedenu standardnu licencu za podatke).

## 3. Pokretanje u Google Colabu

1. U Colabu izaberi **Runtime → Change runtime type → T4 GPU**, ako je dostupan.
2. Prenesi ovaj projekt u Drive i raspakuj ga, npr. u `MyDrive/fingraph-ai`.
3. Prenesi tri stvarna CSV-a u `MyDrive/fingraph-ai/data/raw/`.
4. Otvori priloženi `notebooks/00_colab_quickstart.ipynb` ili pokreni:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/fingraph-ai
%pip install -q -e '.[gnn]'
```

Provjeri učitavanje **prije bilo kojeg treniranja**:

```bash
python -m fingraph.audit --data-dir data/raw
```

Zatim referentni model:

```bash
python -m fingraph.baseline --data-dir data/raw
```

Na kraju GNN:

```bash
python -m fingraph.gnn --data-dir data/raw --epochs 60 --hidden-dim 64
```

U Colab code cell koristi `!python -m ...`, ne obični shell blok.

Rezultati se zapisuju u `outputs/xgboost/` i `outputs/graphsage/`:

- `metrics.json` — PR-AUC, recall, precision i druge metrike.
- `test_predictions.csv` — skorovi samo za poznate testne transakcije.
- `graphsage_state.pt` — naučene težine GNN modela.

**Napomena:** treniranje na kompletnom datasetu nije bilo moguće provjeriti
bez tri originalna CSV fajla i Colab GPU okruženja. Prvo pregledati audit,
pa potvrditi izvodljivost modela.

## 4. Struktura

```
fingraph-ai/
├── data/raw/                     # ručno preuzeti CSV podaci (nije u ZIP-u)
├── docs/
│   ├── PLAN_4_SEDMICE.md
│   └── DATASET_I_IZVORI.md
├── notebooks/
│   └── 00_colab_quickstart.ipynb
├── src/fingraph/
│   ├── data.py                  # učitavanje, ID poravnanje, validacija
│   ├── splits.py                # vremenska podjela i filtriranje veza
│   ├── audit.py                 # početna statistika dataseta
│   ├── metrics.py               # metrike i izbor praga
│   ├── baseline.py              # XGBoost
│   └── gnn.py                   # GraphSAGE
├── tests/
│   └── test_data.py             # testovi nad malim sintetičkim podacima
├── pyproject.toml
└── README.md
```

## 5. Metodološka pravila

1. Glavni zadatak je **klasifikacija transakcija**, uz istraživanje njihovih
   mrežnih odnosa. Ne tvrditi da su označene kriminalne grupe ako nisu.
2. Za inicijalni protokol: treniranje koraci 1–34, validacija 35–41,
   testiranje 42–49. Ako neka particija nema obje poznate klase, auditirati
   i unaprijed argumentovati promjenu protokola.
3. Skaliranje i imputacija koriste samo trening podatke. Trening GNN-a koristi
   veze među čvorovima opaženim do kraja trening perioda. Evaluacija svakog
   vremenskog koraka koristi isključivo veze/čvorove opažene do tog koraka.
4. Prag za F1 bira se na validaciji, ne na testu.
5. Prilikom izvještavanja upisati i ograničenje: objavljene Elliptic++
   karakteristike mogu imati vlastite agregacijske pretpostavke koje treba
   provjeriti prije tvrdnji o strogo kauzalnoj predikciji.
6. Proširenje sa scenarijima iz izvještaja SIPA-e je **sintetičko** i odvojeno
   od glavnog testiranja.

## 6. Testovi

Nakon instaliranja osnovnih paketa:

```bash
python -m unittest discover -s tests -v
```

Testovi koriste izmišljene transakcije i ne zahtijevaju Elliptic++ CSV-ove.

Detaljni raspored i linkovi na izvore su u `docs/`.
