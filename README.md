# FinGraph AI

**Duboko učenje za otkrivanje složenih mreža finansijskih anomalija**  
Projekt iz predmeta *Metode primjene vještačke inteligencije (MPVI)*.

FinGraph AI istražuje detekciju nedozvoljenih finansijskih transakcija u grafu i utjecaj mrežnih odnosa na kvalitet predikcija. Na Elliptic++ datasetu upoređuju se XGBoost, višeslojna neuronska mreža (MLP) i GraphSAGE. Eksperimenti koriste hronološku podjelu podataka kako bi se provjerilo koliko se naučeni obrasci prenose na kasnije vremenske periode.

**Trenutni status:** završene početne implementacije i evaluacije sva tri modela, eksplorativna analiza podataka (EDA) i evaluacija po vremenskim koracima (faza 02). Optimizacija modela i analiza visokorizičnih podgrafova slijede.

## Dataset

Koristi se [Elliptic++ Transactions Dataset](https://github.com/git-disl/EllipticPlusPlus), koji sadrži:

| Podatak | Vrijednost |
|---|---:|
| Transakcije (čvorovi) | 203.769 |
| Usmjerene veze | 234.355 |
| Ulazne karakteristike korištene u modelima | 182 |
| Vremenski koraci | 49 |
| Illicit | 4.545 |
| Licit | 42.019 |
| Unknown | 157.205 |

Oznake u izvornim datotekama su `1 = illicit`, `2 = licit` i `3 = unknown`; interni kod ih mapira na `1`, `0` i `-1`. Transakcije s nepoznatom oznakom ne koriste se kao negativni primjeri. Mogu se koristiti kao neoznačeni čvorovi u grafovskom kontekstu.

Podaci su podijeljeni hronološki: **trening 1–34**, **validacija 35–41** i **test 42–49**. Evaluacija se provodi nad 8.841 poznato označenom testnom transakcijom. Pragovi za klasifikaciju biraju se na validacijskom skupu.

Originalni CSV podaci preuzimaju se iz [foldera autora](https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l?usp=sharing) i smještaju u `data/raw/`:

```text
data/raw/
├── txs_features.csv
├── txs_classes.csv
└── txs_edgelist.csv
```

Sirovi podaci nisu uključeni u repozitorij.

## Implementacija

| Komponenta | Opis |
|---|---|
| `data.py`, `audit.py` | Učitavanje, provjera integriteta i pregled dataseta |
| `splits.py` | Hronološka podjela i vremensko filtriranje veza |
| `baseline.py` | XGBoost referentni model |
| `gnn.py` | GraphSAGE sa dva grafovska sloja |
| `mlp.py` | MLP kontrolni model bez eksplicitnog korištenja `edge_index` |
| `metrics.py` | PR-AUC, ROC-AUC, precision, recall, F1 i top-k metrike |
| `temporal.py` | Provjera spremljenih predikcija i evaluacija po vremenskim koracima |

Eksperimenti su izvedeni u Google Colabu uz Python, PyTorch, PyTorch Geometric, scikit-learn i XGBoost. Imputacija nedostajućih vrijednosti i skaliranje prilagođavaju se podacima iz trening perioda.

### Notebooks

| Notebook | Sadržaj |
|---|---|
| `00_colab_quickstart.ipynb` | Početna verzija okruženja i eksperimenata |
| `01_colab_quickstart.ipynb` | Izvršena verzija: audit dataseta, XGBoost i GraphSAGE |
| `01_eda_and_results_analysis.ipynb` | Raspodjela klasa, karakteristike, vremenski koraci i struktura grafa |
| `02_temporal_evaluation_and_baselines.ipynb` | Evaluacija po vremenu, treniranje MLP-a i poređenje tri modela |

## Rezultati do faze 02

Sva tri modela evaluirana su na istom testnom skupu. Navedene vrijednosti potiču iz izvršenih Colab eksperimenata sa `seed=42`.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| XGBoost | **0,5675** | **0,8718** | **0,8442** | **0,4779** | **0,6103** |
| GraphSAGE | 0,3891 | 0,7876 | 0,4441 | 0,3603 | 0,3978 |
| MLP | 0,3707 | 0,8253 | 0,3271 | 0,4706 | 0,3859 |

XGBoost trenutno ostvaruje najbolji rezultat prema primarnoj metrici PR-AUC. GraphSAGE ostvaruje nešto viši PR-AUC od MLP-a, ali na osnovu jednog pokretanja nije moguće tvrditi da je razlika stabilna ili statistički značajna. MLP je najbolji validacijski PR-AUC (0,7294) ostvario u 28. epohi; treniranje je završilo u 48. epohi primjenom *early stoppinga*.

### Nalaz iz vremenske evaluacije

U testnom koraku **42** nalazi se 239 pozitivno označenih transakcija. Pri validacijskim pragovima XGBoost ih otkriva 192, GraphSAGE 145, a MLP 185. U preostalim koracima **43–49**, od 169 pozitivnih primjera, modeli otkrivaju samo 3, 2 i 7 transakcija, istim redoslijedom.

Ovakva razlika upućuje na izraženu nestabilnost detekcije kroz vrijeme. Udio pozitivnih primjera pada sa 11,58% u treningu na 4,61% u testu, ali sama promjena udjela klase nije dovoljno objašnjenje. Naredni eksperimenti trebaju razdvojiti kvalitet rangiranja, kalibraciju skorova, izbor praga i promjenu distribucije karakteristika.

**Ograničenja:** PR-AUC se mijenja s udjelom pozitivne klase, naročito u koracima s vrlo malo pozitivnih primjera. Elliptic++ sadrži unaprijed izračunate agregirane karakteristike pa MLP i XGBoost nisu potpuno bez mrežnih informacija. Trenutni GraphSAGE prilikom propagacije poruka tretira veze dvosmjerno. Evaluacija se odnosi na oznake transakcija, ne na potvrđene kriminalne grupe ili kompletne podgrafove. Testni rezultati već su korišteni za eksplorativnu analizu; budući hiperparametri birat će se isključivo na validaciji.

## Pokretanje

Preporučeno okruženje je **Google Colab**. Nakon povezivanja Google Drivea, pređi u korijen projekta (folder koji sadrži `pyproject.toml`) i instaliraj zavisnosti:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/fingraph-ai/FinGraph_AI_MPVI_Starter
%pip install -e ".[gnn]"
```

Prije pokretanja faze 02 potrebno je imati originalne CSV datoteke i rezultate prethodnih eksperimenata u `outputs/xgboost/` i `outputs/graphsage/`.

Osnovne komande iz korijena projekta:

```bash
python -m fingraph.audit --data-dir data/raw
python -m fingraph.baseline --data-dir data/raw
python -m fingraph.gnn --data-dir data/raw --epochs 60 --hidden-dim 64
python -m fingraph.mlp --data-dir data/raw
```

U Colab ćeliji pokretanje CLI komandi zahtijeva prefiks `!`. Za vremensku analizu i zajedničko poređenje rezultata pokrenuti `notebooks/02_temporal_evaluation_and_baselines.ipynb` redom. Notebook čita postojeće predikcije i trenira samo novi MLP model.

Rezultati se spremaju u:

- `outputs/xgboost/` — XGBoost metrike i testne predikcije;
- `outputs/graphsage/` — GraphSAGE metrike, predikcije i težine modela;
- `outputs/mlp_v1/` — MLP metrike, predikcije, historija treniranja i checkpoint;
- `outputs/eda/` — grafikoni i tabele eksplorativne analize;
- `outputs/phase02/` — vremenske metrike, uporedne tabele, grafikoni i JSON sažetak.

Folderi `data/raw/` i `outputs/` isključeni su iz Git repozitorija. Za detaljnije upute za posljednju fazu pogledati `docs/FAZA_02_UPUTSTVO.md`.

## Testovi

```bash
python -m unittest discover -s tests -v
```

Testovi provjeravaju učitavanje podataka, oznake, vremensku podjelu, poravnanje predikcija i osnovno treniranje MLP-a na sintetičkim podacima; ne zahtijevaju originalni Elliptic++ dataset.

## Daljnji rad

Sljedeći korak je analiza promjena u raspodjeli karakteristika i skorova kroz vrijeme. Nakon toga slijede kontrolisani eksperimenti s lokalnim i agregiranim karakteristikama, optimizacija GraphSAGE modela i izdvajanje visokorizičnih povezanih transakcija. Dokumentacija i završni poster bit će pripremljeni na osnovu validiranih nalaza.

## Reference

- [Elliptic++ — dataset i kod](https://github.com/git-disl/EllipticPlusPlus)
- [Elmougy i Liu (2023), KDD](https://doi.org/10.1145/3580305.3599803)
- [Weber et al. (2019), AML i GCN](https://arxiv.org/abs/1908.02591)
