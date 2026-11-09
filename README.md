# FinGraph AI

**Duboko učenje za otkrivanje složenih mreža finansijskih anomalija**  
Projekt iz predmeta *Metode primjene vještačke inteligencije (MPVI)*.

FinGraph AI istražuje detekciju nedozvoljenih finansijskih transakcija u grafu i utjecaj mrežnih odnosa na kvalitet predikcija. Na Elliptic++ datasetu upoređuju se XGBoost, višeslojna neuronska mreža (MLP) i GraphSAGE. Eksperimenti koriste hronološku podjelu podataka kako bi se provjerilo koliko se naučeni obrasci prenose na kasnije vremenske periode.

**Trenutni status:** završene početne implementacije i evaluacije XGBoost, GraphSAGE i MLP modela, eksplorativna analiza podataka (EDA), vremenska evaluacija (faza 02) i dijagnostička analiza vremenske generalizacije modela (faza 03). U posljednjoj fazi analizirane su promjene distribucija karakteristika, predikcijski skorovi, klasifikacijski pragovi, kalibracija, grafovska struktura i visokorizični podgrafovi. Modeli su dodatno trenirani na tri hronološke podjele. Sljedeća faza obuhvata ablacijske eksperimente, alternativne postupke predobrade i optimizaciju hiperparametara.

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
| `03_phase04_diagnostics.ipynb` | Dijagnostika vremenske generalizacije, promjena karakteristika, kalibracije i grafovske strukture te ponovno treniranje na različitim hronološkim podjelama |

## Rezultati modela

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

Dodatne hronološke validacije potvrdile su značajne razlike u uspješnosti modela između vremenskih perioda, ali ne predstavljaju nezavisnu potvrdu generalizacije na buduće podatke. U narednim eksperimentima izbor konfiguracija zasnivat će se na razvojnim vremenskim podjelama, bez optimizacije prema prethodno analiziranim testnim koracima 42–49.

### Četvrti eksperiment — dijagnostička analiza

Četvrti eksperiment bio je usmjeren na ispitivanje uzroka slabije generalizacije modela u kasnijim vremenskim koracima. Analizirane su distribucije predikcijskih skorova, osjetljivost na klasifikacijske pragove, Precision@K i Recall@K metrike, kalibracija predikcija, promjene ulaznih karakteristika kroz vrijeme i struktura transakcijskog grafa.

Rezultati su pokazali da se slabija uspješnost u periodu 43–49 ne može objasniti samo izborom klasifikacijskog praga. Kod sva tri modela značajno se pogoršava i kvalitet rangiranja nezakonitih transakcija. XGBoost, naprimjer, u koraku 42 ostvaruje PR-AUC od 0,8976, dok u objedinjenom periodu 43–49 postiže samo 0,0524.

Posebno su izražene promjene karakteristika nezakonitih transakcija. Poređenjem trening perioda i koraka 43–49 utvrđeno je da 150 od 182 karakteristike ima KS statistiku veću od 0,2. To ukazuje na moguće promjene obrazaca nezakonitih transakcija kroz vrijeme, koje mogu otežavati generalizaciju modela.

Analiza grafovske strukture potvrdila je da se svih 234.355 veza nalazi unutar pojedinačnih vremenskih koraka, bez direktnih veza između različitih perioda. Dodatno su vizualizovana tri visokorizična lokalna podgrafa, s ciljem pregleda povezanosti i zastupljenosti neoznačenih transakcija.

#### Evaluacija na dodatnim hronološkim podjelama

Sva tri modela ponovo su trenirana na tri različite vremenske podjele, uz zadržavanje početnih arhitektura i hiperparametara:

| Podjela | Trening | Validacija | XGBoost PR-AUC | GraphSAGE PR-AUC | MLP PR-AUC |
|---|---|---|---:|---:|---:|
| A | 1–20 | 21–27 | **0,8617** | 0,6321 | 0,6648 |
| B | 1–27 | 28–34 | **0,9806** | 0,8315 | 0,6938 |
| C | 1–34 | 35–41 | **0,9341** | 0,5384 | 0,7294 |

XGBoost je ostvario najbolje rezultate u sve tri podjele, dok je GraphSAGE pokazao najveće oscilacije. Rezultati nisu pokazali jednostavan trend kontinuiranog pogoršanja kroz vrijeme, ali su potvrdili da postoje pojedinačni vremenski koraci u kojima sva tri modela postižu znatno slabije rezultate.

Dobijeni nalazi predstavljaju osnovu za naredne eksperimente, koji će se usmjeriti na ablacijske analize karakteristika, drugačiju predobradu podataka i kontrolisanu optimizaciju modela.

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

Za dijagnostičku analizu pokrenuti `notebooks/03_phase04_diagnostics.ipynb` u Google Colabu. Notebook koristi originalni Elliptic++ dataset i prethodno sačuvane predikcije sva tri modela.

Eksperiment uključuje analizu vremenskih promjena, predikcijskih skorova, klasifikacijskih pragova, kalibracije, grafovske strukture i visokorizičnih podgrafova. Dodatno je omogućeno ponovno treniranje početnih konfiguracija modela na tri hronološke podjele.

Ponovljena treniranja uključuju se pomoću sljedećih postavki:

```python
RUN_ROLLING_XGBOOST = True
RUN_ROLLING_MLP = True
RUN_ROLLING_GRAPHSAGE = True
```

Svi rezultati spremaju se u `outputs/phase04_diagnostics/`, uključujući tabele, grafikone, podgrafove i rezultate dodatnih treniranja u direktoriju `rolling_runs/`.

Detaljniji opis metodologije i rezultata nalazi se u `docs/README_EXPERIMENT_04.md`.

## Testovi

```bash
python -m unittest discover -s tests -v
```

Testovi provjeravaju učitavanje podataka, oznake, vremensku podjelu, poravnanje predikcija i osnovno treniranje MLP-a na sintetičkim podacima; ne zahtijevaju originalni Elliptic++ dataset.

## Daljnji rad

Naredna faza projekta usmjerena je na unapređenje postojećih modela na osnovu rezultata dijagnostičke analize. Planirana je ablacijska analiza lokalnih i agregiranih karakteristika, ispitivanje smanjenja dimenzionalnosti pomoću PCA, testiranje alternativnih postupaka predobrade podataka i kontrolisano podešavanje hiperparametara XGBoost, GraphSAGE i MLP modela.

Posebna pažnja bit će posvećena stabilnosti rezultata kroz različite hronološke podjele, kako bi se utvrdilo mogu li se smanjiti problemi generalizacije na kasnije vremenske periode.

U narednim fazama razmotrit će se i proširenje postojećeg transakcijskog grafa podacima o Bitcoin adresama iz Elliptic++ Actors dataseta te primjena hibridnih i self-supervised pristupa.

## Reference

- [Elliptic++ — dataset i kod](https://github.com/git-disl/EllipticPlusPlus)
- [Elmougy i Liu (2023), KDD](https://doi.org/10.1145/3580305.3599803)
- [Weber et al. (2019), AML i GCN](https://arxiv.org/abs/1908.02591)
