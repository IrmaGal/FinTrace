# FinGraph AI

**Duboko učenje za otkrivanje složenih mreža finansijskih anomalija**  
Projekt iz predmeta *Metode primjene vještačke inteligencije (MPVI)*.

FinGraph AI istražuje mogućnost detekcije nezakonitih Bitcoin transakcija primjenom mašinskog i dubokog učenja nad grafovskim podacima. Na Elliptic++ skupu podataka upoređuju se XGBoost, višeslojna neuronska mreža (MLP) i GraphSAGE, s ciljem ispitivanja doprinosa mrežne strukture, karakteristika transakcija i postupaka predobrade kvalitetu klasifikacije.

Poseban fokus istraživanja jeste **vremenska generalizacija modela**, odnosno njihova sposobnost prepoznavanja nezakonitih transakcija u kasnijim vremenskim periodima.

**Trenutni status:** završeno je pet eksperimenata koji obuhvataju početno treniranje i poređenje modela, eksplorativnu analizu podataka, vremensku evaluaciju, dijagnostiku promjena distribucija te sistematsku optimizaciju ulaznih karakteristika, predobrade i hiperparametara. Završna evaluacija pokazala je značajno poboljšanje MLP i GraphSAGE modela, ali i potvrdila da problem generalizacije nakon vremenskog koraka 42 ostaje otvoren.

## Dataset

Koristi se [Elliptic++ Transactions Dataset](https://github.com/git-disl/EllipticPlusPlus), koji sadrži:

| Podatak | Vrijednost |
|---|---:|
| Transakcije (čvorovi) | 203.769 |
| Usmjerene veze | 234.355 |
| Ulazne karakteristike | 182 |
| Vremenski koraci | 49 |
| Illicit transakcije | 4.545 |
| Licit transakcije | 42.019 |
| Unknown transakcije | 157.205 |

Izvorne oznake `1 = illicit`, `2 = licit` i `3 = unknown` interno se mapiraju na `1`, `0` i `-1`.

Transakcije s nepoznatom oznakom ne koriste se kao negativni primjeri, ali mogu učestvovati kao neoznačeni čvorovi u grafovskom kontekstu.

### Vremenska podjela

Osnovna hronološka podjela:

| Skup | Vremenski koraci | Označene transakcije | Illicit |
|---|---|---:|---:|
| Trening | 1–34 | 29.894 | 3.462 |
| Validacija | 35–41 | 7.829 | 675 |
| Test | 42–49 | 8.841 | 408 |

Za razvoj i optimizaciju modela koriste se dodatne hronološke podjele:

| Podjela | Trening | Validacija |
|---|---|---|
| A | 1–20 | 21–27 |
| B | 1–27 | 28–34 |
| C | 1–34 | 35–41 |

Sve transformacije podataka prilagođavaju se isključivo trening periodu. Klasifikacijski pragovi i najbolja epoha neuronskih modela biraju se na validacijskim podacima.

**Napomena:** originalni testni period 42–49 već je analiziran tokom prethodnih eksperimenata. Zbog toga kasnija poređenja na tom periodu predstavljaju retrospektivnu evaluaciju, a ne potpuno nezavisnu potvrdu generalizacije.

### Priprema podataka

Originalni CSV podaci preuzimaju se iz [foldera autora Elliptic++ skupa](https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l?usp=sharing) i smještaju u `data/raw/`:

```text
data/raw/
├── txs_features.csv
├── txs_classes.csv
└── txs_edgelist.csv
```

Sirovi podaci nisu uključeni u Git repozitorij.

## Implementacija

Projekat je organizovan kao Python paket `fingraph`.

| Komponenta | Opis |
|---|---|
| `data.py`, `audit.py` | Učitavanje, provjera integriteta i eksplorativni pregled podataka |
| `splits.py` | Hronološka podjela podataka i vremensko filtriranje grafovskih veza |
| `baseline.py` | Referentni XGBoost model |
| `gnn.py` | Dvoslojni GraphSAGE model |
| `mlp.py` | MLP model bez eksplicitne propagacije kroz graf |
| `metrics.py` | PR-AUC, ROC-AUC, precision, recall, F1 i top-k metrike |
| `temporal.py` | Evaluacija i poređenje predikcija po vremenskim koracima |
| `phase05.py` | Ablacija karakteristika, predobrada i eksperimentalne konfiguracije |
| `phase05_stage_C.py` | Kombinovanje konfiguracija, analiza stabilnosti i podešavanje hiperparametara |

Završna evaluacija petog eksperimenta implementirana je direktno u odgovarajućem Colab notebooku.

Eksperimenti se izvode u **Google Colabu**, koristeći Python, PyTorch, PyTorch Geometric, scikit-learn i XGBoost.

### Notebookovi

| Notebook | Sadržaj |
|---|---|
| `00_colab_quickstart.ipynb` | Početna verzija okruženja |
| `01_colab_quickstart.ipynb` | Audit dataseta i početni XGBoost i GraphSAGE modeli |
| `01_eda_and_results_analysis.ipynb` | Eksplorativna analiza podataka i početnih rezultata |
| `02_temporal_evaluation_and_baselines.ipynb` | MLP i vremenska evaluacija tri modela |
| `03_phase04_diagnostics.ipynb` | Dijagnostika vremenske generalizacije i dodatne hronološke podjele |
| `04_phase05_ablation_preprocessing_tuning.ipynb` | Ablacija karakteristika, predobrada, tuning, stabilnost i završna testna evaluacija |

## Pregled eksperimenata

### Eksperiment 1 — početni modeli

Implementirani su XGBoost i GraphSAGE, uz hronološku podjelu podataka.

XGBoost koristi numeričke karakteristike transakcija, dok GraphSAGE dodatno koristi informacije iz njihovog grafovskog susjedstva.

Na originalnom testnom periodu XGBoost je ostvario PR-AUC od **0,5675**, a GraphSAGE **0,3891**.

### Eksperiment 2 — eksplorativna analiza podataka

Provedena je analiza raspodjele klasa, nedostajućih vrijednosti, numeričkih karakteristika, vremenskih promjena i strukture transakcijskog grafa.

Utvrđena je izražena neravnoteža klasa i promjenjiva zastupljenost nezakonitih transakcija kroz vrijeme.

Također je potvrđeno da se svih 234.355 transakcijskih veza nalazi unutar pojedinačnih vremenskih koraka, bez direktnih veza između različitih perioda.

### Eksperiment 3 — MLP i vremenska evaluacija

Implementiran je MLP kao dodatni referentni neuronski model.

Sva tri modela analizirana su na istim testnim transakcijama, uz evaluaciju po pojedinačnim vremenskim koracima.

Rezultati su pokazali da modeli najveći dio nezakonitih transakcija prepoznaju u koraku 42, dok se nakon tog koraka uspješnost značajno smanjuje.

### Eksperiment 4 — dijagnostika vremenske generalizacije

Detaljnije su analizirani:

- promjene distribucija karakteristika kroz vrijeme;
- predikcijski skorovi i klasifikacijski pragovi;
- Precision@K i Recall@K;
- kalibracija predikcija;
- grafovska struktura i visokorizični podgrafovi;
- stabilnost početnih modela na dodatnim hronološkim podjelama.

Posebno je značajan nalaz da kod nezakonitih transakcija **150 od 182 karakteristike** ima Kolmogorov–Smirnov statistiku veću od 0,2 pri poređenju trening perioda i koraka 43–49.

Ovaj rezultat ukazuje na izražene promjene karakteristika pozitivne klase (*distribution shift*), koje mogu doprinositi slabijoj generalizaciji.

Dodatne hronološke evaluacije pokazale su značajne oscilacije rezultata između vremenskih perioda.

### Eksperiment 5 — ablacija, predobrada i optimizacija modela

Peti eksperiment usmjeren je na sistematsko poboljšanje XGBoost, MLP i GraphSAGE modela na osnovu nalaza prethodnih eksperimenata.

Obuhvata četiri povezane faze.

#### Faza A — ablacijska analiza karakteristika

Ispitan je doprinos različitih skupova ulaznih karakteristika:

- sve 182 karakteristike;
- karakteristike bez agregiranih;
- samo lokalne karakteristike;
- samo agregirane karakteristike;
- samo dodatne imenovane karakteristike;
- PCA transformacija s najmanje 95% objašnjene varijanse.

Izvedena su 54 treniranja na tri hronološke podjele.

XGBoost i GraphSAGE ostvarili su najbolje prosječne rezultate sa svim karakteristikama, dok je MLP uz standardnu predobradu pokazao značajno poboljšanje nakon uklanjanja agregiranih karakteristika.

PCA nije poboljšala rezultate nijednog modela.

#### Faza B — alternativni postupci predobrade

Upoređeni su StandardScaler, RobustScaler, selektivna logaritamska transformacija, QuantileTransformer i PowerTransformer (Yeo–Johnson).

Izvedeno je 36 dodatnih treniranja.

Najvažniji rezultat jeste značajno poboljšanje neuronskih modela primjenom **kvantilske transformacije**.

| Model | Standardna predobrada | Kvantilska transformacija |
|---|---:|---:|
| XGBoost | 0,9276 | 0,9279 |
| MLP | 0,6960 | **0,9030** |
| GraphSAGE | 0,6673 | **0,8587** |

Vrijednosti predstavljaju prosječni validacijski PR-AUC kroz tri hronološke podjele.

XGBoost je pokazao relativno malu osjetljivost na izbor predobrade, dok je kod neuronskih modela odgovarajuća transformacija numeričkih karakteristika bila jedan od ključnih faktora poboljšanja.

#### Faza C — kombinovanje konfiguracija i podešavanje hiperparametara

Najperspektivnije kombinacije karakteristika i predobrade evaluirane su kroz tri hronološke podjele i tri seed vrijednosti (`42`, `123`, `2026`).

Prvo je provedeno 81 treniranje radi poređenja kombinacija i stabilnosti rezultata, nakon čega je izvedeno 108 treniranja za ciljano podešavanje hiperparametara.

Najbolje razvojne konfiguracije:

| Model | Konfiguracija | Prosječni PR-AUC |
|---|---|---:|
| XGBoost | Sve karakteristike, standardna predobrada, početni hiperparametri | 0,9277 |
| MLP | Sve karakteristike, kvantilska transformacija, smanjena težina pozitivne klase i duže treniranje | **0,9102** |
| GraphSAGE | Sve karakteristike, kvantilska transformacija, dva sloja i duže treniranje | **0,8982** |
| GraphSAGE (alternativa) | Sve karakteristike, kvantilska transformacija, tri sloja i duže treniranje | 0,8993 |

Kod MLP-a najbolje rezultate donijela je kombinacija dužeg treniranja i smanjene težine pozitivne klase.

Kod GraphSAGE-a najznačajnije poboljšanje ostvareno je produžavanjem treniranja sa 60 na 120 epoha. Troslojna arhitektura postigla je neznatno veći razvojni PR-AUC, ali je dvoslojni model pokazao dosljednije poboljšanje kroz ponovljena treniranja.

Dodatno podešavanje XGBoost modela nije donijelo uvjerljivo poboljšanje u odnosu na početnu konfiguraciju.

#### Faza C.4 — završna testna evaluacija

Odabrane konfiguracije evaluirane su na originalnom testnom periodu 42–49, koristeći tri seed vrijednosti.

Ukupno je provedeno 12 završnih treniranja, uključujući dodatnu troslojnu GraphSAGE konfiguraciju.

Rezultati su upoređeni sa sačuvanim predikcijama početnih modela, bez ponovnog treniranja prethodnih referentnih verzija.

## Rezultati modela

### Početni i optimizovani modeli

Sljedeća tabela prikazuje rezultate na originalnom testnom periodu 42–49.

| Model | Početni PR-AUC | Nakon eksperimenta 5 | Promjena |
|---|---:|---:|---:|
| XGBoost | 0,5675 | 0,5634 | −0,0041 |
| MLP | 0,3707 | **0,5500** | **+0,1793** |
| GraphSAGE (2 sloja) | 0,3891 | **0,5368** | **+0,1477** |
| GraphSAGE (3 sloja) | 0,3891 | 0,5352 | +0,1461 |

Početne vrijednosti potiču iz ranijih pojedinačnih treniranja, dok optimizovane vrijednosti predstavljaju prosjek tri seed vrijednosti.

MLP je ostvario relativno poboljšanje testnog PR-AUC od približno **48,4%**, a dvoslojni GraphSAGE približno **38,0%**.

XGBoost je zadržao približno početni rezultat i ostao model s najvećim ukupnim testnim PR-AUC.

### Završne klasifikacijske metrike

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| XGBoost | 0,8887 | 0,4730 | **0,6173** |
| MLP | **0,9051** | 0,4641 | 0,6134 |
| GraphSAGE (2 sloja) | 0,7952 | 0,4616 | 0,5840 |
| GraphSAGE (3 sloja) | 0,7789 | 0,4542 | 0,5737 |

Prikazane su prosječne vrijednosti kroz tri seed vrijednosti, uz klasifikacijske pragove odabrane na validaciji.

Optimizacija je naročito poboljšala precision MLP i GraphSAGE modela. Ipak, povećanje ukupnih metrika ne znači nužno da su poboljšanja jednako prisutna u svim vremenskim periodima.

### Vremenska generalizacija

Najvažniji nalaz petog eksperimenta jeste da se poboljšanja neuronskih modela najvećim dijelom odnose na **vremenski korak 42**, dok je kvalitet klasifikacije u periodu 43–49 i dalje veoma nizak.

| Model | PR-AUC korak 42 | PR-AUC koraci 43–49 |
|---|---:|---:|
| XGBoost | 0,8966 | 0,0516 |
| MLP | 0,8868 | 0,0416 |
| GraphSAGE (2 sloja) | 0,8828 | 0,0358 |
| GraphSAGE (3 sloja) | 0,8912 | 0,0360 |

U periodu 43–49 nalazi se 169 označenih nezakonitih transakcija, a udio pozitivne klase iznosi približno 2,53%.

Iako svi optimizovani modeli ostvaruju Precision@100 od 1,00 na kompletnom testnom periodu, svih 100 najbolje rangiranih transakcija pripada koraku 42. Kada se Precision@100 računa samo za korake 43–49, rezultati su znatno slabiji.

Ovi nalazi pokazuju da **optimizacija karakteristika, predobrade i hiperparametara značajno poboljšava neuronske modele, ali ne rješava problem vremenske generalizacije**.

### Zaključak petog eksperimenta

Peti eksperiment pokazao je da početno slabiji rezultati neuronskih mreža ne moraju biti posljedica same arhitekture. Odgovarajućom predobradom i podešavanjem hiperparametara MLP i GraphSAGE mogu ostvariti rezultate vrlo bliske XGBoost modelu.

Posebno je potvrđena važnost kvantilske transformacije za neuronske modele i dužeg treniranja za GraphSAGE.

Međutim, završna vremenska evaluacija pokazala je da visoki rezultati na razvojnim podjelama i prvom testnom koraku ne garantuju uspješnu detekciju nezakonitih transakcija u kasnijim periodima.

Problem vremenske generalizacije ostaje otvoren i predstavlja jednu od glavnih motivacija za nastavak istraživanja.

## Pokretanje

Preporučeno okruženje je **Google Colab**.

Nakon povezivanja Google Drivea, potrebno je otvoriti korijen projekta, odnosno direktorij koji sadrži `pyproject.toml`, i instalirati zavisnosti:

```python
from google.colab import drive
drive.mount("/content/drive")

%cd /content/drive/MyDrive/fingraph-ai/FinGraph_AI_MPVI_Starter
%pip install -e ".[gnn]"
```

Osnovne komande iz korijena projekta:

```bash
python -m fingraph.audit --data-dir data/raw
python -m fingraph.baseline --data-dir data/raw
python -m fingraph.gnn --data-dir data/raw --epochs 60 --hidden-dim 64
python -m fingraph.mlp --data-dir data/raw
```

Za reprodukciju pojedinačnih eksperimenata potrebno je redom pokretati odgovarajuće notebookove.

### Četvrti eksperiment

Notebook:

```text
notebooks/03_phase04_diagnostics.ipynb
```

Sadrži dijagnostiku vremenske generalizacije i mogućnost dodatnog treniranja početnih modela kroz tri hronološke podjele.

Rezultati se spremaju u:

```text
outputs/phase04_diagnostics/
```

Detaljnija dokumentacija dostupna je u `docs/README_EXPERIMENT_04.md`.

### Peti eksperiment

Notebook:

```text
notebooks/04_phase05_ablation_preprocessing_tuning.ipynb
```

Obuhvata:

1. Ablacijsku analizu ulaznih karakteristika.
2. Ispitivanje alternativnih postupaka predobrade.
3. Kombinovanje konfiguracija i provjeru stabilnosti kroz seed vrijednosti.
4. Ciljano podešavanje hiperparametara.
5. Završnu evaluaciju optimizovanih modela i poređenje s početnim rezultatima.

Pojedine faze imaju vlastite kontrolne varijable, pa se već završena treniranja mogu preskočiti prilikom ponovnog izvršavanja notebooka.

Glavni rezultati petog eksperimenta spremaju se u:

```text
outputs/phase05/
├── runs/
├── tables/
└── phaseC_final_test/
    ├── runs/
    ├── tables/
    └── phase05_c4_manifest.json
```

U direktoriju završne evaluacije nalaze se pojedinačne testne predikcije, metrike po vremenskim periodima, historije treniranja i zbirna poređenja s početnim modelima.

Među najvažnijim izlaznim tabelama su:

```text
phase05_c4_all_test_metrics.csv
phase05_c4_summary.csv
phase05_c4_per_step.csv
phase05_c4_original_models_detailed.csv
phase05_c4_vs_original_models.csv
```

Rezultati se mogu arhivirati u `phase05_C4_final_test_results.zip` radi daljnje analize i dokumentovanja.

### Izlazni direktoriji

| Direktorij | Sadržaj |
|---|---|
| `outputs/xgboost/` | Početni XGBoost rezultati |
| `outputs/graphsage/` | Početni GraphSAGE rezultati |
| `outputs/mlp_v1/` | Početni MLP rezultati |
| `outputs/eda/` | EDA grafikoni i tabele |
| `outputs/phase02/` | Vremenska evaluacija i poređenja modela |
| `outputs/phase04_diagnostics/` | Dijagnostika i hronološke podjele |
| `outputs/phase05/` | Ablacije, predobrada, tuning i završna evaluacija |

Direktoriji `data/raw/` i `outputs/` nisu uključeni u Git repozitorij.

## Testovi

Testovi se pokreću iz korijena projekta:

```bash
python -m unittest discover -s tests -v
```

Testovi obuhvataju provjere učitavanja i obrade podataka, oznaka klasa, hronoloških podjela, poravnanja predikcija i osnovnog izvršavanja pojedinih modela na sintetičkim podacima.

## Daljnji rad

Nakon petog eksperimenta naredni istraživački pravac jeste proširenje postojećeg transakcijskog grafa informacijama iz **Elliptic++ Actors dataseta**.

Planirano je konstruisanje heterogenog grafa koji povezuje Bitcoin transakcije i adrese, uz poređenje nekoliko grafovskih arhitektura:

- postojećeg GraphSAGE modela koji koristi samo Transactions graf;
- heterogenog GraphSAGE modela koji koristi Transactions i Actors podatke;
- naprednijeg heterogenog ili hibridnog grafovskog modela koji uključuje attention mehanizam;
- potencijalne kombinacije grafovskih reprezentacija i XGBoost klasifikatora.

Glavno istraživačko pitanje jeste može li dodatni kontekst povezanih Bitcoin adresa poboljšati klasifikaciju nezakonitih transakcija i njihovu stabilnost kroz vrijeme.

Posebna pažnja bit će posvećena provjeri vremenske dostupnosti Actors karakteristika i sprečavanju korištenja budućih informacija (*temporal data leakage*).

Dodatni pravac istraživanja obuhvata vremenski prilagodljive modele, periodično ponovno treniranje i metode koje omogućavaju prilagođavanje promjenama distribucija transakcija.

## Reference

- [Elliptic++ — dataset i izvorni kod](https://github.com/git-disl/EllipticPlusPlus)
- [Elmougy i Liu (2023), Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network for Financial Forensics](https://doi.org/10.1145/3580305.3599803)
- [Weber et al. (2019), Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics](https://arxiv.org/abs/1908.02591)
