# Dataset i izvorne reference

## Elliptic++ — primarni dataset

Autori: Youssef Elmougy, Ling Liu (ACM SIGKDD 2023).

- Repozitorij: https://github.com/git-disl/EllipticPlusPlus
- Podaci: https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l?usp=sharing
- Rad: https://doi.org/10.1145/3580305.3599803
- Deklarisano: 203.769 transakcija, 234.355 veza, 49 vremenskih koraka,
  183 karakteristike, 4.545 illicit, 42.019 licit, 157.205 unknown.
- Preuzimanje: originalni CSV s Google Drivea; GitHub CSV nazivi mogu biti
  samo Git LFS tekstualni pokazivači.
- Licenca: repo trenutno ne izlaže eksplicitno standardno licencno polje za
  dataset; provjeriti uslove redistribucije prije objavljivanja sirovih podataka.

## Pretpostavljeni CSV interfejs (potvrđen iz notebooka autora)

| Datoteka | Obavezne kolone | Namjena |
|---|---|---|
| `txs_features.csv` | `txId`, `Time step`, numeričke kolone | Karakteristike čvorova |
| `txs_classes.csv` | `txId`, `class` | Oznake 1/2/3 |
| `txs_edgelist.csv` | `txId1`, `txId2` | Usmjerene veze |

Datoteke treba **spojiti po `txId`**, ne po rednom broju. Oznaka `class` se
odvaja prije treniranja; Time step se koristi za podjelu, ne kao ciljna oznaka.

## IBM AML — sekundarni dataset (nije dio obaveznog protokola)

- https://github.com/IBM/AML-Data
- https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
- Sintetički bankarski transferi s oznakom laundering.
- IBM naglašava da je licenca **podataka** CDLA-Sharing-1.0, dok repo kod ima
  Apache-2.0. Ne uspoređivati sirove metrike s Elliptic++ kao istu domenu.

## BiH — izvori za tipologije i moguća zasebna analiza

- SIPA FOO izvještaji: https://www.sipa.gov.ba/bs/dokumenti/izvjestaji-foo
- SIPA FOO tipologije 2023 (str. 20–22):
  https://sipa.gov.ba/assets/files/foo-docs/FOOIZVJESTAJ2023.pdf
- Portal e-Nabavke: https://www.ejn.gov.ba/
- Centralna banka BiH: https://www.cbbh.ba/
- ICIJ Offshore Leaks CSV graf: https://offshoreleaks.icij.org/pages/database

**Ograničenje:** nijedan od ovih BiH izvora nije potvrđen kao javni skup
označenih pojedinačnih bankarskih AML transakcija. Za domaći dio možemo
izdvojiti tipologije ili na sintetičkom grafu testirati obrasce. ICIJ
pripadnost mreži nije dokaz nezakonitosti.

## Uvodna literatura

1. Weber et al. (2019), https://arxiv.org/abs/1908.02591
2. Elmougy & Liu (2023), https://doi.org/10.1145/3580305.3599803
3. Bellei et al. (2024), https://arxiv.org/abs/2404.19109
4. LaundroGraph (2022), https://doi.org/10.1145/3533271.3561727
5. GNNExplainer (2019), https://arxiv.org/abs/1903.03894
