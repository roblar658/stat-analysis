# stat-analysis — Native Rust Desktop App for Statistical Diagnostics & Model Quality

Et profesjonelt, lyst og høykontrast desktop-program i **ren Rust** (`eframe` + `egui` + `egui_plot` + `rfd`) – helt uten HTML/nettleser – som automatisk og på forespørsel utfører statistiske forutsetningssjekker, velger og kjører riktige hovedanalyser og robuste alternativer, og genererer interaktive grafer og vektorbaserte PDF-rapporter direkte fra `.csv`-filer.

> **Språkstøtte / Bilingual Support:** Bytt mellom **Norsk (NO)** og **English (EN)** direkte i toppmenyen (`Språk: Norsk ▾` / `Language: English ▾`) – både hele brukergrensesnittet, tolkninger, anbefalinger og eksporterte PDF-/Markdown-rapporter oppdateres umiddelbart.

---

## Høydepunkter

- **Eget Windows-vindu i ren Rust**: Raskt, ryddig, høykontrast lyst design med responsiv layout og horisontal scrolling i brede tabeller.
- **Full CSV- og Excel-støtte**: Dra og slipp `.csv`-filer rett inn i vinduet eller bruk **Åpne CSV-fil...**. Støtter UTF-8, UTF-16 LE/BE, norsk Excel Windows-1252 (`æøå`), `sep=;`, semikolon/komma/tab-skilletegn, desimalkomma og tusenskille.
- **Automatisk analyse + kjøring på forespørsel**:
  - Alle relevante forutsetningssjekker og hovedanalyser kjøres automatisk ved innlasting.
  - Klikk **Kjør alle tester** (*Run All Tests*) i toppfeltet når som helst for å kjøre hele diagnosesuiten på nytt med valgt signifikansnivå ($\alpha \in \{0.01, 0.05, 0.10\}$).
- **Automatisk valg av riktig hovedanalyse**: Sjekker først normalitet, varianshomogenitet, sfærisitet og regresjonsforutsetninger — og utfører deretter riktig test (f.eks. **Student uavhengig t-test** med Cohen's $d$, **Welchs t-test**, **Mann-Whitney U**, **Standard enveis ANOVA** med $\eta^2$, **Welchs ANOVA**, **Kruskal-Wallis H**, **RM-ANOVA (GG/HF)** eller **OLS / HC3 / Newey-West / Ridge**).
- **Manuell testbenk (*Manual Workbench*)**:
  - Kjør valgfrie tester med egne kolonner og parametere på tvers av alle 9 testfamilier.
  - Klikk **Kjør alle relevante tester → Appendiks** for å utføre og logge alle relevante tester rett inn i PDF-appendikset med ett klikk.
- **Ekte flersiders vektor-PDF (`.pdf`) med to appendiks**:
  - Full tekstbryting i alle tabeller (ingen avkuttede celler), innebygde vektorgrafer og sidefot.
  - **APPENDIKS — Matematiske formler og beregninger**: Komplett formeloversikt for alle statistiske metoder samt innsatte tallverdier fra datasettet.
  - **APPENDIKS — Manuelle utregninger**: Alle manuelle utregninger fra testbenken (`Appendiks A.1 .. A.N`).
- **Fleksibel eksport**: Lagre enkeltvise `.svg`-figurer direkte fra hver graf, eller eksporter **PDF (`.pdf`)**, **ZIP-pakke (PDF + Markdown + JSON + alle SVG-figurer)**, **Markdown (`.md`)** eller **JSON (`.json`)**.
- **10 innebygde standard datasett**: Utforsk alle forutsetningsbrudd, tiltak og modelltester umiddelbart fra startsiden eller toppmenyen.

---

## Statistiske forutsetningssjekker, hovedanalyser og innebygde tiltak

| Kategori | Tester og diagnostikk | Automatisk utførte hovedanalyser & anbefalte tiltak |
|---|---|---|
| **Normalfordeling** | Shapiro-Wilk ($n < 50$), Kolmogorov-Smirnov / Lilliefors ($n \ge 50$), skjevhet, eksess-kurtose, Histogram & Q-Q-plott | Mann-Whitney U, Kruskal-Wallis H, Bootstrapping (95 % persentil-KI) |
| **Varianshomogenitet & Grupper** | Levenes test (median / Brown-Forsythe & snitt), Bartletts test, Boksplott | Student uavhengig t-test (Cohen's $d$), Standard enveis ANOVA ($\eta^2$), Welchs t-test, Welchs ANOVA, Mann-Whitney U, Kruskal-Wallis H |
| **Sfærisitet (Repeterte målinger)** | Mauchlys test ($W$, $\chi^2$, $p$), Profilplott | Ukorrigert RM-ANOVA, Greenhouse-Geisser ($\varepsilon_{GG}$), Huynh-Feldt ($\varepsilon_{HF}$) |
| **Multikollinearitet** | VIF per prediktor, Korrelasjonsmatrise (Heatmap, $\|r\| > 0.8$) | Standardisert Ridge-regresjon ($L_2$), Lasso-regresjon ($L_1$) |
| **Autokorrelasjon** | Durbin-Watson, Breusch-Godfrey LM-test, ACF-plott | Newey-West HAC robuste standardfeil, AR(1)-modell ($\phi$, RMSE) |
| **Heteroskedastisitet** | Breusch-Pagan-test, Whites test, Residualplott | HC3 robuste standardfeil, automatisk $\ln(Y)$-transformasjon |
| **Uteliggere & innflytelse** | Cook's Distance ($D > 1$, $D > 4/n$), Mahalanobis $D^2$, Leverage-plott | Automatisk sensitivitetsanalyse (med og uten innflytelsesrike observasjoner) |
| **Manglende data** | Mønsteranalyse, Littles MCAR-test | EM-estimering (FIML) og multippel imputering (MICE med Rubins regler) |
| **Faktoranalyse / PCA** | Samlet KMO, individuell MSA, Bartletts sfærisitetstest, PCA Scree-plott | Forslag til fjerning av svake variabler med ny beregnet KMO |
| **Lineær regresjon** | $R^2$, Justert $R^2$, RMSE, MAE, $F$-test, koeffisienter, residualnormalitet | Modellsammenligning med AIC & BIC (full, null, redusert, kun signifikante) |
| **Logistisk regresjon** | IRLS-estimering, Odds Ratio, Hosmer-Lemeshow goodness-of-fit, kalibreringsplott, ROC-kurve & AUC | Optimal klassifiseringsgrense (Youden's J), konfusjonsmatrise, F1, AIC & BIC |
| **Overlevelsesanalyse (Cox PH)** | Cox proporsjonal hasard (Breslow), Hazard Ratios, Harrells C-index, Schoenfeld-residualer | Tidsavhengige hazard ratios (`HR(t25)` tidlig vs. `HR(t75)` sen) ved brutt PH |
| **Klyngeanalyse** | K-means++, Silhouette-skår, Elbow-metoden (WCSS), 2D PCA-projeksjon | Automatisk DBSCAN-alternativ (tetthetsbasert klynging + støyidentifikasjon) |

---

## 10 innebygde eksempeldatasett (`examples/`)

1. `01_komplett_studie.csv` — Komplett helsestudie (alle tester samlet)
2. `02_normalfordeling_sma_utvalg.csv` — Normalfordeling & små utvalg ($n = 36$, Shapiro-Wilk, Mann-Whitney, Bootstrap)
3. `03_varians_grupper_welch.csv` — Ulik varians i 3 grupper (Levene, Bartlett, Welch ANOVA, Kruskal-Wallis)
4. `04_repeterte_maalinger_sfaerisitet.csv` — Repeterte målinger over 4 tidspunkt (Mauchly, GG/HF RM-ANOVA)
5. `05_boligpriser_multikollinearitet.csv` — Boligpriser (høy VIF, Ridge/Lasso, Cook's D, sensitivitetsanalyse)
6. `06_tidsserie_autokorrelasjon.csv` — Tidsserie med sterk AR(1)-autokorrelasjon (DW, BG, ACF, Newey-West HAC)
7. `07_inntekt_heteroskedastisitet.csv` — Inntektsdata med heteroskedastisitet (BP, White, HC3, Log(Y))
8. `08_undersokelse_manglende_faktor.csv` — Spørreundersøkelse (Littles MCAR, MICE/FIML, KMO, Bartlett, Scree)
9. `09_kreditt_logistisk_roc.csv` — Kredittrisiko (Logistisk regresjon, Hosmer-Lemeshow, ROC & AUC)
10. `10_klinisk_overlevelse_klynger.csv` — Klinisk overlevelse & pasientklynger (Cox PH, Schoenfeld, K-means++, DBSCAN)

---

## Skjermbilder

<img width="959" height="563" alt="image" src="https://github.com/user-attachments/assets/4846dcdf-93a9-403e-a655-dac23f850451" />
<img width="935" height="562" alt="image" src="https://github.com/user-attachments/assets/afb90d1d-9a25-4b1d-a941-0eac006b37a9" />
<img width="959" height="562" alt="image" src="https://github.com/user-attachments/assets/95ed02f7-02b9-4b9c-af88-0ff5168bc3b3" />
<img width="954" height="553" alt="image" src="https://github.com/user-attachments/assets/a7772db8-7051-4e82-b754-7279c66f4bb4" />
<img width="959" height="563" alt="image" src="https://github.com/user-attachments/assets/0dfcd4ef-29f9-47da-b794-33d1b330f133" />
<img width="959" height="558" alt="image" src="https://github.com/user-attachments/assets/80d10ac2-73f8-497f-a875-b758662d5559" />

---

## Installasjon på Windows (WiX v4 MSI)

Prosjektet inkluderer en komplett WiX v4-installasjonspakke ([wix/Package.wxs](wix/Package.wxs), [wix/build-msi.ps1](wix/build-msi.ps1)) som installerer `stat-analysis` til `%LOCALAPPDATA%\Programs\stat-analysis\` med snarveier i Start-menyen og på Skrivebordet (krever ikke administratorrettigheter).

### Bygg MSI-installasjonsfil
```powershell
powershell -ExecutionPolicy Bypass -File .\wix\build-msi.ps1
```
Dette genererer `dist\stat-analysis-0.1.0-x64.msi`.

### Installer fra kommandolinje (valgfritt)
```powershell
msiexec /i .\dist\stat-analysis-0.1.0-x64.msi
```

---

## Bygging, testing og kjøring

### Bygg og kjør tester
```powershell
cargo test --release
cargo build --release
```

### Start desktop-programmet
```powershell
.\target\release\stat-analysis.exe
```
*(eller start **stat-analysis** fra Start-menyen / Skrivebordet etter MSI-installasjon)*

### Kommandolinje (CLI-modus uten vindu)
Kjør full automatisk analyse og lagre PDF, Markdown, JSON, SVG-figurer og ZIP direkte fra terminal:
```powershell
# Norsk rapport (standard)
.\target\release\stat-analysis.exe --csv examples\01_komplett_studie.csv --out min_rapport --alpha 0.05 --lang no

# Engelsk rapport
.\target\release\stat-analysis.exe --csv examples\01_komplett_studie.csv --out report_en --alpha 0.05 --lang en
```

---

## Lisens

Lisensiert til **Robert Larsen** (se [wix/License.rtf](wix/License.rtf)).
