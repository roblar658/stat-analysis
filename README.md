# stat-analysis — Native Rust Desktop UI

Et profesjonelt, lyst og høykontrast desktop-program i **ren Rust** (`eframe` + `egui` + `egui_plot` + `rfd`) – helt uten HTML/nettleser – som automatisk og manuelt utfører alle statistiske forutsetningssjekker, modellkvalitetstester, anbefalte tiltak og interaktive visualiseringer direkte fra `.csv`-filer.

## Høydepunkter i brukergrensesnittet
- **Eget Windows-vindu i ren Rust** med ryddig, høykontrast lyst design og responsiv layout (ingen tekst eller tabeller som flyter over kanten).
- **Dra og slipp** en `.csv`-fil rett inn i vinduet (eller klikk **Åpne CSV-fil...** med Windows-fildialog, eller **Eksempeldata**).
- **Sidemeny med trafikklys-indikatorer** (grønn `OK`, gul `ADVARSEL`, rød `BRUTT`) for alle forutsetningssjekker og modellkvalitetstester.
- **Interaktive GPU-rendrede grafer** (zoom, pan, tooltips) for Histogram, Q-Q-plott, Boksplott, Korrelasjons-heatmap, ACF, Residualplott, Leverage/Cook's D, Scree-plott, ROC-kurve, Kalibrering, Schoenfeld-residualer, Elbow/Silhouette og 2D PCA-klynger.
- **Manuell testbenk**: Velg egne kolonner og parametere for alle 9 testfamilier og kjør på nytt når som helst.
- **Eksport med Windows lagringsdialog**: Lagre enkeltvise SVG-figurer direkte fra hver graf, eller eksporter full rapport som **ekte PDF (.pdf)**, **ZIP (PDF + alle SVG)**, **Markdown (.md)** eller **JSON (.json)**.

## Bilder

<img width="589" height="451" alt="image" src="https://github.com/user-attachments/assets/847c8e17-e909-4037-954d-e45d2ba57bd7" />
<img width="959" height="563" alt="image" src="https://github.com/user-attachments/assets/4846dcdf-93a9-403e-a655-dac23f850451" />
<img width="935" height="562" alt="image" src="https://github.com/user-attachments/assets/afb90d1d-9a25-4b1d-a941-0eac006b37a9" />
<img width="959" height="562" alt="image" src="https://github.com/user-attachments/assets/95ed02f7-02b9-4b9c-af88-0ff5168bc3b3" />
<img width="954" height="553" alt="image" src="https://github.com/user-attachments/assets/a7772db8-7051-4e82-b754-7279c66f4bb4" />
<img width="959" height="563" alt="image" src="https://github.com/user-attachments/assets/0dfcd4ef-29f9-47da-b794-33d1b330f133" />
<img width="959" height="558" alt="image" src="https://github.com/user-attachments/assets/80d10ac2-73f8-497f-a875-b758662d5559" />


## Kjøring

### Start desktop-programmet
```powershell
C:\dev\stat-diagnostics\target\release\stat-analysis.exe
```
*(eller dobbeltklikk `stat-analysis.exe` i Filutforsker)*

### Kommandolinje (CLI-modus uten vindu)
```powershell
C:\dev\stat-diagnostics\target\release\stat-analysis.exe --csv examples\comprehensive_study.csv --out min_rapport --alpha 0.05
```
