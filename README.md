# stat-analysis — Native Rust Desktop UI

Et profesjonelt, lyst og høykontrast desktop-program i **ren Rust** (`eframe` + `egui` + `egui_plot` + `rfd`) – helt uten HTML/nettleser – som automatisk og manuelt utfører alle statistiske forutsetningssjekker, modellkvalitetstester, anbefalte tiltak og interaktive visualiseringer direkte fra `.csv`-filer.

## Høydepunkter i brukergrensesnittet
- **Eget Windows-vindu i ren Rust** med ryddig, høykontrast lyst design og responsiv layout (ingen tekst eller tabeller som flyter over kanten).
- **Dra og slipp** en `.csv`-fil rett inn i vinduet (eller klikk **Åpne CSV-fil...** med Windows-fildialog, eller **Eksempeldata**).
- **Sidemeny med trafikklys-indikatorer** (grønn `OK`, gul `ADVARSEL`, rød `BRUTT`) for alle forutsetningssjekker og modellkvalitetstester.
- **Interaktive GPU-rendrede grafer** (zoom, pan, tooltips) for Histogram, Q-Q-plott, Boksplott, Korrelasjons-heatmap, ACF, Residualplott, Leverage/Cook's D, Scree-plott, ROC-kurve, Kalibrering, Schoenfeld-residualer, Elbow/Silhouette og 2D PCA-klynger.
- **Manuell testbenk**: Velg egne kolonner og parametere for alle 9 testfamilier og kjør på nytt når som helst.
- **Eksport med Windows lagringsdialog**: Lagre enkeltvise SVG-figurer direkte fra hver graf, eller eksporter full rapport som **ekte PDF (.pdf)**, **ZIP (PDF + alle SVG)**, **Markdown (.md)** eller **JSON (.json)**.

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
