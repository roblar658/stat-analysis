pub mod charts;
pub mod manual;
pub mod views;

use crate::auto::{run_automatic_analysis, AutoAnalysisReport};
use crate::data::Dataset;
use crate::report::{
    build_zip_bundle_with_appendix, render_markdown_report_with_appendix,
    render_pdf_report_with_appendix,
};
use crate::ui::charts::{
    primary_button, responsive_two_col, secondary_button, show_card, status_dot_color,
    BORDER_DEFAULT, PRIMARY_BLUE, SURFACE_CARD, TEXT_PRIMARY, TEXT_SECONDARY,
};
use crate::ui::manual::ManualWorkbench;
use crate::ui::views::{
    render_autocorrelation, render_clustering, render_dashboard, render_data_explorer,
    render_factor, render_heteroscedasticity, render_linear_regression, render_logistic,
    render_missing, render_multicollinearity, render_normality, render_outliers,
    render_sphericity, render_survival, render_variance, ViewState,
};
use egui::{Color32, Context, FontId, Frame, Margin, RichText, Stroke, TextStyle, Visuals};
use std::path::Path;

pub struct ExampleDataset {
    pub id: &'static str,
    pub filename: &'static str,
    pub title: &'static str,
    pub focus: &'static str,
    pub csv: &'static str,
}

pub const EXAMPLE_DATASETS: &[ExampleDataset] = &[
    ExampleDataset {
        id: "ex1",
        filename: "01_komplett_studie.csv",
        title: "1. Komplett helsestudie",
        focus: "Alle tester samlet: grupper, kollinearitet, heteroskedastisitet, uteligger, repeterte målinger, manglende, logistisk, Cox og klynger",
        csv: include_str!("../../examples/01_komplett_studie.csv"),
    },
    ExampleDataset {
        id: "ex2",
        filename: "02_normalfordeling_sma_utvalg.csv",
        title: "2. Normalfordeling & små utvalg (n=36)",
        focus: "Shapiro-Wilk (< 50), Q-Q-plott, høyreskjev ventetid, Mann-Whitney U og Bootstrapping",
        csv: include_str!("../../examples/02_normalfordeling_sma_utvalg.csv"),
    },
    ExampleDataset {
        id: "ex3",
        filename: "03_varians_grupper_welch.csv",
        title: "3. Ulik varians i grupper",
        focus: "Levenes test, Bartletts test, boksplott, Welchs ANOVA og Kruskal-Wallis på 3 behandlingsgrupper",
        csv: include_str!("../../examples/03_varians_grupper_welch.csv"),
    },
    ExampleDataset {
        id: "ex4",
        filename: "04_repeterte_maalinger_sfaerisitet.csv",
        title: "4. Repeterte målinger (Sfærisitet)",
        focus: "Mauchlys test, RM-ANOVA med Greenhouse-Geisser- og Huynh-Feldt-korreksjon over 4 målinger",
        csv: include_str!("../../examples/04_repeterte_maalinger_sfaerisitet.csv"),
    },
    ExampleDataset {
        id: "ex5",
        filename: "05_boligpriser_multikollinearitet.csv",
        title: "5. Boligpriser (Multikollinearitet & Uteliggere)",
        focus: "Høy VIF, korrelasjonsmatrise, Ridge/Lasso, Cook's Distance, sensitivitetsanalyse og AIC/BIC",
        csv: include_str!("../../examples/05_boligpriser_multikollinearitet.csv"),
    },
    ExampleDataset {
        id: "ex6",
        filename: "06_tidsserie_autokorrelasjon.csv",
        title: "6. Tidsserie (Autokorrelasjon)",
        focus: "Sterk AR(1) autokorrelasjon: Durbin-Watson, Breusch-Godfrey, ACF-plott og Newey-West HAC",
        csv: include_str!("../../examples/06_tidsserie_autokorrelasjon.csv"),
    },
    ExampleDataset {
        id: "ex7",
        filename: "07_inntekt_heteroskedastisitet.csv",
        title: "7. Inntekt (Heteroskedastisitet)",
        focus: "Breusch-Pagan, Whites test, residualplott, HC3 robuste standardfeil og Log(Y)-transformasjon",
        csv: include_str!("../../examples/07_inntekt_heteroskedastisitet.csv"),
    },
    ExampleDataset {
        id: "ex8",
        filename: "08_undersokelse_manglende_faktor.csv",
        title: "8. Spørreundersøkelse (MCAR & Faktoranalyse)",
        focus: "Systematiske manglende verdier (Littles MCAR, MICE/FIML) + 2 faktorer (KMO, Bartlett, Scree)",
        csv: include_str!("../../examples/08_undersokelse_manglende_faktor.csv"),
    },
    ExampleDataset {
        id: "ex9",
        filename: "09_kreditt_logistisk_roc.csv",
        title: "9. Kredittrisiko (Logistisk & ROC/AUC)",
        focus: "Logistisk regresjon, Hosmer-Lemeshow-tilpasning, kalibreringsplott, ROC-kurve og AUC",
        csv: include_str!("../../examples/09_kreditt_logistisk_roc.csv"),
    },
    ExampleDataset {
        id: "ex10",
        filename: "10_klinisk_overlevelse_klynger.csv",
        title: "10. Klinisk overlevelse & Pasientklynger",
        focus: "Cox proporsjonal hasard, Schoenfeld-residualer, K-means++ Silhouette-skår og Elbow-metoden",
        csv: include_str!("../../examples/10_klinisk_overlevelse_klynger.csv"),
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavSection {
    Home,
    Dashboard,
    Normality,
    Variance,
    Sphericity,
    Multicollinearity,
    Autocorrelation,
    Heteroscedasticity,
    Outliers,
    MissingData,
    FactorSuitability,
    LinearRegression,
    LogisticRegression,
    Survival,
    Clustering,
    ManualWorkbench,
    DataExplorer,
}

pub struct StatApp {
    dataset: Option<Dataset>,
    report: Option<AutoAnalysisReport>,
    alpha: f64,
    nav: NavSection,
    view_state: ViewState,
    manual: ManualWorkbench,
    status_toast: Option<(String, bool)>,
}

impl StatApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        apply_light_theme(&cc.egui_ctx);
        Self {
            dataset: None,
            report: None,
            alpha: 0.05,
            nav: NavSection::Home,
            view_state: ViewState::default(),
            manual: ManualWorkbench::default(),
            status_toast: None,
        }
    }

    fn load_csv_content(&mut self, filename: &str, content: &str) {
        match Dataset::from_csv_str(filename, content) {
            Ok(ds) => {
                let rep = run_automatic_analysis(&ds, self.alpha);
                self.manual.sync_with_dataset(&ds);
                self.view_state = ViewState::default();
                self.status_toast = Some((
                    format!(
                        "Analyserte '{}' ({} rader × {} kolonner) automatisk.",
                        filename,
                        ds.row_count(),
                        ds.headers.len()
                    ),
                    false,
                ));
                self.dataset = Some(ds);
                self.report = Some(rep);
                self.nav = NavSection::Dashboard;
            }
            Err(e) => {
                self.status_toast = Some((format!("Feil ved lesing av CSV: {}", e), true));
            }
        }
    }

    fn load_csv_path(&mut self, path: &Path) {
        match std::fs::read(path) {
            Ok(bytes) => {
                let fname = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("data.csv");
                let decoded = crate::data::decode_csv_bytes(&bytes);
                self.load_csv_content(fname, &decoded);
            }
            Err(e) => {
                self.status_toast = Some((format!("Klarte ikke åpne fil: {}", e), true));
            }
        }
    }

    fn reanalyze_with_alpha(&mut self) {
        if let Some(ref ds) = self.dataset {
            self.report = Some(run_automatic_analysis(ds, self.alpha));
            self.status_toast = Some((format!("Oppdaterte alle tester med α = {:.2}", self.alpha), false));
        }
    }

    fn export_zip(&mut self) {
        let Some(ref rep) = self.report else { return };
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name("stat_analysis_rapport_og_figurer.zip")
            .add_filter("ZIP-arkiv", &["zip"])
            .save_file()
        {
            match build_zip_bundle_with_appendix(rep, &self.manual.history)
                .and_then(|bytes| std::fs::write(&path, bytes).map_err(|e| e.to_string()))
            {
                Ok(_) => self.status_toast = Some((format!("Lagret ZIP-pakke til {}", path.display()), false)),
                Err(e) => self.status_toast = Some((format!("Eksportfeil: {}", e), true)),
            }
        }
    }

    fn export_markdown(&mut self) {
        let Some(ref rep) = self.report else { return };
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name("stat_analysis_rapport.md")
            .add_filter("Markdown", &["md"])
            .save_file()
        {
            match std::fs::write(&path, render_markdown_report_with_appendix(rep, &self.manual.history)) {
                Ok(_) => self.status_toast = Some((format!("Lagret Markdown-rapport til {}", path.display()), false)),
                Err(e) => self.status_toast = Some((format!("Eksportfeil: {}", e), true)),
            }
        }
    }

    fn export_pdf(&mut self) {
        let Some(ref rep) = self.report else { return };
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name("stat_analysis_rapport.pdf")
            .add_filter("PDF-dokument", &["pdf"])
            .save_file()
        {
            let n_app = self.manual.history.len();
            match std::fs::write(&path, render_pdf_report_with_appendix(rep, &self.manual.history)) {
                Ok(_) => {
                    let msg = if n_app > 0 {
                        format!("Lagret PDF-rapport (inkl. {} manuelle utregninger i Appendiks) til {}", n_app, path.display())
                    } else {
                        format!("Lagret PDF-rapport til {}", path.display())
                    };
                    self.status_toast = Some((msg, false));
                }
                Err(e) => self.status_toast = Some((format!("Eksportfeil: {}", e), true)),
            }
        }
    }

    fn export_json(&mut self) {
        let Some(ref rep) = self.report else { return };
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name("stat_analysis_resultater.json")
            .add_filter("JSON", &["json"])
            .save_file()
        {
            match serde_json::to_string_pretty(rep)
                .map_err(|e| e.to_string())
                .and_then(|s| std::fs::write(&path, s).map_err(|e| e.to_string()))
            {
                Ok(_) => self.status_toast = Some((format!("Lagret JSON til {}", path.display()), false)),
                Err(e) => self.status_toast = Some((format!("Eksportfeil: {}", e), true)),
            }
        }
    }

    fn section_status(&self, sec: NavSection) -> Option<&str> {
        let rep = self.report.as_ref()?;
        match sec {
            NavSection::Normality => {
                if rep.normality_checks.is_empty() {
                    None
                } else if rep.normality_checks.iter().all(|n| n.is_normal) {
                    Some("ok")
                } else {
                    Some("warning")
                }
            }
            NavSection::Variance => rep.variance_homogeneity.as_ref().map(|v| v.status.as_str()),
            NavSection::Sphericity => rep.sphericity.as_ref().map(|s| s.status.as_str()),
            NavSection::Multicollinearity => rep
                .linear_regression
                .as_ref()
                .and_then(|r| r.multicollinearity.as_ref())
                .map(|m| m.status.as_str()),
            NavSection::Autocorrelation => rep
                .linear_regression
                .as_ref()
                .map(|r| r.autocorrelation.status.as_str()),
            NavSection::Heteroscedasticity => rep
                .linear_regression
                .as_ref()
                .map(|r| r.heteroscedasticity.status.as_str()),
            NavSection::Outliers => rep
                .linear_regression
                .as_ref()
                .map(|r| r.outliers.status.as_str()),
            NavSection::MissingData => Some(rep.missing_data.status.as_str()),
            NavSection::FactorSuitability => rep.factor_suitability.as_ref().map(|f| f.status.as_str()),
            NavSection::LinearRegression => rep.linear_regression.as_ref().map(|r| {
                if r.adj_r_squared >= 0.5 {
                    "ok"
                } else if r.adj_r_squared >= 0.2 {
                    "warning"
                } else {
                    "fail"
                }
            }),
            NavSection::LogisticRegression => rep.logistic_regression.as_ref().map(|l| l.auc_status.as_str()),
            NavSection::Survival => rep.cox_survival.as_ref().map(|c| c.status.as_str()),
            NavSection::Clustering => rep.clustering.as_ref().map(|c| c.status.as_str()),
            _ => None,
        }
    }
}

/// Load native Windows Segoe UI Regular + Bold fonts when available (with safe fallback)
/// so headings and bold labels before ':' render with true bold font weight.
fn configure_fonts(ctx: &Context) {
    let mut fonts = egui::FontDefinitions::default();

    if let Ok(reg_bytes) = std::fs::read(r"C:\Windows\Fonts\segoeui.ttf") {
        fonts.font_data.insert(
            "segoe_regular".to_owned(),
            std::sync::Arc::new(egui::FontData::from_owned(reg_bytes)),
        );
        fonts
            .families
            .entry(egui::FontFamily::Proportional)
            .or_default()
            .insert(0, "segoe_regular".to_owned());
    }

    let bold_family = egui::FontFamily::Name("bold".into());
    if let Ok(bold_bytes) = std::fs::read(r"C:\Windows\Fonts\segoeuib.ttf") {
        fonts.font_data.insert(
            "segoe_bold".to_owned(),
            std::sync::Arc::new(egui::FontData::from_owned(bold_bytes)),
        );
        fonts
            .families
            .insert(bold_family, vec!["segoe_bold".to_owned()]);
    } else {
        let fallback = fonts
            .families
            .get(&egui::FontFamily::Proportional)
            .cloned()
            .unwrap_or_default();
        fonts.families.insert(bold_family, fallback);
    }

    ctx.set_fonts(fonts);
}

/// High-contrast light theme ONLY (WCAG AA/AAA readable text, crisp borders, no dark mode)
fn apply_light_theme(ctx: &Context) {
    configure_fonts(ctx);
    let mut visuals = Visuals::light();
    visuals.window_corner_radius = 10.0_f32.into();
    visuals.override_text_color = Some(TEXT_PRIMARY);
    visuals.panel_fill = Color32::from_rgb(241, 245, 249); // #F1F5F9
    visuals.window_fill = Color32::WHITE;
    visuals.extreme_bg_color = Color32::WHITE;
    visuals.faint_bg_color = Color32::from_rgb(248, 250, 252);

    // High-contrast selection (soft blue background + dark navy text)
    visuals.selection.bg_fill = Color32::from_rgb(219, 234, 254); // #DBEAFE
    visuals.selection.stroke = Stroke::new(1.2_f32, Color32::from_rgb(30, 58, 138));

    // Crisp, readable widgets
    visuals.widgets.noninteractive.bg_fill = Color32::WHITE;
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, TEXT_PRIMARY);
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, BORDER_DEFAULT);

    visuals.widgets.inactive.corner_radius = 6.0_f32.into();
    visuals.widgets.inactive.bg_fill = Color32::WHITE;
    visuals.widgets.inactive.weak_bg_fill = Color32::WHITE;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, Color32::from_rgb(203, 213, 225));
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, TEXT_PRIMARY);

    visuals.widgets.hovered.corner_radius = 6.0_f32.into();
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(219, 234, 254);
    visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(219, 234, 254);
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.2_f32, PRIMARY_BLUE);
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.2_f32, Color32::from_rgb(30, 58, 138));

    visuals.widgets.active.corner_radius = 6.0_f32.into();
    visuals.widgets.active.bg_fill = Color32::from_rgb(191, 219, 254);
    visuals.widgets.active.weak_bg_fill = Color32::from_rgb(191, 219, 254);
    visuals.widgets.active.bg_stroke = Stroke::new(1.4_f32, PRIMARY_BLUE);
    visuals.widgets.active.fg_stroke = Stroke::new(1.4_f32, Color32::from_rgb(30, 58, 138));

    ctx.set_visuals(visuals);

    let mut style = (*ctx.style()).clone();
    style.wrap_mode = Some(egui::TextWrapMode::Wrap);
    style.text_styles.insert(TextStyle::Body, FontId::proportional(13.5));
    style.text_styles.insert(TextStyle::Button, FontId::proportional(13.0));
    style.text_styles.insert(TextStyle::Heading, FontId::proportional(18.0));
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(10.0, 6.0);
    ctx.set_style(style);
}

impl eframe::App for StatApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        // Handle Drag & Drop CSV files directly onto the native window
        let dropped: Vec<std::path::PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        if let Some(path) = dropped.first() {
            self.load_csv_path(path);
        }

        // Top Toolbar (wraps cleanly on narrower windows)
        egui::TopBottomPanel::top("top_bar")
            .frame(
                Frame::new()
                    .fill(SURFACE_CARD)
                    .stroke(Stroke::new(1.0_f32, BORDER_DEFAULT))
                    .inner_margin(Margin::symmetric(16, 10)),
            )
            .show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    if home_icon_button(ui, self.nav == NavSection::Home)
                        .on_hover_text("Tilbake til startsiden")
                        .clicked()
                    {
                        self.nav = NavSection::Home;
                    }
                    if ui
                        .add(
                            egui::Label::new(
                                RichText::new("stat-analysis")
                                    .font(FontId::new(18.5, egui::FontFamily::Name("bold".into())))
                                    .color(PRIMARY_BLUE),
                            )
                            .sense(egui::Sense::click()),
                        )
                        .on_hover_text("Tilbake til startsiden")
                        .clicked()
                    {
                        self.nav = NavSection::Home;
                    }
                    ui.separator();

                    if primary_button(ui, "Åpne CSV-fil...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("CSV-filer", &["csv", "txt"])
                            .pick_file()
                        {
                            self.load_csv_path(&path);
                        }
                    }

                    ui.menu_button(
                        RichText::new("Eksempelsett (10) ▾").strong().color(TEXT_PRIMARY),
                        |ui| {
                            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                            for ex in EXAMPLE_DATASETS {
                                if ui.button(ex.title).on_hover_text(ex.focus).clicked() {
                                    ui.close_menu();
                                    self.load_csv_content(ex.filename, ex.csv);
                                }
                            }
                        },
                    );

                    ui.separator();
                    ui.label(RichText::new("Signifikansnivå α:").color(TEXT_PRIMARY).strong());
                    let prev_alpha = self.alpha;
                    egui::ComboBox::from_id_salt("alpha_combo")
                        .width(66.0)
                        .selected_text(format!("{:.2}", self.alpha))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.alpha, 0.01, "0.01");
                            ui.selectable_value(&mut self.alpha, 0.05, "0.05");
                            ui.selectable_value(&mut self.alpha, 0.10, "0.10");
                        });
                    if (self.alpha - prev_alpha).abs() > 1e-9 {
                        self.reanalyze_with_alpha();
                    }

                    if self.report.is_some() {
                        ui.separator();
                        if secondary_button(ui, "Kjør alle tester").on_hover_text("Kjør hele den automatiske testpakken på nytt med gjeldende α").clicked() {
                            if let Some(ref ds) = self.dataset {
                                let new_rep = run_automatic_analysis(ds, self.alpha);
                                let msg = format!(
                                    "Kjørte alle {} forutsetningssjekker og {} hovedanalyser på nytt (α = {:.2}).",
                                    new_rep.summary_badges.len(),
                                    new_rep.recommended_analyses.len(),
                                    self.alpha
                                );
                                self.report = Some(new_rep);
                                self.status_toast = Some((msg, false));
                            }
                        }
                        let app_suffix = if self.manual.history.is_empty() {
                            String::new()
                        } else {
                            format!(" — inkl. {} i Appendiks", self.manual.history.len())
                        };
                        ui.menu_button(
                            RichText::new("Eksporter rapport / SVG ▾").strong().color(TEXT_PRIMARY),
                            |ui| {
                                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                                if ui.button(format!("Lagre PDF-rapport (.pdf{})...", app_suffix)).clicked() {
                                    ui.close_menu();
                                    self.export_pdf();
                                }
                                if ui.button(format!("Lagre alt som ZIP (PDF{} + alle SVG-figurer)...", app_suffix)).clicked() {
                                    ui.close_menu();
                                    self.export_zip();
                                }
                                if ui.button("Lagre Markdown-rapport (.md)...").clicked() {
                                    ui.close_menu();
                                    self.export_markdown();
                                }
                                if ui.button("Lagre rådata / resultater (.json)...").clicked() {
                                    ui.close_menu();
                                    self.export_json();
                                }
                            },
                        );
                    }
                });
            });

        // Bottom Status Bar
        if let Some((ref msg, is_err)) = self.status_toast {
            egui::TopBottomPanel::bottom("status_bar")
                .frame(
                    Frame::new()
                        .fill(if is_err {
                            Color32::from_rgb(254, 226, 226)
                        } else {
                            Color32::from_rgb(219, 234, 254)
                        })
                        .stroke(Stroke::new(
                            1.0_f32,
                            if is_err {
                                Color32::from_rgb(252, 165, 165)
                            } else {
                                Color32::from_rgb(147, 197, 253)
                            },
                        ))
                        .inner_margin(Margin::symmetric(16, 6)),
                )
                .show(ctx, |ui| {
                    ui.label(
                        RichText::new(msg)
                            .size(12.5)
                            .strong()
                            .color(if is_err {
                                Color32::from_rgb(127, 29, 29)
                            } else {
                                Color32::from_rgb(30, 58, 138)
                            }),
                    );
                });
        }

        // Left Navigation Sidebar (when dataset is loaded and not on start screen)
        if self.dataset.is_some() && self.nav != NavSection::Home {
            egui::SidePanel::left("sidebar")
                .resizable(true)
                .default_width(245.0)
                .min_width(205.0)
                .max_width(310.0)
                .frame(
                    Frame::new()
                        .fill(SURFACE_CARD)
                        .stroke(Stroke::new(1.0_f32, BORDER_DEFAULT))
                        .inner_margin(Margin::same(10)),
                )
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        self.nav_button(ui, NavSection::Dashboard, "Oversikt (Dashboard)");
                        ui.add_space(10.0);
                        ui.label(RichText::new("FORUTSETNINGER").size(10.5).strong().color(TEXT_SECONDARY));
                        self.nav_button(ui, NavSection::Normality, "1. Normalfordeling");
                        self.nav_button(ui, NavSection::Variance, "2. Varianshomogenitet");
                        self.nav_button(ui, NavSection::Sphericity, "3. Sfærisitet");
                        self.nav_button(ui, NavSection::Multicollinearity, "4. Multikollinearitet");
                        self.nav_button(ui, NavSection::Autocorrelation, "5. Autokorrelasjon");
                        self.nav_button(ui, NavSection::Heteroscedasticity, "6. Heteroskedastisitet");
                        self.nav_button(ui, NavSection::Outliers, "7. Uteliggere & innflytelse");
                        self.nav_button(ui, NavSection::MissingData, "8. Manglende data (MCAR)");
                        self.nav_button(ui, NavSection::FactorSuitability, "9. Faktoranalyse (KMO)");

                        ui.add_space(10.0);
                        ui.label(RichText::new("MODELLKVALITET").size(10.5).strong().color(TEXT_SECONDARY));
                        self.nav_button(ui, NavSection::LinearRegression, "10. Lineær regresjon & AIC");
                        self.nav_button(ui, NavSection::LogisticRegression, "11. Logistisk (HL / ROC)");
                        self.nav_button(ui, NavSection::Survival, "12. Overlevelse (Cox PH)");
                        self.nav_button(ui, NavSection::Clustering, "13. Klyngeanalyse");

                        ui.add_space(10.0);
                        ui.label(RichText::new("VERKTØY").size(10.5).strong().color(TEXT_SECONDARY));
                        self.nav_button(ui, NavSection::ManualWorkbench, "Manuell testbenk");
                        self.nav_button(ui, NavSection::DataExplorer, "Datautforsker");
                    });
                });
        }

        // Main Central Panel (strictly clipped to panel width so nothing overflows right edge)
        egui::CentralPanel::default().show(ctx, |ui| {
            let panel_w = ui.available_width();
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    egui::Frame::new().inner_margin(Margin::same(16)).show(ui, |ui| {
                        let content_w = (panel_w - 46.0).max(260.0);
                        ui.set_max_width(content_w);
                        ui.set_min_width(content_w);

                        if self.nav == NavSection::Home || self.dataset.is_none() || self.report.is_none() {
                            self.render_welcome(ui);
                            return;
                        }

                        let ds = self.dataset.as_ref().unwrap();
                        let rep = self.report.as_ref().unwrap();
                        let mut nav_jump: Option<NavSection> = None;

                        match self.nav {
                            NavSection::Home => {}
                            NavSection::Dashboard => render_dashboard(ui, rep, &mut nav_jump),
                            NavSection::Normality => render_normality(ui, ds, rep, &mut self.view_state),
                            NavSection::Variance => render_variance(ui, ds, rep),
                            NavSection::Sphericity => render_sphericity(ui, rep),
                            NavSection::Multicollinearity => render_multicollinearity(ui, rep),
                            NavSection::Autocorrelation => render_autocorrelation(ui, rep),
                            NavSection::Heteroscedasticity => render_heteroscedasticity(ui, rep),
                            NavSection::Outliers => render_outliers(ui, rep),
                            NavSection::MissingData => render_missing(ui, rep),
                            NavSection::FactorSuitability => render_factor(ui, rep),
                            NavSection::LinearRegression => render_linear_regression(ui, rep),
                            NavSection::LogisticRegression => render_logistic(ui, rep),
                            NavSection::Survival => render_survival(ui, rep, &mut self.view_state),
                            NavSection::Clustering => render_clustering(ui, rep),
                            NavSection::ManualWorkbench => self.manual.render(ui, ds, rep, self.alpha),
                            NavSection::DataExplorer => render_data_explorer(ui, ds, &mut self.view_state),
                        }

                        if let Some(target) = nav_jump {
                            self.nav = target;
                        }
                    });
                });
        });

        // Drag-and-drop overlay
        let hovering = ctx.input(|i| !i.raw.hovered_files.is_empty());
        if hovering {
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("drop_overlay"),
            ));
            let screen = ctx.screen_rect();
            painter.rect_filled(screen, 0.0_f32, Color32::from_rgba_unmultiplied(29, 78, 216, 180));
            painter.text(
                screen.center(),
                egui::Align2::CENTER_CENTER,
                "Slipp CSV-filen her for automatisk statistisk analyse",
                FontId::proportional(24.0),
                Color32::WHITE,
            );
        }
    }
}

impl StatApp {
    fn nav_button(&mut self, ui: &mut egui::Ui, sec: NavSection, label: &str) {
        let selected = self.nav == sec;
        let status_dot = self.section_status(sec).map(status_dot_color);
        ui.horizontal(|ui| {
            if sec == NavSection::Dashboard {
                // Crisp vector arrow to the left of Oversikt (Dashboard) — no blue selection box
                let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                let c = rect.center();
                let arrow_col = if selected { TEXT_PRIMARY } else { TEXT_SECONDARY };
                ui.painter().add(egui::Shape::convex_polygon(
                    vec![
                        egui::pos2(c.x - 3.5, c.y - 4.5),
                        egui::pos2(c.x + 4.5, c.y),
                        egui::pos2(c.x - 3.5, c.y + 4.5),
                    ],
                    arrow_col,
                    Stroke::NONE,
                ));
                let txt = if selected {
                    RichText::new(label)
                        .font(FontId::new(13.2, egui::FontFamily::Name("bold".into())))
                        .color(TEXT_PRIMARY)
                } else {
                    RichText::new(label).size(13.2).color(TEXT_PRIMARY)
                };
                if ui
                    .add(egui::Label::new(txt).sense(egui::Sense::click()))
                    .clicked()
                {
                    self.nav = sec;
                }
                return;
            }

            if let Some(col) = status_dot {
                let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter().circle_filled(rect.center(), 4.5_f32, col);
            } else {
                ui.add_space(14.0);
            }
            let txt = if selected {
                RichText::new(label)
                    .font(FontId::new(13.0, egui::FontFamily::Name("bold".into())))
                    .color(Color32::from_rgb(30, 58, 138))
            } else {
                RichText::new(label).size(13.0).color(TEXT_PRIMARY)
            };
            if ui.selectable_label(selected, txt).clicked() {
                self.nav = sec;
            }
        });
    }

    fn render_welcome(&mut self, ui: &mut egui::Ui) {
        ui.add_space(24.0);
        ui.vertical_centered(|ui| {
            ui.label(
                RichText::new("stat-analysis")
                    .size(32.0)
                    .strong()
                    .color(PRIMARY_BLUE),
            );
            ui.add_space(4.0);
            ui.label(
                RichText::new("Automatisk og manuell statistisk forutsetningssjekk, modellkvalitet og interaktiv visualisering i ren Rust")
                    .size(14.5)
                    .color(TEXT_SECONDARY),
            );
            ui.add_space(20.0);
        });

        show_card(ui, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(12.0);
                ui.label(RichText::new("Dra og slipp en .csv-fil inn i dette vinduet").size(17.0).strong().color(TEXT_PRIMARY));
                ui.label(RichText::new("eller velg fil fra datamaskinen — alle tester og figurer beregnes automatisk").color(TEXT_SECONDARY));
                ui.add_space(14.0);
                ui.horizontal_wrapped(|ui| {
                    if self.dataset.is_some() {
                        if primary_button(ui, "   Tilbake til datasett →   ").clicked() {
                            self.nav = NavSection::Dashboard;
                        }
                    } else if primary_button(ui, "   Åpne egen CSV-fil...   ").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("CSV-filer", &["csv", "txt"])
                            .pick_file()
                        {
                            self.load_csv_path(&path);
                        }
                    }
                });
                ui.add_space(12.0);
            });
        });

        ui.add_space(16.0);
        let bold_font = FontId::new(13.0, egui::FontFamily::Name("bold".into()));
        let reg_font = FontId::proportional(13.0);
        let heading_bold = FontId::new(15.5, egui::FontFamily::Name("bold".into()));

        let render_bullet = |ui: &mut egui::Ui, label: &str, desc: &str| {
            let mut job = egui::text::LayoutJob::default();
            job.append(
                &format!("• {}:", label),
                0.0,
                egui::TextFormat {
                    font_id: bold_font.clone(),
                    color: TEXT_PRIMARY,
                    ..Default::default()
                },
            );
            job.append(
                &format!(" {}", desc),
                0.0,
                egui::TextFormat {
                    font_id: reg_font.clone(),
                    color: TEXT_SECONDARY,
                    ..Default::default()
                },
            );
            ui.label(job);
            ui.add_space(2.0);
        };

        // 10 Standard innebygde eksempelsett
        let mut chosen_example: Option<usize> = None;
        show_card(ui, |ui| {
            ui.label(
                RichText::new("Standard datasett (10 innebygde eksempelsett)")
                    .font(heading_bold.clone())
                    .color(TEXT_PRIMARY),
            );
            ui.label(
                RichText::new("Velg et datasett under for å utforske ulike forutsetningssjekker, tiltak og modelltester umiddelbart:")
                    .size(12.8)
                    .color(TEXT_SECONDARY),
            );
            ui.add_space(8.0);

            let avail = ui.available_width().max(260.0);
            let gap = 10.0_f32;
            let cols = if avail >= 740.0 { 2 } else { 1 };
            let tile_w = ((avail - gap * (cols as f32 - 1.0)) / cols as f32).floor().max(220.0);

            for (row_idx, chunk) in EXAMPLE_DATASETS.chunks(cols).enumerate() {
                ui.horizontal_top(|ui| {
                    for (ci, ex) in chunk.iter().enumerate() {
                        if ci > 0 {
                            ui.add_space(gap);
                        }
                        let idx = row_idx * cols + ci;
                        ui.allocate_ui_with_layout(
                            egui::Vec2::new(tile_w, 0.0),
                            egui::Layout::top_down(egui::Align::LEFT),
                            |ui| {
                                Frame::new()
                                    .fill(Color32::WHITE)
                                    .stroke(Stroke::new(1.0_f32, BORDER_DEFAULT))
                                    .corner_radius(8.0_f32)
                                    .inner_margin(Margin::same(11))
                                    .show(ui, |ui| {
                                        let iw = (tile_w - 24.0).max(160.0);
                                        ui.set_min_width(iw);
                                        ui.set_max_width(iw);
                                        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);

                                        let mut job = egui::text::LayoutJob::default();
                                        job.append(
                                            &format!("{}:", ex.title),
                                            0.0,
                                            egui::TextFormat {
                                                font_id: bold_font.clone(),
                                                color: TEXT_PRIMARY,
                                                ..Default::default()
                                            },
                                        );
                                        job.append(
                                            &format!(" {}", ex.focus),
                                            0.0,
                                            egui::TextFormat {
                                                font_id: FontId::proportional(12.2),
                                                color: TEXT_SECONDARY,
                                                ..Default::default()
                                            },
                                        );
                                        ui.label(job);
                                        ui.add_space(6.0);
                                        if ui
                                            .add(
                                                egui::Button::new(
                                                    RichText::new("Last inn sett →")
                                                        .color(PRIMARY_BLUE)
                                                        .strong(),
                                                )
                                                .fill(Color32::WHITE)
                                                .stroke(Stroke::new(1.0_f32, Color32::from_rgb(191, 219, 254))),
                                            )
                                            .clicked()
                                        {
                                            chosen_example = Some(idx);
                                        }
                                    });
                            },
                        );
                    }
                });
                ui.add_space(gap);
            }
        });

        if let Some(idx) = chosen_example {
            let ex = &EXAMPLE_DATASETS[idx];
            self.load_csv_content(ex.filename, ex.csv);
            return;
        }

        ui.add_space(14.0);

        responsive_two_col(
            ui,
            |ui| {
                show_card(ui, |ui| {
                    ui.label(RichText::new("Forutsetningssjekker og tiltak").font(heading_bold.clone()).color(TEXT_PRIMARY));
                    ui.add_space(8.0);
                    for (label, desc) in [
                        ("Normalfordeling", "Shapiro-Wilk (<50), KS/Lilliefors (>=50), Q-Q, Histogram, Mann-Whitney, Kruskal-Wallis, Bootstrap"),
                        ("Varianshomogenitet", "Levene, Bartlett, Boksplott, Welchs t-test / ANOVA"),
                        ("Sfærisitet", "Mauchlys test, Greenhouse-Geisser & Huynh-Feldt (RM-ANOVA)"),
                        ("Multikollinearitet", "VIF, Korrelasjonsmatrise, Ridge & Lasso"),
                        ("Autokorrelasjon", "Durbin-Watson, Breusch-Godfrey, ACF, Newey-West HAC, AR(1)"),
                        ("Heteroskedastisitet", "Breusch-Pagan, White, Residualplott, HC3 & Log(Y)"),
                        ("Uteliggere", "Cook's distance, Mahalanobis, Leverage-plott, Sensitivitetsanalyse"),
                        ("Manglende data", "Littles MCAR-test, Mønsteranalyse, MICE & FIML"),
                        ("Faktoranalyse", "KMO, MSA, Bartletts sfærisitetstest & PCA Scree-plott"),
                    ] {
                        render_bullet(ui, label, desc);
                    }
                });
            },
            |ui| {
                show_card(ui, |ui| {
                    ui.label(RichText::new("Modellkvalitet og eksport").font(heading_bold.clone()).color(TEXT_PRIMARY));
                    ui.add_space(8.0);
                    for (label, desc) in [
                        ("Lineær regresjon", "R², Justert R², RMSE, MAE, F-test og koeffisienter"),
                        ("Logistisk regresjon", "Hosmer-Lemeshow-test, Kalibreringsplott, ROC-kurve & AUC"),
                        ("Modellsammenligning", "AIC og BIC på tvers av kandidatmodeller"),
                        ("Overlevelsesanalyse (Cox PH)", "Hazard Ratios, C-index & Schoenfeld-residualer"),
                        ("Klyngeanalyse", "K-means++, Silhouette-skår, Elbow-metoden & 2D PCA-projeksjon"),
                        ("Manuell testbenk", "Kjør hver enkelt test manuelt med egne kolonner og parametere"),
                        ("Nedlasting / Eksport", "Lagre ekte PDF-rapport, enkeltvise SVG-figurer, Markdown, JSON eller komplett ZIP"),
                    ] {
                        render_bullet(ui, label, desc);
                    }
                });
            },
        );
    }
}

/// Crisp vector Home icon button rendered with egui::Painter (never shows missing emoji boxes)
fn home_icon_button(ui: &mut egui::Ui, is_active: bool) -> egui::Response {
    let size = egui::vec2(30.0, 28.0);
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
    let painter = ui.painter_at(rect);

    let bg = if resp.hovered() {
        Color32::from_rgb(219, 234, 254)
    } else if is_active {
        Color32::from_rgb(239, 246, 255)
    } else {
        Color32::from_rgb(241, 245, 249)
    };
    let border = if resp.hovered() || is_active {
        PRIMARY_BLUE
    } else {
        Color32::from_rgb(148, 163, 184)
    };
    let icon_col = if resp.hovered() || is_active {
        PRIMARY_BLUE
    } else {
        TEXT_PRIMARY
    };

    painter.rect_filled(rect, 6.0_f32, bg);
    painter.rect_stroke(rect, 6.0_f32, Stroke::new(1.1_f32, border), egui::StrokeKind::Inside);

    let c = rect.center();
    // Roof triangle (clean stroke)
    let roof_top = egui::pos2(c.x, c.y - 7.0);
    let roof_left = egui::pos2(c.x - 8.0, c.y);
    let roof_right = egui::pos2(c.x + 8.0, c.y);
    painter.line_segment([roof_left, roof_top], Stroke::new(2.0_f32, icon_col));
    painter.line_segment([roof_top, roof_right], Stroke::new(2.0_f32, icon_col));

    // House walls
    let wall_rect = egui::Rect::from_min_max(
        egui::pos2(c.x - 5.5, c.y - 0.5),
        egui::pos2(c.x + 5.5, c.y + 7.0),
    );
    painter.rect_stroke(wall_rect, 1.0_f32, Stroke::new(1.7_f32, icon_col), egui::StrokeKind::Inside);

    // Door
    let door_rect = egui::Rect::from_min_max(
        egui::pos2(c.x - 1.8, c.y + 2.2),
        egui::pos2(c.x + 1.8, c.y + 7.0),
    );
    painter.rect_filled(door_rect, 0.5_f32, icon_col);

    resp
}
