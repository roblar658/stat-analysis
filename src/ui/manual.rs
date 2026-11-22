use crate::auto::AutoAnalysisReport;
use crate::data::Dataset;
use crate::plots::{
    svg_acf_plot, svg_boxplot, svg_calibration_plot, svg_clusters_2d, svg_corr_heatmap,
    svg_elbow_silhouette, svg_histogram, svg_leverage_cooks, svg_missingness, svg_qq_plot,
    svg_residuals_vs_fitted, svg_roc_curve, svg_schoenfeld, svg_scree_plot, svg_sphericity_profile,
};
use crate::report::{ManualCalcResult as ManualResult, ManualHistoryEntry};
use crate::stats::clustering::analyze_clustering;
use crate::stats::factor::check_factor_suitability;
use crate::stats::logistic::analyze_logistic_regression;
use crate::stats::missing::analyze_missing_data;
use crate::stats::normality::{bootstrap_mean, check_normality, kruskal_wallis, mann_whitney_u};
use crate::stats::regression::full_linear_regression_diagnostics;
use crate::stats::sphericity::check_sphericity;
use crate::stats::survival::analyze_cox_ph;
use crate::stats::variance::check_variance_homogeneity;
use crate::ui::charts::{
    bold_text, callout_box, kpi_card, plot_acf, plot_boxplot, plot_calibration, plot_clusters_2d,
    plot_corr_heatmap, plot_elbow_and_silhouette, plot_histogram, plot_leverage_cooks,
    plot_missingness_bars, plot_qq, plot_residuals_vs_fitted, plot_roc, plot_schoenfeld,
    plot_scree, plot_sphericity_profile, primary_button, responsive_two_col, save_svg_button,
    secondary_button, show_card, show_table, status_badge, table_header_row, TEXT_PRIMARY,
    TEXT_SECONDARY,
};
use egui::{Color32, ComboBox, RichText, Ui};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManualTestKind {
    Normality,
    GroupVariance,
    Sphericity,
    Regression,
    Missing,
    Factor,
    Logistic,
    Survival,
    Clustering,
}

impl ManualTestKind {
    pub fn all() -> &'static [ManualTestKind] {
        &[
            Self::Normality,
            Self::GroupVariance,
            Self::Sphericity,
            Self::Regression,
            Self::Missing,
            Self::Factor,
            Self::Logistic,
            Self::Survival,
            Self::Clustering,
        ]
    }
    pub fn label(&self) -> &'static str {
        match self {
            Self::Normality => "1. Normalfordeling (SW / KS / Bootstrap)",
            Self::GroupVariance => "2. Varians & Gruppe (Levene / Bartlett / Welch / MW / KW)",
            Self::Sphericity => "3. Sfærisitet (Mauchly / GG / HF / RM-ANOVA)",
            Self::Regression => "4. Lineær regresjon + alle forutsetninger + AIC/BIC",
            Self::Missing => "5. Manglende data (Littles MCAR / MICE / FIML)",
            Self::Factor => "6. Faktoranalyse-egnethet (KMO / Bartlett / PCA)",
            Self::Logistic => "7. Logistisk regresjon (Hosmer-Lemeshow / ROC / AUC)",
            Self::Survival => "8. Overlevelsesanalyse (Cox PH / Schoenfeld)",
            Self::Clustering => "9. Klyngeanalyse (Silhouette / Elbow / 2D PCA)",
        }
    }
}

pub struct ManualWorkbench {
    pub active_test: ManualTestKind,
    pub single_col: String,
    pub group_col: String,
    pub target_col: String,
    pub time_col: String,
    pub event_col: String,
    pub multi_cols: BTreeSet<String>,
    pub bootstrap_n: usize,
    pub bg_lags: usize,
    pub ridge_lambda: f64,
    pub mice_m: usize,
    pub hl_groups: usize,
    pub k_max: usize,
    pub last_error: Option<String>,
    pub last_result: Option<ManualResult>,
    pub history: Vec<ManualHistoryEntry>,
}

impl Default for ManualWorkbench {
    fn default() -> Self {
        Self {
            active_test: ManualTestKind::Normality,
            single_col: String::new(),
            group_col: String::new(),
            target_col: String::new(),
            time_col: String::new(),
            event_col: String::new(),
            multi_cols: BTreeSet::new(),
            bootstrap_n: 2000,
            bg_lags: 2,
            ridge_lambda: 1.0,
            mice_m: 5,
            hl_groups: 10,
            k_max: 8,
            last_error: None,
            last_result: None,
            history: Vec::new(),
        }
    }
}

impl ManualWorkbench {
    pub fn sync_with_dataset(&mut self, ds: &Dataset) {
        let nums = ds.numeric_column_names();
        let grps = ds.grouping_column_names();
        let bins = ds.binary_column_names();

        self.single_col = nums.first().cloned().unwrap_or_default();
        self.group_col = grps
            .first()
            .cloned()
            .or_else(|| ds.headers.first().cloned())
            .unwrap_or_default();
        self.target_col = nums.last().cloned().unwrap_or_default();
        self.time_col = nums.first().cloned().unwrap_or_default();
        self.event_col = bins
            .first()
            .cloned()
            .or_else(|| ds.headers.last().cloned())
            .unwrap_or_default();
        self.multi_cols = nums.iter().take(5).cloned().collect();
        self.last_error = None;
        self.last_result = None;
        self.history.clear();
    }

    pub fn render(&mut self, ui: &mut Ui, ds: &Dataset, rep: &AutoAnalysisReport, alpha: f64) {
        ui.label(RichText::new("Manuell testbenk — kjør alle tester med egne valg").size(22.0).strong().color(TEXT_PRIMARY));
        ui.label(RichText::new("Velg hvilken som helst statistisk test, angi eksakte kolonner og parametere, og eksporter interaktive grafer som SVG.").size(13.0).color(TEXT_SECONDARY));
        ui.add_space(10.0);

        let all_cols = ds.headers.clone();
        let num_cols = ds.all_numeric_capable_names();
        let bin_cols = ds.binary_column_names();
        let grp_cols = ds.grouping_column_names();

        show_card(ui, |ui| {
            let combo_w = (ui.available_width() - 90.0).clamp(200.0, 440.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Velg test:").strong().color(TEXT_PRIMARY));
                ComboBox::from_id_salt("manual_test_pick")
                    .width(combo_w)
                    .selected_text(self.active_test.label())
                    .show_ui(ui, |ui| {
                        for &t in ManualTestKind::all() {
                            if ui.selectable_value(&mut self.active_test, t, t.label()).changed() {
                                self.last_error = None;
                                self.last_result = None;
                                if t == ManualTestKind::Logistic {
                                    if let Some(b) = bin_cols.first() {
                                        self.target_col = b.clone();
                                    }
                                }
                            }
                        }
                    });
            });

            ui.add_space(10.0);
            ui.separator();
            ui.add_space(8.0);

            match self.active_test {
                ManualTestKind::Normality => {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new("Numerisk variabel:").color(TEXT_PRIMARY));
                        combo_col(ui, "man_norm_col", &mut self.single_col, &num_cols);
                        ui.add_space(12.0);
                        ui.label(RichText::new("Bootstrap resamples:").color(TEXT_PRIMARY));
                        ui.add(egui::DragValue::new(&mut self.bootstrap_n).range(200..=10000).speed(100));
                    });
                }
                ManualTestKind::GroupVariance => {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new("Numerisk utfall:").color(TEXT_PRIMARY));
                        combo_col(ui, "man_gv_val", &mut self.single_col, &num_cols);
                        ui.add_space(12.0);
                        ui.label(RichText::new("Gruppekolonne:").color(TEXT_PRIMARY));
                        let g_opts = if grp_cols.is_empty() { &all_cols } else { &grp_cols };
                        combo_col(ui, "man_gv_grp", &mut self.group_col, g_opts);
                    });
                }
                ManualTestKind::Sphericity => {
                    ui.label(RichText::new("Velg minst 3 repeterte målingskolonner:").strong().color(TEXT_PRIMARY));
                    multi_col_picker(ui, &num_cols, &mut self.multi_cols);
                }
                ManualTestKind::Regression => {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new("Avhengig variabel (Y):").color(TEXT_PRIMARY));
                        combo_col(ui, "man_reg_y", &mut self.target_col, &num_cols);
                        ui.add_space(12.0);
                        ui.label(RichText::new("BG lags:").color(TEXT_PRIMARY));
                        ui.add(egui::DragValue::new(&mut self.bg_lags).range(1..=8));
                        ui.add_space(12.0);
                        ui.label(RichText::new("Ridge/Lasso λ:").color(TEXT_PRIMARY));
                        ui.add(egui::DragValue::new(&mut self.ridge_lambda).range(0.01..=100.0).speed(0.2));
                    });
                    ui.add_space(6.0);
                    ui.label(RichText::new("Forklaringsvariabler (X):").strong().color(TEXT_PRIMARY));
                    multi_col_picker(ui, &num_cols, &mut self.multi_cols);
                }
                ManualTestKind::Missing => {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new("Antall MICE-imputeringer (m):").color(TEXT_PRIMARY));
                        ui.add(egui::DragValue::new(&mut self.mice_m).range(3..=15));
                    });
                    ui.add_space(6.0);
                    ui.label(RichText::new("Numeriske kolonner for Littles MCAR & MICE:").strong().color(TEXT_PRIMARY));
                    multi_col_picker(ui, &num_cols, &mut self.multi_cols);
                }
                ManualTestKind::Factor => {
                    ui.label(RichText::new("Velg minst 3 numeriske variabler for KMO & Bartlett:").strong().color(TEXT_PRIMARY));
                    multi_col_picker(ui, &num_cols, &mut self.multi_cols);
                }
                ManualTestKind::Logistic => {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new("Binær målvariabel (Y):").color(TEXT_PRIMARY));
                        let opts = if bin_cols.is_empty() { &all_cols } else { &bin_cols };
                        combo_col(ui, "man_log_y", &mut self.target_col, opts);
                        ui.add_space(12.0);
                        ui.label(RichText::new("Hosmer-Lemeshow grupper (g):").color(TEXT_PRIMARY));
                        ui.add(egui::DragValue::new(&mut self.hl_groups).range(3..=10));
                    });
                    ui.add_space(6.0);
                    ui.label(RichText::new("Forklaringsvariabler (X):").strong().color(TEXT_PRIMARY));
                    multi_col_picker(ui, &num_cols, &mut self.multi_cols);
                }
                ManualTestKind::Survival => {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new("Tidskolonne (> 0):").color(TEXT_PRIMARY));
                        combo_col(ui, "man_surv_t", &mut self.time_col, &num_cols);
                        ui.add_space(12.0);
                        ui.label(RichText::new("Hendelseskolonne (binær):").color(TEXT_PRIMARY));
                        let opts = if bin_cols.is_empty() { &all_cols } else { &bin_cols };
                        combo_col(ui, "man_surv_e", &mut self.event_col, opts);
                    });
                    ui.add_space(6.0);
                    ui.label(RichText::new("Kovariater (X):").strong().color(TEXT_PRIMARY));
                    multi_col_picker(ui, &num_cols, &mut self.multi_cols);
                }
                ManualTestKind::Clustering => {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new("Maks antall klynger (k_max):").color(TEXT_PRIMARY));
                        ui.add(egui::DragValue::new(&mut self.k_max).range(3..=10));
                    });
                    ui.add_space(6.0);
                    ui.label(RichText::new("Numeriske variabler (minst 2):").strong().color(TEXT_PRIMARY));
                    multi_col_picker(ui, &num_cols, &mut self.multi_cols);
                }
            }

            ui.add_space(12.0);
            ui.horizontal_wrapped(|ui| {
                if primary_button(ui, "  Kjør valgt test  ").clicked() {
                    self.execute(ds, alpha);
                }
                if secondary_button(ui, "  Kjør alle relevante tester → Appendiks  ")
                    .on_hover_text("Utfører automatisk alle relevante tester for dette datasettet og legger dem i PDF-appendikset")
                    .clicked()
                {
                    self.execute_all_relevant(ds, rep, alpha);
                }
                if !self.history.is_empty() {
                    ui.add_space(8.0);
                    ui.label(
                        RichText::new(format!(
                            "{} manuelle utregninger inkluderes i Appendiks ved PDF-eksport",
                            self.history.len()
                        ))
                        .color(Color32::from_rgb(20, 83, 45))
                        .strong()
                        .size(12.5),
                    );
                }
            });

            if let Some(ref err) = self.last_error {
                ui.add_space(8.0);
                ui.label(RichText::new(format!("Feil: {}", err)).color(Color32::from_rgb(185, 28, 28)).strong());
            }
        });

        if !self.history.is_empty() {
            ui.add_space(10.0);
            let mut view_idx: Option<usize> = None;
            let mut remove_idx: Option<usize> = None;
            let mut clear_all = false;
            show_card(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(bold_text(
                        &format!("Appendiks-logg for PDF-rapport ({} manuelle utregninger)", self.history.len()),
                        14.0,
                        TEXT_PRIMARY,
                    ));
                    if secondary_button(ui, "Tøm appendiks-logg").clicked() {
                        clear_all = true;
                    }
                });
                ui.add_space(6.0);
                show_table(ui, "manual_hist_tbl", |ui| {
                    table_header_row(ui, &["Ref.", "Tid", "Manuell utregning", "Parametere", "Handlinger"]);
                    for (i, h) in self.history.iter().enumerate() {
                        ui.label(bold_text(&format!("A.{}", i + 1), 12.5, TEXT_PRIMARY));
                        ui.label(RichText::new(&h.timestamp).color(TEXT_SECONDARY));
                        ui.label(bold_text(&h.title, 12.5, TEXT_PRIMARY));
                        ui.label(RichText::new(&h.parameters).color(TEXT_SECONDARY));
                        ui.horizontal(|ui| {
                            if ui.small_button("Vis").clicked() {
                                view_idx = Some(i);
                            }
                            if ui.small_button("Fjern").clicked() {
                                remove_idx = Some(i);
                            }
                        });
                        ui.end_row();
                    }
                });
            });
            if clear_all {
                self.history.clear();
            } else {
                if let Some(i) = view_idx {
                    self.last_result = Some(self.history[i].result.clone());
                }
                if let Some(i) = remove_idx {
                    self.history.remove(i);
                }
            }
        }

        ui.add_space(12.0);
        if let Some(ref res) = self.last_result {
            render_manual_result(ui, res);
        }
    }

    fn execute(&mut self, ds: &Dataset, alpha: f64) {
        self.last_error = None;
        self.last_result = None;
        let run: Result<ManualResult, String> = match self.active_test {
            ManualTestKind::Normality => {
                let vals = ds.get_numeric_values(&self.single_col);
                vals.and_then(|v| {
                    let check = check_normality(&self.single_col, &v, alpha)?;
                    let boot = bootstrap_mean(&self.single_col, &v, self.bootstrap_n).ok();
                    Ok(ManualResult::Normality { check, boot, values: v })
                })
            }
            ManualTestKind::GroupVariance => {
                ds.get_grouped_values(&self.single_col, &self.group_col).and_then(|map| {
                    let groups: Vec<(String, Vec<f64>)> = map.into_iter().filter(|(_, v)| v.len() >= 2).collect();
                    let variance = check_variance_homogeneity(&self.single_col, &self.group_col, &groups, alpha)?;
                    let mw = if groups.len() == 2 {
                        mann_whitney_u(&groups[0].0, &groups[0].1, &groups[1].0, &groups[1].1, alpha).ok()
                    } else {
                        None
                    };
                    let kw = kruskal_wallis(&groups, alpha).ok();
                    Ok(ManualResult::GroupVariance { variance, mw, kw, groups })
                })
            }
            ManualTestKind::Sphericity => {
                let cols: Vec<String> = self.multi_cols.iter().cloned().collect();
                ds.get_complete_numeric_matrix(&cols)
                    .and_then(|rows| check_sphericity(&cols, &rows, alpha))
                    .map(ManualResult::Sphericity)
            }
            ManualTestKind::Regression => {
                let preds: Vec<String> = self
                    .multi_cols
                    .iter()
                    .filter(|c| *c != &self.target_col)
                    .cloned()
                    .collect();
                if preds.is_empty() {
                    Err("Velg minst 1 forklaringsvariabel (X) som er ulik Y.".into())
                } else {
                    let mut all = vec![self.target_col.clone()];
                    all.extend(preds.clone());
                    ds.get_complete_numeric_matrix(&all).and_then(|rows| {
                        let y: Vec<f64> = rows.iter().map(|r| r[0]).collect();
                        let x_cols: Vec<Vec<f64>> = (0..preds.len())
                            .map(|j| rows.iter().map(|r| r[j + 1]).collect())
                            .collect();
                        full_linear_regression_diagnostics(
                            &self.target_col,
                            &y,
                            &x_cols,
                            &preds,
                            alpha,
                            self.bg_lags,
                            self.ridge_lambda,
                        )
                        .map(ManualResult::Regression)
                    })
                }
            }
            ManualTestKind::Missing => {
                let cols: Vec<String> = self.multi_cols.iter().cloned().collect();
                ds.get_numeric_matrix_with_missing(&cols).map(|mat| {
                    ManualResult::Missing(analyze_missing_data(
                        &ds.headers,
                        &ds.raw_rows,
                        &cols,
                        &mat,
                        alpha,
                        self.mice_m,
                    ))
                })
            }
            ManualTestKind::Factor => {
                let cols: Vec<String> = self.multi_cols.iter().cloned().collect();
                ds.get_complete_numeric_matrix(&cols)
                    .and_then(|rows| check_factor_suitability(&cols, &rows, alpha))
                    .map(ManualResult::Factor)
            }
            ManualTestKind::Logistic => {
                let preds: Vec<String> = self
                    .multi_cols
                    .iter()
                    .filter(|c| *c != &self.target_col)
                    .cloned()
                    .collect();
                if preds.is_empty() {
                    Err("Velg minst 1 forklaringsvariabel (X).".into())
                } else {
                    ds.get_binary_series(&self.target_col).and_then(|(y_opt, neg, pos)| {
                        let x_opt = ds.get_numeric_matrix_with_missing(&preds)?;
                        let mut y = Vec::new();
                        let mut x_cols = vec![Vec::new(); preds.len()];
                        for (i, row) in x_opt.iter().enumerate() {
                            if let Some(yv) = y_opt[i] {
                                if row.iter().all(|v| v.is_some()) {
                                    y.push(yv);
                                    for j in 0..preds.len() {
                                        x_cols[j].push(row[j].unwrap());
                                    }
                                }
                            }
                        }
                        analyze_logistic_regression(
                            &self.target_col,
                            &neg,
                            &pos,
                            &y,
                            &x_cols,
                            &preds,
                            self.hl_groups,
                            alpha,
                        )
                        .map(ManualResult::Logistic)
                    })
                }
            }
            ManualTestKind::Survival => {
                let preds: Vec<String> = self
                    .multi_cols
                    .iter()
                    .filter(|c| *c != &self.time_col && *c != &self.event_col)
                    .cloned()
                    .collect();
                if preds.is_empty() {
                    Err("Velg minst 1 kovariat (X).".into())
                } else {
                    ds.get_binary_series(&self.event_col).and_then(|(ev_opt, _, _)| {
                        let mut cols = vec![self.time_col.clone()];
                        cols.extend(preds.clone());
                        let mat_opt = ds.get_numeric_matrix_with_missing(&cols)?;
                        let mut times = Vec::new();
                        let mut events = Vec::new();
                        let mut x_cols = vec![Vec::new(); preds.len()];
                        for (i, row) in mat_opt.iter().enumerate() {
                            if let (Some(t), Some(e)) = (row[0], ev_opt[i]) {
                                if t > 0.0 && row[1..].iter().all(|v| v.is_some()) {
                                    times.push(t);
                                    events.push(e);
                                    for j in 0..preds.len() {
                                        x_cols[j].push(row[j + 1].unwrap());
                                    }
                                }
                            }
                        }
                        analyze_cox_ph(
                            &self.time_col,
                            &self.event_col,
                            &times,
                            &events,
                            &x_cols,
                            &preds,
                            alpha,
                        )
                        .map(ManualResult::Survival)
                    })
                }
            }
            ManualTestKind::Clustering => {
                let cols: Vec<String> = self.multi_cols.iter().cloned().collect();
                ds.get_complete_numeric_matrix(&cols)
                    .and_then(|rows| analyze_clustering(&cols, &rows, self.k_max))
                    .map(ManualResult::Clustering)
            }
        };

        match run {
            Ok(r) => {
                let (title, parameters) = self.describe_entry(alpha);
                self.history.push(ManualHistoryEntry {
                    timestamp: chrono::Local::now().format("%H:%M:%S").to_string(),
                    title,
                    parameters,
                    result: r.clone(),
                });
                self.last_result = Some(r);
            }
            Err(e) => self.last_error = Some(e),
        }
    }

    fn describe_entry(&self, alpha: f64) -> (String, String) {
        let multi: Vec<String> = self.multi_cols.iter().cloned().collect();
        match self.active_test {
            ManualTestKind::Normality => (
                format!("Normalfordeling ({})", self.single_col),
                format!("Variabel = {}, alpha = {:.2}, Bootstrap B = {}", self.single_col, alpha, self.bootstrap_n),
            ),
            ManualTestKind::GroupVariance => (
                format!("Varians & Gruppe ({} per {})", self.single_col, self.group_col),
                format!("Utfall = {}, Gruppe = {}, alpha = {:.2}", self.single_col, self.group_col, alpha),
            ),
            ManualTestKind::Sphericity => (
                "Sfærisitet (Repeated Measures ANOVA)".to_string(),
                format!("Målinger = [{}], alpha = {:.2}", multi.join(", "), alpha),
            ),
            ManualTestKind::Regression => {
                let xs: Vec<String> = multi.into_iter().filter(|c| c != &self.target_col).collect();
                (
                    format!("Lineær regresjon ({})", self.target_col),
                    format!("Y = {}, X = [{}], alpha = {:.2}, BG lags = {}, lambda = {:.2}", self.target_col, xs.join(", "), alpha, self.bg_lags, self.ridge_lambda),
                )
            }
            ManualTestKind::Missing => (
                "Manglende data (Littles MCAR & MICE)".to_string(),
                format!("Kolonner = [{}], alpha = {:.2}, MICE m = {}", multi.join(", "), alpha, self.mice_m),
            ),
            ManualTestKind::Factor => (
                "Faktoranalyse-egnethet (KMO & Bartlett)".to_string(),
                format!("Variabler = [{}], alpha = {:.2}", multi.join(", "), alpha),
            ),
            ManualTestKind::Logistic => {
                let xs: Vec<String> = multi.into_iter().filter(|c| c != &self.target_col).collect();
                (
                    format!("Logistisk regresjon ({})", self.target_col),
                    format!("Y = {}, X = [{}], HL grupper = {}, alpha = {:.2}", self.target_col, xs.join(", "), self.hl_groups, alpha),
                )
            }
            ManualTestKind::Survival => {
                let xs: Vec<String> = multi.into_iter().filter(|c| c != &self.time_col && c != &self.event_col).collect();
                (
                    format!("Cox PH & Schoenfeld ({} / {})", self.time_col, self.event_col),
                    format!("Tid = {}, Hendelse = {}, X = [{}], alpha = {:.2}", self.time_col, self.event_col, xs.join(", "), alpha),
                )
            }
            ManualTestKind::Clustering => (
                "Klyngeanalyse (Silhouette & Elbow)".to_string(),
                format!("Variabler = [{}], k_max = {}", multi.join(", "), self.k_max),
            ),
        }
    }

    /// Execute EVERY relevant statistical test automatically on demand and log all results into the PDF Appendix.
    pub fn execute_all_relevant(&mut self, ds: &Dataset, rep: &AutoAnalysisReport, alpha: f64) {
        self.last_error = None;
        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
        let mut added: Option<ManualResult> = None;

        // 1. Normality + Bootstrap for first numeric column (or primary outcome)
        if let Some(nc) = rep.normality_checks.first() {
            if let Ok(vals) = ds.get_numeric_values(&nc.variable) {
                let boot = bootstrap_mean(&nc.variable, &vals, self.bootstrap_n).ok();
                let res = ManualResult::Normality {
                    check: nc.clone(),
                    boot,
                    values: vals,
                };
                self.history.push(ManualHistoryEntry {
                    timestamp: ts.clone(),
                    title: format!("Normalfordeling ({})", nc.variable),
                    parameters: format!("Variabel = {}, alpha = {:.2}, Bootstrap B = {}", nc.variable, alpha, self.bootstrap_n),
                    result: res.clone(),
                });
                added = Some(res);
            }
        }

        // 2. Group Variance + Student t-test / Welch / ANOVA / Mann-Whitney / Kruskal-Wallis
        if let Some(ref vh) = rep.variance_homogeneity {
            let groups: Vec<(String, Vec<f64>)> = ds
                .get_grouped_values(&vh.value_col, &vh.group_col)
                .map(|m| m.into_iter().filter(|(_, v)| v.len() >= 2).collect())
                .unwrap_or_default();
            let mw = rep.group_fallbacks.as_ref().and_then(|f| f.mann_whitney.clone());
            let kw = rep.group_fallbacks.as_ref().and_then(|f| f.kruskal_wallis.clone());
            let res = ManualResult::GroupVariance {
                variance: vh.clone(),
                mw,
                kw,
                groups,
            };
            self.history.push(ManualHistoryEntry {
                timestamp: ts.clone(),
                title: format!("Varians & Gruppetester ({} per {})", vh.value_col, vh.group_col),
                parameters: format!("Utfall = {}, Gruppe = {}, alpha = {:.2}", vh.value_col, vh.group_col, alpha),
                result: res.clone(),
            });
            added = Some(res);
        }

        // 3. Sphericity (Mauchly / GG / HF)
        if let Some(ref sp) = rep.sphericity {
            let res = ManualResult::Sphericity(sp.clone());
            self.history.push(ManualHistoryEntry {
                timestamp: ts.clone(),
                title: "Sfærisitet (Repeated Measures ANOVA)".into(),
                parameters: format!("Målinger = [{}], alpha = {:.2}", sp.columns.join(", "), alpha),
                result: res.clone(),
            });
            added = Some(res);
        }

        // 4. Linear Regression + all diagnostics
        if let Some(ref reg) = rep.linear_regression {
            let res = ManualResult::Regression(reg.clone());
            self.history.push(ManualHistoryEntry {
                timestamp: ts.clone(),
                title: format!("Lineær regresjon ({})", reg.target),
                parameters: format!("Y = {}, X = [{}], alpha = {:.2}", reg.target, reg.predictors.join(", "), alpha),
                result: res.clone(),
            });
            added = Some(res);
        }

        // 5. Missing Data (Little's MCAR & MICE)
        {
            let res = ManualResult::Missing(rep.missing_data.clone());
            self.history.push(ManualHistoryEntry {
                timestamp: ts.clone(),
                title: "Manglende data (Littles MCAR & MICE)".into(),
                parameters: format!("Alle kolonner, alpha = {:.2}, MICE m = {}", alpha, rep.missing_data.mice_imputations),
                result: res.clone(),
            });
            added = Some(res);
        }

        // 6. Factor Suitability (KMO & Bartlett)
        if let Some(ref fac) = rep.factor_suitability {
            let res = ManualResult::Factor(fac.clone());
            self.history.push(ManualHistoryEntry {
                timestamp: ts.clone(),
                title: "Faktoranalyse-egnethet (KMO & Bartlett)".into(),
                parameters: format!("Variabler = [{}], alpha = {:.2}", fac.variables.join(", "), alpha),
                result: res.clone(),
            });
            added = Some(res);
        }

        // 7. Logistic Regression (Hosmer-Lemeshow & ROC/AUC)
        if let Some(ref lr) = rep.logistic_regression {
            let res = ManualResult::Logistic(lr.clone());
            self.history.push(ManualHistoryEntry {
                timestamp: ts.clone(),
                title: format!("Logistisk regresjon ({})", lr.target),
                parameters: format!("Y = {}, X = [{}], alpha = {:.2}", lr.target, lr.predictors.join(", "), alpha),
                result: res.clone(),
            });
            added = Some(res);
        }

        // 8. Cox Survival (PH & Schoenfeld)
        if let Some(ref cox) = rep.cox_survival {
            let res = ManualResult::Survival(cox.clone());
            self.history.push(ManualHistoryEntry {
                timestamp: ts.clone(),
                title: format!("Cox PH & Schoenfeld ({} / {})", cox.time_col, cox.event_col),
                parameters: format!("Tid = {}, Hendelse = {}, X = [{}]", cox.time_col, cox.event_col, cox.predictors.join(", ")),
                result: res.clone(),
            });
            added = Some(res);
        }

        // 9. Clustering (Silhouette, Elbow & DBSCAN)
        if let Some(ref cl) = rep.clustering {
            let res = ManualResult::Clustering(cl.clone());
            self.history.push(ManualHistoryEntry {
                timestamp: ts.clone(),
                title: "Klyngeanalyse (Silhouette, Elbow & DBSCAN)".into(),
                parameters: format!("Variabler = [{}], beste k = {}", cl.variables.join(", "), cl.best_k_silhouette),
                result: res.clone(),
            });
            added = Some(res);
        }

        if let Some(last) = added {
            self.last_result = Some(last);
        }
    }
}

fn combo_col(ui: &mut Ui, id: &str, selected: &mut String, options: &[String]) {
    ComboBox::from_id_salt(id)
        .selected_text(if selected.is_empty() { "Velg..." } else { selected.as_str() })
        .show_ui(ui, |ui| {
            for opt in options {
                ui.selectable_value(selected, opt.clone(), opt);
            }
        });
}

fn multi_col_picker(ui: &mut Ui, options: &[String], selected: &mut BTreeSet<String>) {
    ui.horizontal_wrapped(|ui| {
        for col in options {
            let mut checked = selected.contains(col);
            if ui.checkbox(&mut checked, col).changed() {
                if checked {
                    selected.insert(col.clone());
                } else {
                    selected.remove(col);
                }
            }
        }
    });
}

fn render_manual_result(ui: &mut Ui, res: &ManualResult) {
    match res {
        ManualResult::Normality { check, boot, values } => {
            show_card(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new(format!("Resultat: Normalfordeling ({})", check.variable)).strong().color(TEXT_PRIMARY).size(15.0));
                    save_svg_button(ui, &format!("hist_{}.svg", check.variable), || {
                        svg_histogram(&format!("Histogram — {}", check.variable), &check.variable, values)
                    });
                    save_svg_button(ui, &format!("qq_{}.svg", check.variable), || {
                        svg_qq_plot(&format!("Q-Q-plott — {}", check.variable), values)
                    });
                });
                ui.add_space(6.0);
                callout_box(ui, &check.status, &check.interpretation, &check.recommendation);
                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    if let Some(ref sw) = check.shapiro_wilk {
                        kpi_card(ui, "Shapiro-Wilk W", &format!("{:.4}", sw.w), Some(&format!("p = {:.4}", sw.p_value)));
                    }
                    kpi_card(ui, "KS-test D", &format!("{:.4}", check.ks_test.d_stat), Some(&format!("Lilliefors p = {:.4}", check.ks_test.p_value_lilliefors)));
                    kpi_card(ui, "Skjevhet / Kurtose", &format!("{:.3} / {:.3}", check.skewness, check.excess_kurtosis), None);
                    if let Some(ref b) = boot {
                        kpi_card(ui, "Bootstrap 95% KI", &format!("[{:.3}, {:.3}]", b.ci_low_95, b.ci_high_95), Some(&format!("SE = {:.3} (B={})", b.bootstrap_se, b.resamples)));
                    }
                });
                ui.add_space(8.0);
                responsive_two_col(
                    ui,
                    |ui| plot_histogram(ui, "man_hist_plot", values, 260.0),
                    |ui| plot_qq(ui, "man_qq_plot", values, 260.0),
                );
            });
        }
        ManualResult::GroupVariance { variance, mw, kw, groups } => {
            show_card(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new(format!("Varianshomogenitet & Gruppetester ({} per {})", variance.value_col, variance.group_col)).strong().color(TEXT_PRIMARY).size(15.0));
                    save_svg_button(ui, "man_boksplott.svg", || {
                        svg_boxplot(&format!("Boksplott: {} per {}", variance.value_col, variance.group_col), &variance.value_col, groups)
                    });
                });
                ui.add_space(6.0);
                callout_box(ui, &variance.status, &variance.interpretation, &variance.recommendation);
                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    kpi_card(ui, "Levene (Median)", &format!("F={:.3}", variance.levene_median.f_stat), Some(&format!("p={:.4}", variance.levene_median.p_value)));
                    kpi_card(ui, "Bartlett", &format!("χ²={:.3}", variance.bartlett.chi2_stat), Some(&format!("p={:.4}", variance.bartlett.p_value)));
                    if let Some(ref st) = variance.student_t_test {
                        kpi_card(ui, "Student t-test", &format!("t={:.3} (d={:.2})", st.t_stat, st.cohens_d), Some(&format!("df={}, p={:.4}", st.df, st.p_value)));
                    }
                    kpi_card(ui, "Standard ANOVA", &format!("F={:.3} (η²={:.2})", variance.standard_anova.f_stat, variance.standard_anova.eta_squared), Some(&format!("p={:.4}", variance.standard_anova.p_value)));
                    kpi_card(ui, "Welchs ANOVA", &format!("F={:.3}", variance.welch_anova.f_stat), Some(&format!("p={:.4}", variance.welch_anova.p_value)));
                    if let Some(ref w) = variance.welch_t_test {
                        kpi_card(ui, "Welchs t-test", &format!("t={:.3}", w.t_stat), Some(&format!("p={:.4}", w.p_value)));
                    }
                    if let Some(ref m) = mw {
                        kpi_card(ui, "Mann-Whitney U", &format!("U={:.1}", m.u_stat), Some(&format!("p={:.4}", m.p_value)));
                    }
                    if let Some(ref k) = kw {
                        kpi_card(ui, "Kruskal-Wallis H", &format!("H={:.3}", k.h_stat), Some(&format!("p={:.4}", k.p_value)));
                    }
                });
                ui.add_space(8.0);
                plot_boxplot(ui, groups, 260.0);
            });
        }
        ManualResult::Sphericity(sp) => {
            show_card(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new("Sfærisitet (Mauchly / GG / HF)").strong().color(TEXT_PRIMARY).size(15.0));
                    save_svg_button(ui, "man_sfaerisitet.svg", || {
                        svg_sphericity_profile("Repeterte målinger", &sp.columns, &sp.condition_means)
                    });
                });
                ui.add_space(6.0);
                callout_box(ui, &sp.status, &sp.interpretation, &sp.recommendation);
                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    kpi_card(ui, "Mauchlys W", &format!("{:.4}", sp.mauchly_w), Some(&format!("p={:.4}", sp.mauchly_p_value)));
                    kpi_card(ui, "RM-ANOVA F", &format!("{:.3}", sp.rm_anova_f), Some(&format!("p={:.4}", sp.p_uncorrected)));
                    kpi_card(ui, "Greenhouse-Geisser", &format!("ε={:.3}", sp.epsilon_gg), Some(&format!("p={:.4}", sp.p_greenhouse_geisser)));
                    kpi_card(ui, "Huynh-Feldt", &format!("ε={:.3}", sp.epsilon_hf), Some(&format!("p={:.4}", sp.p_huynh_feldt)));
                });
                ui.add_space(8.0);
                plot_sphericity_profile(ui, "man_sph_plot", &sp.condition_means, 240.0);
            });
        }
        ManualResult::Regression(reg) => {
            show_card(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new(format!("Lineær regresjon & Diagnostikk (Y = {})", reg.target)).strong().color(TEXT_PRIMARY).size(15.0));
                    save_svg_button(ui, "man_residualer.svg", || {
                        svg_residuals_vs_fitted("Residualer vs Predikert", &reg.fitted, &reg.residuals)
                    });
                    save_svg_button(ui, "man_acf.svg", || {
                        svg_acf_plot("ACF", &reg.autocorrelation.acf_lags, reg.autocorrelation.acf_conf_bound)
                    });
                    save_svg_button(ui, "man_leverage.svg", || {
                        svg_leverage_cooks("Leverage & Cook's D", &reg.outliers.leverage, &reg.outliers.standardized_residuals, &reg.outliers.cooks_d, reg.outliers.cooks_threshold_4_over_n)
                    });
                });
                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    kpi_card(ui, "R² / Just. R²", &format!("{:.3} / {:.3}", reg.r_squared, reg.adj_r_squared), None);
                    kpi_card(ui, "RMSE / MAE", &format!("{:.3} / {:.3}", reg.rmse, reg.mae), None);
                    kpi_card(ui, "AIC / BIC", &format!("{:.1} / {:.1}", reg.aic, reg.bic), None);
                    kpi_card(ui, "Durbin-Watson", &format!("{:.3}", reg.autocorrelation.durbin_watson), Some(&format!("BG p={:.4}", reg.autocorrelation.bg_p_value)));
                    kpi_card(ui, "BP / White p", &format!("{:.4} / {:.4}", reg.heteroscedasticity.bp_p_value, reg.heteroscedasticity.white_p_value), None);
                    if let Some(ref mc) = reg.multicollinearity {
                        kpi_card(ui, "Maks VIF", &format!("{:.2}", mc.max_vif), None);
                    }
                });
                ui.add_space(8.0);
                responsive_two_col(
                    ui,
                    |ui| plot_residuals_vs_fitted(ui, "man_rvf", &reg.fitted, &reg.residuals, 230.0),
                    |ui| plot_acf(ui, "man_acf_p", &reg.autocorrelation.acf_lags, reg.autocorrelation.acf_conf_bound, 230.0),
                );
                ui.add_space(8.0);
                responsive_two_col(
                    ui,
                    |ui| plot_qq(ui, "man_reg_qq", &reg.residuals, 230.0),
                    |ui| plot_leverage_cooks(ui, "man_lev_p", &reg.outliers.leverage, &reg.outliers.standardized_residuals, &reg.outliers.cooks_d, reg.outliers.cooks_threshold_4_over_n, 230.0),
                );
                if let Some(ref mc) = reg.multicollinearity {
                    ui.add_space(10.0);
                    ui.label(RichText::new("Korrelasjonsmatrise (X)").strong().color(TEXT_PRIMARY));
                    plot_corr_heatmap(ui, "man_reg_heat", &mc.predictors, &mc.correlation_matrix);
                }
            });
        }
        ManualResult::Missing(m) => {
            show_card(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new("Manglende data, Littles MCAR & MICE").strong().color(TEXT_PRIMARY).size(15.0));
                    save_svg_button(ui, "man_manglende.svg", || svg_missingness("Manglende verdier (%)", &m.columns));
                });
                callout_box(ui, &m.status, &m.interpretation, &m.recommendation);
                ui.add_space(8.0);
                plot_missingness_bars(ui, "man_miss", &m.columns);
            });
        }
        ManualResult::Factor(fac) => {
            show_card(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new("Egnethet for faktoranalyse (KMO & Bartlett)").strong().color(TEXT_PRIMARY).size(15.0));
                    save_svg_button(ui, "man_scree.svg", || svg_scree_plot("Scree-plott", &fac.eigenvalues));
                    save_svg_button(ui, "man_fac_corr.svg", || svg_corr_heatmap("Korrelasjonsmatrise", &fac.variables, &fac.correlation_matrix));
                });
                callout_box(ui, &fac.status, &fac.interpretation, &fac.recommendation);
                ui.add_space(8.0);
                plot_scree(ui, "man_scree_p", &fac.eigenvalues, 240.0);
                ui.add_space(8.0);
                plot_corr_heatmap(ui, "man_fac_heat", &fac.variables, &fac.correlation_matrix);
            });
        }
        ManualResult::Logistic(lr) => {
            show_card(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new(format!("Logistisk regresjon ({})", lr.target)).strong().color(TEXT_PRIMARY).size(15.0));
                    save_svg_button(ui, "man_roc.svg", || svg_roc_curve("ROC-kurve", &lr.roc_points, lr.auc));
                    save_svg_button(ui, "man_cal.svg", || svg_calibration_plot("Kalibrering", &lr.hosmer_lemeshow.bins));
                });
                callout_box(ui, &lr.auc_status, &lr.interpretation, &lr.recommendation);
                ui.add_space(8.0);
                responsive_two_col(
                    ui,
                    |ui| plot_roc(ui, "man_roc_p", &lr.roc_points, lr.auc, 250.0),
                    |ui| plot_calibration(ui, "man_cal_p", &lr.hosmer_lemeshow.bins, 250.0),
                );
            });
        }
        ManualResult::Survival(cox) => {
            show_card(ui, |ui| {
                ui.label(RichText::new(format!("Cox PH & Schoenfeld ({} / {})", cox.time_col, cox.event_col)).strong().color(TEXT_PRIMARY).size(15.0));
                callout_box(ui, &cox.status, &cox.interpretation, &cox.recommendation);
                ui.add_space(8.0);
                show_table(ui, "man_cox_tbl", |ui| {
                    table_header_row(ui, &["Kovariat", "HR", "Wald p", "Schoenfeld r", "Schoenfeld p", "PH"]);
                    for (i, c) in cox.coefficients.iter().enumerate() {
                        let st = &cox.schoenfeld_tests[i];
                        ui.label(bold_text(&c.name, 13.0, TEXT_PRIMARY));
                        ui.label(format!("{:.3}", c.hazard_ratio));
                        ui.label(format!("{:.4}", c.p_value));
                        ui.label(format!("{:.3}", st.rho_time));
                        ui.label(format!("{:.4}", st.p_value));
                        status_badge(ui, if st.ph_met { "ok" } else { "fail" });
                        ui.end_row();
                    }
                });
                if let Some(st) = cox.schoenfeld_tests.first() {
                    let beta = cox.coefficients[0].coef;
                    ui.add_space(8.0);
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new(format!("Schoenfeld-plott: {}", st.variable)).strong().color(TEXT_PRIMARY));
                        save_svg_button(ui, "man_schoenfeld.svg", || {
                            svg_schoenfeld(&format!("Schoenfeld — {}", st.variable), &st.variable, &st.points, beta)
                        });
                    });
                    plot_schoenfeld(ui, "man_sch_p", &st.points, beta, 250.0);
                }
            });
        }
        ManualResult::Clustering(cl) => {
            show_card(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new("Klyngeanalyse (Silhouette & Elbow)").strong().color(TEXT_PRIMARY).size(15.0));
                    save_svg_button(ui, "man_elbow_sil.svg", || svg_elbow_silhouette("Elbow & Silhouette", &cl.k_evals, cl.elbow_k, cl.best_k_silhouette));
                    save_svg_button(ui, "man_klynger_2d.svg", || svg_clusters_2d("Klynger 2D", &cl.points_2d));
                });
                callout_box(ui, &cl.status, &cl.interpretation, &cl.recommendation);
                ui.add_space(8.0);
                plot_elbow_and_silhouette(ui, "man_cl", &cl.k_evals, 230.0);
                ui.add_space(8.0);
                plot_clusters_2d(ui, "man_cl_2d", &cl.points_2d, 250.0);
            });
        }
    }
}
