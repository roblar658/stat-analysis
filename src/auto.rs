use crate::data::{ColumnInfo, Dataset};
use crate::plots::{
    svg_acf_plot, svg_boxplot, svg_calibration_plot, svg_clusters_2d, svg_corr_heatmap,
    svg_elbow_silhouette, svg_histogram, svg_leverage_cooks, svg_missingness,
    svg_qq_plot, svg_residuals_vs_fitted, svg_roc_curve, svg_schoenfeld, svg_scree_plot,
    svg_sphericity_profile, NamedChart,
};
use crate::stats::clustering::{analyze_clustering, ClusteringReport};
use crate::stats::factor::{check_factor_suitability, FactorSuitabilityReport};
use crate::stats::logistic::{analyze_logistic_regression, LogisticReport};
use crate::stats::missing::{analyze_missing_data, MissingDataReport};
use crate::stats::normality::{
    bootstrap_mean, check_normality, kruskal_wallis, mann_whitney_u, BootstrapResult,
    KruskalWallisResult, MannWhitneyResult, NormalityCheck,
};
use crate::stats::regression::{full_linear_regression_diagnostics, LinearRegressionFullReport};
use crate::stats::sphericity::{check_sphericity, SphericityReport};
use crate::stats::survival::{analyze_cox_ph, CoxSurvivalReport};
use crate::stats::variance::{check_variance_homogeneity, VarianceHomogeneityReport};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryBadge {
    pub check_name: String,
    pub category: String,
    pub status: String, // "ok" | "warning" | "fail" | "na"
    pub headline: String,
    pub action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupFallbackTests {
    pub value_col: String,
    pub group_col: String,
    pub mann_whitney: Option<MannWhitneyResult>,
    pub kruskal_wallis: Option<KruskalWallisResult>,
    pub bootstraps: Vec<BootstrapResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoAnalysisReport {
    pub filename: String,
    pub generated_at: String,
    pub row_count: usize,
    pub col_count: usize,
    pub columns: Vec<ColumnInfo>,
    pub summary_badges: Vec<SummaryBadge>,
    // Part 1: Assumptions
    pub normality_checks: Vec<NormalityCheck>,
    pub variance_homogeneity: Option<VarianceHomogeneityReport>,
    pub group_fallbacks: Option<GroupFallbackTests>,
    pub sphericity: Option<SphericityReport>,
    pub missing_data: MissingDataReport,
    pub factor_suitability: Option<FactorSuitabilityReport>,
    // Part 1 + 2: Linear Regression & Diagnostics
    pub linear_regression: Option<LinearRegressionFullReport>,
    // Part 2: Logistic, Survival, Clustering
    pub logistic_regression: Option<LogisticReport>,
    pub cox_survival: Option<CoxSurvivalReport>,
    pub clustering: Option<ClusteringReport>,
    // All generated SVG charts
    pub charts: Vec<NamedChart>,
    pub notes: Vec<String>,
}

fn sanitize_id(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
        .collect()
}

fn detect_survival_pair(ds: &Dataset) -> Option<(String, String)> {
    let time_keywords = ["time", "tid", "duration", "varighet", "days", "dager", "months", "mnd", "surv", "followup"];
    let event_keywords = ["event", "hendelse", "status", "dead", "dod", "død", "failure", "cens", "outcome_event"];

    let mut time_col: Option<String> = None;
    for col in &ds.columns {
        let lower = col.name.to_ascii_lowercase();
        if time_keywords.iter().any(|kw| lower.contains(kw)) {
            if let (Some(mn), Some(_)) = (col.min, col.max) {
                if mn >= 0.0 && col.unique_count >= 5 {
                    time_col = Some(col.name.clone());
                    break;
                }
            }
        }
    }

    let mut event_col: Option<String> = None;
    for col in &ds.columns {
        if Some(&col.name) == time_col.as_ref() {
            continue;
        }
        let lower = col.name.to_ascii_lowercase();
        if col.unique_count == 2 && event_keywords.iter().any(|kw| lower.contains(kw)) {
            event_col = Some(col.name.clone());
            break;
        }
    }
    if time_col.is_some() && event_col.is_none() {
        // Fallback to any binary column
        event_col = ds.binary_column_names().into_iter().next();
    }

    match (time_col, event_col) {
        (Some(t), Some(e)) if t != e => Some((t, e)),
        _ => None,
    }
}

fn detect_repeated_measures(ds: &Dataset) -> Vec<String> {
    let num_cols = ds.numeric_column_names();
    let rm_prefixes = ["t1", "t2", "t3", "t4", "m1", "m2", "m3", "pre", "mid", "post", "time1", "time2", "time3", "maal1", "maal2", "maal3", "måling", "wave"];
    let matched: Vec<String> = num_cols
        .iter()
        .filter(|name| {
            let l = name.to_ascii_lowercase();
            rm_prefixes.iter().any(|p| l.starts_with(p) || l.contains(&format!("_{}", p)))
        })
        .cloned()
        .collect();
    if matched.len() >= 3 {
        return matched.into_iter().take(6).collect();
    }
    // Otherwise pick up to 4 numeric columns (excluding id-like / time columns)
    let filtered: Vec<String> = num_cols
        .into_iter()
        .filter(|n| {
            let l = n.to_ascii_lowercase();
            l != "id" && !l.ends_with("_id")
        })
        .take(4)
        .collect();
    if filtered.len() >= 3 {
        filtered
    } else {
        vec![]
    }
}

fn pick_regression_target(num_cols: &[String], exclude: &[String]) -> Option<(String, Vec<String>)> {
    let candidates: Vec<String> = num_cols
        .iter()
        .filter(|c| {
            let l = c.to_ascii_lowercase();
            l != "id" && !l.ends_with("_id") && !exclude.contains(c)
        })
        .cloned()
        .collect();
    if candidates.len() < 2 {
        return None;
    }
    let target_keywords = ["y", "target", "outcome", "utfall", "score", "resultat", "pris", "price", "salg", "sales", "response"];
    let target_idx = candidates
        .iter()
        .position(|c| {
            let l = c.to_ascii_lowercase();
            target_keywords.iter().any(|kw| l == *kw || l.contains(kw))
        })
        .unwrap_or(candidates.len() - 1);

    let target = candidates[target_idx].clone();
    let predictors: Vec<String> = candidates
        .into_iter()
        .enumerate()
        .filter(|&(i, _)| i != target_idx)
        .map(|(_, name)| name)
        .take(10)
        .collect();
    Some((target, predictors))
}

pub fn run_automatic_analysis(ds: &Dataset, alpha: f64) -> AutoAnalysisReport {
    let mut charts: Vec<NamedChart> = Vec::new();
    let mut summary_badges: Vec<SummaryBadge> = Vec::new();
    let mut notes: Vec<String> = Vec::new();

    let all_num_cols: Vec<String> = ds
        .numeric_column_names()
        .into_iter()
        .filter(|n| {
            let l = n.to_ascii_lowercase();
            l != "id" && !l.ends_with("_id")
        })
        .collect();

    // 1. Missing Data & Little's MCAR
    let num_for_missing: Vec<String> = all_num_cols.iter().take(12).cloned().collect();
    let num_mat_missing = ds
        .get_numeric_matrix_with_missing(&num_for_missing)
        .unwrap_or_default();
    let missing_data = analyze_missing_data(
        &ds.headers,
        &ds.raw_rows,
        &num_for_missing,
        &num_mat_missing,
        alpha,
        5,
    );
    charts.push(NamedChart {
        id: "missingness".to_string(),
        filename: "01_manglende_verdier.svg".to_string(),
        title: "Manglende verdier per kolonne".to_string(),
        category: "Manglende data".to_string(),
        description: missing_data.interpretation.clone(),
        svg: svg_missingness("Manglende verdier per kolonne (%)", &missing_data.columns),
    });
    summary_badges.push(SummaryBadge {
        check_name: "Tilfeldig manglende data (Littles MCAR)".to_string(),
        category: "Forutsetninger".to_string(),
        status: missing_data.status.clone(),
        headline: if let Some(p) = missing_data.mcar_p_value {
            format!("{:.1}% mangler | MCAR p = {:.4}", missing_data.overall_missing_pct, p)
        } else {
            format!("{:.1}% mangler", missing_data.overall_missing_pct)
        },
        action: missing_data.recommendation.clone(),
    });

    // 2. Normality checks for all numeric columns (up to 8 charts)
    let mut normality_checks = Vec::new();
    for (idx, col) in all_num_cols.iter().enumerate() {
        if let Ok(vals) = ds.get_numeric_values(col) {
            if let Ok(nc) = check_normality(col, &vals, alpha) {
                if idx < 4 {
                    let cid = sanitize_id(col);
                    charts.push(NamedChart {
                        id: format!("hist_{}", cid),
                        filename: format!("02_histogram_{}.svg", cid),
                        title: format!("Histogram: {}", col),
                        category: "Normalfordeling".to_string(),
                        description: nc.interpretation.clone(),
                        svg: svg_histogram(&format!("Histogram og normalfordeling — {}", col), col, &vals),
                    });
                    charts.push(NamedChart {
                        id: format!("qq_{}", cid),
                        filename: format!("02_qq_{}.svg", cid),
                        title: format!("Q-Q-plott: {}", col),
                        category: "Normalfordeling".to_string(),
                        description: format!("Skjevhet = {:.3}, eksess-kurtose = {:.3}", nc.skewness, nc.excess_kurtosis),
                        svg: svg_qq_plot(&format!("Normal Q-Q-plott — {}", col), &vals),
                    });
                }
                normality_checks.push(nc);
            }
        }
    }
    if !normality_checks.is_empty() {
        let non_norm = normality_checks.iter().filter(|n| !n.is_normal).count();
        let status = if non_norm == 0 {
            "ok"
        } else if non_norm < normality_checks.len() {
            "warning"
        } else {
            "fail"
        }
        .to_string();
        summary_badges.push(SummaryBadge {
            check_name: "Normalfordeling (Shapiro-Wilk / KS)".to_string(),
            category: "Forutsetninger".to_string(),
            status,
            headline: format!(
                "{} av {} variabler oppfyller normalitet",
                normality_checks.len() - non_norm,
                normality_checks.len()
            ),
            action: if non_norm == 0 {
                "Parametriske metoder er trygge å bruke.".to_string()
            } else {
                "Bruk Mann-Whitney, Kruskal-Wallis eller bootstrapping for ikke-normale variabler.".to_string()
            },
        });
    }

    // 3. Variance Homogeneity + Non-parametric fallbacks (if grouping column exists)
    let group_cols = ds.grouping_column_names();
    let mut variance_homogeneity = None;
    let mut group_fallbacks = None;

    if let (Some(g_col), Some(v_col)) = (group_cols.first(), all_num_cols.first()) {
        // Prefer a numeric target-like column if available
        let target_v = all_num_cols
            .iter()
            .find(|c| {
                let l = c.to_ascii_lowercase();
                l == "y" || l.contains("score") || l.contains("outcome") || l.contains("utfall")
            })
            .unwrap_or(v_col);

        if let Ok(map) = ds.get_grouped_values(target_v, g_col) {
            let groups: Vec<(String, Vec<f64>)> = map
                .into_iter()
                .filter(|(_, v)| v.len() >= 2)
                .collect();
            if groups.len() >= 2 {
                if let Ok(vh) = check_variance_homogeneity(target_v, g_col, &groups, alpha) {
                    charts.push(NamedChart {
                        id: "boxplot_groups".to_string(),
                        filename: "03_boksplott_grupper.svg".to_string(),
                        title: format!("Boksplott: {} per {}", target_v, g_col),
                        category: "Varianshomogenitet".to_string(),
                        description: vh.interpretation.clone(),
                        svg: svg_boxplot(
                            &format!("Boksplott: {} fordelt på {}", target_v, g_col),
                            target_v,
                            &groups,
                        ),
                    });
                    summary_badges.push(SummaryBadge {
                        check_name: "Varianshomogenitet (Levene / Bartlett)".to_string(),
                        category: "Forutsetninger".to_string(),
                        status: vh.status.clone(),
                        headline: format!(
                            "Levene p = {:.4} | Bartlett p = {:.4}",
                            vh.levene_median.p_value, vh.bartlett.p_value
                        ),
                        action: vh.recommendation.clone(),
                    });
                    variance_homogeneity = Some(vh);

                    // Compute non-parametric & bootstrap fallbacks too
                    let mw = if groups.len() == 2 {
                        mann_whitney_u(&groups[0].0, &groups[0].1, &groups[1].0, &groups[1].1, alpha).ok()
                    } else {
                        None
                    };
                    let kw = kruskal_wallis(&groups, alpha).ok();
                    let bootstraps: Vec<BootstrapResult> = groups
                        .iter()
                        .filter_map(|(gn, gv)| bootstrap_mean(&format!("{} ({})", target_v, gn), gv, 1000).ok())
                        .collect();
                    group_fallbacks = Some(GroupFallbackTests {
                        value_col: target_v.clone(),
                        group_col: g_col.clone(),
                        mann_whitney: mw,
                        kruskal_wallis: kw,
                        bootstraps,
                    });
                }
            }
        }
    } else {
        notes.push("Ingen kategorisk gruppekolonne funnet for automatisk Levene/Bartlett-test — kan kjøres manuelt.".to_string());
    }

    // 4. Sphericity (Repeated Measures ANOVA)
    let rm_cols = detect_repeated_measures(ds);
    let mut sphericity = None;
    if rm_cols.len() >= 3 {
        if let Ok(rows) = ds.get_complete_numeric_matrix(&rm_cols) {
            if let Ok(sp) = check_sphericity(&rm_cols, &rows, alpha) {
                charts.push(NamedChart {
                    id: "sphericity_profile".to_string(),
                    filename: "04_sfaerisitet_profil.svg".to_string(),
                    title: "Profilplott for repeterte målinger".to_string(),
                    category: "Sfærisitet".to_string(),
                    description: sp.interpretation.clone(),
                    svg: svg_sphericity_profile(
                        "Gjennomsnitt på tvers av repeterte målinger",
                        &sp.columns,
                        &sp.condition_means,
                    ),
                });
                summary_badges.push(SummaryBadge {
                    check_name: "Sfærisitet (Mauchlys test)".to_string(),
                    category: "Forutsetninger".to_string(),
                    status: sp.status.clone(),
                    headline: format!(
                        "W = {:.3}, p = {:.4} | GG ε = {:.3}",
                        sp.mauchly_w, sp.mauchly_p_value, sp.epsilon_gg
                    ),
                    action: sp.recommendation.clone(),
                });
                sphericity = Some(sp);
            }
        }
    }

    // 5. Survival Analysis (detect early so we don't accidentally use survival time as OLS target if better target exists)
    let surv_pair = detect_survival_pair(ds);
    let mut cox_survival = None;
    if let Some((ref t_col, ref e_col)) = surv_pair {
        let covs: Vec<String> = all_num_cols
            .iter()
            .filter(|c| *c != t_col && *c != e_col)
            .take(6)
            .cloned()
            .collect();
        if !covs.is_empty() {
            if let Ok((ev_opt, _, _)) = ds.get_binary_series(e_col) {
                let mut cols_needed = vec![t_col.clone()];
                cols_needed.extend(covs.clone());
                if let Ok(mat_opt) = ds.get_numeric_matrix_with_missing(&cols_needed) {
                    let mut times = Vec::new();
                    let mut events = Vec::new();
                    let mut x_cols = vec![Vec::new(); covs.len()];
                    for (i, row) in mat_opt.iter().enumerate() {
                        if let (Some(t), Some(e)) = (row[0], ev_opt[i]) {
                            if t > 0.0 && row[1..].iter().all(|v| v.is_some()) {
                                times.push(t);
                                events.push(e);
                                for j in 0..covs.len() {
                                    x_cols[j].push(row[j + 1].unwrap());
                                }
                            }
                        }
                    }
                    if let Ok(cox) = analyze_cox_ph(t_col, e_col, &times, &events, &x_cols, &covs, alpha) {
                        for st in cox.schoenfeld_tests.iter().take(2) {
                            let beta = cox
                                .coefficients
                                .iter()
                                .find(|c| c.name == st.variable)
                                .map(|c| c.coef)
                                .unwrap_or(0.0);
                            let sid = sanitize_id(&st.variable);
                            charts.push(NamedChart {
                                id: format!("schoenfeld_{}", sid),
                                filename: format!("11_schoenfeld_{}.svg", sid),
                                title: format!("Schoenfeld-residualer: {}", st.variable),
                                category: "Overlevelsesanalyse (Cox)".to_string(),
                                description: format!(
                                    "Korrelasjon med tid r = {:.3}, χ² = {:.3}, p = {:.4}",
                                    st.rho_time, st.chi2_stat, st.p_value
                                ),
                                svg: svg_schoenfeld(
                                    &format!("Schoenfeld-residualer mot tid — {}", st.variable),
                                    &st.variable,
                                    &st.points,
                                    beta,
                                ),
                            });
                        }
                        summary_badges.push(SummaryBadge {
                            check_name: "Cox proporsjonal hasard (Schoenfeld)".to_string(),
                            category: "Modellkvalitet".to_string(),
                            status: cox.status.clone(),
                            headline: format!(
                                "Global p = {:.4} | C-index = {:.3}",
                                cox.schoenfeld_global_p, cox.concordance_c
                            ),
                            action: cox.recommendation.clone(),
                        });
                        cox_survival = Some(cox);
                    }
                }
            }
        }
    } else {
        notes.push("Ingen tydelig overlevelseskolonne (tid + hendelse) gjenkjent automatisk — Cox/Schoenfeld kan kjøres manuelt.".to_string());
    }

    // 6. Linear Regression + Multicollinearity + Autocorrelation + Heteroscedasticity + Outliers + R2/RMSE/AIC/BIC
    let mut exclude_reg: Vec<String> = Vec::new();
    if let Some((ref t_col, ref e_col)) = surv_pair {
        exclude_reg.push(t_col.clone());
        exclude_reg.push(e_col.clone());
    }
    let reg_pair = pick_regression_target(&all_num_cols, &exclude_reg)
        .or_else(|| pick_regression_target(&all_num_cols, &[]));

    let mut linear_regression = None;
    if let Some((target, preds)) = reg_pair {
        let mut all_cols = vec![target.clone()];
        all_cols.extend(preds.clone());
        if let Ok(rows) = ds.get_complete_numeric_matrix(&all_cols) {
            let y: Vec<f64> = rows.iter().map(|r| r[0]).collect();
            let x_cols: Vec<Vec<f64>> = (0..preds.len())
                .map(|j| rows.iter().map(|r| r[j + 1]).collect())
                .collect();
            if let Ok(reg) = full_linear_regression_diagnostics(&target, &y, &x_cols, &preds, alpha, 2, 1.0) {
                // Residual Q-Q & Histogram
                charts.push(NamedChart {
                    id: "reg_resid_qq".to_string(),
                    filename: "05_regresjon_residual_qq.svg".to_string(),
                    title: format!("Residual Q-Q-plott ({})", target),
                    category: "Lineær regresjon".to_string(),
                    description: reg.residual_normality.interpretation.clone(),
                    svg: svg_qq_plot(&format!("Residual Q-Q-plott — {}", target), &reg.residuals),
                });
                charts.push(NamedChart {
                    id: "reg_resid_vs_fitted".to_string(),
                    filename: "06_residualer_vs_predikert.svg".to_string(),
                    title: "Residualplott (Predikert vs. Residualer)".to_string(),
                    category: "Heteroskedastisitet".to_string(),
                    description: reg.heteroscedasticity.interpretation.clone(),
                    svg: svg_residuals_vs_fitted(
                        &format!("Residualer mot predikerte verdier — {}", target),
                        &reg.fitted,
                        &reg.residuals,
                    ),
                });
                charts.push(NamedChart {
                    id: "reg_acf".to_string(),
                    filename: "07_autokorrelasjon_acf.svg".to_string(),
                    title: "ACF-plott (Residual-autokorrelasjon)".to_string(),
                    category: "Autokorrelasjon".to_string(),
                    description: reg.autocorrelation.interpretation.clone(),
                    svg: svg_acf_plot(
                        "Autokorrelasjonsplott (ACF) av residualer",
                        &reg.autocorrelation.acf_lags,
                        reg.autocorrelation.acf_conf_bound,
                    ),
                });
                charts.push(NamedChart {
                    id: "reg_leverage".to_string(),
                    filename: "08_uteliggere_leverage_cooks.svg".to_string(),
                    title: "Leverage- og Cook's Distance-plott".to_string(),
                    category: "Uteliggere og innflytelse".to_string(),
                    description: reg.outliers.interpretation.clone(),
                    svg: svg_leverage_cooks(
                        "Standardiserte residualer mot Leverage (Cook's D)",
                        &reg.outliers.leverage,
                        &reg.outliers.standardized_residuals,
                        &reg.outliers.cooks_d,
                        reg.outliers.cooks_threshold_4_over_n,
                    ),
                });
                if let Some(ref mc) = reg.multicollinearity {
                    charts.push(NamedChart {
                        id: "corr_heatmap".to_string(),
                        filename: "09_korrelasjonsmatrise.svg".to_string(),
                        title: "Korrelasjonsmatrise (Forklaringsvariabler)".to_string(),
                        category: "Multikollinearitet".to_string(),
                        description: mc.interpretation.clone(),
                        svg: svg_corr_heatmap(
                            "Korrelasjonsmatrise (rød ramme: |r| > 0.8)",
                            &mc.predictors,
                            &mc.correlation_matrix,
                        ),
                    });
                    summary_badges.push(SummaryBadge {
                        check_name: "Multikollinearitet (VIF)".to_string(),
                        category: "Forutsetninger".to_string(),
                        status: mc.status.clone(),
                        headline: format!(
                            "Maks VIF = {:.2} | {} par |r| > 0.8",
                            mc.max_vif,
                            mc.high_corr_pairs.len()
                        ),
                        action: mc.recommendation.clone(),
                    });
                }
                summary_badges.push(SummaryBadge {
                    check_name: "Autokorrelasjon (Durbin-Watson / BG)".to_string(),
                    category: "Forutsetninger".to_string(),
                    status: reg.autocorrelation.status.clone(),
                    headline: format!(
                        "DW = {:.3} | BG p = {:.4}",
                        reg.autocorrelation.durbin_watson, reg.autocorrelation.bg_p_value
                    ),
                    action: reg.autocorrelation.recommendation.clone(),
                });
                summary_badges.push(SummaryBadge {
                    check_name: "Heteroskedastisitet (Breusch-Pagan / White)".to_string(),
                    category: "Forutsetninger".to_string(),
                    status: reg.heteroscedasticity.status.clone(),
                    headline: format!(
                        "BP p = {:.4} | White p = {:.4}",
                        reg.heteroscedasticity.bp_p_value, reg.heteroscedasticity.white_p_value
                    ),
                    action: reg.heteroscedasticity.recommendation.clone(),
                });
                summary_badges.push(SummaryBadge {
                    check_name: "Uteliggere & innflytelse (Cook's D / Mahalanobis)".to_string(),
                    category: "Forutsetninger".to_string(),
                    status: reg.outliers.status.clone(),
                    headline: format!(
                        "Cook's D > 4/n: {} | Mahalanobis: {}",
                        reg.outliers.count_cooks_gt_4_n, reg.outliers.count_mahalanobis_outliers
                    ),
                    action: reg.outliers.recommendation.clone(),
                });
                summary_badges.push(SummaryBadge {
                    check_name: format!("Lineær regresjon — Tilpasning ({})", target),
                    category: "Modellkvalitet".to_string(),
                    status: if reg.adj_r_squared >= 0.5 {
                        "ok"
                    } else if reg.adj_r_squared >= 0.2 {
                        "warning"
                    } else {
                        "fail"
                    }
                    .to_string(),
                    headline: format!(
                        "R² = {:.3} (Just. {:.3}) | RMSE = {:.3} | MAE = {:.3} | AIC = {:.1}",
                        reg.r_squared, reg.adj_r_squared, reg.rmse, reg.mae, reg.aic
                    ),
                    action: "Sammenlign AIC/BIC mot reduserte modeller i tabellen under.".to_string(),
                });
                linear_regression = Some(reg);
            }
        }
    }

    // 7. Factor Analysis Suitability (KMO + Bartlett)
    let mut factor_suitability = None;
    let factor_cols: Vec<String> = all_num_cols.iter().take(12).cloned().collect();
    if factor_cols.len() >= 3 {
        if let Ok(rows) = ds.get_complete_numeric_matrix(&factor_cols) {
            if let Ok(fac) = check_factor_suitability(&factor_cols, &rows, alpha) {
                charts.push(NamedChart {
                    id: "factor_scree".to_string(),
                    filename: "10_faktoranalyse_scree.svg".to_string(),
                    title: "Scree-plott (PCA egenverdier)".to_string(),
                    category: "Egnethet for faktoranalyse".to_string(),
                    description: fac.interpretation.clone(),
                    svg: svg_scree_plot("Scree-plott — Egenverdier og Kaiser-kriterium", &fac.eigenvalues),
                });
                summary_badges.push(SummaryBadge {
                    check_name: "Egnethet for faktoranalyse (KMO / Bartlett)".to_string(),
                    category: "Forutsetninger".to_string(),
                    status: fac.status.clone(),
                    headline: format!(
                        "KMO = {:.3} | Bartlett p = {:.4}",
                        fac.kmo_overall, fac.bartlett_p_value
                    ),
                    action: fac.recommendation.clone(),
                });
                factor_suitability = Some(fac);
            }
        }
    }

    // 8. Logistic Regression (if any binary column exists)
    let mut logistic_regression = None;
    let bin_cols = ds.binary_column_names();
    if let Some(bin_target) = bin_cols
        .iter()
        .find(|c| surv_pair.as_ref().map(|(_, e)| e != *c).unwrap_or(true))
        .or_else(|| bin_cols.first())
    {
        let log_preds: Vec<String> = all_num_cols
            .iter()
            .filter(|c| *c != bin_target)
            .take(8)
            .cloned()
            .collect();
        if !log_preds.is_empty() {
            if let Ok((y_opt, neg_lbl, pos_lbl)) = ds.get_binary_series(bin_target) {
                if let Ok(x_opt) = ds.get_numeric_matrix_with_missing(&log_preds) {
                    let mut y = Vec::new();
                    let mut x_cols = vec![Vec::new(); log_preds.len()];
                    for (i, row) in x_opt.iter().enumerate() {
                        if let Some(yv) = y_opt[i] {
                            if row.iter().all(|v| v.is_some()) {
                                y.push(yv);
                                for j in 0..log_preds.len() {
                                    x_cols[j].push(row[j].unwrap());
                                }
                            }
                        }
                    }
                    if let Ok(lr) = analyze_logistic_regression(
                        bin_target,
                        &neg_lbl,
                        &pos_lbl,
                        &y,
                        &x_cols,
                        &log_preds,
                        10,
                        alpha,
                    ) {
                        charts.push(NamedChart {
                            id: "logistic_roc".to_string(),
                            filename: "12_logistisk_roc.svg".to_string(),
                            title: format!("ROC-kurve ({})", bin_target),
                            category: "Logistisk regresjon".to_string(),
                            description: format!("AUC = {:.3} — {}", lr.auc, lr.auc_label),
                            svg: svg_roc_curve(
                                &format!("ROC-kurve — {} ({})", bin_target, pos_lbl),
                                &lr.roc_points,
                                lr.auc,
                            ),
                        });
                        charts.push(NamedChart {
                            id: "logistic_calibration".to_string(),
                            filename: "13_hosmer_lemeshow_kalibrering.svg".to_string(),
                            title: "Hosmer-Lemeshow kalibreringsplott".to_string(),
                            category: "Logistisk regresjon".to_string(),
                            description: lr.interpretation.clone(),
                            svg: svg_calibration_plot(
                                "Hosmer-Lemeshow: Predikert vs. Observert rate",
                                &lr.hosmer_lemeshow.bins,
                            ),
                        });
                        summary_badges.push(SummaryBadge {
                            check_name: format!("Logistisk tilpasning — Hosmer-Lemeshow ({})", bin_target),
                            category: "Modellkvalitet".to_string(),
                            status: lr.hl_status.clone(),
                            headline: format!(
                                "χ²({}) = {:.2}, p = {:.4}",
                                lr.hosmer_lemeshow.df,
                                lr.hosmer_lemeshow.chi2_stat,
                                lr.hosmer_lemeshow.p_value
                            ),
                            action: lr.recommendation.clone(),
                        });
                        summary_badges.push(SummaryBadge {
                            check_name: format!("Klassifisering — ROC & AUC ({})", bin_target),
                            category: "Modellkvalitet".to_string(),
                            status: lr.auc_status.clone(),
                            headline: format!("AUC = {:.3} ({}) | AIC = {:.1}", lr.auc, lr.auc_label, lr.aic),
                            action: format!(
                                "Optimal grense = {:.3} (Nøyaktighet {:.1}%, F1 = {:.3}).",
                                lr.confusion_matrix.threshold,
                                lr.confusion_matrix.accuracy * 100.0,
                                lr.confusion_matrix.f1_score
                            ),
                        });
                        logistic_regression = Some(lr);
                    }
                }
            }
        }
    } else {
        notes.push("Ingen binær utfallskolonne funnet automatisk for logistisk regresjon (Hosmer-Lemeshow / ROC) — kan kjøres manuelt.".to_string());
    }

    // 9. Clustering (Silhouette + Elbow)
    let mut clustering = None;
    let clust_cols: Vec<String> = all_num_cols.iter().take(8).cloned().collect();
    if clust_cols.len() >= 2 {
        if let Ok(rows) = ds.get_complete_numeric_matrix(&clust_cols) {
            if let Ok(cl) = analyze_clustering(&clust_cols, &rows, 8) {
                charts.push(NamedChart {
                    id: "clustering_elbow_sil".to_string(),
                    filename: "14_klynge_elbow_silhouette.svg".to_string(),
                    title: "Klyngekvalitet: Elbow & Silhouette".to_string(),
                    category: "Klyngeanalyse".to_string(),
                    description: cl.interpretation.clone(),
                    svg: svg_elbow_silhouette(
                        "Klyngeanalyse — Elbow-metoden og Silhouette-skår",
                        &cl.k_evals,
                        cl.elbow_k,
                        cl.best_k_silhouette,
                    ),
                });
                charts.push(NamedChart {
                    id: "clustering_pca2d".to_string(),
                    filename: "15_klynger_pca2d.svg".to_string(),
                    title: format!("Klynger i 2D PCA-rom (k = {})", cl.best_k_silhouette),
                    category: "Klyngeanalyse".to_string(),
                    description: cl.recommendation.clone(),
                    svg: svg_clusters_2d(
                        &format!("K-means klynger (k = {}, Silhouette = {:.3})", cl.best_k_silhouette, cl.best_silhouette),
                        &cl.points_2d,
                    ),
                });
                summary_badges.push(SummaryBadge {
                    check_name: "Klyngeanalyse (Silhouette / Elbow)".to_string(),
                    category: "Modellkvalitet".to_string(),
                    status: cl.status.clone(),
                    headline: format!(
                        "Beste Silhouette = {:.3} (k={}) | Elbow k={}",
                        cl.best_silhouette, cl.best_k_silhouette, cl.elbow_k
                    ),
                    action: cl.recommendation.clone(),
                });
                clustering = Some(cl);
            }
        }
    }

    AutoAnalysisReport {
        filename: ds.filename.clone(),
        generated_at: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        row_count: ds.row_count(),
        col_count: ds.headers.len(),
        columns: ds.columns.clone(),
        summary_badges,
        normality_checks,
        variance_homogeneity,
        group_fallbacks,
        sphericity,
        missing_data,
        factor_suitability,
        linear_regression,
        logistic_regression,
        cox_survival,
        clustering,
        charts,
        notes,
    }
}
