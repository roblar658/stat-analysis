use crate::auto::run_automatic_analysis;
use crate::data::Dataset;
use crate::report::{build_zip_bundle, render_markdown_report, render_pdf_report};
use crate::stats::normality::{check_normality, shapiro_wilk};
use crate::stats::regression::check_multicollinearity;

#[test]
fn test_full_automatic_pipeline_on_example_csv() {
    let csv_data = include_str!("../examples/comprehensive_study.csv");
    let ds = Dataset::from_csv_str("comprehensive_study.csv", csv_data).expect("parse csv");
    assert_eq!(ds.row_count(), 120);

    let rep = run_automatic_analysis(&ds, 0.05);

    // Part 1 & Part 2 modules must all execute cleanly on the comprehensive study dataset
    assert!(!rep.normality_checks.is_empty());
    assert!(rep.variance_homogeneity.is_some());
    assert!(rep.group_fallbacks.is_some());
    assert!(rep.sphericity.is_some());
    assert!(rep.missing_data.mcar_applicable);
    assert!(rep.missing_data.mcar_p_value.unwrap().is_finite());
    assert!(!rep.missing_data.mice_summary.is_empty());
    assert!(rep.factor_suitability.is_some());
    assert!(rep.linear_regression.is_some());
    assert!(rep.logistic_regression.is_some());
    assert!(rep.cox_survival.is_some());
    assert!(rep.clustering.is_some());

    // Check regression diagnostics
    let reg = rep.linear_regression.as_ref().unwrap();
    assert!(reg.r_squared > 0.0 && reg.r_squared <= 1.0);
    assert!(reg.rmse > 0.0 && reg.mae > 0.0);
    assert!(reg.multicollinearity.is_some());
    let mc = reg.multicollinearity.as_ref().unwrap();
    // x1_inntekt and x2_forbruk are collinear by design
    assert!(mc.max_vif > 5.0, "Expected high VIF, got {}", mc.max_vif);
    assert!(reg.outliers.count_cooks_gt_4_n >= 1);
    assert!(reg.model_comparison.len() >= 2);

    // Check Logistic
    let lr = rep.logistic_regression.as_ref().unwrap();
    assert!(lr.auc > 0.55 && lr.auc <= 1.0);
    assert!(lr.hosmer_lemeshow.p_value.is_finite());

    // Check Cox PH & Schoenfeld
    let cox = rep.cox_survival.as_ref().unwrap();
    assert!(cox.n_events >= 10);
    assert!(cox.schoenfeld_global_p.is_finite());

    // Check Clustering
    let cl = rep.clustering.as_ref().unwrap();
    assert!(cl.best_silhouette > 0.1 && cl.best_silhouette <= 1.0);
    assert!(cl.k_evals.len() >= 5);

    // Check Charts & Reports & ZIP
    assert!(rep.charts.len() >= 12);
    for ch in &rep.charts {
        assert!(ch.svg.starts_with("<svg"));
        assert!(ch.svg.ends_with("</svg>"));
    }
    let pdf = render_pdf_report(&rep);
    assert!(pdf.starts_with(b"%PDF-1.4"));
    assert!(pdf.ends_with(b"%%EOF\n"));
    assert!(pdf.len() > 4000, "PDF too small: {} bytes", pdf.len());
    let md = render_markdown_report(&rep);
    assert!(md.contains("Littles MCAR"));
    let zip_bytes = build_zip_bundle(&rep).expect("zip creation");
    assert!(zip_bytes.len() > 2000);
}

#[test]
fn test_shapiro_wilk_distinguishes_normal_vs_skewed() {
    let normal_like = vec![
        -1.6, -1.2, -0.9, -0.6, -0.4, -0.2, 0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.7,
    ];
    let sw_norm = shapiro_wilk(&normal_like).unwrap();
    assert!(sw_norm.w > 0.95);
    assert!(sw_norm.p_value > 0.2, "Expected normal p > 0.2, got {}", sw_norm.p_value);

    let skewed = vec![1.0, 1.1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.0, 3.0, 8.0, 25.0, 90.0];
    let sw_skew = shapiro_wilk(&skewed).unwrap();
    assert!(sw_skew.p_value < 0.01, "Expected skewed p < 0.01, got {}", sw_skew.p_value);

    let nc = check_normality("skewed", &skewed, 0.05).unwrap();
    assert!(!nc.is_normal);
}

#[test]
fn test_vif_detects_collinearity() {
    let x1: Vec<f64> = (1..=30).map(|i| i as f64).collect();
    let x2: Vec<f64> = x1.iter().enumerate().map(|(i, &v)| 2.0 * v + ((i % 3) as f64) * 0.1).collect();
    let x3: Vec<f64> = (1..=30).map(|i| ((i * 7) % 11) as f64).collect();
    let names = vec!["x1".into(), "x2".into(), "x3".into()];
    let mc = check_multicollinearity(None, &[x1, x2, x3], &names, 1.0).unwrap();
    assert!(mc.max_vif > 50.0);
    assert_eq!(mc.status, "fail");
}

#[test]
fn test_excel_windows1252_csv_norwegian() {
    // Raw Windows-1252 bytes: "sep=;\r\nMåling;Lønn;Grønn\r\n12,5;1\xA0250,5;Bær\r\n15,2;1\xA0400,0;Øl\r\n"
    let mut raw: Vec<u8> = Vec::new();
    raw.extend_from_slice(b"sep=;\r\nM");
    raw.push(0xE5); // å in CP1252
    raw.extend_from_slice(b"ling;L");
    raw.push(0xF8); // ø in CP1252
    raw.extend_from_slice(b"nn;Kategori\r\n12,5;1\xA0250,5;B");
    raw.push(0xE6); // æ in CP1252
    raw.extend_from_slice(b"r\r\n15,2;1\xA0400,0;");
    raw.push(0xD8); // Ø in CP1252
    raw.extend_from_slice(b"l\r\n");

    let ds = Dataset::from_csv_bytes("excel_norsk.csv", &raw).expect("parse Excel CP1252 CSV");
    assert_eq!(ds.delimiter, ';');
    assert_eq!(ds.headers, vec!["Måling", "Lønn", "Kategori"]);
    let vals = ds.get_numeric_values("Lønn").unwrap();
    assert!((vals[0] - 1250.5).abs() < 1e-6);
    assert!((vals[1] - 1400.0).abs() < 1e-6);
}

#[test]
fn test_all_10_standard_example_datasets() {
    assert_eq!(crate::ui::EXAMPLE_DATASETS.len(), 10);
    for ex in crate::ui::EXAMPLE_DATASETS {
        let ds = Dataset::from_csv_str(ex.filename, ex.csv)
            .unwrap_or_else(|e| panic!("Feil i {}: {}", ex.filename, e));
        let rep = run_automatic_analysis(&ds, 0.05);
        assert!(
            !rep.summary_badges.is_empty(),
            "Ingen sjekker kjørt for {}",
            ex.filename
        );
        assert!(
            !rep.charts.is_empty(),
            "Ingen figurer generert for {}",
            ex.filename
        );
    }
}

#[test]
fn test_pdf_report_includes_manual_appendix() {
    use crate::report::{render_pdf_report_with_appendix, ManualCalcResult, ManualHistoryEntry};
    use crate::stats::normality::bootstrap_mean;

    let csv_data = include_str!("../examples/01_komplett_studie.csv");
    let ds = Dataset::from_csv_str("01_komplett_studie.csv", csv_data).unwrap();
    let rep = run_automatic_analysis(&ds, 0.05);

    let vals = ds.get_numeric_values("utfall_y").unwrap();
    let nc = check_normality("utfall_y", &vals, 0.05).unwrap();
    let boot = bootstrap_mean("utfall_y", &vals, 500).ok();

    let entries = vec![
        ManualHistoryEntry {
            timestamp: "16:30:00".into(),
            title: "Normalfordeling (utfall_y)".into(),
            parameters: "Variabel = utfall_y, alpha = 0.05, Bootstrap B = 500".into(),
            result: ManualCalcResult::Normality {
                check: nc,
                boot,
                values: vals,
            },
        },
        ManualHistoryEntry {
            timestamp: "16:31:15".into(),
            title: "Lineær regresjon (utfall_y)".into(),
            parameters: "Y = utfall_y, X = [alder, x1_inntekt]".into(),
            result: ManualCalcResult::Regression(rep.linear_regression.clone().unwrap()),
        },
    ];

    let pdf = render_pdf_report_with_appendix(&rep, &entries);
    assert!(pdf.starts_with(b"%PDF-1.4"));
    assert!(pdf.ends_with(b"%%EOF\n"));
    let pdf_str = String::from_utf8_lossy(&pdf);
    assert!(pdf_str.contains("APPENDIKS"), "PDF missing APPENDIKS banner");
    assert!(pdf_str.contains("Appendiks A.1"), "PDF missing Appendiks A.1");
    assert!(pdf_str.contains("Appendiks A.2"), "PDF missing Appendiks A.2");
    assert!(pdf_str.contains("Matematiske formler"), "PDF missing formulas section");
    assert!(pdf_str.contains("Shapiro-Wilk"), "PDF missing Shapiro-Wilk formula");
    assert!(pdf_str.contains("Cohen's d"), "PDF missing Student t-test formula");
}

#[test]
fn test_automatic_t_test_selection_when_assumptions_met() {
    // 2 normal groups with equal variance -> automatic decision engine MUST choose Student's independent t-test
    let csv = "gruppe,score\n\
A,10.1\nA,10.5\nA,9.8\nA,10.3\nA,9.9\nA,10.2\nA,10.0\nA,9.7\nA,10.4\nA,10.1\n\
B,12.2\nB,12.6\nB,11.9\nB,12.4\nB,12.0\nB,12.3\nB,12.1\nB,11.8\nB,12.5\nB,12.2\n";
    let ds = Dataset::from_csv_str("ttest_case.csv", csv).unwrap();
    let rep = run_automatic_analysis(&ds, 0.05);
    let chosen = rep
        .recommended_analyses
        .iter()
        .find(|r| r.outcome == "score")
        .expect("Expected recommended analysis for score");
    assert_eq!(chosen.chosen_test, "Student uavhengig t-test");
    assert!(chosen.significant, "Expected significant t-test difference");
    assert!(chosen.formula_and_calc.contains("Cohen's d"));
}

#[test]
fn test_all_individual_statistical_methods_and_fallbacks() {
    use crate::stats::clustering::analyze_clustering;
    use crate::stats::factor::check_factor_suitability;
    use crate::stats::logistic::analyze_logistic_regression;
    use crate::stats::missing::analyze_missing_data;
    use crate::stats::normality::{
        bootstrap_mean, kolmogorov_smirnov_normal, kruskal_wallis, mann_whitney_u,
    };
    use crate::stats::regression::full_linear_regression_diagnostics;
    use crate::stats::sphericity::check_sphericity;
    use crate::stats::survival::analyze_cox_ph;
    use crate::stats::variance::check_variance_homogeneity;

    // 1. Normality (KS/Lilliefors n>=50), Mann-Whitney, Kruskal-Wallis, Bootstrap
    let ds2 = Dataset::from_csv_str("02", include_str!("../examples/02_normalfordeling_sma_utvalg.csv")).unwrap();
    let skew_vals = ds2.get_numeric_values("ventetid_skjev").unwrap();
    let ks = kolmogorov_smirnov_normal(&skew_vals).unwrap();
    assert!(ks.d_stat > 0.0 && ks.p_value_lilliefors.is_finite());
    let boot = bootstrap_mean("ventetid_skjev", &skew_vals, 500).unwrap();
    assert!(boot.ci_low_95 < boot.observed_mean && boot.observed_mean < boot.ci_high_95);
    let grp2: Vec<(String, Vec<f64>)> = ds2.get_grouped_values("ventetid_skjev", "avdeling").unwrap().into_iter().collect();
    let mw = mann_whitney_u(&grp2[0].0, &grp2[0].1, &grp2[1].0, &grp2[1].1, 0.05).unwrap();
    assert!(mw.p_value.is_finite());
    let kw = kruskal_wallis(&grp2, 0.05).unwrap();
    assert!(kw.h_stat >= 0.0 && kw.p_value.is_finite());

    // 2. Variance homogeneity (Levene, Bartlett, Welch t/ANOVA) — dataset 03 has unequal variance
    let ds3 = Dataset::from_csv_str("03", include_str!("../examples/03_varians_grupper_welch.csv")).unwrap();
    let grp3: Vec<(String, Vec<f64>)> = ds3.get_grouped_values("effekt_score", "behandlingsgruppe").unwrap().into_iter().collect();
    let vh = check_variance_homogeneity("effekt_score", "behandlingsgruppe", &grp3, 0.05).unwrap();
    assert!(!vh.levene_median.equal_variances, "Expected unequal variance in dataset 03");
    assert!(!vh.bartlett.equal_variances);
    assert!(vh.welch_anova.significant);

    // 3. Sphericity (Mauchly, Greenhouse-Geisser, Huynh-Feldt) — dataset 04 violates sphericity
    let ds4 = Dataset::from_csv_str("04", include_str!("../examples/04_repeterte_maalinger_sfaerisitet.csv")).unwrap();
    let rm_cols: Vec<String> = vec!["t1_for", "t2_uke2", "t3_uke6", "t4_uke12"].into_iter().map(String::from).collect();
    let rm_rows = ds4.get_complete_numeric_matrix(&rm_cols).unwrap();
    let sp = check_sphericity(&rm_cols, &rm_rows, 0.05).unwrap();
    assert!(!sp.sphericity_met, "Expected sphericity violation in dataset 04");
    assert!(sp.epsilon_gg < 0.85 && sp.epsilon_hf >= sp.epsilon_gg);

    // 4. Autocorrelation (DW, BG, ACF, Newey-West, AR1) — dataset 06 has strong AR(1)
    let ds6 = Dataset::from_csv_str("06", include_str!("../examples/06_tidsserie_autokorrelasjon.csv")).unwrap();
    let cols6: Vec<String> = vec!["salg_enheter", "annonse_tusen", "temperatur_c", "rabatt_pst"].into_iter().map(String::from).collect();
    let m6 = ds6.get_complete_numeric_matrix(&cols6).unwrap();
    let y6: Vec<f64> = m6.iter().map(|r| r[0]).collect();
    let x6: Vec<Vec<f64>> = (1..4).map(|j| m6.iter().map(|r| r[j]).collect()).collect();
    let reg6 = full_linear_regression_diagnostics("salg_enheter", &y6, &x6, &cols6[1..], 0.05, 2, 1.0).unwrap();
    assert!(reg6.autocorrelation.durbin_watson < 1.2, "Expected strong positive autocorrelation, got DW={}", reg6.autocorrelation.durbin_watson);
    assert!(reg6.autocorrelation.bg_p_value < 0.01);
    assert!(reg6.autocorrelation.ar1_phi > 0.4);

    // 5. Heteroscedasticity (BP, White, HC3, Log-Y) — dataset 07 is log-linear heteroscedastic
    let ds7 = Dataset::from_csv_str("07", include_str!("../examples/07_inntekt_heteroskedastisitet.csv")).unwrap();
    let cols7: Vec<String> = vec!["aarslonn_tusen", "utdanning_aar", "erfaring_aar", "timer_uke"].into_iter().map(String::from).collect();
    let m7 = ds7.get_complete_numeric_matrix(&cols7).unwrap();
    let y7: Vec<f64> = m7.iter().map(|r| r[0]).collect();
    let x7: Vec<Vec<f64>> = (1..4).map(|j| m7.iter().map(|r| r[j]).collect()).collect();
    let reg7 = full_linear_regression_diagnostics("aarslonn_tusen", &y7, &x7, &cols7[1..], 0.05, 2, 1.0).unwrap();
    assert!(reg7.heteroscedasticity.bp_p_value < 0.05, "Expected BP < 0.05 in dataset 07");
    assert!(reg7.heteroscedasticity.log_y_fallback.applicable);
    assert!(reg7.heteroscedasticity.log_y_fallback.improved, "Expected Log(Y) to improve homoscedasticity");

    // 6. Missing data (Little's MCAR + MICE) & Factor suitability (KMO + Bartlett) — dataset 08
    let ds8 = Dataset::from_csv_str("08", include_str!("../examples/08_undersokelse_manglende_faktor.csv")).unwrap();
    let cols8 = ds8.numeric_column_names();
    let m8_miss = ds8.get_numeric_matrix_with_missing(&cols8).unwrap();
    let miss = analyze_missing_data(&ds8.headers, &ds8.raw_rows, &cols8, &m8_miss, 0.05, 5);
    assert!(miss.mcar_applicable && miss.total_missing_cells > 0);
    assert!(!miss.mice_summary.is_empty());
    let m8_comp = ds8.get_complete_numeric_matrix(&cols8).unwrap();
    let fac = check_factor_suitability(&cols8, &m8_comp, 0.05).unwrap();
    assert!(fac.kmo_overall > 0.65, "Expected KMO > 0.65, got {}", fac.kmo_overall);
    assert!(fac.bartlett_p_value < 0.001);
    assert!(fac.suggested_drop.iter().any(|v| v == "tilfeldig_spm"), "Expected noise item flagged");

    // 7. Logistic Regression (Hosmer-Lemeshow, ROC & AUC) — dataset 09
    let ds9 = Dataset::from_csv_str("09", include_str!("../examples/09_kreditt_logistisk_roc.csv")).unwrap();
    let (y9_opt, neg, pos) = ds9.get_binary_series("mislighold").unwrap();
    let preds9: Vec<String> = vec!["alder", "inntekt_tusen", "gjeldsgrad", "kreditt_mnd"].into_iter().map(String::from).collect();
    let x9_opt = ds9.get_numeric_matrix_with_missing(&preds9).unwrap();
    let y9: Vec<f64> = y9_opt.iter().map(|v| v.unwrap()).collect();
    let x9: Vec<Vec<f64>> = (0..preds9.len()).map(|j| x9_opt.iter().map(|r| r[j].unwrap()).collect()).collect();
    let lr = analyze_logistic_regression("mislighold", &neg, &pos, &y9, &x9, &preds9, 10, 0.05).unwrap();
    assert!(lr.auc > 0.72, "Expected AUC > 0.72, got {}", lr.auc);
    assert!(lr.hosmer_lemeshow.p_value.is_finite());

    // 8. Cox PH (Schoenfeld) & Clustering (Silhouette / Elbow) — dataset 10
    let ds10 = Dataset::from_csv_str("10", include_str!("../examples/10_klinisk_overlevelse_klynger.csv")).unwrap();
    let (ev10_opt, _, _) = ds10.get_binary_series("hendelse").unwrap();
    let cols10: Vec<String> = vec!["overlevelse_mnd", "biomarkor_a", "biomarkor_b", "alder", "bmi"].into_iter().map(String::from).collect();
    let m10 = ds10.get_complete_numeric_matrix(&cols10).unwrap();
    let t10: Vec<f64> = m10.iter().map(|r| r[0]).collect();
    let ev10: Vec<f64> = ev10_opt.iter().map(|v| v.unwrap()).collect();
    let x10: Vec<Vec<f64>> = (1..5).map(|j| m10.iter().map(|r| r[j]).collect()).collect();
    let cox = analyze_cox_ph("overlevelse_mnd", "hendelse", &t10, &ev10, &x10, &cols10[1..], 0.05).unwrap();
    assert!(cox.n_events > 20 && cox.concordance_c > 0.55);
    assert!(cox.schoenfeld_global_p.is_finite());

    let cl_cols: Vec<String> = vec!["biomarkor_a", "biomarkor_b", "alder", "bmi"].into_iter().map(String::from).collect();
    let cl_rows = ds10.get_complete_numeric_matrix(&cl_cols).unwrap();
    let cl = analyze_clustering(&cl_cols, &cl_rows, 7).unwrap();
    assert!(cl.best_silhouette > 0.45, "Expected strong 3-cluster silhouette > 0.45, got {}", cl.best_silhouette);
    assert_eq!(cl.best_k_silhouette, 3, "Expected optimal k=3 for 3 generated clusters");
}
