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
