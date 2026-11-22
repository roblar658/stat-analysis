use crate::stats::utils::{chi2_upper_p, normal_two_tailed_p};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticCoef {
    pub name: String,
    pub estimate: f64,
    pub odds_ratio: f64,
    pub std_error: f64,
    pub z_stat: f64,
    pub p_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HosmerLemeshowBin {
    pub group: usize,
    pub n: usize,
    pub observed_events: f64,
    pub expected_events: f64,
    pub observed_rate: f64,
    pub expected_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HosmerLemeshowResult {
    pub groups: usize,
    pub chi2_stat: f64,
    pub df: usize,
    pub p_value: f64,
    pub good_fit: bool,
    pub bins: Vec<HosmerLemeshowBin>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RocPoint {
    pub fpr: f64,
    pub tpr: f64,
    pub threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    pub threshold: f64,
    pub tp: usize,
    pub fp: usize,
    pub tn: usize,
    pub fn_count: usize,
    pub accuracy: f64,
    pub sensitivity: f64,
    pub specificity: f64,
    pub f1_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticModelCompare {
    pub model_name: String,
    pub k_params: usize,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticReport {
    pub target: String,
    pub positive_label: String,
    pub negative_label: String,
    pub predictors: Vec<String>,
    pub n: usize,
    pub positive_count: usize,
    pub coefficients: Vec<LogisticCoef>,
    pub log_likelihood: f64,
    pub null_log_likelihood: f64,
    pub mcfadden_r2: f64,
    pub aic: f64,
    pub bic: f64,
    pub hosmer_lemeshow: HosmerLemeshowResult,
    pub auc: f64,
    pub auc_label: String,
    pub roc_points: Vec<RocPoint>,
    pub confusion_matrix: ConfusionMatrix,
    pub model_comparison: Vec<LogisticModelCompare>,
    pub hl_status: String,
    pub auc_status: String,
    pub interpretation: String,
    pub recommendation: String,
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

struct IrlsFit {
    beta: Vec<f64>,
    std_errors: Vec<f64>,
    probs: Vec<f64>,
    log_likelihood: f64,
}

fn fit_logistic_irls(y: &[f64], x_cols: &[Vec<f64>]) -> Result<IrlsFit, String> {
    let n = y.len();
    let p = x_cols.len();
    let k = p + 1;
    if n <= k + 2 {
        return Err(format!(
            "For få observasjoner (n={}) for logistisk regresjon med {} parametere.",
            n, k
        ));
    }

    // Standardize predictors internally for stable Newton-Raphson convergence
    let mut means = vec![0.0; p];
    let mut stds = vec![1.0; p];
    let mut x = DMatrix::zeros(n, k);
    for i in 0..n {
        x[(i, 0)] = 1.0;
    }
    for j in 0..p {
        let m = x_cols[j].iter().sum::<f64>() / n as f64;
        let v = x_cols[j].iter().map(|val| (val - m).powi(2)).sum::<f64>() / (n - 1) as f64;
        let s = v.sqrt();
        means[j] = m;
        stds[j] = if s > 1e-12 { s } else { 1.0 };
        for i in 0..n {
            x[(i, j + 1)] = (x_cols[j][i] - means[j]) / stds[j];
        }
    }

    let mut beta_s = DVector::zeros(k);
    let p_bar = (y.iter().sum::<f64>() / n as f64).clamp(1e-4, 1.0 - 1e-4);
    beta_s[0] = (p_bar / (1.0 - p_bar)).ln();

    let ridge = 1e-4; // tiny L2 stabilization to prevent divergence under quasi-separation
    for _iter in 0..50 {
        let eta = &x * &beta_s;
        let mut grad = DVector::zeros(k);
        let mut hess = DMatrix::zeros(k, k);

        for i in 0..n {
            let pi = sigmoid(eta[i]).clamp(1e-9, 1.0 - 1e-9);
            let w = pi * (1.0 - pi);
            let r = y[i] - pi;
            let xi = x.row(i).transpose();
            grad += &xi * r;
            hess += (&xi * xi.transpose()) * w;
        }
        // Ridge penalty on non-intercept terms
        for j in 1..k {
            grad[j] -= ridge * beta_s[j];
            hess[(j, j)] += ridge;
        }
        hess[(0, 0)] += 1e-8;

        let h_inv = hess
            .clone()
            .try_inverse()
            .unwrap_or_else(|| hess.pseudo_inverse(1e-8).unwrap_or_else(|_| DMatrix::identity(k, k)));
        let step = &h_inv * grad;
        let max_step = step.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let damping = if max_step > 5.0 { 5.0 / max_step } else { 1.0 };
        beta_s += step * damping;
        if max_step < 1e-7 {
            break;
        }
    }

    // Compute final probabilities, log-likelihood, and original-scale covariance
    let eta = &x * &beta_s;
    let mut probs = Vec::with_capacity(n);
    let mut ll = 0.0;
    for i in 0..n {
        let pi = sigmoid(eta[i]).clamp(1e-12, 1.0 - 1e-12);
        probs.push(pi);
        ll += y[i] * pi.ln() + (1.0 - y[i]) * (1.0 - pi).ln();
    }

    // Transform beta back to original scale
    let mut beta_orig = vec![0.0; k];
    beta_orig[0] = beta_s[0];
    for j in 0..p {
        let bj = beta_s[j + 1] / stds[j];
        beta_orig[j + 1] = bj;
        beta_orig[0] -= bj * means[j];
    }

    // Compute Fisher information on original scale X for accurate SEs
    let mut x_orig = DMatrix::zeros(n, k);
    for i in 0..n {
        x_orig[(i, 0)] = 1.0;
        for j in 0..p {
            x_orig[(i, j + 1)] = x_cols[j][i];
        }
    }
    let mut info = DMatrix::zeros(k, k);
    for i in 0..n {
        let w = probs[i] * (1.0 - probs[i]);
        let xi = x_orig.row(i).transpose();
        info += (&xi * xi.transpose()) * w;
    }
    for j in 0..k {
        info[(j, j)] += 1e-8;
    }
    let cov = info
        .clone()
        .try_inverse()
        .unwrap_or_else(|| info.pseudo_inverse(1e-8).unwrap_or_else(|_| DMatrix::identity(k, k)));
    let std_errors: Vec<f64> = (0..k).map(|j| cov[(j, j)].max(0.0).sqrt()).collect();

    Ok(IrlsFit {
        beta: beta_orig,
        std_errors,
        probs,
        log_likelihood: ll,
    })
}

pub fn hosmer_lemeshow(y: &[f64], probs: &[f64], g_req: usize, alpha: f64) -> HosmerLemeshowResult {
    let n = y.len();
    let g = g_req.clamp(3, (n / 4).max(3).min(10));
    let mut paired: Vec<(f64, f64)> = probs.iter().copied().zip(y.iter().copied()).collect();
    paired.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut bins = Vec::with_capacity(g);
    let mut chi2 = 0.0;

    for b in 0..g {
        let start = b * n / g;
        let end = (b + 1) * n / g;
        let slice = &paired[start..end];
        let ng = slice.len();
        if ng == 0 {
            continue;
        }
        let obs1: f64 = slice.iter().map(|&(_, yi)| yi).sum();
        let exp1: f64 = slice.iter().map(|&(pi, _)| pi).sum();
        let obs0 = ng as f64 - obs1;
        let exp0 = ng as f64 - exp1;

        if exp1 > 1e-9 {
            chi2 += (obs1 - exp1).powi(2) / exp1;
        }
        if exp0 > 1e-9 {
            chi2 += (obs0 - exp0).powi(2) / exp0;
        }

        bins.push(HosmerLemeshowBin {
            group: b + 1,
            n: ng,
            observed_events: obs1,
            expected_events: exp1,
            observed_rate: obs1 / ng as f64,
            expected_rate: exp1 / ng as f64,
        });
    }

    let df = bins.len().saturating_sub(2).max(1);
    let p_value = chi2_upper_p(chi2, df as f64);

    HosmerLemeshowResult {
        groups: bins.len(),
        chi2_stat: chi2,
        df,
        p_value,
        good_fit: p_value >= alpha,
        bins,
    }
}

pub fn compute_roc_auc(y: &[f64], probs: &[f64]) -> (Vec<RocPoint>, f64, ConfusionMatrix) {
    let n = y.len();
    let mut paired: Vec<(f64, f64)> = probs.iter().copied().zip(y.iter().copied()).collect();
    paired.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let total_pos: f64 = y.iter().filter(|&&v| v > 0.5).count() as f64;
    let total_neg: f64 = (n as f64) - total_pos;

    let mut points = vec![RocPoint {
        fpr: 0.0,
        tpr: 0.0,
        threshold: 1.0,
    }];

    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut i = 0;
    while i < n {
        let thresh = paired[i].0;
        while i < n && (paired[i].0 - thresh).abs() <= 1e-12 {
            if paired[i].1 > 0.5 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            i += 1;
        }
        points.push(RocPoint {
            fpr: if total_neg > 0.0 { fp / total_neg } else { 0.0 },
            tpr: if total_pos > 0.0 { tp / total_pos } else { 0.0 },
            threshold: thresh,
        });
    }
    if points.last().map(|p| p.fpr < 1.0 || p.tpr < 1.0).unwrap_or(true) {
        points.push(RocPoint {
            fpr: 1.0,
            tpr: 1.0,
            threshold: 0.0,
        });
    }

    // Trapezoidal AUC
    let mut auc = 0.0;
    for w in points.windows(2) {
        auc += (w[1].fpr - w[0].fpr) * (w[1].tpr + w[0].tpr) * 0.5;
    }

    // Find optimal Youden's J threshold (tpr - fpr)
    let best_pt = points
        .iter()
        .max_by(|a, b| {
            (a.tpr - a.fpr)
                .partial_cmp(&(b.tpr - b.fpr))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned()
        .unwrap_or(RocPoint {
            fpr: 0.0,
            tpr: 0.0,
            threshold: 0.5,
        });

    let thr = best_pt.threshold;
    let (mut tp_c, mut fp_c, mut tn_c, mut fn_c) = (0usize, 0usize, 0usize, 0usize);
    for (&yi, &pi) in y.iter().zip(probs.iter()) {
        let pred = pi >= thr;
        let actual = yi > 0.5;
        match (pred, actual) {
            (true, true) => tp_c += 1,
            (true, false) => fp_c += 1,
            (false, false) => tn_c += 1,
            (false, true) => fn_c += 1,
        }
    }
    let accuracy = (tp_c + tn_c) as f64 / n.max(1) as f64;
    let sensitivity = tp_c as f64 / (tp_c + fn_c).max(1) as f64;
    let specificity = tn_c as f64 / (tn_c + fp_c).max(1) as f64;
    let precision = tp_c as f64 / (tp_c + fp_c).max(1) as f64;
    let f1_score = if (precision + sensitivity) > 1e-12 {
        2.0 * precision * sensitivity / (precision + sensitivity)
    } else {
        0.0
    };

    (
        points,
        auc,
        ConfusionMatrix {
            threshold: thr,
            tp: tp_c,
            fp: fp_c,
            tn: tn_c,
            fn_count: fn_c,
            accuracy,
            sensitivity,
            specificity,
            f1_score,
        },
    )
}

pub fn analyze_logistic_regression(
    target: &str,
    neg_label: &str,
    pos_label: &str,
    y: &[f64],
    x_cols: &[Vec<f64>],
    predictor_names: &[String],
    hl_groups: usize,
    alpha: f64,
) -> Result<LogisticReport, String> {
    let n = y.len();
    let pos_count = y.iter().filter(|&&v| v > 0.5).count();
    if pos_count < 3 || (n - pos_count) < 3 {
        return Err(format!(
            "Binær målvariabel '{}' har for få hendelser i en klasse ({} positive, {} negative).",
            target,
            pos_count,
            n - pos_count
        ));
    }

    let fit = fit_logistic_irls(y, x_cols)?;
    let p_bar = (pos_count as f64 / n as f64).clamp(1e-12, 1.0 - 1e-12);
    let null_ll = (pos_count as f64) * p_bar.ln() + ((n - pos_count) as f64) * (1.0 - p_bar).ln();
    let mcfadden_r2 = (1.0 - fit.log_likelihood / null_ll).clamp(0.0, 1.0);

    let k = x_cols.len() + 1;
    let aic = 2.0 * k as f64 - 2.0 * fit.log_likelihood;
    let bic = (n as f64).ln() * k as f64 - 2.0 * fit.log_likelihood;

    let mut names = vec!["(Konstantledd)".to_string()];
    names.extend(predictor_names.iter().cloned());

    let coefficients: Vec<LogisticCoef> = (0..k)
        .map(|j| {
            let est = fit.beta[j];
            let se = fit.std_errors[j];
            let z = if se > 1e-15 { est / se } else { 0.0 };
            LogisticCoef {
                name: names[j].clone(),
                estimate: est,
                odds_ratio: est.clamp(-50.0, 50.0).exp(),
                std_error: se,
                z_stat: z,
                p_value: normal_two_tailed_p(z),
            }
        })
        .collect();

    let hl = hosmer_lemeshow(y, &fit.probs, hl_groups, alpha);
    let (roc_points, auc, confusion_matrix) = compute_roc_auc(y, &fit.probs);

    let auc_label = if auc >= 0.9 {
        crate::i18n::tr("Fremragende (> 0.90)", "Outstanding (> 0.90)")
    } else if auc >= 0.8 {
        crate::i18n::tr("God (> 0.80)", "Good (> 0.80)")
    } else if auc >= 0.7 {
        crate::i18n::tr("Akseptabel (> 0.70)", "Acceptable (> 0.70)")
    } else if auc >= 0.6 {
        crate::i18n::tr("Svak (0.60–0.70)", "Weak (0.60–0.70)")
    } else {
        crate::i18n::tr("Dårlig / nær tilfeldig (<= 0.60)", "Poor / near random (<= 0.60)")
    }
    .to_string();

    let mut model_comparison = vec![
        LogisticModelCompare {
            model_name: crate::i18n::tr("Full logistisk modell", "Full logistic model").to_string(),
            k_params: k,
            log_likelihood: fit.log_likelihood,
            aic,
            bic,
        },
        LogisticModelCompare {
            model_name: crate::i18n::tr("Nullmodell (kun konstantledd)", "Null model (intercept only)").to_string(),
            k_params: 1,
            log_likelihood: null_ll,
            aic: 2.0 - 2.0 * null_ll,
            bic: (n as f64).ln() - 2.0 * null_ll,
        },
    ];
    if x_cols.len() >= 2 {
        let worst_idx = (0..x_cols.len())
            .max_by(|&a, &b| {
                coefficients[a + 1]
                    .p_value
                    .partial_cmp(&coefficients[b + 1].p_value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        let sub_x: Vec<Vec<f64>> = (0..x_cols.len())
            .filter(|&j| j != worst_idx)
            .map(|j| x_cols[j].clone())
            .collect();
        if let Ok(sub_fit) = fit_logistic_irls(y, &sub_x) {
            let ks = sub_x.len() + 1;
            model_comparison.push(LogisticModelCompare {
                model_name: format!("Redusert (uten {})", predictor_names[worst_idx]),
                k_params: ks,
                log_likelihood: sub_fit.log_likelihood,
                aic: 2.0 * ks as f64 - 2.0 * sub_fit.log_likelihood,
                bic: (n as f64).ln() * ks as f64 - 2.0 * sub_fit.log_likelihood,
            });
        }
    }
    model_comparison.sort_by(|a, b| a.aic.partial_cmp(&b.aic).unwrap_or(std::cmp::Ordering::Equal));

    let hl_status = if hl.good_fit { "ok" } else { "fail" }.to_string();
    let auc_status = if auc >= 0.8 {
        "ok"
    } else if auc >= 0.7 {
        "warning"
    } else {
        "fail"
    }
    .to_string();

    let interpretation = format!(
        "Hosmer-Lemeshow χ²({}) = {:.3}, p = {:.4} ({}). ROC AUC = {:.3} ({}), McFadden R² = {:.3}.",
        hl.df,
        hl.chi2_stat,
        hl.p_value,
        if hl.good_fit {
            crate::i18n::tr("god tilpasning", "good fit")
        } else {
            crate::i18n::tr("tegn til manglende tilpasning", "signs of lack of fit")
        },
        auc,
        auc_label,
        mcfadden_r2
    );

    let recommendation = if hl.good_fit && auc >= 0.7 {
        crate::i18n::tr(
            "Logistisk modell viser både god kalibrering (Hosmer-Lemeshow p > 0.05) og akseptabel/god diskriminering (AUC > 0.7).",
            "Logistic model shows both good calibration (Hosmer-Lemeshow p > 0.05) and acceptable/good discrimination (AUC > 0.7).",
        )
        .to_string()
    } else if !hl.good_fit {
        crate::i18n::tr(
            "Hosmer-Lemeshow p < 0.05 indikerer feilkalibrering. Vurder interaksjonsledd, ikke-lineære ledd (splines/polynomer) eller færre uteliggere.",
            "Hosmer-Lemeshow p < 0.05 indicates miscalibration. Consider interaction terms, nonlinear terms (splines/polynomials), or outlier review.",
        )
        .to_string()
    } else {
        crate::i18n::tr(
            "Lav AUC (< 0.70): modellen skiller svakt mellom klassene. Vurder sterkere prediktorer eller ikke-lineære klassifikasjonsmodeller.",
            "Low AUC (< 0.70): weak class separation. Consider stronger predictors or nonlinear classification models.",
        )
        .to_string()
    };

    Ok(LogisticReport {
        target: target.to_string(),
        positive_label: pos_label.to_string(),
        negative_label: neg_label.to_string(),
        predictors: predictor_names.to_vec(),
        n,
        positive_count: pos_count,
        coefficients,
        log_likelihood: fit.log_likelihood,
        null_log_likelihood: null_ll,
        mcfadden_r2,
        aic,
        bic,
        hosmer_lemeshow: hl,
        auc,
        auc_label,
        roc_points,
        confusion_matrix,
        model_comparison,
        hl_status,
        auc_status,
        interpretation,
        recommendation,
    })
}
