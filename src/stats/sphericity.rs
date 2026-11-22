use crate::stats::utils::{chi2_upper_p, covariance_matrix, f_upper_p};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SphericityReport {
    pub columns: Vec<String>,
    pub n_subjects: usize,
    pub k_conditions: usize,
    pub mauchly_w: f64,
    pub chi2_stat: f64,
    pub chi2_df: usize,
    pub mauchly_p_value: f64,
    pub sphericity_met: bool,
    pub rm_anova_f: f64,
    pub df1_uncorrected: f64,
    pub df2_uncorrected: f64,
    pub p_uncorrected: f64,
    pub epsilon_gg: f64,
    pub p_greenhouse_geisser: f64,
    pub epsilon_hf: f64,
    pub p_huynh_feldt: f64,
    pub condition_means: Vec<f64>,
    pub status: String,
    pub interpretation: String,
    pub recommendation: String,
}

/// Build an orthonormal (k-1) x k Helmert contrast matrix C such that C * C^T = I_{k-1}
/// and each row sums to 0.
fn helmert_orthonormal(k: usize) -> DMatrix<f64> {
    let mut c = DMatrix::zeros(k - 1, k);
    for i in 0..(k - 1) {
        let m = (i + 1) as f64;
        let norm = (m * (m + 1.0)).sqrt();
        for j in 0..=i {
            c[(i, j)] = 1.0 / norm;
        }
        c[(i, i + 1)] = -m / norm;
    }
    c
}

/// Mauchly's test of sphericity + Repeated Measures ANOVA with Greenhouse-Geisser and Huynh-Feldt corrections.
pub fn check_sphericity(
    columns: &[String],
    rows: &[Vec<f64>],
    alpha: f64,
) -> Result<SphericityReport, String> {
    let k = columns.len();
    let n = rows.len();
    if k < 3 {
        return Err("Mauchlys sfærisitetstest krever minst 3 repeterte målinger/kolonner.".to_string());
    }
    if n <= k {
        return Err(format!(
            "For få observasjoner (n={}) for {} repeterte målinger.",
            n, k
        ));
    }

    let s = covariance_matrix(rows);
    let c = helmert_orthonormal(k);
    let m = &c * &s * c.transpose(); // (k-1) x (k-1) symmetric matrix

    let p = (k - 1) as f64;
    let trace_m = m.trace();
    if trace_m <= 1e-15 {
        return Err("Null varians i differansene mellom repeterte målinger.".to_string());
    }

    // Eigenvalues of symmetric M for stable determinant & trace(M^2)
    let eig = m.clone().symmetric_eigen();
    let evals: Vec<f64> = eig.eigenvalues.iter().map(|&v| v.max(1e-15)).collect();
    let log_det: f64 = evals.iter().map(|v| v.ln()).sum();
    let log_w = log_det - p * (trace_m / p).ln();
    let mauchly_w = log_w.exp().clamp(0.0, 1.0);

    // Chi-square approximation for Mauchly's W
    let df_chi2 = (k * (k - 1)) / 2 - 1;
    let f_factor = (2.0 * p * p + p + 2.0) / (6.0 * p * (n - 1) as f64);
    let chi2_stat = (-((n - 1) as f64) * (1.0 - f_factor) * log_w).max(0.0);
    let mauchly_p_value = chi2_upper_p(chi2_stat, df_chi2 as f64);
    let sphericity_met = mauchly_p_value >= alpha;

    // Greenhouse-Geisser epsilon: (trace(M))^2 / (p * trace(M^2))
    let trace_m2: f64 = evals.iter().map(|v| v * v).sum();
    let epsilon_gg = ((trace_m * trace_m) / (p * trace_m2)).clamp(1.0 / p, 1.0);

    // Huynh-Feldt epsilon: (n * p * eps_gg - 2) / (p * (n - 1 - p * eps_gg))
    let nf = n as f64;
    let hf_denom = p * (nf - 1.0 - p * epsilon_gg);
    let epsilon_hf = if hf_denom > 1e-12 {
        ((nf * p * epsilon_gg - 2.0) / hf_denom).clamp(epsilon_gg, 1.0)
    } else {
        1.0
    };

    // Repeated Measures One-Way ANOVA
    let mut grand_sum = 0.0;
    let mut col_sums = vec![0.0; k];
    let mut subj_sums = vec![0.0; n];
    for (i, r) in rows.iter().enumerate() {
        for j in 0..k {
            grand_sum += r[j];
            col_sums[j] += r[j];
            subj_sums[i] += r[j];
        }
    }
    let grand_mean = grand_sum / (n * k) as f64;
    let condition_means: Vec<f64> = col_sums.iter().map(|&cs| cs / nf).collect();

    let mut ss_cond = 0.0;
    for j in 0..k {
        ss_cond += nf * (condition_means[j] - grand_mean).powi(2);
    }
    let mut ss_subj = 0.0;
    for i in 0..n {
        let sm = subj_sums[i] / k as f64;
        ss_subj += (k as f64) * (sm - grand_mean).powi(2);
    }
    let mut ss_total = 0.0;
    for r in rows {
        for &v in r {
            ss_total += (v - grand_mean).powi(2);
        }
    }
    let ss_error = (ss_total - ss_cond - ss_subj).max(1e-15);

    let df1 = p;
    let df2 = p * (nf - 1.0);
    let rm_anova_f = (ss_cond / df1) / (ss_error / df2);
    let p_uncorrected = f_upper_p(rm_anova_f, df1, df2);
    let p_greenhouse_geisser = f_upper_p(rm_anova_f, df1 * epsilon_gg, df2 * epsilon_gg);
    let p_huynh_feldt = f_upper_p(rm_anova_f, df1 * epsilon_hf, df2 * epsilon_hf);

    let status = if sphericity_met { "ok" } else { "fail" }.to_string();
    let interpretation = if sphericity_met {
        if crate::i18n::is_en() {
            format!(
                "Sphericity is met (Mauchly's W = {:.4}, p = {:.4} >= {:.2}). Uncorrected RM-ANOVA F = {:.3}, p = {:.4}.",
                mauchly_w, mauchly_p_value, alpha, rm_anova_f, p_uncorrected
            )
        } else {
            format!(
                "Sfærisitet er oppfylt (Mauchlys W = {:.4}, p = {:.4} >= {:.2}). Ukorrigert RM-ANOVA F = {:.3}, p = {:.4}.",
                mauchly_w, mauchly_p_value, alpha, rm_anova_f, p_uncorrected
            )
        }
    } else if crate::i18n::is_en() {
        format!(
            "Sphericity is violated (Mauchly's W = {:.4}, p = {:.4} < {:.2}). GG ε = {:.3} (p = {:.4}), HF ε = {:.3} (p = {:.4}).",
            mauchly_w, mauchly_p_value, alpha, epsilon_gg, p_greenhouse_geisser, epsilon_hf, p_huynh_feldt
        )
    } else {
        format!(
            "Sfærisitet er brutt (Mauchlys W = {:.4}, p = {:.4} < {:.2}). GG ε = {:.3} (p = {:.4}), HF ε = {:.3} (p = {:.4}).",
            mauchly_w, mauchly_p_value, alpha, epsilon_gg, p_greenhouse_geisser, epsilon_hf, p_huynh_feldt
        )
    };

    let recommendation = if sphericity_met {
        crate::i18n::tr(
            "Du kan bruke standard ukorrigerte frihetsgrader i Repeated Measures ANOVA.",
            "Standard uncorrected degrees of freedom in Repeated Measures ANOVA can be used.",
        )
        .to_string()
    } else if epsilon_gg < 0.75 {
        if crate::i18n::is_en() {
            format!(
                "Use Greenhouse-Geisser correction (ε = {:.3} < 0.75, corrected p = {:.4}).",
                epsilon_gg, p_greenhouse_geisser
            )
        } else {
            format!(
                "Bruk Greenhouse-Geisser-korreksjon (ε = {:.3} < 0.75, korrigert p = {:.4}).",
                epsilon_gg, p_greenhouse_geisser
            )
        }
    } else if crate::i18n::is_en() {
        format!(
            "Use Huynh-Feldt correction (ε_GG = {:.3} >= 0.75, HF corrected p = {:.4}) or Greenhouse-Geisser (p = {:.4}).",
            epsilon_gg, p_huynh_feldt, p_greenhouse_geisser
        )
    } else {
        format!(
            "Bruk Huynh-Feldt-korreksjon (ε_GG = {:.3} >= 0.75, HF korrigert p = {:.4}) eller Greenhouse-Geisser (p = {:.4}).",
            epsilon_gg, p_huynh_feldt, p_greenhouse_geisser
        )
    };

    Ok(SphericityReport {
        columns: columns.to_vec(),
        n_subjects: n,
        k_conditions: k,
        mauchly_w,
        chi2_stat,
        chi2_df: df_chi2,
        mauchly_p_value,
        sphericity_met,
        rm_anova_f,
        df1_uncorrected: df1,
        df2_uncorrected: df2,
        p_uncorrected,
        epsilon_gg,
        p_greenhouse_geisser,
        epsilon_hf,
        p_huynh_feldt,
        condition_means,
        status,
        interpretation,
        recommendation,
    })
}
