use crate::stats::utils::{chi2_upper_p, fit_ols, mean, variance};
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnMissingInfo {
    pub column: String,
    pub missing_count: usize,
    pub missing_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingPattern {
    pub pattern: String, // e.g. "1101" where 1 = present, 0 = missing
    pub missing_cols: Vec<String>,
    pub row_count: usize,
    pub pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiceColumnSummary {
    pub column: String,
    pub complete_case_mean: f64,
    pub fiml_em_mean: f64,
    pub mice_pooled_mean: f64,
    pub mice_pooled_se: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingDataReport {
    pub total_rows: usize,
    pub total_cells: usize,
    pub total_missing_cells: usize,
    pub overall_missing_pct: f64,
    pub complete_rows: usize,
    pub columns: Vec<ColumnMissingInfo>,
    pub patterns: Vec<MissingPattern>,
    pub mcar_applicable: bool,
    pub mcar_chi2: Option<f64>,
    pub mcar_df: Option<usize>,
    pub mcar_p_value: Option<f64>,
    pub is_mcar: Option<bool>,
    pub mice_imputations: usize,
    pub mice_summary: Vec<MiceColumnSummary>,
    pub status: String,
    pub interpretation: String,
    pub recommendation: String,
}

/// EM algorithm for multivariate normal mean and covariance under MAR (FIML)
fn em_mvn(data: &[Vec<Option<f64>>], max_iter: usize) -> (Vec<f64>, DMatrix<f64>) {
    let n = data.len();
    let p = data[0].len();

    // Initialize with available-case means and diagonal variances
    let mut mu = vec![0.0; p];
    let mut sigma = DMatrix::identity(p, p);
    for j in 0..p {
        let vals: Vec<f64> = data.iter().filter_map(|r| r[j]).collect();
        if !vals.is_empty() {
            mu[j] = mean(&vals);
            let v = if vals.len() > 1 {
                variance(&vals).max(1e-6)
            } else {
                1.0
            };
            sigma[(j, j)] = v;
        }
    }

    for _iter in 0..max_iter {
        let mut sum_x = vec![0.0; p];
        let mut sum_xx: DMatrix<f64> = DMatrix::zeros(p, p);

        for row in data {
            let obs: Vec<usize> = (0..p).filter(|&j| row[j].is_some()).collect();
            let mis: Vec<usize> = (0..p).filter(|&j| row[j].is_none()).collect();

            if obs.is_empty() {
                for j in 0..p {
                    sum_x[j] += mu[j];
                }
                for a in 0..p {
                    for b in 0..p {
                        sum_xx[(a, b)] += sigma[(a, b)] + mu[a] * mu[b];
                    }
                }
                continue;
            }

            let mut x_imp = vec![0.0; p];
            for &j in &obs {
                x_imp[j] = row[j].unwrap();
            }

            let mut cov_cond = DMatrix::zeros(p, p);
            if !mis.is_empty() {
                let no = obs.len();
                let nm = mis.len();
                let mut s_oo = DMatrix::zeros(no, no);
                for (a, &ja) in obs.iter().enumerate() {
                    for (b, &jb) in obs.iter().enumerate() {
                        s_oo[(a, b)] = sigma[(ja, jb)];
                    }
                }
                // Slight ridge for stability
                for a in 0..no {
                    s_oo[(a, a)] += 1e-8;
                }
                let s_oo_inv = s_oo
                    .clone()
                    .try_inverse()
                    .unwrap_or_else(|| s_oo.pseudo_inverse(1e-8).unwrap_or_else(|_| DMatrix::identity(no, no)));

                let mut s_mo = DMatrix::zeros(nm, no);
                for (a, &ja) in mis.iter().enumerate() {
                    for (b, &jb) in obs.iter().enumerate() {
                        s_mo[(a, b)] = sigma[(ja, jb)];
                    }
                }
                let mut s_mm = DMatrix::zeros(nm, nm);
                for (a, &ja) in mis.iter().enumerate() {
                    for (b, &jb) in mis.iter().enumerate() {
                        s_mm[(a, b)] = sigma[(ja, jb)];
                    }
                }

                let diff_o = DVector::from_iterator(no, obs.iter().map(|&j| row[j].unwrap() - mu[j]));
                let b_mat = &s_mo * &s_oo_inv;
                let pred_m = &b_mat * diff_o;
                for (a, &jm) in mis.iter().enumerate() {
                    x_imp[jm] = mu[jm] + pred_m[a];
                }
                let c_mm = s_mm - &b_mat * s_mo.transpose();
                for (a, &ja) in mis.iter().enumerate() {
                    for (b, &jb) in mis.iter().enumerate() {
                        cov_cond[(ja, jb)] = c_mm[(a, b)];
                    }
                }
            }

            for j in 0..p {
                sum_x[j] += x_imp[j];
            }
            for a in 0..p {
                for b in 0..p {
                    sum_xx[(a, b)] += x_imp[a] * x_imp[b] + cov_cond[(a, b)];
                }
            }
        }

        let new_mu: Vec<f64> = sum_x.iter().map(|s| s / n as f64).collect();
        let mut new_sigma: DMatrix<f64> = DMatrix::zeros(p, p);
        for a in 0..p {
            for b in 0..p {
                new_sigma[(a, b)] = (sum_xx[(a, b)] / n as f64) - new_mu[a] * new_mu[b];
            }
            new_sigma[(a, a)] = new_sigma[(a, a)].max(1e-6);
        }

        let diff: f64 = mu
            .iter()
            .zip(new_mu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        mu = new_mu;
        sigma = new_sigma;
        if diff < 1e-6 {
            break;
        }
    }

    (mu, sigma)
}

/// Multiple Imputation by Chained Equations (MICE) using Bayesian/noisy linear regression & Rubin's rules
fn run_mice(
    col_names: &[String],
    data: &[Vec<Option<f64>>],
    em_means: &[f64],
    m_imputations: usize,
) -> Vec<MiceColumnSummary> {
    let n = data.len();
    let p = col_names.len();
    let m = m_imputations.clamp(3, 15);
    let mut rng = StdRng::seed_from_u64(1337);

    // Precompute complete-case column means
    let cc_means: Vec<f64> = (0..p)
        .map(|j| {
            let vals: Vec<f64> = data.iter().filter_map(|r| r[j]).collect();
            if vals.is_empty() {
                0.0
            } else {
                mean(&vals)
            }
        })
        .collect();

    let mut per_imp_means = vec![vec![0.0; m]; p];
    let mut per_imp_vars = vec![vec![0.0; m]; p];

    for imp in 0..m {
        // Initialize missing values by random draw from observed values of each column
        let mut mat = vec![vec![0.0; p]; n];
        let mut missing_idx_per_col: Vec<Vec<usize>> = vec![vec![]; p];
        for j in 0..p {
            let obs_vals: Vec<f64> = data.iter().filter_map(|r| r[j]).collect();
            for i in 0..n {
                if let Some(v) = data[i][j] {
                    mat[i][j] = v;
                } else {
                    missing_idx_per_col[j].push(i);
                    if !obs_vals.is_empty() {
                        mat[i][j] = obs_vals[rng.gen_range(0..obs_vals.len())];
                    } else {
                        mat[i][j] = 0.0;
                    }
                }
            }
        }

        // 5 chained equation sweeps
        for _sweep in 0..5 {
            for j in 0..p {
                if missing_idx_per_col[j].is_empty() || p < 2 {
                    continue;
                }
                let obs_rows: Vec<usize> = (0..n).filter(|&i| data[i][j].is_some()).collect();
                if obs_rows.len() <= p + 2 {
                    continue;
                }
                let y_obs: Vec<f64> = obs_rows.iter().map(|&i| mat[i][j]).collect();
                let other_cols: Vec<usize> = (0..p).filter(|&k| k != j).collect();
                let x_obs: Vec<Vec<f64>> = other_cols
                    .iter()
                    .map(|&k| obs_rows.iter().map(|&i| mat[i][k]).collect())
                    .collect();
                let other_names: Vec<String> = other_cols
                    .iter()
                    .map(|&k| col_names[k].clone())
                    .collect();

                if let Ok(fit) = fit_ols(&y_obs, &x_obs, &other_names) {
                    let resid_sd = (fit.rss / (obs_rows.len() - fit.k).max(1) as f64).sqrt();
                    for &mi in &missing_idx_per_col[j] {
                        let mut pred = fit.beta[0];
                        for (idx_c, &k) in other_cols.iter().enumerate() {
                            pred += fit.beta[idx_c + 1] * mat[mi][k];
                        }
                        // Box-Muller normal noise
                        let u1: f64 = rng.gen::<f64>().max(1e-12);
                        let u2: f64 = rng.gen::<f64>();
                        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                        mat[mi][j] = pred + z * resid_sd;
                    }
                }
            }
        }

        for j in 0..p {
            let col_vals: Vec<f64> = (0..n).map(|i| mat[i][j]).collect();
            per_imp_means[j][imp] = mean(&col_vals);
            per_imp_vars[j][imp] = variance(&col_vals) / n as f64;
        }
    }

    // Rubin's rules pooling
    (0..p)
        .map(|j| {
            let q_bar = mean(&per_imp_means[j]);
            let u_bar = mean(&per_imp_vars[j]);
            let b_var = variance(&per_imp_means[j]);
            let total_var = u_bar + (1.0 + 1.0 / m as f64) * b_var;
            MiceColumnSummary {
                column: col_names[j].clone(),
                complete_case_mean: cc_means[j],
                fiml_em_mean: em_means[j],
                mice_pooled_mean: q_bar,
                mice_pooled_se: total_var.max(0.0).sqrt(),
            }
        })
        .collect()
}

/// Missingness analysis + Little's MCAR test + EM (FIML) & MICE multiple imputation
pub fn analyze_missing_data(
    all_headers: &[String],
    raw_rows: &[Vec<Option<String>>],
    numeric_col_names: &[String],
    numeric_matrix: &[Vec<Option<f64>>],
    alpha: f64,
    mice_m: usize,
) -> MissingDataReport {
    let n = raw_rows.len();
    let total_cols = all_headers.len();
    let total_cells = n * total_cols;
    let mut col_missing = vec![0usize; total_cols];
    let mut pattern_counts: BTreeMap<String, (usize, Vec<String>)> = BTreeMap::new();
    let mut complete_rows = 0usize;

    for row in raw_rows {
        let mut pat = String::with_capacity(total_cols);
        let mut miss_names = Vec::new();
        for (j, cell) in row.iter().enumerate() {
            if cell.is_none() {
                col_missing[j] += 1;
                pat.push('0');
                miss_names.push(all_headers[j].clone());
            } else {
                pat.push('1');
            }
        }
        if miss_names.is_empty() {
            complete_rows += 1;
        }
        let entry = pattern_counts.entry(pat).or_insert((0, miss_names));
        entry.0 += 1;
    }

    let total_missing_cells: usize = col_missing.iter().sum();
    let overall_missing_pct = if total_cells > 0 {
        100.0 * total_missing_cells as f64 / total_cells as f64
    } else {
        0.0
    };

    let columns: Vec<ColumnMissingInfo> = all_headers
        .iter()
        .enumerate()
        .map(|(j, name)| ColumnMissingInfo {
            column: name.clone(),
            missing_count: col_missing[j],
            missing_pct: if n > 0 {
                100.0 * col_missing[j] as f64 / n as f64
            } else {
                0.0
            },
        })
        .collect();

    let mut patterns: Vec<MissingPattern> = pattern_counts
        .into_iter()
        .map(|(pattern, (row_count, missing_cols))| MissingPattern {
            pattern,
            missing_cols,
            row_count,
            pct: 100.0 * row_count as f64 / n.max(1) as f64,
        })
        .collect();
    patterns.sort_by(|a, b| b.row_count.cmp(&a.row_count));

    // Check if numeric columns have missing values for Little's MCAR test
    let num_p = numeric_col_names.len();
    let num_has_missing = numeric_matrix
        .iter()
        .any(|r| r.iter().any(|v| v.is_none()));

    if !num_has_missing || num_p < 2 {
        let status = if total_missing_cells == 0 {
            "ok"
        } else {
            "warning"
        }
        .to_string();
        let interpretation = if total_missing_cells == 0 {
            crate::i18n::tr(
                "Datasettet har ingen manglende verdier (0% missing). Littles MCAR-test er ikke nødvendig.",
                "Dataset has no missing values (0% missing). Little's MCAR test is not needed.",
            )
            .to_string()
        } else if crate::i18n::is_en() {
            format!(
                "Found {} missing cells ({:.2}%), but fewer than 2 numeric columns with missing values for Little's MCAR test.",
                total_missing_cells, overall_missing_pct
            )
        } else {
            format!(
                "Fant {} manglende celler ({:.2}%), men færre enn 2 numeriske kolonner med manglende verdier for Littles MCAR-test.",
                total_missing_cells, overall_missing_pct
            )
        };
        let recommendation = if total_missing_cells == 0 {
            crate::i18n::tr(
                "Alle observasjoner er komplette — ingen imputering nødvendig.",
                "All observations are complete — no imputation required.",
            )
            .to_string()
        } else {
            crate::i18n::tr(
                "Undersøk mønsteret i kategoriske/manglende felt før videre modellering.",
                "Inspect the missingness pattern in categorical/missing fields before further modeling.",
            )
            .to_string()
        };

        return MissingDataReport {
            total_rows: n,
            total_cells,
            total_missing_cells,
            overall_missing_pct,
            complete_rows,
            columns,
            patterns,
            mcar_applicable: false,
            mcar_chi2: None,
            mcar_df: None,
            mcar_p_value: None,
            is_mcar: None,
            mice_imputations: 0,
            mice_summary: vec![],
            status,
            interpretation,
            recommendation,
        };
    }

    // Run EM algorithm (FIML) to estimate mu and Sigma
    let (em_mu, em_sigma) = em_mvn(numeric_matrix, 60);

    // Group rows by observed-variable pattern
    let mut pat_groups: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();
    for (i, row) in numeric_matrix.iter().enumerate() {
        let obs_cols: Vec<usize> = (0..num_p).filter(|&j| row[j].is_some()).collect();
        if !obs_cols.is_empty() {
            pat_groups.entry(obs_cols).or_default().push(i);
        }
    }

    // Little's d^2 = sum_j m_j * (y_bar_obs,j - mu_obs,j)^T * Sigma_obs,j^{-1} * (y_bar_obs,j - mu_obs,j)
    // df = sum_j p_j - p
    let mut d2 = 0.0;
    let mut sum_pj = 0usize;
    for (obs_cols, row_indices) in &pat_groups {
        let m_j = row_indices.len();
        let p_j = obs_cols.len();
        sum_pj += p_j;

        let mut y_bar = DVector::zeros(p_j);
        for (a, &col_idx) in obs_cols.iter().enumerate() {
            let s: f64 = row_indices
                .iter()
                .map(|&ri| numeric_matrix[ri][col_idx].unwrap())
                .sum();
            y_bar[a] = s / m_j as f64 - em_mu[col_idx];
        }

        let mut s_obs = DMatrix::zeros(p_j, p_j);
        for (a, &ca) in obs_cols.iter().enumerate() {
            for (b, &cb) in obs_cols.iter().enumerate() {
                s_obs[(a, b)] = em_sigma[(ca, cb)];
            }
        }
        // Small-sample unbiased scaling n/(n-1)
        let scale = (n as f64) / ((n - 1).max(1) as f64);
        s_obs *= scale;
        for a in 0..p_j {
            s_obs[(a, a)] += 1e-8;
        }
        let s_obs_inv = s_obs
            .clone()
            .try_inverse()
            .unwrap_or_else(|| s_obs.pseudo_inverse(1e-8).unwrap_or_else(|_| DMatrix::identity(p_j, p_j)));

        let quad = (y_bar.transpose() * s_obs_inv * &y_bar)[(0, 0)].max(0.0);
        d2 += (m_j as f64) * quad;
    }

    let df = sum_pj.saturating_sub(num_p).max(1);
    let p_val = chi2_upper_p(d2, df as f64);
    let is_mcar = p_val >= alpha;

    let mice_summary = run_mice(numeric_col_names, numeric_matrix, &em_mu, mice_m);

    let status = if is_mcar { "warning" } else { "fail" }.to_string();
    let interpretation = if is_mcar {
        if crate::i18n::is_en() {
            format!(
                "Little's MCAR test: χ²({}) = {:.3}, p = {:.4} (>= {:.2}). Data are consistent with MCAR (Missing Completely At Random), with {:.2}% missing cells.",
                df, d2, p_val, alpha, overall_missing_pct
            )
        } else {
            format!(
                "Littles MCAR-test: χ²({}) = {:.3}, p = {:.4} (>= {:.2}). Data er forenlig med MCAR (tilfeldig manglende), men {:.2}% av cellene mangler.",
                df, d2, p_val, alpha, overall_missing_pct
            )
        }
    } else if crate::i18n::is_en() {
        format!(
            "Little's MCAR test rejected! χ²({}) = {:.3}, p = {:.4} (< {:.2}). Missing data are NOT completely random (MAR/MNAR).",
            df, d2, p_val, alpha
        )
    } else {
        format!(
            "Littles MCAR-test forkastet! χ²({}) = {:.3}, p = {:.4} (< {:.2}). Manglende data er IKKE helt tilfeldig (MAR/MNAR).",
            df, d2, p_val, alpha
        )
    };

    let recommendation = if is_mcar {
        crate::i18n::tr(
            "MCAR-forutsetningen er ikke forkastet, men multippel imputering (MICE) eller FIML (beregnet under) bevarer statistisk styrke fremfor listwise deletion.",
            "MCAR assumption is not rejected, yet multiple imputation (MICE) or FIML (computed below) preserves statistical power over listwise deletion.",
        )
        .to_string()
    } else {
        crate::i18n::tr(
            "Unngå sletting av rader (listwise deletion) da det gir skjevhet! Bruk multippel imputering (MICE) eller FIML (EM-estimatene og MICE-sammendraget er beregnet under).",
            "Avoid listwise deletion as it introduces bias! Use multiple imputation (MICE) or FIML (EM estimates and MICE summary are computed below).",
        )
        .to_string()
    };

    MissingDataReport {
        total_rows: n,
        total_cells,
        total_missing_cells,
        overall_missing_pct,
        complete_rows,
        columns,
        patterns,
        mcar_applicable: true,
        mcar_chi2: Some(d2),
        mcar_df: Some(df),
        mcar_p_value: Some(p_val),
        is_mcar: Some(is_mcar),
        mice_imputations: mice_m.clamp(3, 15),
        mice_summary,
        status,
        interpretation,
        recommendation,
    }
}
