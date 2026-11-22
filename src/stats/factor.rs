use crate::stats::utils::{chi2_upper_p, correlation_matrix};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableMsa {
    pub variable: String,
    pub msa: f64,
    pub max_abs_corr: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorSuitabilityReport {
    pub variables: Vec<String>,
    pub n: usize,
    pub kmo_overall: f64,
    pub kmo_label: String,
    pub per_variable_msa: Vec<VariableMsa>,
    pub bartlett_chi2: f64,
    pub bartlett_df: usize,
    pub bartlett_p_value: f64,
    pub eigenvalues: Vec<f64>,
    pub variance_explained_pct: Vec<f64>,
    pub kaiser_num_factors: usize,
    pub correlation_matrix: Vec<Vec<f64>>,
    pub suggested_drop: Vec<String>,
    pub kmo_after_drop: Option<f64>,
    pub status: String,
    pub interpretation: String,
    pub recommendation: String,
}

fn compute_kmo_and_msa(corr: &[Vec<f64>]) -> (f64, Vec<f64>, f64) {
    let p = corr.len();
    let mut r_mat = DMatrix::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            r_mat[(i, j)] = corr[i][j];
        }
    }

    // Eigenvalues for determinant check and slight ridge if near-singular
    let eig = r_mat.clone().symmetric_eigen();
    let min_ev = eig
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let r_reg = if min_ev < 1e-6 {
        let mut rr = r_mat.clone();
        for i in 0..p {
            rr[(i, i)] += 1e-4;
        }
        rr
    } else {
        r_mat.clone()
    };

    let inv = r_reg
        .clone()
        .try_inverse()
        .unwrap_or_else(|| r_reg.pseudo_inverse(1e-8).unwrap_or_else(|_| DMatrix::identity(p, p)));

    // Partial correlation matrix: u_ij = -inv_ij / sqrt(inv_ii * inv_jj)
    let mut sum_r2 = 0.0;
    let mut sum_u2 = 0.0;
    let mut msa = vec![0.0; p];

    for i in 0..p {
        let mut row_r2 = 0.0;
        let mut row_u2 = 0.0;
        for j in 0..p {
            if i == j {
                continue;
            }
            let r2 = corr[i][j] * corr[i][j];
            let denom = (inv[(i, i)].abs() * inv[(j, j)].abs()).max(1e-15).sqrt();
            let u = -inv[(i, j)] / denom;
            let u2 = u * u;
            row_r2 += r2;
            row_u2 += u2;
            sum_r2 += r2;
            sum_u2 += u2;
        }
        msa[i] = if (row_r2 + row_u2) > 1e-15 {
            row_r2 / (row_r2 + row_u2)
        } else {
            0.0
        };
    }

    let kmo = if (sum_r2 + sum_u2) > 1e-15 {
        sum_r2 / (sum_r2 + sum_u2)
    } else {
        0.0
    };

    let det = eig
        .eigenvalues
        .iter()
        .map(|&ev| ev.max(1e-12).ln())
        .sum::<f64>(); // returns ln(det(R))

    (kmo, msa, det)
}

fn kmo_rating(kmo: f64) -> &'static str {
    if kmo >= 0.9 {
        "Utmerket (Marvelous, >= 0.90)"
    } else if kmo >= 0.8 {
        "Svært god (Meritorious, 0.80–0.89)"
    } else if kmo >= 0.7 {
        "Middels god (Middling, 0.70–0.79)"
    } else if kmo >= 0.6 {
        "Akseptabel (Mediocre, 0.60–0.69)"
    } else if kmo >= 0.5 {
        "Svak (Miserable, 0.50–0.59)"
    } else {
        "Uegnet (Unacceptable, < 0.50)"
    }
}

pub fn check_factor_suitability(
    variables: &[String],
    rows: &[Vec<f64>],
    alpha: f64,
) -> Result<FactorSuitabilityReport, String> {
    let p = variables.len();
    let n = rows.len();
    if p < 3 {
        return Err("Faktoranalyse-egnethet (KMO / Bartlett) krever minst 3 numeriske variabler.".to_string());
    }
    if n <= p + 2 {
        return Err(format!(
            "For få observasjoner (n={}) for {} variabler i KMO/Bartlett.",
            n, p
        ));
    }

    let corr = correlation_matrix(rows);
    let (kmo_overall, msa_vec, log_det_r) = compute_kmo_and_msa(&corr);

    // Bartlett's test of sphericity: chi2 = -(n - 1 - (2p + 5)/6) * ln|R|, df = p(p-1)/2
    let bartlett_chi2 = (-((n as f64) - 1.0 - (2.0 * p as f64 + 5.0) / 6.0) * log_det_r).max(0.0);
    let bartlett_df = p * (p - 1) / 2;
    let bartlett_p_value = chi2_upper_p(bartlett_chi2, bartlett_df as f64);

    // PCA eigenvalues of correlation matrix
    let mut r_mat = DMatrix::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            r_mat[(i, j)] = corr[i][j];
        }
    }
    let eig = r_mat.symmetric_eigen();
    let mut eigenvalues: Vec<f64> = eig.eigenvalues.iter().map(|&v| v.max(0.0)).collect();
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let sum_ev: f64 = eigenvalues.iter().sum::<f64>().max(1e-12);
    let variance_explained_pct: Vec<f64> = eigenvalues.iter().map(|&e| 100.0 * e / sum_ev).collect();
    let kaiser_num_factors = eigenvalues.iter().filter(|&&e| e > 1.0).count().max(1);

    let mut per_variable_msa = Vec::with_capacity(p);
    let mut suggested_drop = Vec::new();
    for i in 0..p {
        let max_abs_corr = (0..p)
            .filter(|&j| j != i)
            .map(|j| corr[i][j].abs())
            .fold(0.0_f64, f64::max);
        if msa_vec[i] < 0.5 || max_abs_corr < 0.3 {
            suggested_drop.push(variables[i].clone());
        }
        per_variable_msa.push(VariableMsa {
            variable: variables[i].clone(),
            msa: msa_vec[i],
            max_abs_corr,
        });
    }

    // Compute KMO after dropping weak variables (if >= 3 remain)
    let kmo_after_drop = if !suggested_drop.is_empty() && (p - suggested_drop.len()) >= 3 {
        let keep_idx: Vec<usize> = (0..p)
            .filter(|&i| !suggested_drop.contains(&variables[i]))
            .collect();
        let sub_rows: Vec<Vec<f64>> = rows
            .iter()
            .map(|r| keep_idx.iter().map(|&j| r[j]).collect())
            .collect();
        let sub_corr = correlation_matrix(&sub_rows);
        let (new_kmo, _, _) = compute_kmo_and_msa(&sub_corr);
        Some(new_kmo)
    } else {
        None
    };

    let suitable = kmo_overall >= 0.6 && bartlett_p_value < alpha;
    let status = if suitable {
        "ok"
    } else if kmo_overall >= 0.5 && bartlett_p_value < alpha {
        "warning"
    } else {
        "fail"
    }
    .to_string();

    let interpretation = format!(
        "KMO = {:.3} ({}), Bartletts sfærisitetstest: χ²({}) = {:.2}, p = {:.4}. Kaisers kriterium (egenverdi > 1) foreslår {} faktorer/komponenter.",
        kmo_overall,
        kmo_rating(kmo_overall),
        bartlett_df,
        bartlett_chi2,
        bartlett_p_value,
        kaiser_num_factors
    );

    let recommendation = if suitable {
        "Dataene er godt egnet for PCA / Eksplorativ faktoranalyse (KMO > 0.6 og Bartlett p < 0.05).".to_string()
    } else if !suggested_drop.is_empty() {
        if let Some(new_kmo) = kmo_after_drop {
            format!(
                "Fjern variabler med lav korrelasjon/MSA ({}) — dette endrer KMO til {:.3}.",
                suggested_drop.join(", "),
                new_kmo
            )
        } else {
            format!(
                "Fjern variabler med lav korrelasjon til resten: {}.",
                suggested_drop.join(", ")
            )
        }
    } else {
        "Korrelasjonene mellom variablene er for svake for meningsfull faktoranalyse.".to_string()
    };

    Ok(FactorSuitabilityReport {
        variables: variables.to_vec(),
        n,
        kmo_overall,
        kmo_label: kmo_rating(kmo_overall).to_string(),
        per_variable_msa,
        bartlett_chi2,
        bartlett_df,
        bartlett_p_value,
        eigenvalues,
        variance_explained_pct,
        kaiser_num_factors,
        correlation_matrix: corr,
        suggested_drop,
        kmo_after_drop,
        status,
        interpretation,
        recommendation,
    })
}
