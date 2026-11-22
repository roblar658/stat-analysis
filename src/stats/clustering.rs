use crate::stats::utils::{covariance_matrix, mean, std_dev};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterKEval {
    pub k: usize,
    pub wcss: f64,
    pub silhouette: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterSummary {
    pub cluster_id: usize,
    pub size: usize,
    pub mean_silhouette: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterPoint2D {
    pub pc1: f64,
    pub pc2: f64,
    pub cluster: usize,
    pub silhouette: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbscanFallback {
    pub eps: f64,
    pub min_pts: usize,
    pub n_clusters: usize,
    pub noise_points: usize,
    pub silhouette: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringReport {
    pub variables: Vec<String>,
    pub n: usize,
    pub k_evals: Vec<ClusterKEval>,
    pub best_k_silhouette: usize,
    pub best_silhouette: f64,
    pub elbow_k: usize,
    pub cluster_summaries: Vec<ClusterSummary>,
    pub points_2d: Vec<ClusterPoint2D>,
    pub dbscan_fallback: DbscanFallback,
    pub status: String,
    pub interpretation: String,
    pub recommendation: String,
}

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn kmeans_single_run(data: &[Vec<f64>], k: usize, seed: u64) -> (Vec<usize>, Vec<Vec<f64>>, f64) {
    let n = data.len();
    let p = data[0].len();
    let mut rng = StdRng::seed_from_u64(seed);

    // K-means++ initialization
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    centroids.push(data[rng.gen_range(0..n)].clone());

    for _ in 1..k {
        let mut dists = Vec::with_capacity(n);
        let mut total = 0.0;
        for pt in data {
            let d = centroids
                .iter()
                .map(|c| sq_dist(pt, c))
                .fold(f64::INFINITY, f64::min);
            dists.push(d);
            total += d;
        }
        if total <= 1e-15 {
            centroids.push(data[rng.gen_range(0..n)].clone());
        } else {
            let mut r = rng.gen::<f64>() * total;
            let mut chosen = n - 1;
            for (i, &d) in dists.iter().enumerate() {
                r -= d;
                if r <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            centroids.push(data[chosen].clone());
        }
    }

    let mut labels = vec![0usize; n];
    for _iter in 0..100 {
        let mut changed = false;
        for (i, pt) in data.iter().enumerate() {
            let mut best_c = 0;
            let mut best_d = f64::INFINITY;
            for (c, cen) in centroids.iter().enumerate() {
                let d = sq_dist(pt, cen);
                if d < best_d {
                    best_d = d;
                    best_c = c;
                }
            }
            if labels[i] != best_c {
                labels[i] = best_c;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        let mut sums = vec![vec![0.0; p]; k];
        let mut counts = vec![0usize; k];
        for (i, pt) in data.iter().enumerate() {
            let c = labels[i];
            counts[c] += 1;
            for j in 0..p {
                sums[c][j] += pt[j];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..p {
                    centroids[c][j] = sums[c][j] / counts[c] as f64;
                }
            } else {
                centroids[c] = data[rng.gen_range(0..n)].clone();
            }
        }
    }

    let wcss: f64 = data
        .iter()
        .enumerate()
        .map(|(i, pt)| sq_dist(pt, &centroids[labels[i]]))
        .sum();
    (labels, centroids, wcss)
}

fn kmeans_best(data: &[Vec<f64>], k: usize, restarts: usize) -> (Vec<usize>, Vec<Vec<f64>>, f64) {
    let mut best = kmeans_single_run(data, k, 100 + k as u64);
    for r in 1..restarts {
        let cand = kmeans_single_run(data, k, 100 + (k as u64) * 31 + r as u64);
        if cand.2 < best.2 {
            best = cand;
        }
    }
    best
}

fn silhouette_scores(data: &[Vec<f64>], labels: &[usize], k: usize) -> (f64, Vec<f64>) {
    let n = data.len();
    if k < 2 || n <= k {
        return (0.0, vec![0.0; n]);
    }

    let mut cluster_members: Vec<Vec<usize>> = vec![vec![]; k];
    for (i, &c) in labels.iter().enumerate() {
        cluster_members[c].push(i);
    }

    let mut sils = vec![0.0; n];
    for i in 0..n {
        let ci = labels[i];
        let own = &cluster_members[ci];
        if own.len() <= 1 {
            sils[i] = 0.0;
            continue;
        }
        let a_i: f64 = own
            .iter()
            .filter(|&&j| j != i)
            .map(|&j| sq_dist(&data[i], &data[j]).sqrt())
            .sum::<f64>()
            / (own.len() - 1) as f64;

        let mut b_i = f64::INFINITY;
        for c in 0..k {
            if c == ci || cluster_members[c].is_empty() {
                continue;
            }
            let d_c: f64 = cluster_members[c]
                .iter()
                .map(|&j| sq_dist(&data[i], &data[j]).sqrt())
                .sum::<f64>()
                / cluster_members[c].len() as f64;
            if d_c < b_i {
                b_i = d_c;
            }
        }
        let denom = a_i.max(b_i);
        sils[i] = if denom > 1e-12 {
            (b_i - a_i) / denom
        } else {
            0.0
        };
    }

    let mean_sil = sils.iter().sum::<f64>() / n as f64;
    (mean_sil, sils)
}

/// Density-based clustering fallback (DBSCAN) with automatic k-distance eps selection.
fn run_dbscan(data: &[Vec<f64>]) -> DbscanFallback {
    let n = data.len();
    let p = data[0].len();
    let min_pts = (p + 1).clamp(4, 8).min(n.saturating_sub(1).max(2));

    // Estimate eps as the 75th percentile of each point's min_pts-th nearest neighbor distance
    let mut k_dists = Vec::with_capacity(n);
    for i in 0..n {
        let mut ds: Vec<f64> = (0..n)
            .filter(|&j| j != i)
            .map(|j| sq_dist(&data[i], &data[j]).sqrt())
            .collect();
        ds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (min_pts - 1).min(ds.len().saturating_sub(1));
        k_dists.push(ds[idx]);
    }
    k_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let eps = k_dists[(n * 3 / 4).min(n - 1)].max(0.15);
    let eps2 = eps * eps;

    let mut labels: Vec<i32> = vec![-2; n]; // -2 unvisited, -1 noise, >=0 cluster id
    let mut cid: i32 = 0;

    let region_query = |pt_idx: usize| -> Vec<usize> {
        (0..n)
            .filter(|&j| sq_dist(&data[pt_idx], &data[j]) <= eps2)
            .collect()
    };

    for i in 0..n {
        if labels[i] != -2 {
            continue;
        }
        let neighbors = region_query(i);
        if neighbors.len() < min_pts {
            labels[i] = -1;
            continue;
        }
        labels[i] = cid;
        let mut queue = neighbors;
        let mut head = 0;
        while head < queue.len() {
            let q = queue[head];
            head += 1;
            if labels[q] == -1 {
                labels[q] = cid;
            }
            if labels[q] != -2 {
                continue;
            }
            labels[q] = cid;
            let q_nb = region_query(q);
            if q_nb.len() >= min_pts {
                for nb in q_nb {
                    if labels[nb] == -2 || labels[nb] == -1 {
                        queue.push(nb);
                    }
                }
            }
        }
        cid += 1;
    }

    let n_clusters = cid.max(0) as usize;
    let noise_points = labels.iter().filter(|&&l| l < 0).count();
    let silhouette = if n_clusters >= 2 {
        let valid_pts: Vec<Vec<f64>> = data
            .iter()
            .enumerate()
            .filter(|&(i, _)| labels[i] >= 0)
            .map(|(_, r)| r.clone())
            .collect();
        let valid_lbls: Vec<usize> = labels.iter().filter(|&&l| l >= 0).map(|&l| l as usize).collect();
        if valid_pts.len() > n_clusters {
            Some(silhouette_scores(&valid_pts, &valid_lbls, n_clusters).0)
        } else {
            None
        }
    } else {
        None
    };

    DbscanFallback {
        eps,
        min_pts,
        n_clusters,
        noise_points,
        silhouette,
    }
}

/// Clustering evaluation: K-means++, Elbow method (WCSS), Silhouette score, DBSCAN fallback, and 2D PCA projection.
pub fn analyze_clustering(
    variables: &[String],
    rows: &[Vec<f64>],
    k_max_req: usize,
) -> Result<ClusteringReport, String> {
    let n = rows.len();
    let p = variables.len();
    if p < 2 {
        return Err("Klyngeanalyse krever minst 2 numeriske variabler.".to_string());
    }
    if n < 6 {
        return Err("Klyngeanalyse krever minst 6 observasjoner.".to_string());
    }

    // Standardize features (z-score)
    let mut std_rows = vec![vec![0.0; p]; n];
    for j in 0..p {
        let col: Vec<f64> = rows.iter().map(|r| r[j]).collect();
        let m = mean(&col);
        let s = std_dev(&col);
        let sd = if s > 1e-12 { s } else { 1.0 };
        for i in 0..n {
            std_rows[i][j] = (rows[i][j] - m) / sd;
        }
    }

    // Cap n for exact O(n^2) silhouette if huge dataset (> 800 rows)
    let sil_sample_rows: Vec<Vec<f64>> = if n > 600 {
        let step = n / 600;
        std_rows.iter().step_by(step.max(1)).take(600).cloned().collect()
    } else {
        std_rows.clone()
    };

    let k_max = k_max_req.clamp(3, 10).min(n - 1);
    let mut k_evals = Vec::new();

    // k = 1 for Elbow curve baseline
    let (_, _, wcss1) = kmeans_best(&std_rows, 1, 1);
    k_evals.push(ClusterKEval {
        k: 1,
        wcss: wcss1,
        silhouette: 0.0,
    });

    let mut best_k = 2;
    let mut best_sil = -1.0_f64;
    let mut best_labels = vec![0usize; n];
    let mut best_point_sils = vec![0.0; n];

    for k in 2..=k_max {
        let (labels, _, wcss) = kmeans_best(&std_rows, k, 6);
        let (sil, pt_sils) = if n <= 600 {
            silhouette_scores(&std_rows, &labels, k)
        } else {
            let step = (n / 600).max(1);
            let sub_labels: Vec<usize> = labels.iter().step_by(step).take(600).copied().collect();
            let (s, _) = silhouette_scores(&sil_sample_rows, &sub_labels, k);
            (s, vec![s; n])
        };

        k_evals.push(ClusterKEval {
            k,
            wcss,
            silhouette: sil,
        });

        if sil > best_sil {
            best_sil = sil;
            best_k = k;
            best_labels = labels;
            best_point_sils = pt_sils;
        }
    }

    // Automatic Elbow detection (maximum perpendicular distance to line from k=1 to k=k_max)
    let x1 = k_evals.first().unwrap().k as f64;
    let y1 = k_evals.first().unwrap().wcss;
    let x2 = k_evals.last().unwrap().k as f64;
    let y2 = k_evals.last().unwrap().wcss;
    let line_len = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt().max(1e-12);
    let mut elbow_k = 2;
    let mut max_dist = -1.0_f64;
    for ev in &k_evals {
        if ev.k < 2 || ev.k >= k_max {
            continue;
        }
        // Normalize axes so distance is scale-invariant
        let xn = (ev.k as f64 - x1) / (x2 - x1).max(1.0);
        let yn = (ev.wcss - y2) / (y1 - y2).max(1e-12);
        // Line from (0, 1) to (1, 0): x + y - 1 = 0
        let dist = (xn + yn - 1.0).abs() / std::f64::consts::SQRT_2;
        if dist > max_dist {
            max_dist = dist;
            elbow_k = ev.k;
        }
    }
    let _ = line_len;

    // Summaries for best_k
    let mut cluster_summaries = Vec::with_capacity(best_k);
    for c in 0..best_k {
        let members: Vec<f64> = best_labels
            .iter()
            .enumerate()
            .filter(|&(_, &lbl)| lbl == c)
            .map(|(i, _)| best_point_sils[i])
            .collect();
        cluster_summaries.push(ClusterSummary {
            cluster_id: c + 1,
            size: members.len(),
            mean_silhouette: if members.is_empty() {
                0.0
            } else {
                mean(&members)
            },
        });
    }

    // 2D PCA projection of standardized data for visualization
    let cov = covariance_matrix(&std_rows);
    let eig = cov.symmetric_eigen();
    let mut eig_order: Vec<usize> = (0..p).collect();
    eig_order.sort_by(|&a, &b| {
        eig.eigenvalues[b]
            .partial_cmp(&eig.eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let pc1_vec: Vec<f64> = (0..p).map(|j| eig.eigenvectors[(j, eig_order[0])]).collect();
    let pc2_vec: Vec<f64> = (0..p).map(|j| eig.eigenvectors[(j, eig_order[1])]).collect();

    let points_2d: Vec<ClusterPoint2D> = (0..n)
        .map(|i| {
            let pc1: f64 = (0..p).map(|j| std_rows[i][j] * pc1_vec[j]).sum();
            let pc2: f64 = (0..p).map(|j| std_rows[i][j] * pc2_vec[j]).sum();
            ClusterPoint2D {
                pc1,
                pc2,
                cluster: best_labels[i] + 1,
                silhouette: best_point_sils[i],
            }
        })
        .collect();

    let dbscan_fallback = run_dbscan(&std_rows);

    let status = if best_sil >= 0.5 {
        "ok"
    } else if best_sil >= 0.25 {
        "warning"
    } else {
        "fail"
    }
    .to_string();

    let interpretation = if crate::i18n::is_en() {
        format!(
            "Highest mean Silhouette score = {:.3} at k = {} clusters. Elbow method suggests k = {} clusters. DBSCAN (eps={:.2}) found {} clusters ({} noise points).",
            best_sil, best_k, elbow_k, dbscan_fallback.eps, dbscan_fallback.n_clusters, dbscan_fallback.noise_points
        )
    } else {
        format!(
            "Høyeste gjennomsnittlige Silhouette-skår = {:.3} ved k = {} klynger. Elbow-metoden foreslår k = {} klynger. DBSCAN (eps={:.2}) fant {} klynger ({} støypunkter).",
            best_sil, best_k, elbow_k, dbscan_fallback.eps, dbscan_fallback.n_clusters, dbscan_fallback.noise_points
        )
    };

    let recommendation = if best_sil >= 0.5 {
        if crate::i18n::is_en() {
            format!(
                "Clear cluster structure (Silhouette = {:.3} >= 0.50). K-means with k = {} is recommended.",
                best_sil, best_k
            )
        } else {
            format!(
                "Tydelig klyngestruktur (Silhouette = {:.3} >= 0.50). K-means med k = {} anbefales.",
                best_sil, best_k
            )
        }
    } else if best_sil >= 0.25 {
        if crate::i18n::is_en() {
            format!(
                "Moderate cluster separation (Silhouette = {:.3}). Compare K-means k = {} (Silhouette) / k = {} (Elbow) with DBSCAN ({} clusters).",
                best_sil, best_k, elbow_k, dbscan_fallback.n_clusters
            )
        } else {
            format!(
                "Moderat klyngeseparasjon (Silhouette = {:.3}). Sammenlign K-means k = {} (Silhouette) / k = {} (Elbow) med DBSCAN ({} klynger).",
                best_sil, best_k, elbow_k, dbscan_fallback.n_clusters
            )
        }
    } else if crate::i18n::is_en() {
        format!(
            "Weak K-means separation (Silhouette = {:.3} < 0.25). DBSCAN fallback computed (eps = {:.2}, {} clusters, {} noise points).",
            best_sil, dbscan_fallback.eps, dbscan_fallback.n_clusters, dbscan_fallback.noise_points
        )
    } else {
        format!(
            "Svak K-means-separasjon (Silhouette = {:.3} < 0.25). DBSCAN-alternativet er beregnet (eps = {:.2}, {} klynger, {} støypunkter).",
            best_sil, dbscan_fallback.eps, dbscan_fallback.n_clusters, dbscan_fallback.noise_points
        )
    };

    Ok(ClusteringReport {
        variables: variables.to_vec(),
        n,
        k_evals,
        best_k_silhouette: best_k,
        best_silhouette: best_sil,
        elbow_k,
        cluster_summaries,
        points_2d,
        dbscan_fallback,
        status,
        interpretation,
        recommendation,
    })
}
