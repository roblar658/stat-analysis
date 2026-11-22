use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ColumnKind {
    Numeric,
    Binary,
    Categorical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnInfo {
    pub index: usize,
    pub name: String,
    pub kind: ColumnKind,
    pub total_count: usize,
    pub valid_count: usize,
    pub missing_count: usize,
    pub unique_count: usize,
    pub mean: Option<f64>,
    pub std_dev: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub categories: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub filename: String,
    pub delimiter: char,
    pub headers: Vec<String>,
    pub raw_rows: Vec<Vec<Option<String>>>,
    pub numeric_cols: HashMap<String, Vec<Option<f64>>>,
    pub columns: Vec<ColumnInfo>,
}

fn is_missing_token(s: &str) -> bool {
    let t = s.trim();
    if t.is_empty() {
        return true;
    }
    matches!(
        t.to_ascii_lowercase().as_str(),
        "na" | "n/a" | "nan" | "null" | "none" | "nil" | "." | "-" | "?" | "missing"
    )
}

fn detect_delimiter(content: &str) -> char {
    let first_lines: Vec<&str> = content.lines().take(10).filter(|l| !l.trim().is_empty()).collect();
    if first_lines.is_empty() {
        return ',';
    }
    let candidates = [',', ';', '\t'];
    let mut best_delim = ',';
    let mut best_score = 0usize;

    for &d in &candidates {
        let counts: Vec<usize> = first_lines.iter().map(|l| l.matches(d).count()).collect();
        let first = counts[0];
        if first > 0 && counts.iter().all(|&c| c == first) {
            if first > best_score {
                best_score = first;
                best_delim = d;
            }
        }
    }
    if best_score == 0 {
        // Fallback to whichever appears most in header
        for &d in &candidates {
            let c = first_lines[0].matches(d).count();
            if c > best_score {
                best_score = c;
                best_delim = d;
            }
        }
    }
    best_delim
}

/// Decode raw CSV file bytes supporting UTF-8 (with/without BOM), UTF-16 LE/BE (Excel Unicode Text),
/// and Windows-1252 / Latin-1 (Excel standard ANSI CSV in Norway/Europe with æ, ø, å).
pub fn decode_csv_bytes(bytes: &[u8]) -> String {
    // UTF-8 BOM
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        if let Ok(s) = std::str::from_utf8(&bytes[3..]) {
            return s.to_string();
        }
    }
    // UTF-16 LE BOM
    if bytes.starts_with(&[0xFF, 0xFE]) && bytes.len() >= 2 {
        let u16s: Vec<u16> = bytes[2..]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        return String::from_utf16_lossy(&u16s);
    }
    // UTF-16 BE BOM
    if bytes.starts_with(&[0xFE, 0xFF]) && bytes.len() >= 2 {
        let u16s: Vec<u16> = bytes[2..]
            .chunks_exact(2)
            .map(|c| u16::from_be_bytes([c[0], c[1]]))
            .collect();
        return String::from_utf16_lossy(&u16s);
    }
    // Strict UTF-8
    if let Ok(s) = std::str::from_utf8(bytes) {
        return s.to_string();
    }
    // Fallback: Windows-1252 (Excel ANSI CSV — æ, ø, å, Æ, Ø, Å, €, –, …, etc.)
    const CP1252_80_9F: [char; 32] = [
        '\u{20AC}', '\u{FFFD}', '\u{201A}', '\u{0192}', '\u{201E}', '\u{2026}', '\u{2020}', '\u{2021}',
        '\u{02C6}', '\u{2030}', '\u{0160}', '\u{2039}', '\u{0152}', '\u{FFFD}', '\u{017D}', '\u{FFFD}',
        '\u{FFFD}', '\u{2018}', '\u{2019}', '\u{201C}', '\u{201D}', '\u{2022}', '\u{2013}', '\u{2014}',
        '\u{02DC}', '\u{2122}', '\u{0161}', '\u{203A}', '\u{0153}', '\u{FFFD}', '\u{017E}', '\u{0178}',
    ];
    let mut out = String::with_capacity(bytes.len());
    for &b in bytes {
        if b < 0x80 {
            out.push(b as char);
        } else if b < 0xA0 {
            out.push(CP1252_80_9F[(b - 0x80) as usize]);
        } else {
            out.push(b as char); // 0xA0..=0xFF maps 1:1 to Unicode U+00A0..=U+00FF (Latin-1, æøåÆØÅ)
        }
    }
    out
}

fn parse_number(s: &str, allow_comma_decimal: bool) -> Option<f64> {
    // Strip regular spaces and Excel non-breaking spaces (thousands separator in Norwegian locale)
    let t: String = s
        .trim()
        .chars()
        .filter(|&c| c != '\u{00A0}' && c != '\u{202F}')
        .collect();
    let t = t.trim();
    if let Ok(v) = t.parse::<f64>() {
        if v.is_finite() {
            return Some(v);
        }
    }
    if allow_comma_decimal && t.contains(',') && !t.contains('.') {
        let replaced = t.replace(' ', "").replace(',', ".");
        if let Ok(v) = replaced.parse::<f64>() {
            if v.is_finite() {
                return Some(v);
            }
        }
    }
    // Also handle spaces as thousands separators with dot decimal
    if t.contains(' ') {
        let compact = t.replace(' ', "");
        if let Ok(v) = compact.parse::<f64>() {
            if v.is_finite() {
                return Some(v);
            }
        }
    }
    None
}

impl Dataset {
    pub fn from_csv_bytes(filename: &str, bytes: &[u8]) -> Result<Self, String> {
        let decoded = decode_csv_bytes(bytes);
        Self::from_csv_str(filename, &decoded)
    }

    pub fn from_csv_str(filename: &str, input: &str) -> Result<Self, String> {
        let mut cleaned = input.strip_prefix('\u{feff}').unwrap_or(input);
        // Handle Excel's optional `sep=;` / `sep=,` first line
        let mut forced_delim: Option<char> = None;
        if let Some(first_line) = cleaned.lines().next() {
            let fl = first_line.trim();
            if fl.len() == 5 && fl.to_ascii_lowercase().starts_with("sep=") {
                forced_delim = fl.chars().nth(4);
                if let Some(pos) = cleaned.find('\n') {
                    cleaned = &cleaned[pos + 1..];
                }
            }
        }
        let delimiter = forced_delim.unwrap_or_else(|| detect_delimiter(cleaned));
        let allow_comma_decimal = delimiter != ',';

        let mut reader = csv::ReaderBuilder::new()
            .delimiter(delimiter as u8)
            .flexible(true)
            .trim(csv::Trim::All)
            .from_reader(cleaned.as_bytes());

        let header_record = reader
            .headers()
            .map_err(|e| format!("Klarte ikke lese CSV-header: {}", e))?;

        let mut headers: Vec<String> = header_record
            .iter()
            .enumerate()
            .map(|(i, h)| {
                let h = h.trim();
                if h.is_empty() {
                    format!("col_{}", i + 1)
                } else {
                    h.to_string()
                }
            })
            .collect();

        if headers.is_empty() {
            return Err("CSV-filen inneholder ingen kolonner.".to_string());
        }

        // Ensure unique header names
        let mut seen: HashMap<String, usize> = HashMap::new();
        for h in &mut headers {
            let entry = seen.entry(h.clone()).or_insert(0);
            *entry += 1;
            if *entry > 1 {
                *h = format!("{}_{}", h, *entry);
            }
        }

        let ncols = headers.len();
        let mut raw_rows: Vec<Vec<Option<String>>> = Vec::new();
        for result in reader.records() {
            let record = result.map_err(|e| format!("CSV-lesefeil: {}", e))?;
            if record.iter().all(|f| f.trim().is_empty()) {
                continue;
            }
            let mut row = Vec::with_capacity(ncols);
            for i in 0..ncols {
                let val = record.get(i).unwrap_or("");
                if is_missing_token(val) {
                    row.push(None);
                } else {
                    row.push(Some(val.trim().to_string()));
                }
            }
            raw_rows.push(row);
        }

        if raw_rows.is_empty() {
            return Err("CSV-filen inneholder ingen datarader.".to_string());
        }

        let n = raw_rows.len();
        let mut numeric_cols: HashMap<String, Vec<Option<f64>>> = HashMap::new();
        let mut columns = Vec::with_capacity(ncols);

        for (ci, name) in headers.iter().enumerate() {
            let mut valid_strs: Vec<String> = Vec::new();
            let mut parsed_nums: Vec<Option<f64>> = Vec::with_capacity(n);
            let mut num_success = 0usize;

            for row in &raw_rows {
                match &row[ci] {
                    None => parsed_nums.push(None),
                    Some(s) => {
                        valid_strs.push(s.clone());
                        if let Some(v) = parse_number(s, allow_comma_decimal) {
                            parsed_nums.push(Some(v));
                            num_success += 1;
                        } else {
                            parsed_nums.push(None);
                        }
                    }
                }
            }

            let valid_count = valid_strs.len();
            let missing_count = n - valid_count;
            let unique_set: HashSet<&String> = valid_strs.iter().collect();
            let unique_count = unique_set.len();

            let is_numeric = valid_count > 0 && (num_success as f64 / valid_count as f64) >= 0.9;

            let (kind, mean, std_dev, min, max, categories) = if is_numeric {
                let vals: Vec<f64> = parsed_nums.iter().filter_map(|&v| v).collect();
                let mut distinct_vals: Vec<f64> = Vec::new();
                for &v in &vals {
                    if !distinct_vals.iter().any(|&x| (x - v).abs() < 1e-12) {
                        distinct_vals.push(v);
                    }
                }
                distinct_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let is_binary = distinct_vals.len() == 2
                    && ((distinct_vals[0] == 0.0 && distinct_vals[1] == 1.0)
                        || (distinct_vals[0] == 1.0 && distinct_vals[1] == 2.0));

                let mean_v = if vals.is_empty() {
                    None
                } else {
                    Some(vals.iter().sum::<f64>() / vals.len() as f64)
                };
                let std_v = if vals.len() > 1 {
                    let m = mean_v.unwrap();
                    let var = vals.iter().map(|x| (x - m).powi(2)).sum::<f64>()
                        / (vals.len() - 1) as f64;
                    Some(var.sqrt())
                } else {
                    None
                };
                let min_v = vals.iter().copied().reduce(f64::min);
                let max_v = vals.iter().copied().reduce(f64::max);

                numeric_cols.insert(name.clone(), parsed_nums);
                let kind = if is_binary {
                    ColumnKind::Binary
                } else {
                    ColumnKind::Numeric
                };
                let cats = if is_binary {
                    distinct_vals.iter().map(|v| format!("{}", v)).collect()
                } else {
                    vec![]
                };
                (kind, mean_v, std_v, min_v, max_v, cats)
            } else {
                // Check if boolean/binary string
                let mut cats: Vec<String> = unique_set.into_iter().cloned().collect();
                cats.sort();
                let kind = if cats.len() == 2 {
                    // Also map to 0/1 in numeric_cols so logistic/survival can use it
                    let c1 = &cats[1];
                    let mapped: Vec<Option<f64>> = raw_rows
                        .iter()
                        .map(|r| {
                            r[ci].as_ref().map(|s| if s == c1 { 1.0 } else { 0.0 })
                        })
                        .collect();
                    numeric_cols.insert(name.clone(), mapped);
                    ColumnKind::Binary
                } else {
                    ColumnKind::Categorical
                };
                (kind, None, None, None, None, cats)
            };

            columns.push(ColumnInfo {
                index: ci,
                name: name.clone(),
                kind,
                total_count: n,
                valid_count,
                missing_count,
                unique_count,
                mean,
                std_dev,
                min,
                max,
                categories,
            });
        }

        Ok(Dataset {
            filename: filename.to_string(),
            delimiter,
            headers,
            raw_rows,
            numeric_cols,
            columns,
        })
    }

    pub fn row_count(&self) -> usize {
        self.raw_rows.len()
    }

    pub fn col_index(&self, name: &str) -> Option<usize> {
        self.headers.iter().position(|h| h == name)
    }

    pub fn numeric_column_names(&self) -> Vec<String> {
        self.columns
            .iter()
            .filter(|c| c.kind == ColumnKind::Numeric)
            .map(|c| c.name.clone())
            .collect()
    }

    pub fn all_numeric_capable_names(&self) -> Vec<String> {
        self.columns
            .iter()
            .filter(|c| matches!(c.kind, ColumnKind::Numeric | ColumnKind::Binary))
            .map(|c| c.name.clone())
            .collect()
    }

    pub fn binary_column_names(&self) -> Vec<String> {
        self.columns
            .iter()
            .filter(|c| c.kind == ColumnKind::Binary)
            .map(|c| c.name.clone())
            .collect()
    }

    pub fn grouping_column_names(&self) -> Vec<String> {
        self.columns
            .iter()
            .filter(|c| {
                (c.kind == ColumnKind::Categorical || c.kind == ColumnKind::Binary)
                    && c.unique_count >= 2
                    && c.unique_count <= 15
            })
            .map(|c| c.name.clone())
            .collect()
    }

    /// Return non-missing numeric values for a single column.
    pub fn get_numeric_values(&self, col: &str) -> Result<Vec<f64>, String> {
        let vals = self
            .numeric_cols
            .get(col)
            .ok_or_else(|| format!("Kolonnen '{}' er ikke numerisk.", col))?;
        Ok(vals.iter().filter_map(|&v| v).collect())
    }

    /// Return complete-case numeric rows for the given column names.
    pub fn get_complete_numeric_matrix(&self, cols: &[String]) -> Result<Vec<Vec<f64>>, String> {
        let mut col_vecs: Vec<&Vec<Option<f64>>> = Vec::with_capacity(cols.len());
        for c in cols {
            let v = self
                .numeric_cols
                .get(c)
                .ok_or_else(|| format!("Kolonnen '{}' er ikke numerisk.", c))?;
            col_vecs.push(v);
        }
        let n = self.row_count();
        let mut rows = Vec::new();
        for i in 0..n {
            let mut row = Vec::with_capacity(cols.len());
            let mut ok = true;
            for cv in &col_vecs {
                if let Some(val) = cv[i] {
                    row.push(val);
                } else {
                    ok = false;
                    break;
                }
            }
            if ok {
                rows.push(row);
            }
        }
        Ok(rows)
    }

    /// Group values of a numeric column by a grouping column (categorical, binary, or low-cardinality numeric).
    pub fn get_grouped_values(
        &self,
        value_col: &str,
        group_col: &str,
    ) -> Result<BTreeMap<String, Vec<f64>>, String> {
        let g_idx = self
            .col_index(group_col)
            .ok_or_else(|| format!("Fant ikke gruppekolonne '{}'.", group_col))?;
        let vals = self
            .numeric_cols
            .get(value_col)
            .ok_or_else(|| format!("Kolonnen '{}' er ikke numerisk.", value_col))?;

        let mut groups: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for (i, row) in self.raw_rows.iter().enumerate() {
            if let (Some(g), Some(v)) = (&row[g_idx], vals[i]) {
                groups.entry(g.clone()).or_default().push(v);
            }
        }
        // Filter out empty or singleton? Keep all and let tests validate.
        Ok(groups)
    }

    /// Extract binary (0.0 or 1.0) values for a column, normalizing {a, b} -> {0.0, 1.0} if needed.
    pub fn get_binary_series(&self, col: &str) -> Result<(Vec<Option<f64>>, String, String), String> {
        let ci = self
            .col_index(col)
            .ok_or_else(|| format!("Fant ikke kolonne '{}'.", col))?;

        let mut unique: Vec<String> = Vec::new();
        for r in &self.raw_rows {
            if let Some(v) = &r[ci] {
                if !unique.contains(v) {
                    unique.push(v.clone());
                }
            }
        }
        unique.sort();
        if unique.len() != 2 {
            return Err(format!(
                "Kolonnen '{}' må ha nøyaktig 2 unike verdier for binær/logistisk/event-analyse (fant {}).",
                col,
                unique.len()
            ));
        }

        // If numeric 0/1, ensure 0 is negative and 1 is positive
        let (neg, pos) = if let (Ok(a), Ok(b)) = (unique[0].parse::<f64>(), unique[1].parse::<f64>()) {
            if a < b {
                (unique[0].clone(), unique[1].clone())
            } else {
                (unique[1].clone(), unique[0].clone())
            }
        } else {
            (unique[0].clone(), unique[1].clone())
        };

        let out = self
            .raw_rows
            .iter()
            .map(|r| r[ci].as_ref().map(|s| if s == &pos { 1.0 } else { 0.0 }))
            .collect();
        Ok((out, neg, pos))
    }

    /// Return numeric matrix with Option<f64> preserving missingness for missing data analysis.
    pub fn get_numeric_matrix_with_missing(
        &self,
        cols: &[String],
    ) -> Result<Vec<Vec<Option<f64>>>, String> {
        let mut col_vecs: Vec<&Vec<Option<f64>>> = Vec::with_capacity(cols.len());
        for c in cols {
            let v = self
                .numeric_cols
                .get(c)
                .ok_or_else(|| format!("Kolonnen '{}' er ikke numerisk.", c))?;
            col_vecs.push(v);
        }
        let n = self.row_count();
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(cols.len());
            for cv in &col_vecs {
                row.push(cv[i]);
            }
            rows.push(row);
        }
        Ok(rows)
    }
}
