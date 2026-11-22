#[cfg(not(test))]
use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Lang {
    #[default]
    No,
    En,
}

impl Lang {
    pub fn code(self) -> &'static str {
        match self {
            Lang::No => "NO",
            Lang::En => "EN",
        }
    }

    pub fn native_name(self) -> &'static str {
        match self {
            Lang::No => "Norsk",
            Lang::En => "English",
        }
    }

    pub fn from_str_code(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "en" | "eng" | "english" => Lang::En,
            _ => Lang::No,
        }
    }
}

#[cfg(not(test))]
static CURRENT_LANG: AtomicU8 = AtomicU8::new(0);

#[cfg(test)]
thread_local! {
    static TEST_LANG: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

pub fn current_lang() -> Lang {
    #[cfg(test)]
    let raw = TEST_LANG.with(|c| c.get());
    #[cfg(not(test))]
    let raw = CURRENT_LANG.load(Ordering::Relaxed);
    match raw {
        1 => Lang::En,
        _ => Lang::No,
    }
}

pub fn set_lang(lang: Lang) {
    let val = match lang {
        Lang::No => 0,
        Lang::En => 1,
    };
    #[cfg(test)]
    TEST_LANG.with(|c| c.set(val));
    #[cfg(not(test))]
    CURRENT_LANG.store(val, Ordering::Relaxed);
}

#[inline]
pub fn is_en() -> bool {
    current_lang() == Lang::En
}

#[inline]
pub fn tr<'a>(no: &'a str, en: &'a str) -> &'a str {
    if is_en() { en } else { no }
}

pub fn status_label(status: &str) -> &'static str {
    match (status, is_en()) {
        ("ok", _) => "OK",
        ("warning", false) => "ADVARSEL",
        ("warning", true) => "WARNING",
        ("fail", false) => "BRUTT",
        ("fail", true) => "VIOLATED",
        _ => "N/A",
    }
}
