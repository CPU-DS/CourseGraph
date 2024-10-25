use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use regex::Regex;

#[pyfunction]
pub fn get_title_from_latex(latex: String) -> PyResult<Vec<String>> {
    let mut titles = Vec::new();

    let commands = vec![
        "title",
        "part",
        "chapter",
        "section",
        "subsection",
        "subsubsection",
        "paragraph",
        "subparagraph",
    ];

    for command in commands {
        let pattern = format!(r"\\({})\{{(.*?)\}}", command);
        let re = Regex::new(&pattern).map_err(|e| PyValueError::new_err(e.to_string()))?;

        for caps in re.captures_iter(&latex) {
            if let Some(title) = caps.get(1) {
                titles.push(title.as_str().to_string());
            }
        }
    }

    Ok(titles)
}

#[pyfunction]
pub fn get_list_from_string(text: &str) -> PyResult<Vec<String>> {
    let mut list_string = String::new();
    let mut stack = 0;
    let mut chars = text.chars();
    let mut result = Vec::new();

    while let Some(s) = chars.next() {
        if s == '[' {
            stack += 1;
        }
        if stack > 0 {
            list_string.push(s);
        }
        if s == ']' {
            stack -= 1;
            if stack == 0 {
                // Here, you would parse the list_string to a Vec
                // For simplicity, let's assume it's a list of strings
                result = vec![list_string.to_string()];
                break;
            }
        }
    }

    Ok(result)
}

#[pyfunction]
pub fn replace_linefeed(sentence: &str, ignore_end: bool, replace: &str) -> PyResult<String> {
    let sentence_endings = r"[。！？.!?]";
    let pattern = if ignore_end {
        format!(r"(?<!{})\n", sentence_endings)
    } else {
        r"\n".to_string()
    };

    let re = Regex::new(&pattern).unwrap();
    Ok(re.replace_all(sentence, replace).into_owned())
}
