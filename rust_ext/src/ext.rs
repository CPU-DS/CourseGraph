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
