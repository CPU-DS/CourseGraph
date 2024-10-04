mod ext;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pymodule]
fn rust_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ext::get_title_from_latex, m)?)?;
    Ok(())
}
