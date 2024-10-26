mod ext;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pymodule]
fn course_graph_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ext::common::get_title_from_latex, m)?)?;
    m.add_function(wrap_pyfunction!(ext::common::get_list_from_string, m)?)?;
    m.add_function(wrap_pyfunction!(ext::structure::post_process, m)?)?;
    Ok(())
}
