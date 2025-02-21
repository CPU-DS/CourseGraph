mod ext;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pymodule]
fn extension(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ext::common::get_title_from_latex, m)?)?;
    m.add_function(wrap_pyfunction!(ext::common::get_list_from_string, m)?)?;
    m.add_function(wrap_pyfunction!(ext::structure::structure_post_process, m)?)?;
    m.add_function(wrap_pyfunction!(
        ext::common::find_longest_consecutive_sequence,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(ext::common::optimize_strings_length, m)?)?;
    m.add_function(wrap_pyfunction!(ext::common::merge_strings, m)?)?;
    Ok(())
}
