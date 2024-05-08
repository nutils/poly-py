use ndarray::{ArrayBase, ArrayD, ArrayViewD, Data, Dimension};
use numpy::{
    PyArray, PyArray2, PyArrayDyn, PyArrayMethods, PyReadonlyArray2, PyReadonlyArrayDyn,
    PyUntypedArrayMethods, ToPyArray,
};
use nutils_poly::{MulPlan, MulVar, PartialDerivPlan, Power};
use pyo3::class::basic::CompareOp;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use sha1::{Digest, Sha1};
use sqnc::traits::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::iter;

// INTERNAL UTILITIES

/// Concatenate iterables of `usize`.
macro_rules! shape {
    ($($part:expr),*) => {{
        let mut shape: Vec<usize> = Vec::new();
        $(shape.extend($part.into_iter().map(|item| item.to_owned()));)*
        shape
    }};
    ($($part:expr,)*) => { shape![$($part),*] };
}

/// Broadcast the input shapes into a single shape.
fn broadcast_shapes(shapes: &[&[usize]]) -> PyResult<Vec<usize>> {
    let Some(n) = shapes.iter().map(|shape| shape.len()).max() else {
        return Err(PyValueError::new_err("expected at least one shape"));
    };
    let mut common: Vec<usize> = iter::repeat(1).take(n).collect();
    for shape in shapes {
        for (i, j) in iter::zip(common.iter_mut().rev(), shape.iter().rev()) {
            if *i == 1 {
                *i = *j;
            } else if *i != *j && *j != 1 {
                return Err(PyValueError::new_err(format!(
                    "cannot broadcast shapes: {shapes:?}"
                )));
            }
        }
    }
    Ok(common)
}

// Function `has_contiguous_rows` tests whether the last axis of an array is
// contiguous. This is used by several methods below to iterate over slices
// instead of `ArrayView`s, like so:
//
// ```
// if has_contiguous_rows(array) {
//     for row in array.rows() {
//         let row = row.as_slice().unwrap();
//         ...
//     }
// }
// ```
//
// Ideally we would combine this into a `contiguous_rows` function that returns
// an iterator over slices if possible:
//
// ```
// fn contiguous_rows<S, D>(array: &ArrayBase<S, D>) -> Option<impl Iterator<Item = &[S::Item]>> {
//     if has_contiguous_rows(array) {
//         Some(array.rows().map(|row| row.as_slice().unwrap()))
//     } else {
//         None
//     }
// }
//
// if let Some(rows) = contiguous_rows(array) {
//     ...
// }
// ```
//
// Unfortunately this doesn't seem to be possible, because `array.rows()`
// returns `ArrayView`s and `ArrayView::as_slice()` returns a slice with a
// lifetime bound by the view of the row, not the original array, and the view
// of the row lives only for the duration of `std::iter::Map::next()`.

/// Returns true if the last axis of the input array is contiguous.
#[inline]
fn has_contiguous_rows<S, D>(array: &ArrayBase<S, D>) -> bool
where
    S: Data,
    D: Dimension,
{
    array.strides().last().map_or(false, |stride| *stride == 1)
        || array.shape().last().map_or(false, |n| *n <= 1)
}

#[inline]
fn as_coeffs_dyn<'a>(
    coeffs: &'a PyReadonlyArrayDyn<f64>,
    nvars: usize,
    name: &str,
) -> PyResult<(ArrayViewD<'a, f64>, Power, usize, &'a [usize])> {
    let Some((ncoeffs, leading_shape)) = coeffs.shape().split_last() else {
        return Err(PyValueError::new_err(format!("expected `{name}` with at least one axis")));
    };
    let degree = degree(nvars, *ncoeffs)?;
    Ok((coeffs.as_array(), degree, *ncoeffs, leading_shape))
}

#[inline]
fn as_coeffs_dyn_with_ncoeffs<'a>(
    coeffs: &'a PyReadonlyArrayDyn<f64>,
    desired_ncoeffs: usize,
    name: &str,
) -> PyResult<(ArrayViewD<'a, f64>, &'a [usize])> {
    let Some((ncoeffs, leading_shape)) = coeffs.shape().split_last() else {
        return Err(PyValueError::new_err(format!("expected `{name}` with shape `(...,{desired_ncoeffs})` but got `()`")));
    };
    if *ncoeffs != desired_ncoeffs {
        return Err(PyValueError::new_err(format!(
            "expected `coeffs` with shape `(...,{desired_ncoeffs})` but got `(...,{ncoeffs})`"
        )));
    }
    Ok((coeffs.as_array(), leading_shape))
}

// PUBLIC INTERFACE

/// Returns the degree of a polynomial for the given number of coefficients and variables.
///
/// Args
/// ----
/// nvars : :class:`int`
///     The number of variables.
/// ncoeffs : :class:`int`
///     The number of coefficients.
///
/// Returns
/// -------
/// degree : :class:`int`
///     The degree of the polynomial.
///
/// Raises
/// ------
/// :class:`ValueError`
///     If the number of coefficients or the number of variables are negative.
/// :class:`ValueError`
///     If there is no degree that matches the given number of coefficients and
///     variables. For example a polynomial in two variables has one coefficient
///     for degree zero and three coefficients for degree one, but there is
///     nothing in between.
///
/// See also
/// --------
/// :func:`ncoeffs` : The inverse operation.
#[pyfunction]
#[pyo3(text_signature = "(nvars, ncoeffs)")]
fn degree(nvars: usize, ncoeffs: usize) -> PyResult<Power> {
    nutils_poly::degree(nvars, ncoeffs).ok_or_else(|| {
        let variables = match nvars {
            1 => "variable",
            _ => "variables",
        };
        PyValueError::new_err(format!(
            "invalid number of coefficients for a polynomial with {nvars} {variables}: {ncoeffs}"
        ))
    })
}

/// Returns the number of coefficients for a polynomial of given degree and number of variables.
///
/// Args
/// ----
/// degree : :class:`int`
///     The degree of the polynomial.
/// nvars : :class:`int`
///     The number of variables.
///
/// Returns
/// -------
/// ncoeffs : :class:`int`
///     The number of coefficients of the polynomial.
///
/// Raises
/// ------
/// :class:`ValueError`
///     If the degree or the number of variables are negative.
///
/// See also
/// --------
/// :func:`ncoeffs` : The inverse operation.
#[pyfunction]
#[pyo3(text_signature = "(nvars, degree)")]
fn ncoeffs(nvars: usize, degree: Power) -> usize {
    nutils_poly::ncoeffs(nvars, degree)
}

/// Evaluates polynomials for the given values.
///
/// The degree of the polynomial and the number of variables are automatically
/// determined from the lengths of the last axes of the arguments.
///
/// Args
/// ----
/// coeffs : :class:`numpy.ndarray`
///     The polynomial coefficients. The last axis constitutes the axis of
///     coefficients.
/// values : :class:`numpy.ndarray`
///     The values for the variables of the coefficients. The last axis
///     constitutes the axis of variables.
///
/// Returns
/// -------
/// :class:`numpy.ndarray`
///     The evaluated polynomial. The shape is the broadcasted shape of
///     ``coeffs`` and ``values`` with last axes stripped.
///
/// See also
/// --------
/// :func:`eval_outer` : Evaluates all pairs of coefficients and values.
#[pyfunction]
#[pyo3(text_signature = "(coeffs, coords)")]
fn eval<'py>(
    py: Python<'py>,
    coeffs: PyReadonlyArrayDyn<f64>,
    values: PyReadonlyArrayDyn<f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let values = values.as_array();
    let Some((nvars, values_leading_shape)) = values.shape().split_last() else {
        return Err(PyValueError::new_err("expected `values` with at least one axis"));
    };
    let (coeffs, degree, ncoeffs, coeffs_leading_shape) = as_coeffs_dyn(&coeffs, *nvars, "coeffs")?;
    let result_shape = broadcast_shapes(&[coeffs_leading_shape, values_leading_shape])?;
    let coeffs = coeffs.broadcast(shape![&result_shape, [ncoeffs]]).unwrap();
    let values = values.broadcast(shape![&result_shape, [nvars]]).unwrap();
    let mut result = Vec::with_capacity(result_shape.iter().copied().product());
    if has_contiguous_rows(&coeffs) && has_contiguous_rows(&values) {
        for (coeffs, values) in iter::zip(coeffs.rows(), values.rows()) {
            result.push(
                nutils_poly::eval(
                    coeffs.as_slice().unwrap(),
                    values.as_slice().unwrap(),
                    degree,
                )
                .unwrap(),
            );
        }
    } else {
        for (coeffs, values) in iter::zip(coeffs.rows(), values.rows()) {
            result.push(nutils_poly::eval(coeffs, &values, degree).unwrap());
        }
    }
    PyArray::from_vec_bound(py, result).reshape(result_shape)
}

/// Evaluates all pairs of polynomials and values.
///
/// The degree of the polynomial and the number of variables are automatically
/// determined from the lengths of the last axes of the arguments.
///
/// Args
/// ----
/// coeffs : :class:`numpy.ndarray`
///     The polynomial coefficients. The last axis constitutes the axis of coefficients.
/// values : :class:`numpy.ndarray`
///     The values for the variables of the coefficients. The last axis
///     constitutes the axis of variables.
///
/// Returns
/// -------
/// :class:`numpy.ndarray`
///     The evaluated polynomial. The leading axes of ``values`` come first,
///     followed by the leading axes of ``coeffs``.
///
/// See also
/// --------
/// :func:`eval` : Evaluates joined polynomials and values.
#[pyfunction]
#[pyo3(text_signature = "(coeffs, coords)")]
fn eval_outer<'py>(
    py: Python<'py>,
    coeffs: PyReadonlyArrayDyn<f64>,
    values: PyReadonlyArrayDyn<f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let values = values.as_array();
    let Some((nvars, values_leading_shape)) = values.shape().split_last() else {
        return Err(PyValueError::new_err("expected `values` with at least one axis"));
    };
    let (coeffs, degree, _, coeffs_leading_shape) = as_coeffs_dyn(&coeffs, *nvars, "coeffs")?;
    let result_shape = shape![values_leading_shape, coeffs_leading_shape];
    let mut result = Vec::with_capacity(result_shape.iter().copied().product());
    if has_contiguous_rows(&coeffs) && has_contiguous_rows(&values) {
        for values in values.rows() {
            for coeffs in coeffs.rows() {
                result.push(
                    nutils_poly::eval(
                        coeffs.as_slice().unwrap(),
                        values.as_slice().unwrap(),
                        degree,
                    )
                    .unwrap(),
                );
            }
        }
    } else if has_contiguous_rows(&coeffs) {
        let mut contig_values: Box<[f64]> = iter::repeat(0.0).take(*nvars).collect();
        let mut contig_values = contig_values.as_mut_sqnc();
        for values in values.rows() {
            contig_values.assign(values.copied().iter()).unwrap();
            for coeffs in coeffs.rows() {
                result.push(
                    nutils_poly::eval(coeffs.as_slice().unwrap(), &contig_values, degree).unwrap(),
                );
            }
        }
    } else {
        let mut contig_values: Box<[f64]> = iter::repeat(0.0).take(*nvars).collect();
        let mut contig_values = contig_values.as_mut_sqnc();
        for values in values.rows() {
            contig_values.assign(values.copied().iter()).unwrap();
            for coeffs in coeffs.rows() {
                result.push(nutils_poly::eval(coeffs, &contig_values, degree).unwrap());
            }
        }
    }
    PyArray::from_vec_bound(py, result).reshape(result_shape)
}

/// Plan for the partial derivative of a polynomial.
///
/// The plan can be applied to coefficients (``plan(coeffs)``) to obtain the
/// coefficients for the partial derivative of the polynomial.
///
/// Args
/// ----
/// nvars : :class:`int`
///     The number of variables of the polynomial.
/// degree : :class:`int`
///     The degree of the polynomial.
/// var : :class:`int`
///     The (index of the) variable to compute the partial derivative for.
///
/// Example
/// -------
///
/// The partial derivative of polynomial :math:`f(x) = x_1^2 + 2 x_0 x_1 + 3
/// x_0` to :math:`x_1`:
///
/// >>> import numpy
/// >>> pd = PartialDerivPlan(2, 2, 1)
/// >>> numpy.testing.assert_allclose(
/// ...     pd(numpy.array([1, 2, 0, 0, 3, 0], dtype=float)),
/// ...     [2, 2, 0],
/// ... )
///
/// See also
/// --------
/// :func:`partial_deriv` : Compute the partial derivative without a plan.
/// :func:`GradPlan` : Plan for the gradient of a polynomial.
#[pyclass]
#[pyo3(name = "PartialDerivPlan", module = "nutils_poly")]
#[derive(Debug, Clone)]
struct PyPartialDerivPlan(PartialDerivPlan);

#[pymethods]
impl PyPartialDerivPlan {
    #[new]
    #[pyo3(text_signature = "(nvars, degree, var)")]
    fn new(nvars: usize, degree: Power, var: usize) -> PyResult<Self> {
        if let Some(plan) = PartialDerivPlan::new(nvars, degree, var) {
            Ok(Self(plan))
        } else {
            Err(PyValueError::new_err(format!("the (index of the) variable to plan the partial derivative for ({var}) exceeds the number of variables ({nvars})")))
        }
    }
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        coeffs: PyReadonlyArrayDyn<f64>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let (coeffs, leading_shape) =
            as_coeffs_dyn_with_ncoeffs(&coeffs, self.0.ncoeffs_input(), "coeffs")?;
        let result_shape = shape![leading_shape, [self.0.ncoeffs_output()]];
        let mut result = Vec::with_capacity(result_shape.iter().copied().product());
        if has_contiguous_rows(&coeffs) {
            for coeffs in coeffs.rows() {
                let coeffs = coeffs.as_slice().unwrap().as_sqnc();
                result.extend(self.0.apply(coeffs).unwrap().iter());
            }
        } else {
            for coeffs in coeffs.rows() {
                result.extend(self.0.apply(coeffs).unwrap().iter());
            }
        }
        PyArray::from_vec_bound(py, result).reshape(result_shape)
    }
}

/// Returns the coefficients for the partial derivative of a polynomial.
///
/// Args
/// ----
/// coeffs : :class:`numpy.ndarray`
///     The coefficients of the polynomials. The last axis constitutes the axis
///     of coefficients.
/// nvars : :class:`int`
///     The number of variables of the polynomial.
/// var : :class:`int`
///     The (index of the) variable to compute the partial derivative for.
///
/// Returns
/// -------
/// :class:`numpy.ndarray`
///     The coefficients for the partial derivative. The last axis constitutes
///     the axis of coefficients. The leading axes are the same as the leading
///     axes of ``coeffs``.
///
/// Example
/// -------
///
/// The partial derivative of polynomial :math:`f(x) = x_1^2 + 2 x_0 x_1 + 3
/// x_0` to :math:`x_1`:
///
/// >>> import numpy
/// >>> numpy.testing.assert_allclose(
/// ...     partial_deriv(numpy.array([1, 2, 0, 0, 3, 0], dtype=float), 2, 1),
/// ...     [2, 2, 0],
/// ... )
///
/// Notes
/// -----
///
/// If you are going to compute the partial derivative multiple times for the
/// same degree and number of variables, then consider using a
/// :class:`PartialDerivPlan`.
///
/// See also
/// --------
/// :func:`PartialDerivPlan` : Compute the partial derivative with a plan.
/// :func:`grad` : Compute the gradient of a polynomial.
#[pyfunction]
#[pyo3(text_signature = "(coeffs, nvars)")]
fn partial_deriv<'py>(
    py: Python<'py>,
    coeffs: PyReadonlyArrayDyn<f64>,
    nvars: usize,
    var: usize,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let Some(ncoeffs) = coeffs.shape().last() else {
        return Err(PyValueError::new_err("expected `coeffs` with at least one axis"));
    };
    let degree = degree(nvars, *ncoeffs)?;
    PyPartialDerivPlan::new(nvars, degree, var)?.__call__(py, coeffs)
}

/// Plan for the gradient of a polynomial.
///
/// The plan can be applied to coefficients (``plan(coeffs)``) to obtain the
/// coefficients for the gradient of the polynomial.
///
/// Args
/// ----
/// nvars : :class:`int`
///     The number of variables of the polynomial.
/// degree : :class:`int`
///     The degree of the polynomial.
///
/// See also
/// --------
/// :func:`grad` : Compute the gradient without a plan.
/// :func:`PartialDerivPlan` : Plan for the partial derivative of a polynomial.
#[pyclass]
#[pyo3(name = "GradPlan", module = "nutils_poly")]
#[derive(Debug, Clone)]
struct PyGradPlan {
    plans: Box<[PartialDerivPlan]>,
    ncoeffs_input: usize,
    ncoeffs_output: usize,
}

#[pymethods]
impl PyGradPlan {
    #[new]
    #[pyo3(text_signature = "(nvars, degree)")]
    fn new(nvars: usize, degree: Power) -> Self {
        let plans: Box<[_]> = Iterator::map(0..nvars, |var| {
            PartialDerivPlan::new(nvars, degree, var).unwrap()
        })
        .collect();
        let (ncoeffs_input, ncoeffs_output) = if let Some(plan) = plans.first() {
            (plan.ncoeffs_input(), plan.ncoeffs_output())
        } else {
            (1, 1)
        };
        Self {
            plans,
            ncoeffs_input,
            ncoeffs_output,
        }
    }
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        coeffs: PyReadonlyArrayDyn<f64>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let (coeffs, leading_shape) =
            as_coeffs_dyn_with_ncoeffs(&coeffs, self.ncoeffs_input, "coeffs")?;
        let result_shape = shape![leading_shape, [self.plans.len(), self.ncoeffs_output]];
        let mut result = Vec::with_capacity(result_shape.iter().copied().product());
        if has_contiguous_rows(&coeffs) {
            for coeffs in coeffs.rows() {
                for plan in self.plans.iter() {
                    let coeffs = coeffs.as_slice().unwrap().as_sqnc();
                    result.extend(plan.apply(coeffs).unwrap().iter());
                }
            }
        } else {
            for coeffs in coeffs.rows() {
                for plan in self.plans.iter() {
                    result.extend(plan.apply(coeffs).unwrap().iter());
                }
            }
        }
        PyArray::from_vec_bound(py, result).reshape(result_shape)
    }
    #[getter]
    fn ncoeffs_output(&self) -> usize {
        self.ncoeffs_output
    }
}

/// Returns the coefficients for the gradient of a polynomial.
///
/// Args
/// ----
/// coeffs : :class:`numpy.ndarray`
///     The coefficients of the polynomials. The last axis constitutes the axis
///     of coefficients.
/// nvars : :class:`int`
///     The number of variables of the polynomial.
///
/// Returns
/// -------
/// :class:`numpy.ndarray`
///     The coefficients for the partial derivative. The last axis constitutes
///     the axis of coefficients and the next to last axis the axis of
///     variables. The leading axes are the same as the leading axes of
///     ``coeffs``.
///
/// Notes
/// -----
///
/// If you are going to compute the gradient multiple times for the
/// same degree and number of variables, then consider using a
/// :class:`GradPlan`.
///
/// See also
/// --------
/// :func:`GradPlan` : Compute the gradient with a plan.
/// :func:`partial_deriv` : Compute the partial derivative of a polynomial.
#[pyfunction]
#[pyo3(text_signature = "(coeffs, nvars)")]
fn grad<'py>(
    py: Python<'py>,
    coeffs: PyReadonlyArrayDyn<f64>,
    nvars: usize,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let Some(ncoeffs) = coeffs.shape().last() else {
        return Err(PyValueError::new_err("expected `coeffs` with at least one axis"));
    };
    let degree = degree(nvars, *ncoeffs)?;
    PyGradPlan::new(nvars, degree).__call__(py, coeffs)
}

/// Multiplication variable.
///
/// The multiplication variable defines if the variable exists in the left
/// operand, the right operand of both.
///
/// .. py:attribute:: Left
///    :type: MulVar
///
///    The variable exists only in the left operand.
///
/// .. py:attribute:: Right
///    :type: MulVar
///
///    The variable exists only in the right operand.
///
/// .. py:attribute:: Both
///    :type: MulVar
///
///    The variable exists in both operands.
#[pyclass]
#[pyo3(name = "MulVar", module = "nutils_poly")]
#[derive(Debug, Clone)]
struct PyMulVar(MulVar);

#[pymethods]
impl PyMulVar {
    /// Left
    #[allow(non_snake_case)]
    #[classattr]
    fn Left() -> Self {
        Self(MulVar::Left)
    }
    /// Right
    #[allow(non_snake_case)]
    #[classattr]
    fn Right() -> Self {
        Self(MulVar::Right)
    }
    /// Both
    #[allow(non_snake_case)]
    #[classattr]
    fn Both() -> Self {
        Self(MulVar::Both)
    }
    fn __repr__(&self) -> &'static str {
        match self.0 {
            MulVar::Left => "MulVar.Left",
            MulVar::Right => "MulVar.Right",
            MulVar::Both => "MulVar.Both",
        }
    }
    fn __reduce__(&self) -> &'static str {
        self.__repr__()
    }
    fn __richcmp__(&self, other: &Self, op: CompareOp, py: Python<'_>) -> PyObject {
        match op {
            CompareOp::Eq => (self.0 == other.0).into_py(py),
            CompareOp::Ne => (self.0 != other.0).into_py(py),
            _ => py.NotImplemented(),
        }
    }
    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.0.hash(&mut hasher);
        hasher.finish()
    }
    #[getter]
    fn __nutils_hash__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let mut hasher = Sha1::new();
        hasher.update(self.__repr__().as_bytes());
        let digest = hasher.finalize();
        PyBytes::new_bound(py, &digest)
    }
}

/// Plan for the product of two polynomials.
///
/// The plan can be applied to a pair of coefficients (``plan(coeffs_left,
/// coeffs_right, vars)``) to obtain the coefficients for the product of
/// the polynomials.
///
/// Args
/// ----
/// vars : sequence of :class:`MulVar`
///     List of variables. For each output variable ``vars`` specifies if the
///     variable exists in the left polynomial, the right polynomial or both.
/// degree_left : :class:`int`
///     The degree of the left polynomial.
/// degree_right : :class:`int`
///     The degree of the right polynomial.
///
/// Example
/// -------
///
/// Consider the following polynomials:
///
/// .. math:: f(x) = x^2 + 2 x + 3
/// .. math:: g(x) = 3 x + 1
///
/// The coefficients for the product :math:`p(x) = f(x) g(x)`:
///
/// >>> import numpy
/// >>> plan = MulPlan([MulVar.Both], 2, 1)
/// >>> numpy.testing.assert_allclose(
/// ...     plan(numpy.array([1, 2, 3], dtype=float), numpy.array([3, 1], dtype=float)),
/// ...     [3, 7, 11, 3],
/// ... )
///
/// The coefficients for the product :math:`q(x, y) = f(x) g(y)`:
///
/// >>> import numpy
/// >>> plan = MulPlan([MulVar.Left, MulVar.Right], 2, 1)
/// >>> numpy.testing.assert_allclose(
/// ...     plan(numpy.array([1, 2, 3], dtype=float), numpy.array([3, 1], dtype=float)),
/// ...     [0, 0, 0, 3, 6, 9, 0, 1, 2, 3],
/// ... )
///
/// See also
/// --------
/// :func:`mul` : Compute the multiplication without a plan.
/// :meth:`MulPlan.same_vars` : Create a plan for the multiplication of two polynomials in the same variables.
/// :meth:`MulPlan.different_vars` : Create a plan for the multiplication of two polynomials in different variables.
#[pyclass]
#[pyo3(name = "MulPlan", module = "nutils_poly")]
#[derive(Debug, Clone)]
struct PyMulPlan(MulPlan);

#[pymethods]
impl PyMulPlan {
    #[new]
    #[pyo3(text_signature = "(vars, degree_left, degree_right)")]
    fn new(vars: Vec<PyMulVar>, degree_left: Power, degree_right: Power) -> Self {
        Self(MulPlan::new(
            &vars.as_sqnc().map(|var| var.0),
            degree_left,
            degree_right,
        ))
    }
    /// Create a plan for the multiplication of two polynomials in the same variables.
    ///
    /// Args
    /// ----
    /// nvars : class:`int`
    ///     The number of variables of the left and right polynomials.
    /// degree_left : :class:`int`
    ///     The degree of the left polynomial.
    /// degree_right : :class:`int`
    ///     The degree of the right polynomial.
    #[staticmethod]
    fn same_vars(nvars: usize, degree_left: Power, degree_right: Power) -> Self {
        Self(MulPlan::same_vars(nvars, degree_left, degree_right))
    }
    /// Create a plan for the multiplication of two polynomials in different variables.
    ///
    /// Args
    /// ----
    /// nvars_left : :class:`int`
    ///     The number of variables of the left polynomial.
    /// nvars_right : :class:`int`
    ///     The number of variables of the right polynomial.
    /// degree_left : :class:`int`
    ///     The degree of the left polynomial.
    /// degree_right : :class:`int`
    ///     The degree of the right polynomial.
    #[staticmethod]
    fn different_vars(
        nvars_left: usize,
        nvars_right: usize,
        degree_left: Power,
        degree_right: Power,
    ) -> PyResult<Self> {
        Ok(Self(
            MulPlan::different_vars(nvars_left, nvars_right, degree_left, degree_right)
                .map_err(|err| PyValueError::new_err(err.to_string()))?,
        ))
    }
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        coeffs_left: PyReadonlyArrayDyn<f64>,
        coeffs_right: PyReadonlyArrayDyn<f64>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let (coeffs_left, leading_shape_left) =
            as_coeffs_dyn_with_ncoeffs(&coeffs_left, self.0.ncoeffs_left(), "coeffs_left")?;
        let (coeffs_right, leading_shape_right) =
            as_coeffs_dyn_with_ncoeffs(&coeffs_right, self.0.ncoeffs_right(), "coeffs_right")?;
        let leading_shape = broadcast_shapes(&[leading_shape_left, leading_shape_right])?;
        let coeffs_left = coeffs_left
            .broadcast(shape![&leading_shape, [self.0.ncoeffs_left()]])
            .unwrap();
        let coeffs_right = coeffs_right
            .broadcast(shape![&leading_shape, [self.0.ncoeffs_right()]])
            .unwrap();
        let result_shape = shape![&leading_shape, [self.0.ncoeffs_output()]];
        let mut result = Vec::with_capacity(result_shape.iter().copied().product());
        if has_contiguous_rows(&coeffs_left) && has_contiguous_rows(&coeffs_right) {
            for (coeffs_left, coeffs_right) in iter::zip(coeffs_left.rows(), coeffs_right.rows()) {
                let coeffs_left = coeffs_left.as_slice().unwrap().as_sqnc();
                let coeffs_right = coeffs_right.as_slice().unwrap().as_sqnc();
                result.extend(self.0.apply(coeffs_left, coeffs_right).unwrap().iter());
            }
        } else {
            for (coeffs_left, coeffs_right) in iter::zip(coeffs_left.rows(), coeffs_right.rows()) {
                result.extend(self.0.apply(coeffs_left, coeffs_right).unwrap().iter());
            }
        }
        PyArray::from_vec_bound(py, result).reshape(result_shape)
    }
    #[getter]
    fn ncoeffs_output(&self) -> usize {
        self.0.ncoeffs_output()
    }
}

/// Returns the coefficients for the product of two polynomials.
///
/// Args
/// ----
/// coeffs_left : :class:`int`
///     The coefficents of the left polynomial. The last axis constitutes the
///     axis of coefficients.
/// coeffs_right : :class:`int`
///     The coefficents of the right polynomial. The last axis constitutes the
///     axis of coefficients.
/// vars : sequence of :class:`MulVar`
///     List of variables. For each output variable ``vars`` specifies if the
///     variable exists in the left polynomial, the right polynomial or both.
///
/// Returns
/// -------
/// :class:`numpy.ndarray`
///     The coefficients for the product. The last axis constitutes
///     the axis of coefficients. The leading axes are the same as the leading axes of
///     ``coeffs_left`` and ``coeffs_right``.
///
/// Example
/// -------
///
/// Consider the following polynomials:
///
/// .. math:: f(x) = x^2 + 2 x + 3
/// .. math:: g(x) = 3 x + 1
///
/// The coefficients for the product :math:`p(x) = f(x) g(x)`:
///
/// >>> import numpy
/// >>> numpy.testing.assert_allclose(
/// ...     mul(
/// ...         numpy.array([1, 2, 3], dtype=float),
/// ...         numpy.array([3, 1], dtype=float),
/// ...         [MulVar.Both],
/// ...     ),
/// ...     [3, 7, 11, 3],
/// ... )
///
/// The coefficients for the product :math:`q(x, y) = f(x) g(y)`:
///
/// >>> import numpy
/// >>> numpy.testing.assert_allclose(
/// ...     mul(
/// ...         numpy.array([1, 2, 3], dtype=float),
/// ...         numpy.array([3, 1], dtype=float),
/// ...         [MulVar.Left, MulVar.Right],
/// ...     ),
/// ...     [0, 0, 0, 3, 6, 9, 0, 1, 2, 3],
/// ... )
///
/// Notes
/// -----
///
/// If you are going to compute the product multiple times for the same degree
/// and definition of variables, then consider using a :class:`MulPlan`.
///
/// See also
/// --------
/// :func:`MulPlan` : Compute the multiplication with a plan.
#[pyfunction]
#[pyo3(text_signature = "(coeffs_left, coeffs_right, vars)")]
fn mul<'py>(
    py: Python<'py>,
    coeffs_left: PyReadonlyArrayDyn<f64>,
    coeffs_right: PyReadonlyArrayDyn<f64>,
    vars: Vec<PyMulVar>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let Some(ncoeffs_left) = coeffs_left.shape().last() else {
        return Err(PyValueError::new_err("expected `coeffs_left` with at least one axis"));
    };
    let Some(ncoeffs_right) = coeffs_right.shape().last() else {
        return Err(PyValueError::new_err("expected `coeffs_right` with at least one axis"));
    };
    let nvars_left = vars.iter().filter(|var| var.0 != MulVar::Right).count();
    let nvars_right = vars.iter().filter(|var| var.0 != MulVar::Left).count();
    let degree_left = degree(nvars_left, *ncoeffs_left)?;
    let degree_right = degree(nvars_right, *ncoeffs_right)?;
    PyMulPlan::new(vars, degree_left, degree_right).__call__(py, coeffs_left, coeffs_right)
}

/// Returns the coefficients for the product of two polynomials in the same variables.
///
/// Args
/// ----
/// coeffs_left : :class:`int`
///     The coefficents of the left polynomial. The last axis constitutes the
///     axis of coefficients.
/// coeffs_right : :class:`int`
///     The coefficents of the right polynomial. The last axis constitutes the
///     axis of coefficients.
/// nvars : class:`int`
///     The number of variables of the left and right polynomials.
///
/// Returns
/// -------
/// :class:`numpy.ndarray`
///     The coefficients for the product. The last axis constitutes
///     the axis of coefficients. The leading axes are the same as the leading axes of
///     ``coeffs_left`` and ``coeffs_right``.
///
/// Example
/// -------
///
/// Consider the following polynomials:
///
/// .. math:: f(x) = x^2 + 2 x + 3
/// .. math:: g(x) = 3 x + 1
///
/// The coefficients for the product :math:`p(x) = f(x) g(x)`:
///
/// >>> import numpy
/// >>> numpy.testing.assert_allclose(
/// ...     mul_same_vars(
/// ...         numpy.array([1, 2, 3], dtype=float),
/// ...         numpy.array([3, 1], dtype=float),
/// ...         1,
/// ...     ),
/// ...     [3, 7, 11, 3],
/// ... )
///
/// Notes
/// -----
///
/// If you are going to compute the product multiple times for the same degree
/// and definition of variables, then consider using a :class:`MulPlan`.
///
/// See also
/// --------
/// :func:`MulPlan` : Compute the multiplication with a plan.
#[pyfunction]
#[pyo3(text_signature = "(coeffs_left, coeffs_right, nvars)")]
fn mul_same_vars<'py>(
    py: Python<'py>,
    coeffs_left: PyReadonlyArrayDyn<f64>,
    coeffs_right: PyReadonlyArrayDyn<f64>,
    nvars: usize,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let Some(ncoeffs_left) = coeffs_left.shape().last() else {
        return Err(PyValueError::new_err("expected `coeffs_left` with at least one axis"));
    };
    let Some(ncoeffs_right) = coeffs_right.shape().last() else {
        return Err(PyValueError::new_err("expected `coeffs_right` with at least one axis"));
    };
    let degree_left = degree(nvars, *ncoeffs_left)?;
    let degree_right = degree(nvars, *ncoeffs_right)?;
    PyMulPlan::same_vars(nvars, degree_left, degree_right).__call__(py, coeffs_left, coeffs_right)
}

/// Returns the coefficients for the product of two polynomials.
///
/// Args
/// ----
/// coeffs_left : :class:`int`
///     The coefficents of the left polynomial. The last axis constitutes the
///     axis of coefficients.
/// coeffs_right : :class:`int`
///     The coefficents of the right polynomial. The last axis constitutes the
///     axis of coefficients.
/// nvars_left : :class:`int`
///     The number of variables of the left polynomial.
/// nvars_right : :class:`int`
///     The number of variables of the right polynomial.
///
/// Returns
/// -------
/// :class:`numpy.ndarray`
///     The coefficients for the product. The last axis constitutes
///     the axis of coefficients. The leading axes are the same as the leading axes of
///     ``coeffs_left`` and ``coeffs_right``.
///
/// Example
/// -------
///
/// Consider the following polynomials:
///
/// .. math:: f(x) = x^2 + 2 x + 3
/// .. math:: g(x) = 3 x + 1
///
/// The coefficients for the product :math:`q(x, y) = f(x) g(y)`:
///
/// >>> import numpy
/// >>> numpy.testing.assert_allclose(
/// ...     mul_different_vars(
/// ...         numpy.array([1, 2, 3], dtype=float),
/// ...         numpy.array([3, 1], dtype=float),
/// ...         1,
/// ...         1,
/// ...     ),
/// ...     [0, 0, 0, 3, 6, 9, 0, 1, 2, 3],
/// ... )
///
/// Notes
/// -----
///
/// If you are going to compute the product multiple times for the same degree
/// and definition of variables, then consider using a :class:`MulPlan`.
///
/// See also
/// --------
/// :func:`MulPlan` : Compute the multiplication with a plan.
#[pyfunction]
#[pyo3(text_signature = "(coeffs_left, coeffs_right, vars)")]
fn mul_different_vars<'py>(
    py: Python<'py>,
    coeffs_left: PyReadonlyArrayDyn<f64>,
    coeffs_right: PyReadonlyArrayDyn<f64>,
    nvars_left: usize,
    nvars_right: usize,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let Some(ncoeffs_left) = coeffs_left.shape().last() else {
        return Err(PyValueError::new_err("expected `coeffs_left` with at least one axis"));
    };
    let Some(ncoeffs_right) = coeffs_right.shape().last() else {
        return Err(PyValueError::new_err("expected `coeffs_right` with at least one axis"));
    };
    let degree_left = degree(nvars_left, *ncoeffs_left)?;
    let degree_right = degree(nvars_right, *ncoeffs_right)?;
    PyMulPlan::different_vars(nvars_left, nvars_right, degree_left, degree_right)?.__call__(
        py,
        coeffs_left,
        coeffs_right,
    )
}

/// Returns a coefficient transformation matrix.
///
/// The matrix is such that the following two expressions are equivalent:
///
///     eval(outer_coeffs, eval(inner_coeffs, inner_values))
///     eval(matrix @ outer_coeffs, inner_values)
///
/// where `matrix` is the result of
///
///     composition_with_inner_matrix(inner_coeffs, inner_nvars, outer_nvars, outer_degree)
#[pyfunction]
#[pyo3(text_signature = "(inner_coeffs, inner_nvars, outer_nvars, outer_degree)")]
fn composition_with_inner_matrix<'py>(
    py: Python<'py>,
    inner_coeffs: PyReadonlyArray2<f64>,
    inner_nvars: usize,
    outer_nvars: usize,
    outer_degree: Power,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inner_coeffs = inner_coeffs.as_array();
    if inner_coeffs.shape()[0] != outer_nvars {
        return Err(PyValueError::new_err(format!(
            "expected `inner_coeffs` with shape ({}, ?) but got ({}, {})",
            outer_nvars,
            inner_coeffs.shape()[0],
            inner_coeffs.shape()[1],
        )));
    }
    let inner_degree = degree(inner_nvars, inner_coeffs.shape()[1])?;
    let result = nutils_poly::composition_with_inner_matrix(
        inner_coeffs
            .as_standard_layout()
            .as_slice()
            .unwrap()
            .iter()
            .copied(),
        inner_nvars,
        inner_degree,
        outer_nvars,
        outer_degree,
    )
    .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(result.to_pyarray_bound(py))
}

/// Return coefficients for the given degree.
///
/// Given coefficients for a polynomial in ``nvars`` variables of implied
/// degree ``old_degree``, this function returns coefficients for degree
/// ``new_degree``.
///
/// Args
/// ----
/// coeffs : :class:`numpy.ndarray`
///     The coefficients.
/// nvars : :class:`int`
///     The number of variables.
/// new_degree : :class:`int`
///     The degree of the returned coefficients.
///
/// Returns
/// -------
/// :class:`numpy.ndarray`
///     Coefficients for a polynomial of degree ``new_degree``.
///
/// Raises
/// ------
/// :class:`ValueError`
///     If the number of variables is negative.
/// :class:`ValueError`
///     If the ``old_degree`` is larger than the ``new_degree``.
///
/// Example
/// -------
///
/// >>> import numpy
/// >>> numpy.testing.assert_allclose(
/// ...     change_degree(numpy.array([2, 1], dtype=float), 1, 2),
/// ...     [0, 2, 1],
/// ... )
#[pyfunction]
#[pyo3(text_signature = "(coeffs, nvars, new_degree)")]
fn change_degree<'py>(
    py: Python<'py>,
    coeffs: PyReadonlyArrayDyn<f64>,
    nvars: usize,
    new_degree: Power,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let (coeffs, degree, _, leading_shape) = as_coeffs_dyn(&coeffs, nvars, "coeffs")?;
    let Some(new_indices) = nutils_poly::MapDegree::new(nvars, degree, new_degree) else {
        return Err(PyValueError::new_err(
            "the new degree is lower than the old degree",
        ));
    };
    // `MapDegree` is expensive. Since we are using `new_indices` multiple
    // times, we convert the sequence into a `Vec`.
    let new_indices: Vec<_> = new_indices.iter().collect();
    let new_ncoeffs = nutils_poly::ncoeffs(nvars, new_degree);
    let new_shape = shape![leading_shape, [new_ncoeffs]];
    let mut new_coeffs: ArrayD<f64> = ArrayD::zeros(&new_shape[..]);
    for (mut new_coeffs, coeffs) in iter::zip(new_coeffs.rows_mut(), coeffs.rows()) {
        for (coeff, new_index) in iter::zip(coeffs, &new_indices) {
            new_coeffs[*new_index] = *coeff;
        }
    }
    Ok(PyArray::from_owned_array_bound(py, new_coeffs))
}

/// Low-level functions for evaluating and manipulating polynomials.
///
/// The polynomials considered in this module are `power series`_ in zero or
/// more variables centered at zero and truncated to order :math:`p`,
///
/// .. math:: Σ_{k ∈ ℤ^n | Σ_i k_i ≤ p} c_k ∏_i x_i^{k_i}
///
/// where :math:`c` is a vector of coefficients, :math:`x` a vector of
/// :math:`n` variables and :math:`p` a nonnegative integer degree.
///
/// This module requires the coefficients to be stored in a linear array in
/// reverse `lexicographic order`_: the coefficient for powers :math:`j ∈ ℤ^n`
/// comes before the coefficient for powers :math:`k ∈ ℤ^n / \{j\}` iff
/// :math:`j_i > k_i`, where :math:`i = \max_l(j_l ≠ k_l)`, the index of the
/// *last* non-matching power.
///
/// Examples
/// --------
///
/// The vector of coefficients for the polynomial :math:`f(x) = x_0^2 - x_0 + 2` is
/// ``[1, -1, 2]``.
///
/// With :func:`eval()` we can evaluate this polynomial. Evaluation for
/// :math:`x_0 = 0`, :math:`x_0 = 1` and :math:`x_0 = 2` simulatenously:
///
/// >>> import numpy
/// >>> numpy.testing.assert_allclose(
/// ...     eval(numpy.array([1, -1, 2], dtype=float), numpy.array([[0], [1], [2]], dtype=float)),
/// ...     [2, 2, 4],
/// ... )
///
/// :func:`partial_deriv()` computes the coefficients for the partial
/// derivative of a polynomial to one of the variables:
///
/// >>> numpy.testing.assert_allclose(
/// ...     partial_deriv(numpy.array([1, -1, 2], dtype=float), 1, 0),
/// ...     [2, -1],
/// ... )
///
/// Similarly, :func:`grad` computes the coefficients for all partial
/// derivatives at once, which is equivalent to stacking all results of
/// :func:`partial_deriv()` along the next to last axis. Example for polynomial
/// :math:`g(x) = x_1^2 + 2 x_0 x_1 + 3 x_0` (vector of coefficients: ``[1, 2,
/// 0, 0, 3, 0]``):
///
/// >>> numpy.testing.assert_allclose(
/// ...     grad(numpy.array([1, 2, 0, 0, 3, 0], dtype=float), 2),
/// ...     [[2, 0, 3], [2, 2, 0]],
/// ... )
///
/// .. _power series: https://en.wikipedia.org/wiki/Power_series
/// .. _lexicographic order: https://en.wikipedia.org/wiki/Lexicographic_order
#[pymodule]
#[pyo3(name = "nutils_poly")]
fn pymod(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyMulVar>()?;
    m.add_class::<PyMulPlan>()?;
    m.add_class::<PyPartialDerivPlan>()?;
    m.add_class::<PyGradPlan>()?;

    macro_rules! wrap_and_add_pyfunctions {
        ($m:ident $(, $f:ident)* $(,)?) => {
            $(
                let f = wrap_pyfunction!($f, $m)?;
                // Maturin wraps this module in a parent module (with the same
                // name) and re-exports all of the former into the latter. The
                // name of `m` (assigned to attribute `__module__` of `f`) is
                // `nutils_poly.nutils_poly`. While not wrong, we'd prefer to
                // hide the nested structure and rewrite the `__module__`
                // attribute to `nutils_poly`.
                f.setattr("__module__", "nutils_poly")?;
                $m.add_function(f)?;
            )*
        };
    }
    wrap_and_add_pyfunctions!(
        m,
        degree,
        ncoeffs,
        eval,
        eval_outer,
        grad,
        partial_deriv,
        mul,
        mul_same_vars,
        mul_different_vars,
        composition_with_inner_matrix,
        change_degree
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, ShapeBuilder};

    #[test]
    fn has_contiguous_rows() {
        // Tests whether `has_contiguous_rows(array)` and `array.as_slice()`
        // are compatible for some length `l` and stride `s`.
        fn test(l: usize, s: usize) -> bool {
            let array = Array1::from_shape_vec((l,).strides((s,)), vec![0, 1, 2, 3]).unwrap();
            super::has_contiguous_rows(&array) == array.as_slice().is_some()
        }
        assert!(test(1, 0));
        assert!(test(1, 1));
        assert!(test(1, 2));
        assert!(test(2, 1));
        assert!(test(2, 2));
    }
}
