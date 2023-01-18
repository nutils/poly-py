from doctest import DocTestSuite
import numpy
import nutils_poly
from unittest import TestCase



class Degree(TestCase):

    def test(self):
        self.assertEqual(nutils_poly.degree(0, 1), 0)
        self.assertEqual(nutils_poly.degree(1, 1), 0)
        self.assertEqual(nutils_poly.degree(1, 2), 1)
        self.assertEqual(nutils_poly.degree(1, 3), 2)
        self.assertEqual(nutils_poly.degree(1, 4), 3)
        self.assertEqual(nutils_poly.degree(2, 1), 0)
        self.assertEqual(nutils_poly.degree(2, 3), 1)
        self.assertEqual(nutils_poly.degree(2, 6), 2)
        self.assertEqual(nutils_poly.degree(2, 10), 3)
        with self.assertRaisesRegex(ValueError, 'invalid number of coefficients'):
            nutils_poly.degree(0, 0)
        with self.assertRaisesRegex(ValueError, 'invalid number of coefficients'):
            nutils_poly.degree(1, 0)
        with self.assertRaisesRegex(ValueError, 'invalid number of coefficients'):
            nutils_poly.degree(2, 2)


class NCoefs(TestCase):

    def test(self):
        self.assertEqual(nutils_poly.ncoeffs(0, 0), 1)
        self.assertEqual(nutils_poly.ncoeffs(1, 0), 1)
        self.assertEqual(nutils_poly.ncoeffs(1, 1), 2)
        self.assertEqual(nutils_poly.ncoeffs(1, 2), 3)
        self.assertEqual(nutils_poly.ncoeffs(1, 3), 4)
        self.assertEqual(nutils_poly.ncoeffs(2, 0), 1)
        self.assertEqual(nutils_poly.ncoeffs(2, 1), 3)
        self.assertEqual(nutils_poly.ncoeffs(2, 2), 6)
        self.assertEqual(nutils_poly.ncoeffs(2, 3), 10)


class Eval(TestCase):

    def assert_eval(self, coeffs, values, desired):
        for order_coeffs in None, 'F':
            for order_values in None, 'F':
                with self.subTest(order_coeffs=order_coeffs, order_values=order_values):
                    coeffs_ = numpy.array(coeffs, dtype=numpy.float64, order=order_coeffs)
                    values_ = numpy.array(values, dtype=numpy.float64, order=order_values)
                    actual = nutils_poly.eval(coeffs_, values_)
                    numpy.testing.assert_allclose(actual, desired)

    def test_no_leading_shape(self):
        self.assert_eval([1], [], 1)
        self.assert_eval([2, 1], [2], 5)
        self.assert_eval([3, 2, 1], [2], 17)
        self.assert_eval([6, 5, 4, 3, 2, 1], [2, 3], 113)
        self.assert_eval([4, 3, 2, 1], [2, 3, 4], 30)

    def test_equal_leading_shape(self):
        self.assert_eval([[4, 3], [2, 1]], [[3], [2]], [15, 5])
        self.assert_eval([[6, 5, 4], [3, 2, 1]], [[2, 3], [4, 5]], [32, 24])

    def test_unequal_leading_shape(self):
        self.assert_eval(
            numpy.array([[4, 3], [2, 1]])[:,None,:],
            numpy.array([3, 2])[None,:,None],
            [[15, 11], [7, 5]])
        self.assert_eval(
            numpy.array([[4, 3], [2, 1]])[:,None,:],
            numpy.array([3, 2])[:,None],
            [[15, 11], [7, 5]])


class EvalOuter(TestCase):

    def assert_eval_outer(self, coeffs, values, desired):
        for order_coeffs in None, 'F':
            for order_values in None, 'F':
                with self.subTest(order_coeffs=order_coeffs, order_values=order_values):
                    coeffs_ = numpy.array(coeffs, dtype=numpy.float64, order=order_coeffs)
                    values_ = numpy.array(values, dtype=numpy.float64, order=order_values)
                    actual = nutils_poly.eval_outer(coeffs_, values_)
                    numpy.testing.assert_allclose(actual, desired)

    def test_outer_no_leading_shape(self):
        self.assert_eval_outer([1], [], 1)
        self.assert_eval_outer([2, 1], [2], 5)
        self.assert_eval_outer([3, 2, 1], [2], 17)
        self.assert_eval_outer([6, 5, 4, 3, 2, 1], [2, 3], 113)
        self.assert_eval_outer([4, 3, 2, 1], [2, 3, 4], 30)

    def test_outer_leading_shape(self):
        self.assert_eval_outer([[4, 3], [2, 1]], [[4], [3], [2]], [[19, 9], [15, 7], [11, 5]])


class PartialDeriv(TestCase):

    def assert_partial_derivs(self, coeffs, nvars, desired):
        desired = numpy.array(desired, dtype=float)
        assert nvars == desired.shape[-2]
        for order_coeffs in None, 'F':
            coeffs_ = numpy.array(coeffs, dtype=numpy.float64, order=order_coeffs)
            degree = nutils_poly.degree(nvars, coeffs_.shape[-1])
            with self.subTest(order_coeffs=order_coeffs, planned=True, var='all'):
                plan = nutils_poly.GradPlan(nvars, degree)
                actual = plan(coeffs_)
                numpy.testing.assert_allclose(actual, desired)
            with self.subTest(order_coeffs=order_coeffs, planned=False, var='all'):
                actual = nutils_poly.grad(coeffs_, nvars)
                numpy.testing.assert_allclose(actual, desired)
            for var in range(nvars):
                with self.subTest(order_coeffs=order_coeffs, planned=True, var=var):
                    plan = nutils_poly.PartialDerivPlan(nvars, degree, var)
                    actual = plan(coeffs_)
                    numpy.testing.assert_allclose(actual, desired[...,var,:])
                with self.subTest(order_coeffs=order_coeffs, planned=False, var=var):
                    actual = nutils_poly.partial_deriv(coeffs_, nvars, var)
                    numpy.testing.assert_allclose(actual, desired[...,var,:])

    def test(self):
        self.assert_partial_derivs([1], 0, numpy.zeros((0, 1)))
        self.assert_partial_derivs([1], 1, [[0]])
        self.assert_partial_derivs([2, 1], 1, [[2]])
        self.assert_partial_derivs([3, 2, 1], 1, [[6, 2]])
        self.assert_partial_derivs([3, 2, 1], 2, [[2], [3]])
        self.assert_partial_derivs([6, 5, 4, 3, 2, 1], 2, [[5, 6, 2], [12, 5, 4]])

        with self.assertRaisesRegex(ValueError, 'invalid number of coefficients'):
            nutils_poly.partial_deriv(numpy.array([], dtype=float), 1, 0)
        with self.assertRaisesRegex(ValueError, 'invalid number of coefficients'):
            nutils_poly.partial_deriv(numpy.array([2, 1], dtype=float), 2, 0)
        with self.assertRaisesRegex(ValueError, 'invalid number of coefficients'):
            nutils_poly.partial_deriv(numpy.array([2, 1], dtype=float), 2, 0)


class Mul(TestCase):

    def assert_mul(self, coeffs_left, coeffs_right, vars, desired):
        desired = numpy.array(desired, dtype=float)
        for order_coeffs_left in None, 'F':
            coeffs_left_ = numpy.array(coeffs_left, dtype=numpy.float64, order=order_coeffs_left)
            degree_left = nutils_poly.degree(sum(var != nutils_poly.MulVar.Right for var in vars), coeffs_left_.shape[-1])
            for order_coeffs_right in None, 'F':
                coeffs_right_ = numpy.array(coeffs_right, dtype=numpy.float64, order=order_coeffs_right)
                degree_right = nutils_poly.degree(sum(var != nutils_poly.MulVar.Left for var in vars), coeffs_right_.shape[-1])
                with self.subTest(order_coeffs_left=order_coeffs_left, order_coeffs_right=order_coeffs_right, planned=True):
                    plan = nutils_poly.MulPlan(vars, degree_left, degree_right)
                    actual = plan(coeffs_left_, coeffs_right_)
                    numpy.testing.assert_allclose(actual, desired)
                with self.subTest(order_coeffs_left=order_coeffs_left, order_coeffs_right=order_coeffs_right, planned=False):
                    actual = nutils_poly.mul(coeffs_left_, coeffs_right_, vars)
                    numpy.testing.assert_allclose(actual, desired)

    def test(self):
        self.assert_mul([1], [2], (nutils_poly.MulVar.Both,), [2])
        self.assert_mul([2, 1], [3], (nutils_poly.MulVar.Both,), [6, 3])
        self.assert_mul([2, 1], [4, 3], (nutils_poly.MulVar.Both,), [8, 10, 3])
        self.assert_mul([3, 2, 1], [6, 5, 4], (nutils_poly.MulVar.Both, nutils_poly.MulVar.Both), [18, 27, 18, 10, 13, 4])
        self.assert_mul([1], [2], (nutils_poly.MulVar.Left, nutils_poly.MulVar.Right,), [2])
        self.assert_mul([2, 1], [3], (nutils_poly.MulVar.Left, nutils_poly.MulVar.Right,), [0, 6, 3])
        self.assert_mul([2, 1], [4, 3], (nutils_poly.MulVar.Left, nutils_poly.MulVar.Right,), [0, 8, 4, 0, 6, 3])
        self.assert_mul([3, 2, 1], [5, 4], (nutils_poly.MulVar.Left, nutils_poly.MulVar.Left, nutils_poly.MulVar.Right), [0, 15, 10, 5, 0, 0, 12, 0, 8, 4])

    def test_same_var(self):
        plan = nutils_poly.MulPlan.same_vars(1, 2, 1)
        coeffs_left = numpy.array([1, 2, 3], dtype=float)
        coeffs_right = numpy.array([3, 1], dtype=float)
        desired = [3, 7, 11, 3]
        numpy.testing.assert_allclose(plan(coeffs_left, coeffs_right), desired)
        numpy.testing.assert_allclose(nutils_poly.mul_same_vars(coeffs_left, coeffs_right, 1), desired)

    def test_different_vars(self):
        plan = nutils_poly.MulPlan.different_vars(1, 1, 2, 1)
        coeffs_left = numpy.array([1, 2, 3], dtype=float)
        coeffs_right = numpy.array([3, 1], dtype=float)
        desired = [0, 0, 0, 3, 6, 9, 0, 1, 2, 3]
        numpy.testing.assert_allclose(plan(coeffs_left, coeffs_right), desired)
        numpy.testing.assert_allclose(nutils_poly.mul_different_vars(coeffs_left, coeffs_right, 1, 1), desired)


class CompositionWithInnerMatrix(TestCase):

    def assert_composition(self, inner_coeffs, inner_nvars, outer_nvars, outer_degree):
        inner_coeffs = numpy.array(inner_coeffs, dtype=float)
        matrix = nutils_poly.composition_with_inner_matrix(inner_coeffs, inner_nvars, outer_nvars, outer_degree)
        outer_ncoeffs = nutils_poly.ncoeffs(outer_nvars, outer_degree)
        inner_ncoeffs = inner_coeffs.shape[-1]
        for outer_coeffs in (numpy.eye(outer_ncoeffs, dtype=float) if outer_ncoeffs else (numpy.zeros((0,), dtype=float),)):
            for inner_values in (numpy.eye(inner_nvars, dtype=float) if inner_nvars else (numpy.zeros((0,), dtype=float),)):
                numpy.testing.assert_allclose(
                    nutils_poly.eval(matrix @ outer_coeffs, inner_values),
                    nutils_poly.eval(outer_coeffs, nutils_poly.eval(inner_coeffs, inner_values)),
                )

    def test_inner_0d_outer_0d(self):
        self.assert_composition(numpy.zeros((0, 1)), 0, 0, 0)

    def test_inner_1d_outer_0d(self):
        self.assert_composition([[2]], 0, 1, 2)

    def test_inner_1d_outer_1d(self):
        self.assert_composition([[3, 2, 1]], 1, 1, 2)

    def test_inner_2d_outer_1d(self):
        self.assert_composition([[3, 2, 1]], 2, 1, 2)

    def test_inner_2d_outer_2d(self):
        self.assert_composition([[3, 2, 1]], 2, 1, 3)


class ChangeDegree(TestCase):

    def assert_change_degree(self, coeffs, nvars, new_degree, desired):
        coeffs_ = numpy.array(coeffs, dtype=numpy.float64)
        actual = nutils_poly.change_degree(coeffs_, nvars, new_degree)
        numpy.testing.assert_allclose(actual, desired)

    def test(self):
        self.assert_change_degree([2, 1], 1, 1, [2, 1])
        self.assert_change_degree([2, 1], 1, 2, [0, 2, 1])
        self.assert_change_degree([2, 1], 1, 3, [0, 0, 2, 1])
        self.assert_change_degree([3, 2, 1], 2, 2, [0, 0, 3, 0, 2, 1])
        with self.assertRaisesRegex(ValueError, 'invalid number of coefficients'):
            nutils_poly.change_degree(numpy.array([2, 1], dtype=float), 2, 3)
        with self.assertRaisesRegex(ValueError, 'the new degree is lower than the old degree'):
            nutils_poly.change_degree(numpy.array([2, 1], dtype=float), 1, 0)

def load_tests(loader, tests, ignore):
    tests.addTests(DocTestSuite(nutils_poly))
    return tests
