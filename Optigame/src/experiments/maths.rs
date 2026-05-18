use crate::common::math_ext::Ordf64Ext;
use ndarray::ArrayView2;

/// Defines max norm as just the biggest absolute value among the coefficients
///$$
/// \|A\|_{m} = \max_{i,j} |a_{i,j}|
///$$
pub(super) fn max_norm<'a>(mat: ArrayView2<'a, f64>) -> f64 {
    mat.map(|&val| val.abs()).max_f64()
}

/// Define infinity norm as the biggest value of the row-wise sum of absolute value of the coefficients
/// $$ \|A\|_{\infty} = \max_{i} \sum_{j=1}^{n} |a_{i,j}| $$
pub(super) fn infinity_norm<'a>(mat: ArrayView2<'a, f64>) -> f64 {
    mat.rows()
        .into_iter()
        .map(|row| row.iter().map(|&val| val.abs()).sum())
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Define the Frobenius norm for a square matrix as
/// $$ \|A\|_{F} = \sum_{i=1}^{n} \sum_{j=1}^{n} |a_{i,j}|^2 $$
pub(super) fn frobenius_norm<'a>(mat: ArrayView2<'a, f64>) -> f64 {
    mat.fold(0., |acc, &val| acc + val * val).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_max_norm() {
        let mat = array![[1.0, -5.0], [3.0, 2.0]];
        assert_relative_eq!(max_norm(mat.view()), 5.0);
    }

    #[test]
    fn test_infinity_norm() {
        let mat = array![
            [1.0, -5.0], // sum of abs = 6.0
            [3.0, 2.0]   // sum of abs = 5.0
        ];
        // The infinity norm is the maximum absolute row sum.
        assert_relative_eq!(infinity_norm(mat.view()), 6.0);
    }

    #[test]
    fn test_frobenius_norm() {
        let mat = array![[1.0, 2.0], [2.0, 4.0]];
        // Sum of squares: 1 + 4 + 4 + 16 = 25. sqrt(25) = 5.
        assert_relative_eq!(frobenius_norm(mat.view()), 5.0);
    }
}
