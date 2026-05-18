use ndarray::{ArrayView1, s};
use super::types::MetricMethodType;
use crate::domain::structure::GameResult;

pub(crate) fn vec_results_to_metrics(
    list_results: Vec<GameResult>,
    method: MetricMethodType,
) -> Vec<f64> {
    let mut vec_metrics: Vec<f64> = Vec::with_capacity(list_results.len());
    for result in list_results {
        let metric = compute_single_metric(&result, method);
        vec_metrics.push(metric);
    }
    vec_metrics
}

pub(crate) fn compute_single_metric(game_result: &GameResult, method: MetricMethodType) -> f64 {
    let gaps = game_result.gaps();
    match method {
        MetricMethodType::MaxLast(cutoff_percent) => {
            let tail = get_tail(gaps, cutoff_percent);
            tail.into_iter()
                .fold(f64::NEG_INFINITY, |acc, &elem| acc.max(elem))
        }
        MetricMethodType::VarLast(cutoff_percent) => {
            let tail = get_tail(gaps, cutoff_percent);
            let tail_len = tail.len() as f64;
            if tail_len == 0.0 { return 0.0; }
            let mean = tail.iter().sum::<f64>() / tail_len;
            tail.iter().fold(0.0, |acc, &val| {
                let diff = val - mean;
                acc + diff * diff / tail_len
            })
        }
        MetricMethodType::TotalVar => gaps
            .iter()
            .zip(gaps.iter().skip(1))
            .fold(0.0, |acc, (&a, &b)| acc + (b - a).abs()),
    }
}

fn get_tail<'a>(array: ArrayView1<'a, f64>, cutoff_percent: f64) -> ArrayView1<'a, f64> {
    let cutoff_idx = ((1. - cutoff_percent) * array.len() as f64) as usize;
    array.slice_move(s![cutoff_idx..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use approx::assert_relative_eq;

    fn make_mock_result(gaps: Vec<f64>) -> GameResult {
        let n = gaps.len();
        GameResult {
            gaps_history: Array1::from_vec(gaps),
            x_history: ndarray::Array2::zeros((n, 2)),
            y_history: ndarray::Array2::zeros((n, 2)),
        }
    }

    #[test]
    fn test_total_variation() {
        let res = make_mock_result(vec![1.0, 3.0, 2.0]);
        assert_relative_eq!(compute_single_metric(&res, MetricMethodType::TotalVar), 3.0);
    }

    #[test]
    fn test_max_last() {
        let res = make_mock_result(vec![10.0, 1.0, 5.0, 2.0]);
        assert_relative_eq!(compute_single_metric(&res, MetricMethodType::MaxLast(0.5)), 5.0);
    }
}
