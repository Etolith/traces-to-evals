use crate::evaluation::ScoreScale;

pub(super) fn raw_score_in_scale(raw_score: f32, score_scale: ScoreScale) -> bool {
    match score_scale {
        ScoreScale::Binary | ScoreScale::Unit => unit_interval(raw_score),
        ScoreScale::FourPoint => (1.0..=4.0).contains(&raw_score),
    }
}

pub(super) fn unit_interval(value: f32) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}
