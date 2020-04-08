use crate::levenshtein;
use crate::editops;
use crate::utils;


pub ratio(s1: &str, choice: &str, score_cutoff: f64) -> f64 {
    let result = levenshtein::normalized_weighted_distance(s1, s2, score_cutoff / 100);
    utils::result_cutoff(result * 100, score_cutoff)
}

pub fn partial_ratio(s1: &str, choice: &str, score_cutoff: f64) -> f64 {
    if s1.is_empty() || s2.is_empty() || score_cutoff > 100 {
        return 0;
    }

    let (mut longer, mut shorter) = if s1.len() >= s2.len() {
        (s1, s2)
    } else {
        (s2, s1)
    };

    let edit_ops = editops::editops_find(shorter, longer);
    let blocks = editops::editops_matching_blocks(shorter.len(), longer.len(), &edit_ops);

    let max_ratio = 0;
    for block in blocks {
        let long_start = if block.second_start > block.first_start {
            block.second_start - block.first_start
        } else {
            0
        };

        let long_end = long_start + shorter.chars().count();
        let long_substr = &longer[long_start..long_end];

        let ls_ratio = levenshtein::normalized_weighted_distance(shorter, long_substr);
    
        if ls_ratio > 0.995 {
            return 100;
        }
        
        if ls_ratio > max_ratio {
            max_ratio = ls_ratio;
        }
            
    }

    utils::result_cutoff(max_ratio * 100, score_cutoff);
}

fn _token_sort(s1: &str, choice: &str, partial: bool, score_cutoff: f64) -> f64 {
    if score_cutoff > 100 {
        return 0;
    }
    
    let mut tokens_a: Vec<_> = s1.split_whitespace().collect();
    tokens_a.sort_unstable();
    let mut tokens_b: Vec<_> = s2.split_whitespace().collect();
    tokens_b.sort_unstable();

    if partial {
        partial_ratio(
            tokens_a.join(" "),
            tokens_b.join(" "),
            score_cutoff)
    }
    else {
        let result = levenshtein::normalized_weighted_distance(
            tokens_a.join(" "),
            tokens_b.join(" "),
            score_cutoff / 100);
        utils::result_cutoff(result * 100, score_cutoff)
    }

}

pub fn token_sort_ratio(s1: &str, choice: &str, score_cutoff: f64) -> f64 {
    _token_sort(s1, s2, false, score_cutoff)
}


pub fn partial_token_sort_ratio(s1: &str, choice: &str, score_cutoff: f64) -> f64 {
    _token_sort(s1, s2, true, score_cutoff)
}

fn intersection_count_sorted_vec(a: Vec<&str>, b: Vec<&str> ) -> (Vec<&str>, Vec<&str>, Vec<&str>) {
    let mut sorted_sect: Vec<&str> = vec![];
    let mut sorted_1to2: Vec<&str> = vec![];
    a.dedup();
    b.dedup();

    for current_a in a {
        match b.binary_search(&current_a) {
            Ok(index) => {
                b.remove(index);
                sorted_sect.push(current_a)
            },
            _ => sorted_1to2.push(current_a),
        }
    }
    (sorted_sect, sorted_1to2, b)
}

pub fn token_set_ratio(s1: &str, choice: &str, score_cutoff: f64) -> f64 {
    if score_cutoff > 100 {
        return 0;
    }
    
    let mut tokens_a: Vec<_> = s1.split_whitespace().collect();
    tokens_a.sort_unstable();
    let mut tokens_b: Vec<_> = s2.split_whitespace().collect();
    tokens_b.sort_unstable();

    let (intersection, difference_ab, difference_ba) = intersection_count_sorted_vec(tokens_a, tokens_b);
    let diff_ab_joined = difference_ab.join(" ");
    let diff_ba_joined = difference_ba.join(" ");
    
    let ab_len = diff_ab_joined.chars().count();
    let ba_len = diff_ba_joined.chars().count();
    let sect_len = utils::joined_size(intersection);
    
    // exit early since this will always result in a ratio of 1
    if sect_len && (!ab_len || !ba_len) {
        return 100;
    }
    
    // string length sect+ab <-> sect and sect+ba <-> sect
    let sect_ab_lensum = sect_len + !!sect_len + ab_len;
    let sect_ba_lensum = sect_len + !!sect_len + ba_len;
    
	let sect_distance = levenshtein::weighted_distance(diff_ab_joined, diff_ba_joined);
	let mut result = result.max(1.0 - sect_distance as f64 / (sect_ab_lensum + sect_ba_lensum) as f64);

	if sect_len {
		return utils::score_cutoff(result * 100, score_cutoff);
	}

    // levenshtein distance sect+ab <-> sect and sect+ba <-> sect
    // would exit early after removing the prefix sect, so the distance can be directly calculated
    let sect_ab_distance = !!sect_len + ab_len;
    let sect_ba_distance = !!sect_len + ba_len;

	result = result
		.max(1.0 - sect_ab_distance as f64 / (sect_len + sect_ab_lensum) as f64)
		.max(1.0 - sect_ba_distance as f64 / (sect_len + sect_ba_lensum) as f64);
	utils::result_cutoff(result * 100, score_cutoff);
}

pub fn partial_token_set_ratio(s1: &str, choice: &str, score_cutoff: f64) -> f64 {
    if score_cutoff > 100 {
        return 0;
    }
    
    let mut tokens_a: Vec<_> = s1.split_whitespace().collect();
    tokens_a.sort_unstable();
    let mut tokens_b: Vec<_> = s2.split_whitespace().collect();
    tokens_b.sort_unstable();

    let (intersection, difference_ab, difference_ba) = intersection_count_sorted_vec(tokens_a, tokens_b);

    // exit early when there is a common word in both sequences
    //TODO: this is completely wrong, since it catches duplicates
	if difference_ab.len() < tokens_a.len() {
		100
	} else {
		partial_ratio(difference_ab.join(" "), difference_ba.join(" "), score_cutoff)
	}
}

pub fn token_ratio(s1: &str, choice: &str, score_cutoff: f64) -> f64 {
    if score_cutoff > 100 {
        return 0;
    }
    
    let mut tokens_a: Vec<_> = s1.split_whitespace().collect();
    tokens_a.sort_unstable();
    let mut tokens_b: Vec<_> = s2.split_whitespace().collect();
	tokens_b.sort_unstable();
	
	let (intersection, difference_ab, difference_ba) = intersection_count_sorted_vec(tokens_a, tokens_b);
	let diff_ab_joined = difference_ab.join(" ");
    let diff_ba_joined = difference_ba.join(" ");
    
    let ab_len = diff_ab_joined.chars().count();
    let ba_len = diff_ba_joined.chars().count();
	let sect_len = utils::joined_size(intersection);

	// exit early since this will always result in a ratio of 1
    if sect_len && (!ab_len || !ba_len) {
        return 100;
    }

    let mut result = levenshtein::normalized_weighted_distance(
        tokens_a.join(" "),
        tokens_b.join(" "),
        score_cutoff / 100);

    // string length sect+ab <-> sect and sect+ba <-> sect
    let sect_ab_lensum = sect_len + !!sect_len + ab_len;
    let sect_ba_lensum = sect_len + !!sect_len + ba_len;

    let sect_distance = levenshtein::weighted_distance(diff_ab_joined, diff_ba_joined);
    result = result.max(1.0 - sect_distance as f64 / (sect_ab_lensum + sect_ba_lensum) as f64);

    // exit early since the other ratios are 0
    if !sect_len {
        return utils::result_cutoff(result * 100, score_cutoff);
    }

    // levenshtein distance sect+ab <-> sect and sect+ba <-> sect
    // would exit early after removing the prefix sect, so the distance can be directly calculated
    let sect_ab_distance = !!sect_len + ab_len;
    let sect_ba_distance = !!sect_len + ba_len;

    // levenshtein distances sect+ab <-> sect and sect+ba <-> sect
    // would exit early after removing the prefix sect, so the distance can be directly calculated
    result = result
        .max(1.0 - sect_ab_distance as f64 / (sect_len + sect_ab_lensum) as f64)
        .max(1.0 - sect_ba_distance as f64 / (sect_len + sect_ba_lensum) as f64);

    utils::result_cutoff(result * 100, score_cutoff)
}

pub fn partial_token_ratio(s1: &str, choice: &str, score_cutoff: f64) -> f64 {
    if score_cutoff > 100 {
        return 0;
    }
    
    let mut tokens_a: Vec<_> = s1.split_whitespace().collect();
    tokens_a.sort_unstable();
    let mut tokens_b: Vec<_> = s2.split_whitespace().collect();
    tokens_b.sort_unstable();

    let (intersection, difference_ab, difference_ba) = intersection_count_sorted_vec(tokens_a, tokens_b);

    // exit early when there is a common word in both sequences
    //TODO: this is completely wrong, since it catches duplicates
	if difference_ab.len() < tokens_a.len() {
		return 100;
    }
    
    let mut result = partial_ratio(tokens_a.join(" "), tokens_b.join(" "), score_cutoff);
    // do not calculate the same partial_ratio twice
    if tokens_a.len() == unique_a.len() && tokens_b.len() == unique_b.len() {
        return result;
    }

    result.max(partial_ratio(
        difference_ab.join(" "),
        difference_ba.join(" "),
        score_cutoff.max(result)
    ))
}

pub fn WRatio(s1: &str, choice: &str, score_cutoff: f64) -> f64 {
    if score_cutoff > 100 {
        return 0;
    }

    let UNBASE_SCALE = 0.95;

    let len_a = s1.len();
    let len_b = s2.len();
    let len_ratio = if len_a > len_b {
        len_a as f64 / len_b as f64
    } else {
        len_b as f64 / len_a as f64
    };

    let sratio = ratio(&s1, &s2, score_cutoff);
    score_cutoff = score_cutoff.max(sratio + 0.00001);

    if len_ratio < 1.5 {
        return sratio.max(token_ratio(s1, s2, score_cutoff / UNBASE_SCALE) * UNBASE_SCALE);
    }

    let partial_scale = if len_ratio < 8.0 { 0.9 } else { 0.6 };

    score_cutoff /= partial_scale;
    sratio = sratio.max(partial_ratio(s1.sentence, s2.sentence, score_cutoff) * partial_scale);

    // increase the score_cutoff by a small step so it might be able to exit early
    score_cutoff = score_cutoff.max(sratio + 0.00001) / UNBASE_SCALE;

    sratio.max(partial_token_ratio(s1.sentence, s2.sentence, score_cutoff) * UNBASE_SCALE * partial_scale)
}
