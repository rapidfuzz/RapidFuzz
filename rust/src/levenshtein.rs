use crate::utils;
use std::cmp::min;

pub fn weighted_distance(sentence1: &str, sentence2: &str) -> usize {
    // check which is the bigger string for later operations
	let (mut longer, mut shorter) = if sentence1.len() >= sentence2.len() {
		(sentence1, sentence2)
	} else {
		(sentence2, sentence1)
	};

	let string_affix = utils::StringAffix::find(&longer, &shorter);

	let longer_len = string_affix.first_string_len;
	let shorter_len = string_affix.second_string_len;

	if shorter_len == 0 {
		return longer_len;
	}

	let prefix_len = string_affix.prefix_len;
	longer = &longer[prefix_len..prefix_len + longer_len];
	shorter = &shorter[prefix_len..prefix_len + shorter_len];
  
  	// calculate edit distance
	let mut cache: Vec<usize> = (1..=longer_len).collect();
	let mut result = longer_len;
  
	for (i, char1) in shorter.chars().enumerate() {
		result = i + 1;
		let mut distance_b = i;
  
		for (j, char2) in longer.chars().enumerate() {
			if char1 == char2 {
				result = distance_b;
			} else {
				result += 1;
			}
			distance_b = cache[j];
			result = result.min(distance_b + 1);
			cache[j] = result;
		}
	}
	result
}

pub fn normalized_weighted_distance(sentence1: &str, sentence2: &str, min_ratio: f64) -> f64 {
	if sentence1.is_empty() || sentence2.is_empty() {
		return sentence1.is_empty() && sentence2.is_empty();
	}

	let sentence1_len = sentence1.chars().count();
    let sentence2_len = sentence2.chars().count();
	let lensum = sentence1_len + sentence2_len;

	// constant time calculation to find a string ratio based on the string length
    // so it can exit early without running any levenshtein calculations
	let min_distance = if sentence1_len > sentence2_len {
		sentence1_len - sentence2_len
	} else {
		sentence2_len - sentence1_len
	}

	let len_ratio = 1.0 - min_distance as f64 / lensum as f64;
	if len_ratio < min_ratio {
		return 0.0;
	}

	let lensum = a.chars().count() + b.chars().count();
	let dist = weighted_distance(a, b);

	let ratio = 1.0 - dist as f64 / lensum as f64
	if ratio >= min_ratio { ratio } else { 0.0 }

}
