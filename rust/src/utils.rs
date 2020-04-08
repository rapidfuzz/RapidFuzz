
struct Affix {
    usize prefix_len,
    usize suffix_len,
}

impl StringAffix {
	//TODO split this into two functions trim_prefix and trim_suffix
	pub fn find(first_string: &str, second_string: &str) -> StringAffix {
		// remove common prefix and suffix (linear vs square runtime for levensthein)
		let prefix_len = first_string.char_indices()
    		.zip(second_string.char_indices())
			.take_while(|&(a_char, b_char)| a_char == b_char)
			.count();

		let string = &first_string[prefix_len..];
		let string2 = &second_string[prefix_len..];

		let first_string_len = string.chars().count();
		let second_string_len = string2.chars().count();

		let suffix_len = string.chars().rev()
    		.zip(string2.chars().rev())
			.take_while(|&(a_char, b_char)| a_char == b_char)
			.count();

		StringAffix {
			prefix_len,
			first_string_len: first_string_len-suffix_len,
			second_string_len: second_string_len-suffix_len,
		}
	}
}