#pragma once

template<typename T>
concept bool Iterable = requires(T a, T b) {
	{ std::begin(a) != std::end(b) } -> bool;
	{ ++std::begin(a) };
	{ *std::begin(a) };
};

template<typename T>
concept bool IterableOfIterables =
    Iterable<T> &&
	requires(T a, T b) {
		{ std::begin(*std::begin(a)) != std::end(*std::begin(b)) } -> bool;
		{ ++std::begin(*std::begin(a)) };
		{ *std::begin(*std::begin(a)) };
	};