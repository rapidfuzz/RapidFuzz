#pragma once

template<typename T>
concept bool Iterable = requires(T a, T b) {
	{ std::begin(a) != std::end(b) } -> bool;
};

template<typename T>
concept bool IterableOfIterables =
    Iterable<T> &&
	requires(T a, T b) {
		{ std::begin(*std::begin(a)) != std::end(*std::begin(b)) } -> bool;
	};