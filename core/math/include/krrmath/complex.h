#pragma once

#include "common.h"
#include "constants.h"
#include <iostream>
#include <math.h>
#include <algorithm>

ELAINA_NAMESPACE_BEGIN

template <typename T>
struct Complex {
	ELAINA_CALLABLE Complex(T re) : re(re), im(0) {}
	ELAINA_CALLABLE Complex(T re, T im) : re(re), im(im) {}

	ELAINA_CALLABLE Complex operator-() const { return { -re, -im }; }

	ELAINA_CALLABLE Complex operator+(Complex z) const { return { re + z.re, im + z.im }; }

	ELAINA_CALLABLE Complex operator-(Complex z) const { return { re - z.re, im - z.im }; }

	ELAINA_CALLABLE Complex operator*(Complex z) const {
		return { re * z.re - im * z.im, re * z.im + im * z.re };
	}

	ELAINA_CALLABLE Complex operator/(Complex z) const {
		T scale = 1 / (z.re * z.re + z.im * z.im);
		return { scale * (re * z.re + im * z.im), scale * (im * z.re - re * z.im) };
	}

	friend ELAINA_CALLABLE Complex operator+(T value, Complex z) {
		return Complex(value) + z;
	}

	friend ELAINA_CALLABLE Complex operator-(T value, Complex z) {
		return Complex(value) - z;
	}

	friend ELAINA_CALLABLE Complex operator*(T value, Complex z) {
		return Complex(value) * z;
	}

	friend ELAINA_CALLABLE Complex operator/(T value, Complex z) {
		return Complex(value) / z;
	}

	ELAINA_CALLABLE T real() { return re; }

	ELAINA_CALLABLE T imag() { return im; }

	ELAINA_CALLABLE T norm() { return re * re + im * im; }
	
	T re, im;
};

template <typename T>
ELAINA_CALLABLE T real(const Complex<T>& z) {
	return z.re;
}

template <typename T>
ELAINA_CALLABLE T imag(const Complex<T>& z) {
	return z.im;
}

template <typename T>
ELAINA_CALLABLE T norm(const Complex<T>& z) {
	return z.re * z.re + z.im * z.im;
}

template <typename T>
ELAINA_CALLABLE T abs(const Complex<T>& z) {
	return sqrt(norm(z));
}

template <typename T>
ELAINA_CALLABLE Complex<T> sqrt(const Complex<T>& z) {
	T n = abs(z), t1 = sqrt(T(.5) * (n + abs(z.re))),
		t2 = T(.5) * z.im / t1;

	if (n == 0)
		return 0;

	if (z.re >= 0)
		return { t1, t2 };
	else
		return { abs(t2), copysign(t1, z.im) };
}


ELAINA_NAMESPACE_END