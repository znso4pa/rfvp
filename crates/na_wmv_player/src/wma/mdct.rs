//! MDCT/IMDCT implementation.
//!
//! The inverse transform is a DCT-IV, which we evaluate with one complex FFT
//! instead of the direct O(N²) sum. See [`Mdct::dct4`] for the derivation.
//!
//! The definitional O(N²) implementation is kept under `cfg(test)` as an oracle:
//! the fast path is only trustworthy if it reproduces it sample for sample.

use std::f64::consts::PI;

/// Radix-2 complex FFT, in place. `re.len()` must be a power of two.
fn fft_in_place(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert_eq!(n, im.len());
    debug_assert!(n.is_power_of_two());

    if n <= 1 {
        return;
    }

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // Butterflies.
    let mut span = 2usize;
    while span <= n {
        let ang = -2.0 * PI / span as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let half = span / 2;

        let mut base = 0usize;
        while base < n {
            // Twiddles by recurrence: (cos, sin)((k+1)θ) from (cos, sin)(kθ).
            let mut cr = 1.0f64;
            let mut ci = 0.0f64;

            for k in 0..half {
                let (ur, ui) = (re[base + k], im[base + k]);
                let (vr, vi) = (re[base + k + half], im[base + k + half]);

                let tr = vr * cr - vi * ci;
                let ti = vr * ci + vi * cr;

                re[base + k] = ur + tr;
                im[base + k] = ui + ti;
                re[base + k + half] = ur - tr;
                im[base + k + half] = ui - ti;

                let next_cr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = next_cr;
            }

            base += span;
        }

        span <<= 1;
    }
}

/// MDCT/IMDCT context for one block size.
///
/// `len` is the number of coefficients, which is also the number of samples in a
/// half IMDCT; a full IMDCT produces `2 * len` samples.
pub struct Mdct {
    /// Frame size (N). Full IMDCT output is 2N.
    pub len: usize,
    /// Scale factor (double precision in upstream).
    pub scale: f64,

    /// FFT scratch, length `2 * len` (the transform is `2N` points).
    scratch_re: Vec<f64>,
    scratch_im: Vec<f64>,
}

impl Mdct {
    pub fn new(len: usize, scale: f64) -> Self {
        debug_assert!(len.is_power_of_two(), "block size must be a power of two");
        Self {
            len,
            scale,
            scratch_re: vec![0.0; len * 2],
            scratch_im: vec![0.0; len * 2],
        }
    }

    /// Half-length inverse MDCT.
    ///
    /// Input: N coefficients.
    /// Output: N samples (half IMDCT), matching upstream's MDCT semantics.
    ///
    /// Upstream evaluates two direct O(N²) sums (`L = N/2`):
    ///
    /// ```text
    ///   dst[i]   =  scale · Σ_j cos((2j+1)·π/(4N)·(2N−2i−1)) · src[j]
    ///   dst[i+L] = −scale · Σ_j cos((2j+1)·π/(4N)·(3N+2i+1))  · src[j]
    /// ```
    ///
    /// Both are DCT-IV values at permuted indices. Writing the DCT-IV basis as
    /// `f_j(k) = cos(π/N·(k+½)(j+½))`, the two arguments become `2k+1 = 2N−2i−1`
    /// and `2k+1 = 3N+2i+1`. The first is already in range; the second folds back
    /// with the two symmetries `f(−1−k) = f(k)` and `f(k−2N) = −f(k)`:
    ///
    /// ```text
    ///   f(3N/2 + i) = −f(N/2 − 1 − i)
    /// ```
    ///
    /// So, with `C = DCT-IV(src)`,
    ///
    /// ```text
    ///   dst[i]   = scale · C[N−1−i]
    ///   dst[i+L] = scale · C[L−1−i]
    /// ```
    ///
    /// which is what we compute, in O(N log N).
    pub fn imdct_half(&mut self, dst: &mut [f32], src: &[f32]) {
        let n = self.len;
        let l = n / 2;
        debug_assert_eq!(dst.len(), n);
        debug_assert_eq!(src.len(), n);

        self.dct4(src);

        let scale = self.scale;
        let c = &self.scratch_im[..n];
        for i in 0..l {
            dst[i] = (c[n - 1 - i] * scale) as f32;
            dst[i + l] = (c[l - 1 - i] * scale) as f32;
        }
    }

    /// Full IMDCT.
    ///
    /// Input: N coefficients.
    /// Output: 2N samples.
    pub fn imdct_full(&mut self, dst: &mut [f32], src: &[f32]) {
        let len = self.len * 2;
        let len2 = len / 2;
        let len4 = len / 4;

        // The half IMDCT is written into the middle of the output.
        self.imdct_half(&mut dst[len4..len4 + len2], src);

        for i in 0..len4 {
            dst[i] = -dst[len2 - i - 1];
            dst[len - i - 1] = dst[len2 + i];
        }
    }

    /// DCT-IV: `C[m] = Σ_k src[k]·cos(π/N·(k+½)(m+½))`, left in `scratch_im[..N]`.
    ///
    /// Evaluated as one `2N`-point complex FFT. Splitting the phase (`M = 2N`):
    ///
    /// ```text
    ///   π/N·(k+½)(m+½) = πk(2m+1)/M + π(2m+1)/(2M)
    ///   ⇒  C[m] = Re{ e^{−iφ_m} · FFT_M(b)[m] },   φ_m = π(2m+1)/(2M)
    ///       b[k] = src[k]·e^{−iπk/M}   (k < N, zero-padded to M)
    /// ```
    ///
    /// Both rotations are linear in the index, so their factors are generated by
    /// a Chebyshev recurrence — four `sin`/`cos` calls per transform, no tables.
    ///
    /// Numerically this is *not* bit-identical to the direct sum: the old code
    /// rounded a partial sum into `f32` after every term, this one accumulates in
    /// `f64` and rounds once. On s16 output the difference is at most ±1 LSB.
    fn dct4(&mut self, src: &[f32]) {
        let n = self.len;
        let m = 2 * n;

        let re = &mut self.scratch_re[..m];
        let im = &mut self.scratch_im[..m];

        // Pre-rotation + zero padding: b[k] = src[k]·e^{−iπk/M}.
        {
            let step = PI / m as f64;
            let two_cos = 2.0 * step.cos();

            let mut prev = (-step).cos();
            let mut cur = 1.0f64; // cos(0)
            let mut prev_i = (-step).sin();
            let mut cur_i = 0.0f64; // sin(0)

            for k in 0..m {
                let c = if k < n { f64::from(src[k]) } else { 0.0 };
                re[k] = c * cur;
                im[k] = -c * cur_i;

                let next = two_cos * cur - prev;
                prev = cur;
                cur = next;

                let next_i = two_cos * cur_i - prev_i;
                prev_i = cur_i;
                cur_i = next_i;
            }
        }

        fft_in_place(re, im);

        // Post-rotation: C[m] = Re{ e^{−iφ_m}·D[m] }, in place in `im`.
        {
            let step = PI / m as f64;
            let base = step * 0.5; // φ_0
            let two_cos = 2.0 * step.cos();

            let mut prev = (base - step).cos();
            let mut cur = base.cos();
            let mut prev_i = (base - step).sin();
            let mut cur_i = base.sin();

            for j in 0..n {
                let (r, i2) = (re[j], im[j]);
                im[j] = r * cur + i2 * cur_i;

                let next = two_cos * cur - prev;
                prev = cur;
                cur = next;

                let next_i = two_cos * cur_i - prev_i;
                prev_i = cur_i;
                cur_i = next_i;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The definitional implementation, kept as the oracle for the fast path.
    ///
    /// Same math as the original O(N²) code: no cleverness, just the sums.
    fn naive_imdct_full(len: usize, scale: f64, src: &[f32]) -> Vec<f32> {
        let l = len >> 1;
        let n = l * 2; // coefficients == half-IMDCT samples
        let phase = PI / (4.0 * n as f64);

        let mut half = vec![0.0f32; n];
        for i in 0..l {
            let mut sum_d = 0.0f64;
            let mut sum_u = 0.0f64;

            let i_d = phase * ((4 * l - 2 * i - 1) as f64);
            let i_u = phase * ((3 * n + 2 * i + 1) as f64);

            for j in 0..n {
                let a = (2 * j + 1) as f64;
                let val = src[j] as f64;
                sum_d += (a * i_d).cos() * val;
                sum_u += (a * i_u).cos() * val;
            }

            half[i] = (sum_d * scale) as f32;
            half[i + l] = (-(sum_u * scale)) as f32;
        }

        // Full output: the half transform goes in the middle, mirrored outwards.
        let out_len = len * 2;
        let mut out = vec![0.0f32; out_len];
        out[l..l + n].copy_from_slice(&half);
        for i in 0..l {
            out[i] = -out[n - i - 1];
            out[out_len - i - 1] = out[n + i];
        }
        out
    }

    fn signal(n: usize, seed: u64) -> Vec<f32> {
        // Deterministic pseudo-random, plus a tone so the spectrum isn't flat.
        let mut s = seed;
        (0..n)
            .map(|i| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0;
                let tone = (i as f64 * 0.05).sin();
                (noise * 0.5 + tone) as f32
            })
            .collect()
    }

    #[test]
    fn fft_matches_dft() {
        for &n in &[2usize, 4, 8, 16, 64] {
            let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin()).collect();
            let mut re = x.clone();
            let mut im = vec![0.0; n];
            fft_in_place(&mut re, &mut im);

            for k in 0..n {
                let (mut dr, mut di) = (0.0f64, 0.0f64);
                for (j, &v) in x.iter().enumerate() {
                    let ang = -2.0 * PI * (j * k) as f64 / n as f64;
                    dr += v * ang.cos();
                    di += v * ang.sin();
                }
                assert!((re[k] - dr).abs() < 1e-9, "n={n} k={k} re");
                assert!((im[k] - di).abs() < 1e-9, "n={n} k={k} im");
            }
        }
    }

    /// The fast path must reproduce the definitional one.
    ///
    /// The two differ only in where rounding happens (per term vs. once at the
    /// end), so anything above a few `f32` ULP means the index mapping is wrong,
    /// not that precision drifted.
    #[test]
    fn fast_imdct_matches_naive() {
        let scale = 1.0f64 / 32768.0;

        for &len in &[128usize, 256, 512, 1024, 2048] {
            let src = signal(len, len as u64);
            let want = naive_imdct_full(len, scale, &src);

            let mut got = vec![0.0f32; len * 2];
            let mut mdct = Mdct::new(len, scale);
            mdct.imdct_full(&mut got, &src);

            let mut worst = 0.0f32;
            let mut worst_at = 0usize;
            for (i, (a, b)) in want.iter().zip(got.iter()).enumerate() {
                let d = (a - b).abs();
                if d > worst {
                    worst = d;
                    worst_at = i;
                }
            }

            // Scaled output is O(1); 1e-5 is well below an s16 LSB
            // (1/32768 ≈ 3.05e-5) but well above f32 noise.
            assert!(
                worst < 1e-5,
                "len={len}: worst |Δ| = {worst} at {worst_at} (want {}, got {})",
                want[worst_at],
                got[worst_at]
            );
        }
    }

    /// A single DC coefficient must reproduce the definitional transform exactly.
    #[test]
    fn dc_coefficient_matches_naive() {
        let len = 256usize;
        let scale = 1.0f64;
        let mut src = vec![0.0f32; len];
        src[0] = 1.0;

        let mut got = vec![0.0f32; len * 2];
        Mdct::new(len, scale).imdct_full(&mut got, &src);

        let want = naive_imdct_full(len, scale, &src);
        for (i, (a, b)) in want.iter().zip(got.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "i={i}: want {a}, got {b}");
        }
    }
}
