//! Bitstream reader.
//!
//! This is a simplified but semantically equivalent translation of upstream's
//! `GetBitContext` (MSB-first bit order).

use crate::error::{DecoderError, Result};

#[derive(Clone)]
pub struct GetBitContext<'a> {
    buf: &'a [u8],
    size_in_bits: usize,
    bit_pos: usize,
}

impl<'a> GetBitContext<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            size_in_bits: buf.len() * 8,
            bit_pos: 0,
        }
    }

    #[inline]
    pub fn bits_left(&self) -> isize {
        self.size_in_bits as isize - self.bit_pos as isize
    }

    #[inline]
    pub fn bits_read(&self) -> usize {
        self.bit_pos
    }

    #[inline]
    pub fn align_to_byte(&mut self) {
        self.bit_pos = (self.bit_pos + 7) & !7;
    }

    #[inline]
    pub fn skip_bits(&mut self, n: usize) -> Result<()> {
        // Past the end is not an error: upstream's buffers are zero-padded
        // (`AV_INPUT_BUFFER_PADDING_SIZE`), so the reader is allowed to run a little
        // beyond the last real byte. `bits_left()` goes negative, which is exactly
        // what the callers that check it expect.
        self.bit_pos += n;
        Ok(())
    }

    #[inline]
    pub fn get_bits1(&mut self) -> Result<u32> {
        self.get_bits(1)
    }

    /// Read up to 32 bits, zero-extending past the end of the buffer.
    ///
    /// Upstream reads from a buffer with `AV_INPUT_BUFFER_PADDING_SIZE` zero bytes
    /// appended, so a read that starts within the last few bits of the payload
    /// silently yields zeros instead of failing. A frame that ends two bits short of
    /// its declared length is normal in a superframe whose carry was rounded up to a
    /// byte boundary — erroring on it turns a decodable packet into a lost one.
    #[inline]
    pub fn get_bits(&mut self, n: usize) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(DecoderError::InvalidData("get_bits > 32".into()));
        }

        let mut out: u32 = 0;
        let mut remaining = n;
        while remaining > 0 {
            if self.bit_pos >= self.size_in_bits {
                out <<= remaining;
                self.bit_pos += remaining;
                break;
            }

            let byte_idx = self.bit_pos >> 3;
            let bit_in_byte = self.bit_pos & 7; // 0..7, MSB-first
            let avail = 8 - bit_in_byte;
            let take = remaining.min(avail);

            let byte = self.buf[byte_idx] as u32;
            let shift = (avail - take) as u32;
            let mask = (1u32 << take) - 1;
            let bits = (byte >> shift) & mask;

            out = (out << take) | bits;
            self.bit_pos += take;
            remaining -= take;
        }

        Ok(out)
    }

    #[inline]
    pub fn show_bits(&self, n: usize) -> Result<u32> {
        let mut tmp = self.clone();
        tmp.get_bits(n)
    }

    #[inline]
    pub fn get_bits_long(&mut self, n: usize) -> Result<u32> {
        self.get_bits(n)
    }
}
