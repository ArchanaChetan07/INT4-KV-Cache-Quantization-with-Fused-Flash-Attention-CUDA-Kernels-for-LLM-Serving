"""Tests for INT4 nibble packing (2 values/byte storage format)."""

import numpy as np
import pytest

from src.int4_pack import pack_int4, unpack_int4, packed_nbytes
from src.quantize_int4_ref import quantize_int4_ref


class TestPackRoundTrip:

    def test_even_last_dim_exact(self):
        np.random.seed(0)
        q = np.random.randint(0, 16, (256, 64), dtype=np.uint8)
        packed = pack_int4(q)
        assert packed.shape == (256, 32)
        np.testing.assert_array_equal(unpack_int4(packed, 64), q)

    def test_odd_last_dim_exact(self):
        np.random.seed(1)
        q = np.random.randint(0, 16, (10, 7), dtype=np.uint8)
        packed = pack_int4(q)
        assert packed.shape == (10, 4)  # padded to 8, packed to 4
        np.testing.assert_array_equal(unpack_int4(packed, 7), q)

    def test_3d_shapes(self):
        np.random.seed(2)
        q = np.random.randint(0, 16, (4, 128, 32), dtype=np.uint8)
        packed = pack_int4(q)
        assert packed.shape == (4, 128, 16)
        np.testing.assert_array_equal(unpack_int4(packed, 32), q)

    def test_extreme_values(self):
        q = np.array([[0, 15, 15, 0, 7, 8]], dtype=np.uint8)
        np.testing.assert_array_equal(unpack_int4(pack_int4(q), 6), q)

    def test_rejects_out_of_range(self):
        q = np.array([[16]], dtype=np.uint8)
        with pytest.raises(AssertionError):
            pack_int4(q)


class TestPackedStorage:

    def test_storage_halves(self):
        q = np.random.randint(0, 16, (1024, 128), dtype=np.uint8)
        packed = pack_int4(q)
        assert packed.nbytes == q.nbytes // 2
        assert packed.nbytes == packed_nbytes(q.shape)

    def test_end_to_end_with_quantizer(self):
        """Quantize -> pack -> unpack -> dequantize stays within error bounds."""
        np.random.seed(3)
        kv = np.random.randn(256, 64).astype(np.float32)

        q, scale, zp = quantize_int4_ref(kv, per_channel=True)
        q_restored = unpack_int4(pack_int4(q), 64)
        np.testing.assert_array_equal(q_restored, q)

        dequant = q_restored.astype(np.float32) * scale - zp * scale
        assert np.abs(kv - dequant).max() <= scale.max() / 1.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
