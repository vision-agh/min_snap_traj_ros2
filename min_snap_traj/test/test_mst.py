"""Tests for the mst module."""

# -------------------------------------------------------------------------------
# Copyright (c) 2025 Hubert Szolc, EVS AGH University of Krakow, Poland
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -------------------------------------------------------------------------------
import min_snap_traj.mst as mst


def test_t_vec_degs():
    """Test t_vec generation for different degrees."""
    for deg in range(0, 12):
        t_vec = mst.t_vec(1.0, deg)
        assert (
            len(t_vec) == deg + 1
        ), f"[deg={deg}] t_vec length for degree is incorrect ({len(t_vec)})."
        assert all(
            isinstance(t, float) for t in t_vec
        ), f"[deg={deg}] All elements in t_vec should be floats."
        assert all(
            t == 1.0 for t in t_vec
        ), f"[[deg={deg}] All elements in t_vec should be equal to 1.0."

    for deg in range(0, 12):
        t_vec = mst.t_vec(2.0, deg)
        assert (
            len(t_vec) == deg + 1
        ), f"[deg={deg}] t_vec length for degree is incorrect ({len(t_vec)})."
        assert all(
            isinstance(t, float) for t in t_vec
        ), f"[deg={deg}] All elements in t_vec should be floats."
        assert all(
            t == 2.0**i for i, t in enumerate(t_vec)
        ), f"[deg={deg}] Elements in t_vec should be powers of 2.0."
