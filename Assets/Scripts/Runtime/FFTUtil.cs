// FFTUtil.cs – simple radix-2 FFT helper with magnitude output for Mel.cs
using System;
using System.Numerics;

public static class FFTUtil
{
    // In-place Radix-2 FFT (copy of ClickDetector.FFT but public)
    public static void InPlaceRadix2(Complex[] a, bool inverse)
    {
        int n = a.Length;
        for (int j = 1, i = 0; j < n; j++)
        {
            int bit = n >> 1;
            for (; i >= bit; bit >>= 1) i -= bit;
            i += bit;
            if (j < i) { var t = a[j]; a[j] = a[i]; a[i] = t; }
        }
        for (int len = 2; len <= n; len <<= 1)
        {
            double ang = 2 * Math.PI / len * (inverse ? -1 : 1);
            Complex wlen = new Complex(Math.Cos(ang), Math.Sin(ang));
            for (int i = 0; i < n; i += len)
            {
                Complex w = Complex.One;
                for (int j = 0; j < len / 2; j++)
                {
                    Complex u = a[i + j], v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
        if (inverse) for (int i = 0; i < n; i++) a[i] /= n;
    }

    // Returns magnitude spectrum (N/2+1) from real input array length N
    public static float[] Magnitude(float[] real)
    {
        int n = real.Length;
        Complex[] buf = new Complex[n];
        for (int i = 0; i < n; i++) buf[i] = new Complex(real[i], 0);
        InPlaceRadix2(buf, false);
        int bins = n / 2 + 1;
        float[] mag = new float[bins];
        for (int k = 0; k < bins; k++) mag[k] = (float)buf[k].Magnitude;
        return mag;
    }
} 