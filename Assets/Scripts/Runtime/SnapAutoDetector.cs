using System;
using System.Numerics;
using UnityEngine;

[RequireComponent(typeof(HandJointLogger))]
public class SnapAutoDetector : MonoBehaviour
{
    [Header("Save folder")]
    public string subDir = "snaps";          // under Application.persistentDataPath

    [Header("Gates (tune in Inspector)")]
    public int blockSize = 512;   // ≈10.7 ms @48 kHz
    [Range(0, 6)] public float rmsGate = 0.001f;
    [Range(1, 10)] public float ratioGate = 4f;
    public float centroidGateLow = 55f;
    public float centroidGateHigh = 75f;
    [Range(.90f, 1f)] public float floorSmooth = .97f;

    [Header("Debounce (seconds)")]
    public float cooldown = 0.25f;           // avoid double-saves

    // ───────── internals ─────────
    HandJointLogger _logger;
    float[] _hann, _buf;
    Complex[] _fft;
    float _noiseFloor, _lastSave;
    string _path;

    const int BIN_LO_HI = 22;       // 1 kHz
    const int BIN_HI_HI = 128;      // 8 kHz
    const int BIN_LO_LOW = 0;
    const int BIN_HI_LOW = 13;       // 600 Hz
    const int VEC_LEN = BIN_HI_HI - BIN_LO_HI;

    void Start()
    {
        _logger = GetComponent<HandJointLogger>();

        _hann = new float[blockSize];
        for (int i = 0; i < blockSize; i++)
            _hann[i] = 0.5f * (1f - Mathf.Cos(2f * Mathf.PI * i / (blockSize - 1)));

        _buf = new float[blockSize];
        _fft = new Complex[blockSize];

        _path = System.IO.Path.Combine(Application.persistentDataPath, subDir);
        System.IO.Directory.CreateDirectory(_path);
    }

    void Update()
    {
        // pull live mic block --------------------------------------------
        if (!_logger.TryGetMicBlock(blockSize, _buf)) return;

        // debounce
        if (Time.time - _lastSave < cooldown) return;

        ApplyWindow(_buf);

        // FFT + energy measures
        float[] hi = new float[VEC_LEN];
        float lowE = BandEnergy(_buf, BIN_LO_LOW, BIN_HI_LOW);
        float hiE = BandMagnitude(_buf, hi);

        // 1) RMS gate (adaptive)
        float rmsHi = Mathf.Sqrt(hiE / VEC_LEN);
        _noiseFloor = Mathf.Lerp(rmsHi, _noiseFloor, floorSmooth);
        Debug.Log($"[SnapAutoDetector] -5 RMS={rmsHi:F2}  noiseFloor={_noiseFloor:F2}");
        if (rmsHi < _noiseFloor * rmsGate) return;
        Debug.Log($"[SnapAutoDetector] -4 RMS gate passed");

        // 2) Hi/Low ratio
        float ratio = hiE / (lowE + 1e-6f);
        if (ratio < ratioGate) return;
 
        // 3) Spectral centroid
        float centroid = 0f, sum = 0f;
        for (int k = 0; k < VEC_LEN; k++) { centroid += hi[k] * k; sum += hi[k]; }
        centroid /= sum + 1e-6f;
        Debug.Log($"[SnapAutoDetector] -3 ratio={ratio:F1}  centroid={centroid:F1} rmsHi={rmsHi:F6}");

        if (centroid < centroidGateLow || centroid > centroidGateHigh) return;

        // passed all gates → save ----------------------------------------
        string f = $"snap_{DateTime.Now:yyyyMMdd_HHmmss_fff}.wav";
        string wavPath = System.IO.Path.Combine(_path, f);
        ClickAutoRecorder.SaveBlock(_buf, 48_000, wavPath);
        Debug.Log($"[SnapAutoDetector] saved {wavPath}");
        _lastSave = Time.time;
    }

    // ───────── helpers ─────────
    void ApplyWindow(float[] buf)
    {
        for (int i = 0; i < buf.Length; i++) buf[i] *= _hann[i];
    }

    float BandMagnitude(float[] time, float[] dstHi)
    {
        for (int i = 0; i < blockSize; i++) _fft[i] = new Complex(time[i], 0);
        FFT.InPlaceRadix2(_fft, false);
        float sumSq = 0f;
        for (int k = 0; k < VEC_LEN; k++)
        {
            float m = (float)_fft[BIN_LO_HI + k].Magnitude;
            dstHi[k] = m; sumSq += m * m;
        }
        return sumSq;
    }

    float BandEnergy(float[] time, int lo, int hi)
    {
        for (int i = 0; i < blockSize; i++) _fft[i] = new Complex(time[i], 0);
        FFT.InPlaceRadix2(_fft, false);
        float sum = 0f;
        for (int k = lo; k <= hi; k++)
            sum += (float)_fft[k].Magnitude * (float)_fft[k].Magnitude;
        return sum;
    }

    // tiny radix-2 FFT (unchanged)
    static class FFT
    {
        public static void InPlaceRadix2(Complex[] a, bool inv)
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
                double ang = 2 * Math.PI / len * (inv ? -1 : 1);
                Complex wlen = new Complex(Math.Cos(ang), Math.Sin(ang));
                for (int i = 0; i < n; i += len)
                {
                    Complex w = Complex.One;
                    for (int j = 0; j < len / 2; j++)
                    {
                        Complex u = a[i + j], v = a[i + j + len / 2] * w;
                        a[i + j] = u + v; a[i + j + len / 2] = u - v; w *= wlen;
                    }
                }
            }
            if (inv) for (int i = 0; i < n; i++) a[i] /= n;
        }
    }
}