using System;
using System.IO;
using UnityEngine;
using System.Numerics;

[RequireComponent(typeof(HandJointLogger))]
public class ClickAutoRecorder : MonoBehaviour
{
  [Header("Settings")]
  public int blockSize = 1024;   // ≈21.3 ms @48 kHz
  public float debounceTime = .20f;
  public float minRmsForNoise = .03f;

  [Header("Enable auto-save flags")]
  public bool saveClicks = true;
  public bool saveNoise = true;

  string outDir = null;

  // ─────────────────────────────────────────────────────────────
  HandJointLogger _logger;
  float[] _buf;
  float _lastSave;

  void Start()
  {
    _logger = GetComponent<HandJointLogger>();
    _buf = new float[blockSize];

    outDir = Application.persistentDataPath;
    Directory.CreateDirectory(Path.Combine(outDir, "clicks"));
    Directory.CreateDirectory(Path.Combine(outDir, "noise"));

    Debug.Log($"[AutoRec] saving to {outDir}");
  }

  void Update()
  {
    if (!_logger.TryGetMicBlock(blockSize, _buf)) return;

    // Quick RMS
    float rms = 0f; foreach (var s in _buf) rms += s * s;
    rms = Mathf.Sqrt(rms / blockSize);

    bool clickLike = ClickDetectorStaticGates.Pass(_buf, rms);
    if (Time.time - _lastSave < debounceTime) return;

    if (clickLike)
    {
      if (saveClicks)
      {
        SaveBlock(_buf, 48_000, Path.Combine(outDir, "clicks"));
        _lastSave = Time.time;
      }
    }
    else if (rms > minRmsForNoise)
    {
      if (saveNoise)
      {
        SaveBlock(_buf, 48_000, Path.Combine(outDir, "noise"));
        _lastSave = Time.time;
      }
    }
  }

  public static void SaveBlock(float[] buf, int sampleRate, string path)
  {
    float peak = 0f;
    foreach (var s in buf) peak = Mathf.Max(peak, Mathf.Abs(s));
    float gain = peak < 1e-4f ? 1f : 0.9f / peak;

    // build WAV in memory
    using var ms = new MemoryStream();
    using var bw = new BinaryWriter(ms);
    int bytes = buf.Length * 2;
    bw.Write("RIFF".ToCharArray()); bw.Write(36 + bytes);
    bw.Write("WAVE".ToCharArray());
    bw.Write("fmt ".ToCharArray()); bw.Write(16);
    bw.Write((short)1); bw.Write((short)1);
    bw.Write(sampleRate); bw.Write(sampleRate * 2);
    bw.Write((short)2); bw.Write((short)16);
    bw.Write("data".ToCharArray()); bw.Write(bytes);

    foreach (float f in buf)
    {
      short s16 = (short)Mathf.Clamp(Mathf.RoundToInt(f * gain * 32767f), -32768, 32767);
      bw.Write(s16);
    }

    string ts = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
    string clipPath = Path.Combine(path, $"clip_{ts}.wav");

    File.WriteAllBytes(clipPath, ms.ToArray());
  }

}

/* -------------------------------------------------------------------------
   Static gate helper identical to ClickDetector’s *pre-similarity* logic.
   If you change thresholds in ClickDetector, mirror them here.
---------------------------------------------------------------------------*/
static class ClickDetectorStaticGates
{
  const int N = 1024;            // FFT size must match blockSize above
  static readonly float[] hann;
  static readonly System.Numerics.Complex[] fft;
  const float logRatioGate = 2.0f;   // log10(hi/lo)
  const float centroidGate = 30;   // double because bin index scale doubled (≈4 kHz)

  static ClickDetectorStaticGates()
  {
    hann = new float[N];
    for (int i = 0; i < N; i++)
      hann[i] = 0.5f * (1f - Mathf.Cos(2 * Mathf.PI * i / (N - 1)));
    fft = new System.Numerics.Complex[N];
  }

  public static bool Pass(float[] block, float rms)
  {
    // avoid indexing beyond block bounds
    if (block == null || block.Length < N) return false;
    float[] buf = new float[N];
    for (int i = 0; i < N; i++) buf[i] = block[i] * hann[i];

    for (int i = 0; i < N; i++) fft[i] = new System.Numerics.Complex(buf[i], 0);
    FFT.InPlaceRadix2(fft, false);

    double hi = 0, lo = 0, cent = 0, magSum = 0;
    for (int k = 0; k < N / 2; k++)
    {
      double m = fft[k].Magnitude;
      if (k <= 13) lo += m * m;          // <600 Hz @1024
      if (k >= 22 && k < 171)            // 1–8 kHz @1024
      {
        hi += m * m;
        cent += m * k;
        magSum += m;
      }
    }
    double logRatio = Math.Log10(hi / (lo + 1e-9));
    double centroid = (cent / (magSum + 1e-9)) * 2 + 22;   // formula unchanged, bin scale doubled handled by gate above
    return logRatio >= logRatioGate && centroid >= centroidGate;
  }

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
        Complex wlen = new(Math.Cos(ang), Math.Sin(ang));
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