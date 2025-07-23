// Assets/Scripts/Editor/ClickFingerprintBuilder.cs
#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;
using System.Linq;
using System.Numerics;

public static class ClickFingerprintBuilder
{
  const int FFT_SIZE = 1024;    // must be power-of-two ≥ click length
  const int TOP_K = 6;       // how many peaks to keep

  [MenuItem("Assets/Audio/Build Click Fingerprint", false, 2000)]
  static void BuildFingerprint()
  {
    var clip = Selection.activeObject as AudioClip;
    if (!clip) { Debug.LogWarning("Select a single WAV/AIFF AudioClip first."); return; }

    // 1 ─ load samples (mono 48 kHz recommended)
    float[] buf = new float[clip.samples];
    clip.GetData(buf, 0);

    // 2 ─ pad / truncate to FFT_SIZE and apply Hann window
    float[] window = new float[FFT_SIZE];
    int N = Mathf.Min(buf.Length, FFT_SIZE);
    for (int i = 0; i < N; i++)
    {
      float w = 0.5f * (1f - Mathf.Cos(2f * Mathf.PI * i / (N - 1)));
      window[i] = buf[i] * w;
    }

    // 3 ─ compute FFT (naïve DFT – fine for a 1-off editor step)
    Complex[] X = new Complex[FFT_SIZE];
    for (int k = 0; k < FFT_SIZE; k++)
    {
      Complex sum = Complex.Zero;
      for (int n = 0; n < FFT_SIZE; n++)
        sum += window[n] * Complex.Exp(-Complex.ImaginaryOne * 2 * Mathf.PI * k * n / FFT_SIZE);
      X[k] = sum;
    }
    float[] mag = X.Select(c => (float)c.Magnitude).ToArray();

    // 4 ─ keep top-K bins
    var top = mag
        .Select((m, i) => (m, i))
        .OrderByDescending(pair => pair.m)
        .Take(TOP_K)
        .ToArray();

    // 5 ─ create asset
    var fp = ScriptableObject.CreateInstance<ClickFingerprint>();
    fp.fftSize = FFT_SIZE;
    fp.freqs = top.Select(p => (float)p.i).ToArray();
    fp.mags = top.Select(p => p.m / top[0].m).ToArray();   // normalise

    string path = AssetDatabase.GenerateUniqueAssetPath("Assets/Audio/ClickFingerprint.asset");
    AssetDatabase.CreateAsset(fp, path);
    AssetDatabase.SaveAssets();
    Debug.Log($"Click fingerprint saved → {path}");
  }

  [MenuItem("Assets/Audio/Build Click Fingerprint", true)]
  static bool ValidateMenu() => Selection.activeObject is AudioClip;
}
#endif
