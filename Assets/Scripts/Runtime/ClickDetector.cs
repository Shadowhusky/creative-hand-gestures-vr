using System;
using System.Numerics;
using System.Linq;
using UnityEngine;
using TMPro;
using Unity.Barracuda;
using System.Collections.Generic;

[DisallowMultipleComponent]
public class ClickDetector : MonoBehaviour
{
  [Header("CNN ONNX Model (preferred)")]
  public NNModel onnxModel;

  [Header("CNN Meta JSON")]
  public TextAsset cnnMetaJson;

  [Header("Lightweight LogReg JSON (fallback)")]
  public TextAsset logRegJson;

  [Header("Shared mic (HandJointLogger)")]
  public HandJointLogger handLogger;

  [Header("Similarity / gates")]
  public int blockSize = 1024;                     // ≈21 ms @48 kHz
  [Range(0, 1)] public float similarityThreshold = 0.80f;
  [Range(0, 1)] public float smooth = 0.85f;

  [Header("Energy gate")] public float rmsGate = 0.01f;
  [Header("RMS upper gate")] public float rmsUpperGate = 0.2f;
  [Header("Spectral log-ratio gate (log10(hi/lo))")] public float ratioGate = 2f;
  [Header("Centroid gate")] public float centroidGate = 50f;

  [Header("Debug (optional)")] public TMP_Text debugText;

  public static event Action OnClickDetected;
  public static float CurrentSimilarity { get; private set; }

  // ----- constants / buffers (unchanged) -----------------------------
  const int BIN_LO_HI = 22, BIN_HI_HI = 150;         // 1–8 kHz @1024 → 128 bins
  const int BIN_LO_LOW = 0, BIN_HI_LOW = 13;         // <600 Hz
  const int VEC_LEN = BIN_HI_HI - BIN_LO_HI;         // 128

  float[] _hann, _liveBuf;
  Complex[] _fft;
  float[] _m, _s, _w; float _b;
  bool _useModel;

  float _emaSim, _noiseFloor;

  // ----- CNN specifics -----
  private List<float> audioHistory = new List<float>();
  private Model model;
  private IWorker worker;
  private Meta cnnMeta;
  private bool _useCNN;
  private int targetSamples;
  private float srRatio;

  [Serializable]
  class Meta { public float mean, std; public int n_mels, hop, sr; public float excerpt_sec; }

  [Serializable]
  class LRjson { public float[] mean, scale, weight; public float bias; }

  void Start()
  {
    // Initialize buffers
    _hann = new float[blockSize];
    for (int i = 0; i < blockSize; i++)
      _hann[i] = 0.5f * (1f - Mathf.Cos(2f * Mathf.PI * i / (blockSize - 1)));

    _fft = new Complex[blockSize];
    _liveBuf = new float[blockSize];

    // Find HandJointLogger if not assigned
    if (!handLogger)
    {
      handLogger = FindObjectOfType<HandJointLogger>();
      if (!handLogger)
      {
        Debug.LogError("ClickDetector: HandJointLogger not found");
        enabled = false;
        return;
      }
    }

    // Choose detection backend
    if (onnxModel && cnnMetaJson)
    {
      cnnMeta = JsonUtility.FromJson<Meta>(cnnMetaJson.text);
      if (cnnMeta == null)
      {
        Debug.LogError("ClickDetector: Failed to parse cnnMetaJson");
        enabled = false;
        return;
      }
      model = ModelLoader.Load(onnxModel);
      worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, model);
      // Verify model has a single output
      if (model.outputs.Count != 1)
      {
        Debug.LogError("ClickDetector: Model must have exactly one output");
        enabled = false;
        return;
      }
      _useCNN = true;
      targetSamples = (int)(cnnMeta.excerpt_sec * cnnMeta.sr);
      srRatio = handLogger.MicSampleRate / (float)cnnMeta.sr;
      Debug.Log($"[ClickDetector] using CNN model, targetSamples={targetSamples}, srRatio={srRatio:F2}");
    }
    else if (logRegJson)
    {
      ParseJsonModel(logRegJson.text);
      _useModel = true;
      Debug.Log("[ClickDetector] using logistic-reg model");
    }
    else
    {
      Debug.LogError("ClickDetector: Assign either CNN ONNX/meta or LogReg JSON");
      enabled = false;
    }
  }

  void Update()
  {
    if (!handLogger || !handLogger.TryGetMicBlock(blockSize, _liveBuf)) return;

    if (_useCNN)
    {
      // Downsample to 16kHz
      int dsCount = (int)(blockSize / srRatio);
      float[] dsBuf = new float[dsCount];
      float audioMin = float.MaxValue, audioMax = float.MinValue;
      for (int i = 0; i < blockSize; i++)
      {
        audioMin = Mathf.Min(audioMin, _liveBuf[i]);
        audioMax = Mathf.Max(audioMax, _liveBuf[i]);
      }
      for (int i = 0; i < dsCount; i++)
      {
        int start = (int)(i * srRatio);
        float sum = 0f;
        int count = Mathf.Min((int)srRatio, blockSize - start);
        for (int j = 0; j < count; j++) sum += _liveBuf[start + j];
        dsBuf[i] = count > 0 ? sum / count : 0f;
      }
      audioHistory.AddRange(dsBuf);
      if (audioHistory.Count > targetSamples)
        audioHistory.RemoveRange(0, audioHistory.Count - targetSamples);
      Debug.Log($"[ClickDetector] Audio input min={audioMin:F2} max={audioMax:F2}, history length={audioHistory.Count}");
    }

    ApplyWindow(_liveBuf);

    float[] hi = new float[VEC_LEN];
    float lowE = BandEnergy(_liveBuf, BIN_LO_LOW, BIN_HI_LOW);
    float hiE = BandMagnitude(_liveBuf, hi);

    float rmsHi = Mathf.Sqrt(hiE / VEC_LEN);

    if (rmsHi > rmsUpperGate) { DecaySim(); PushDebug(_emaSim, rmsHi, 0, 0, false); return; }

    _noiseFloor = Mathf.Lerp(rmsHi, _noiseFloor, 0.98f);
    if (rmsHi < _noiseFloor * rmsGate) { DecaySim(); PushDebug(_emaSim, rmsHi, 0, 0, false); return; }

    float ratio = Mathf.Log10(hiE / (lowE + 1e-6f));

    float centroid = 0, sumMag = 0;
    for (int k = 0; k < VEC_LEN; k++) { centroid += hi[k] * k; sumMag += hi[k]; }
    centroid /= sumMag + 1e-6f;

    if (ratio < ratioGate) { DecaySim(); PushDebug(_emaSim, rmsHi, ratio, centroid, false); return; }

    if (centroid < centroidGate) { DecaySim(); PushDebug(_emaSim, rmsHi, ratio, centroid, false); return; }

    float sim;
    if (_useCNN)
    {
      if (audioHistory.Count < targetSamples) { DecaySim(); PushDebug(_emaSim, rmsHi, ratio, centroid, false); return; }
      sim = GetCnnProb();
    }
    else if (_useModel)
    {
      // time-domain RMS for regression feature
      double rmsTime = Math.Sqrt(_liveBuf.Select(v => v * v).Average());
      sim = LogisticProb(rmsTime, ratio, centroid, SpectralFlatness(hi));
    }
    else
    {
      Debug.LogError("ClickDetector: No valid detection backend");
      enabled = false;
      return;
    }

    _emaSim = _emaSim * smooth + sim * (1f - smooth);
    CurrentSimilarity = _emaSim;

    bool click = _emaSim > similarityThreshold;
    if (click) OnClickDetected?.Invoke();
    PushDebug(_emaSim, rmsHi, ratio, centroid, click);
  }

  float GetCnnProb()
  {
    float[] wave = audioHistory.ToArray();
    float[] logMel = Mel.ToLogMel(wave);
    float melMin = float.MaxValue, melMax = float.MinValue;
    for (int i = 0; i < logMel.Length; i++)
    {
      melMin = Mathf.Min(melMin, logMel[i]);
      melMax = Mathf.Max(melMax, logMel[i]);
    }
    Debug.Log($"[ClickDetector] logMel min={melMin:F2} max={melMax:F2}");
    // Normalize to match training data
    float[] normMel = new float[logMel.Length];
    for (int i = 0; i < logMel.Length; i++)
    {
      normMel[i] = (logMel[i] - cnnMeta.mean) / (cnnMeta.std != 0 ? cnnMeta.std : 1f);
    }
    float normMin = float.MaxValue, normMax = float.MinValue;
    for (int i = 0; i < normMel.Length; i++)
    {
      normMin = Mathf.Min(normMin, normMel[i]);
      normMax = Mathf.Max(normMax, normMel[i]);
    }
    Debug.Log($"[ClickDetector] Normalized mel min={normMin:F2} max={normMax:F2}");
    int n_frames = (int)(cnnMeta.excerpt_sec * cnnMeta.sr / cnnMeta.hop);
    TensorShape shape = new TensorShape(1, cnnMeta.n_mels, n_frames, 1);
    using Tensor input = new Tensor(shape, normMel);
    worker.Execute(input);
    using Tensor output = worker.PeekOutput();
    var data = output.ToReadOnlyArray();
    float sim = Mathf.Clamp01(data[0]); // Ensure output is [0,1]
    Debug.Log($"[ClickDetector] CNN sim={sim:F2} emaSim={_emaSim:F2}");
    return sim;
  }

  void ParseJsonModel(string jsonText)
  {
    var tmp = JsonUtility.FromJson<LRjson>(jsonText);
    if (tmp == null || tmp.mean == null || tmp.scale == null || tmp.weight == null)
    {
      Debug.LogError("ClickDetector: Failed to parse logRegJson");
      enabled = false;
      return;
    }
    _m = tmp.mean; _s = tmp.scale; _w = tmp.weight; _b = tmp.bias;
  }

  float LogisticProb(double rmsTime, double logRatio, double centroid, double flatness)
  {
    double[] feat = { rmsTime, logRatio, centroid, flatness };
    double z = _b;
    for (int i = 0; i < 4; i++)
    {
      double denom = Math.Abs(_s[i]) > 1e-6 ? _s[i] : 1.0; // avoid /0
      z += _w[i] * (float)((feat[i] - _m[i]) / denom);
    }
    float prob = (float)(1.0 / (1.0 + Math.Exp(-z)));
    Debug.Log($"[ClickDetector] Logistic sim={prob:F2} emaSim={_emaSim:F2} rms={rmsTime:F3} logR={logRatio:F2} cent={centroid:F1}");
    return prob;
  }

  void ApplyWindow(float[] b) { for (int i = 0; i < b.Length; i++) b[i] *= _hann[i]; }

  float BandMagnitude(float[] time, float[] dst)
  {
    for (int i = 0; i < blockSize; i++) _fft[i] = new Complex(time[i], 0);
    FFT.InPlaceRadix2(_fft, false);
    float sum = 0;
    for (int k = 0; k < VEC_LEN; k++)
    {
      float m = (float)_fft[BIN_LO_HI + k].Magnitude;
      dst[k] = m; sum += m * m;
    }
    return sum;
  }

  float BandEnergy(float[] time, int lo, int hi)
  {
    for (int i = 0; i < blockSize; i++) _fft[i] = new Complex(time[i], 0);
    FFT.InPlaceRadix2(_fft, false);
    float s = 0; for (int k = lo; k <= hi; k++) s += (float)_fft[k].Magnitude * (float)_fft[k].Magnitude;
    return s;
  }

  static void Normalise(float[] v)
  {
    double s = 0; foreach (var x in v) s += x * x;
    float inv = 1f / Mathf.Sqrt((float)s + 1e-6f); for (int i = 0; i < v.Length; i++) v[i] *= inv;
  }

  static float SpectralFlatness(float[] hi)
  {
    double g = 0, m = 0; foreach (var x in hi) { g += Math.Log(x + 1e-6); m += x; }
    g = Math.Exp(g / hi.Length); return (float)(g / (m / hi.Length + 1e-6));
  }

  void DecaySim()
  {
    _emaSim *= smooth;
    CurrentSimilarity = _emaSim;
  }

  void PushDebug(float sim, float rms, float ratio, float centroid, bool ok)
  {
    if (!debugText) return;
    debugText.text = $"sim {sim:F2} \nClick {ok} \nRMS {rms:F3} \nratio {ratio:F1} \ncentroid {centroid:F1}";
  }

  static class FFT
  {
    public static void InPlaceRadix2(Complex[] a, bool inv)
    {
      int n = a.Length; for (int j = 1, i = 0; j < n; j++)
      {
        int bit = n >> 1; for (; i >= bit; bit >>= 1) i -= bit; i += bit;
        if (j < i) { var t = a[j]; a[j] = a[i]; a[i] = t; }
      }
      for (int len = 2; len <= n; len <<= 1)
      {
        double ang = 2 * Math.PI / len * (inv ? -1 : 1);
        Complex wl = new Complex(Math.Cos(ang), Math.Sin(ang));
        for (int i = 0; i < n; i += len)
        {
          Complex w = Complex.One;
          for (int j = 0; j < len / 2; j++)
          {
            Complex u = a[i + j], v = a[i + j + len / 2] * w;
            a[i + j] = u + v; a[i + j + len / 2] = u - v; w *= wl;
          }
        }
      }
      if (inv) { for (int i = 0; i < n; i++) a[i] /= n; }
    }
  }

  void OnDestroy()
  {
    if (_useCNN && worker != null) worker.Dispose();
  }
}