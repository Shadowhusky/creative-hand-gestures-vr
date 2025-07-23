// Mel.cs  – single static method
using System;
using UnityEngine;

public static class Mel
{
  const int N_MELS = 64;
  const int N_FFT = 1024;
  const int HOP = 256;
  const int SAMPLE_RATE = 16000;

  // input 0.64 s mono PCM, length = 10240
  public static float[] ToLogMel(float[] wave)
  {
    // ❶ window & FFT (using Unity's FFT utility)
    int nFrames = 40;
    float[] mel = new float[N_MELS * nFrames];

    float[] window = new float[N_FFT];
    for (int i = 0; i < N_FFT; ++i)
      window[i] = 0.5f - 0.5f * Mathf.Cos(2 * Mathf.PI * i / (N_FFT - 1));

    int melOffset = 0;
    for (int frame = 0; frame < nFrames; ++frame)
    {
      int start = frame * HOP;
      float[] frameBuf = new float[N_FFT];
      for (int i = 0; i < N_FFT; ++i)
        frameBuf[i] = (start + i < wave.Length ? wave[start + i] : 0) * window[i];

      float[] mag = FFTUtil.Magnitude(frameBuf);   // length N_FFT/2+1

      float[] melBins = MelFilters.Apply(mag, SAMPLE_RATE); // 64 bins

      // ❸ log-scale
      for (int m = 0; m < N_MELS; ++m)
      {
        float db = 10f * Mathf.Log10(melBins[m] + 1e-8f); // dB scale
        db = Mathf.Clamp(db, -80f, 0f);
        mel[melOffset++] = db;
      }
    }
    return mel;    // length 64*40, row-major (mels slow, frames fast)
  }
}