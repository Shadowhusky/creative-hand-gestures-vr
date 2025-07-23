// MelFilters.cs – lightweight triangular mel filter-bank (64 × (N_FFT/2+1))
using System;

public static class MelFilters
{
    const int N_MELS = 64;
    const int N_FFT = 1024;
    static readonly float[,] _fb = BuildFilters();

    static float[,] BuildFilters()
    {
        int bins = N_FFT / 2 + 1;
        float[,] fb = new float[N_MELS, bins];
        // mel scale helpers
        float hzToMel(float hz) => 2595f * (float)Math.Log10(1 + hz / 700f);
        float melToHz(float mel) => 700f * ((float)Math.Pow(10, mel / 2595f) - 1);

        float fMin = 0, fMax = 8000;          // up to Nyquist (16 kHz sample used later)
        float melMin = hzToMel(fMin);
        float melMax = hzToMel(fMax);
        float melStep = (melMax - melMin) / (N_MELS + 1);
        float[] melCenters = new float[N_MELS + 2];
        for (int m = 0; m < melCenters.Length; m++) melCenters[m] = melMin + m * melStep;
        float[] hzCenters = new float[melCenters.Length];
        for (int m = 0; m < melCenters.Length; m++) hzCenters[m] = melToHz(melCenters[m]);
        float[] binCenters = new float[hzCenters.Length];
        for (int m = 0; m < hzCenters.Length; m++) binCenters[m] = (hzCenters[m] / fMax) * (bins - 1);

        for (int m = 1; m <= N_MELS; m++)
        {
            int fLeft = (int)Math.Floor(binCenters[m - 1]);
            int fCenter = (int)Math.Floor(binCenters[m]);
            int fRight = (int)Math.Floor(binCenters[m + 1]);
            for (int k = fLeft; k < fCenter; k++)
                fb[m - 1, k] = (k - binCenters[m - 1]) / (binCenters[m] - binCenters[m - 1]);
            for (int k = fCenter; k < fRight; k++)
                fb[m - 1, k] = (binCenters[m + 1] - k) / (binCenters[m + 1] - binCenters[m]);
        }
        return fb;
    }

    // Apply filter-bank to magnitude spectrum → 64-bin vector
    public static float[] Apply(float[] mag, int sampleRate)
    {
        int bins = N_FFT / 2 + 1;
        if (mag.Length != bins) throw new ArgumentException("mag length mismatch");
        float[] mel = new float[N_MELS];
        for (int m = 0; m < N_MELS; m++)
        {
            float sum = 0;
            for (int k = 0; k < bins; k++) sum += _fb[m, k] * mag[k];
            mel[m] = sum;
        }
        return mel;
    }
} 