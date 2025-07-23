using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR;
using UnityEngine.XR.Hands;
using UnityEngine.XR.Management;
// removed generic System.Numerics import to avoid Vector3 ambiguity
using Complex = System.Numerics.Complex;

[DefaultExecutionOrder(50)]
public class HandJointLogger : MonoBehaviour
{
    [Header("Detection thresholds")]
    public XRNode handNode = XRNode.RightHand;
    public float contactThresh = 0.05f,
                  velThresh = 0.80f,
                  palmThresh = 0.08f,
                  maxWindow = 0.25f,
                  debounceTime = 0.80f;

    [Header("Optional audio gate")]
    public bool useAudioGate = false;
    public float peakThresh = 0.10f,
                 rmsThresh = 0.02f;

    // ────── new adaptive snap gate parameters (mirrors SnapAutoDetector) ──────
    [Header("Snap audio gate (adaptive)")]
    public int blockSize = 512;           // ≈10.7 ms @48 kHz
    [Range(0, 6)] public float rmsGate = 0.001f;
    [Range(1, 10)] public float ratioGate = 4f;
    public float centroidGateLow = 55f;
    public float centroidGateHigh = 75f;
    [Range(.90f, 1f)] public float floorSmooth = .97f;

    Text _uiText;          // optional Matrics_Content display
    const string UI_PARENT = "MatricsContent";


    [Header("HUD colour")] public Color hudColor = Color.green;

    // ───────────────── constants ─────────────────
    static readonly XRHandJointID[] Tips = {
        XRHandJointID.ThumbTip, XRHandJointID.IndexTip, XRHandJointID.MiddleTip,
        XRHandJointID.RingTip,  XRHandJointID.LittleTip };
    const int TIP_CT = 5;
    const int WIN = 11;              // ±5 frames  ≈ 0.16 s @60 Hz
    const int BUF = 1024;            // 20 ms  microphone block
    const float AUDIO_WINDOW = 0.12f;   // accept peak within last 120 ms

    XRHandSubsystem _hands;
    AudioClip _mic;
    float _lastPeak, _lastRms, _lastPeakTime;
    StreamWriter _writer;

    // ────── buffers for adaptive audio gate ──────
    float[] _hann, _audioBuf;             // window + mic block
    Complex[] _fft;                       // FFT working array
    float _noiseFloor;                    // adaptive floor (hi-band RMS)

    // frequency-band constants (assumes 48 kHz / 512-sample block)
    const int BIN_LO_HI = 22;    // ≈1 kHz
    const int BIN_HI_HI = 128;   // ≈8 kHz
    const int BIN_LO_LOW = 0;
    const int BIN_HI_LOW = 13;   // ≈600 Hz
    const int VEC_LEN = BIN_HI_HI - BIN_LO_HI;

    // ───────────────── runtime ─────────────────
    int _frame, _snap; float _lastSnap = -999f;
    Vector3 _prevMidRelThumb; bool _havePrevRel;

    enum State { Idle, Contact }
    State _st = State.Idle; float _contactTime;

    public bool recordSnap = true;  // record snap events to CSV

    public bool TryGetMicBlock(int sampleCount, float[] dest)
    {
        if (_mic == null || dest == null || dest.Length < sampleCount) return false;
        int pos = Microphone.GetPosition(null) - sampleCount;
        if (pos < 0) return false;
        _mic.GetData(dest, pos);
        return true;
    }

    public int MicSampleRate => 48000;  // constant in this logger

    struct Frame
    {
        public float dist, speed, palm;
        public Vector3[] tipRel;               // 5× tip coords (wrist-rel)
        public Vector3[] vel;                  // 5× velocities
    }
    readonly Queue<Frame> _win = new();        // ≤WIN frames

    Text _hud;

    IEnumerator Start()
    {
        yield return null;                     // let OpenXR loader finish
        _hands = XRGeneralSettings.Instance?.Manager?.activeLoader
                    ?.GetLoadedSubsystem<XRHandSubsystem>();
        if (_hands == null || !_hands.running)
        {
            Debug.LogError("XRHands not running");
            enabled = false;
            yield break;
        }

        if (useAudioGate)
        {
            _mic = Microphone.Start(null, true, 1, 48000);
            if (!_mic) useAudioGate = false;
            else
            {
                // allocate buffers for adaptive gate
                _hann = new float[blockSize];
                for (int i = 0; i < blockSize; i++)
                    _hann[i] = 0.5f * (1f - Mathf.Cos(2f * Mathf.PI * i / (blockSize - 1)));
                _audioBuf = new float[blockSize];
                _fft = new Complex[blockSize];
            }
        }

        if (recordSnap)
        {
            StartCsv();
        }

        BuildHud();            // world-space HUD  (unchanged)
        HookUiPanel();         // ← NEW
    }

    void HookUiPanel()
    {
        var parent = GameObject.Find(UI_PARENT);
        if (!parent) return;

        var t = new GameObject("MetricsText").AddComponent<Text>();
        t.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        t.fontSize = 26;
        t.color = Color.white;
        t.alignment = TextAnchor.UpperLeft;
        t.rectTransform.sizeDelta = new Vector2(250, 500);
        t.supportRichText = true;
        t.horizontalOverflow = HorizontalWrapMode.Wrap;
        t.verticalOverflow = VerticalWrapMode.Truncate;

        // center in parent
        t.rectTransform.localPosition = Vector3.zero;
        t.rectTransform.localRotation = Quaternion.identity;
        t.rectTransform.localScale = Vector3.one;
        t.transform.SetParent(parent.transform, false);

        _uiText = t;
    }


    void Update()
    {
        _frame++;
        XRHand h = handNode == XRNode.LeftHand ? _hands.leftHand : _hands.rightHand;
        if (!h.isTracked) { _st = State.Idle; return; }
        if (!h.GetJoint(XRHandJointID.Wrist).TryGetPose(out Pose wrist)) return;

        // ① tip relative positions + velocities
        Vector3[] rel = new Vector3[TIP_CT];
        Vector3[] vel = new Vector3[TIP_CT];
        for (int i = 0; i < TIP_CT; i++)
        {
            h.GetJoint(Tips[i]).TryGetPose(out Pose p);
            rel[i] = p.position - wrist.position;
            vel[i] = _win.Count == 0 ? Vector3.zero
                     : (rel[i] - _win.Peek().tipRel[i]) / Time.deltaTime;
        }

        // ② metrics
        h.GetJoint(XRHandJointID.ThumbTip).TryGetPose(out Pose thumb);
        h.GetJoint(XRHandJointID.MiddleTip).TryGetPose(out Pose mid);
        h.GetJoint(XRHandJointID.Palm).TryGetPose(out Pose palm);

        float dist = Vector3.Distance(thumb.position, mid.position);
        float palmD = Vector3.Distance(mid.position, palm.position);

        // relative speed  (middle ↔ thumb)
        Vector3 midRelThumb = mid.position - thumb.position;
        float speedRel = _havePrevRel
            ? Vector3.Distance(midRelThumb, _prevMidRelThumb) / Time.deltaTime
            : 0f;
        _prevMidRelThumb = midRelThumb; _havePrevRel = true;

        if (useAudioGate) (_lastPeak, _lastRms) = GetAudioMetrics();

        // ③ FSM snap detection
        bool snap = false;


        bool flick = speedRel > velThresh && palmD < palmThresh;
        bool audioOK = AudioGateOk();   // ← replaced simple peak gate
        bool debounce = Time.time - _lastSnap > debounceTime;

        if (audioOK && debounce) { snap = true; _st = State.Idle; }
        else if (Time.time - _contactTime > maxWindow) _st = State.Idle;


        // record snap event
        if (snap && recordSnap) { _snap++; _lastSnap = Time.time; }

        // ④ CSV
        if (_writer != null)
        {
            _writer.Write(Time.time.ToString("F4", CultureInfo.InvariantCulture) + ',');
            foreach (var v in rel) _writer.Write($"{v.x:F4},{v.y:F4},{v.z:F4},");
            _writer.WriteLine($"{dist:F4},{speedRel:F3},{palmD:F4},{(snap ? "Snap" : "None")}");
        }
        
        // ⑤ enqueue frame
        _win.Enqueue(new Frame
        {
            dist = dist,
            speed = speedRel,
            palm = palmD,
            tipRel = rel,
            vel = vel
        });
        if (_win.Count > WIN) _win.Dequeue();

        // ⑥ HUD
        if (_hud && _frame % 15 == 0)
        {
            string msg =
                $"Frames : {_frame}\n" +
                $"Snaps  : {_snap}\n" +
                $"Dist   : {dist * 100f:F1} mm\n" +
                $"Speed  : {speedRel:F2} m/s\n" +
                $"Palm   : {palmD * 100f:F1} mm\n" +
                $"Peak   : {_lastPeak:F6}";

            if (_uiText) _uiText.text = msg;
            else
            {
                _hud.text = msg;
            }

        }
    }

    public bool TryGetWindowFeatures(out double[] feat)
    {
        feat = null; if (_win.Count < WIN) return false;

        float dMin = float.MaxValue, dSum = 0,
              sMax = float.MinValue, sSum = 0,
              pMin = float.MaxValue, pSum = 0;
        Frame first = default, last = default;
        Vector3[] velSum = new Vector3[TIP_CT];
        int idx = 0;
        foreach (var f in _win)
        {
            if (idx == 0) first = f; last = f; idx++;
            dMin = Mathf.Min(dMin, f.dist); dSum += f.dist;
            sMax = Mathf.Max(sMax, f.speed); sSum += f.speed;
            pMin = Mathf.Min(pMin, f.palm); pSum += f.palm;
            for (int t = 0; t < TIP_CT; t++) velSum[t] += f.vel[t];
        }

        var list = new List<double> { dMin, dSum / WIN, sMax, sSum / WIN, pMin, pSum / WIN };
        for (int t = 0; t < TIP_CT; t++)
        {   // ΔXYZ
            Vector3 d = last.tipRel[t] - first.tipRel[t];
            list.Add(d.x); list.Add(d.y); list.Add(d.z);
        }
        feat = list.ToArray();
        return true;
    }

    void StartCsv()
    {
        string file = Path.Combine(Application.persistentDataPath,
                         $"finger_snap_{DateTime.Now:yyyyMMdd_HHmmss}.csv");
        _writer = new StreamWriter(file, false);
        _writer.Write("Time,");
        foreach (var id in Tips) _writer.Write($"{id}.x,{id}.y,{id}.z,");
        _writer.WriteLine("Dist,Speed,PalmDist,Label");
        Debug.Log("CSV → " + file);
    }

    (float, float) GetAudioMetrics()
    {
        if (!_mic) return (0, 0);
        int pos = Microphone.GetPosition(null) - BUF;
        if (pos < 0) return (_lastPeak, _lastRms);

        float[] buf = new float[BUF];
        _mic.GetData(buf, pos);

        double sum = 0; float pk = 0;
        for (int i = 0; i < BUF; i++)
        {
            float a = Mathf.Abs(buf[i]);
            pk = Mathf.Max(pk, a);
            sum += a * a;
        }
        float rms = Mathf.Sqrt((float)(sum / BUF));

        _lastPeak = pk; _lastRms = rms;
        return (pk, rms);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Adaptive spectral gate (ported from SnapAutoDetector)
    bool AudioGateOk()
    {
        if (!useAudioGate) return true;
        if (!TryGetMicBlock(blockSize, _audioBuf)) return false;

        ApplyWindow(_audioBuf);

        float[] hi = new float[VEC_LEN];
        float lowE = BandEnergy(_audioBuf, BIN_LO_LOW, BIN_HI_LOW);
        float hiE = BandMagnitude(_audioBuf, hi);

        // 1) adaptive RMS gate
        float rmsHi = Mathf.Sqrt(hiE / VEC_LEN);
        if (_noiseFloor == 0f) _noiseFloor = rmsHi;                 // initialise
        _noiseFloor = Mathf.Lerp(rmsHi, _noiseFloor, floorSmooth);  // smooth

        Debug.Log($"[HandJointLogger] -5 RMS={rmsHi:F2}  noiseFloor={_noiseFloor:F2}");
        if (rmsHi < _noiseFloor * rmsGate) return false;
        Debug.Log($"[HandJointLogger] -4 RMS gate passed");

        // 2) high/low-band energy ratio
        float ratio = hiE / (lowE + 1e-6f);
        if (ratio < ratioGate) return false;

        // 3) spectral centroid within hi band
        float centroid = 0f, sum = 0f;
        for (int k = 0; k < VEC_LEN; k++) { centroid += hi[k] * k; sum += hi[k]; }
        centroid /= sum + 1e-6f;

        Debug.Log($"[HandJointLogger] -3 ratio={ratio:F1}  centroid={centroid:F1} rmsHi={rmsHi:F6}");


        return centroid >= centroidGateLow && centroid <= centroidGateHigh;
    }

    // window + spectral helpers ------------------------------------------------
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

    // tiny radix-2 FFT (identical to SnapAutoDetector)
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

    // ───────────────────────── UI helpers / cleanup ─────────────────────────
    void BuildHud()
    {
        var c = new GameObject("HUD_Canvas").AddComponent<Canvas>();
        c.renderMode = RenderMode.WorldSpace;
        c.transform.position = new Vector3(0, 0.5f, 1f);
        c.transform.localScale = Vector3.one * 0.002f;

        var t = new GameObject("HUD_Text").AddComponent<Text>();
        t.transform.SetParent(c.transform, false);
        t.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        t.fontSize = 110;
        t.color = hudColor;
        t.alignment = TextAnchor.UpperLeft;
        t.rectTransform.sizeDelta = new Vector2(950, 950);
        _hud = t;
    }

    void OnDestroy()
    {
        _writer?.Close();
        if (Microphone.IsRecording(null)) Microphone.End(null);
    }

    internal bool TryGetPalmPose(out Pose palmPose)
    {
        palmPose = default;
        var sub = XRGeneralSettings.Instance?.Manager?.activeLoader
                     ?.GetLoadedSubsystem<XRHandSubsystem>();
        if (sub == null || !sub.running) return false;

        XRHand h = handNode == XRNode.LeftHand ? sub.leftHand : sub.rightHand;
        if (!h.isTracked) return false;
        return h.GetJoint(XRHandJointID.Palm).TryGetPose(out palmPose);
    }
}
