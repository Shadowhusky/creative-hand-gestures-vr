using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;
using UnityEngine.XR;
using UnityEngine.XR.Hands;
using UnityEngine.XR.Management;

#if UNITY_ANDROID || UNITY_STANDALONE
using OVR;                                    // for OVRPassthroughLayer
#endif

public class SnapPredictor : MonoBehaviour
{
    public static event Action OnSnapDetected;   // raised on each snap

    [Header("Scene References")]
    public HandJointLogger logger;
    public ParticleSystem snapEffectPrefab;

    [Header("HUD")]
    public Text hudText;
    public Color hudColor = Color.yellow;

    [Header("Model file (StreamingAssets)")]
    public string svmJson = "snap_svm.json";
    [Range(0f, 1f)] public float threshold = 0.60f;

    [Header("Particle offset (metres)")]
    public float particleYOffset = 0.02f;        // 2 cm upward tweak

    [Header("Passthrough (optional)")]
    public OVRPassthroughLayer passthroughLayer;

    // ─── internal ─────────────────────────────────────────────
    SvmModel _svm;
    double[] _featBuf;
    Canvas _hudCanvas;
    Camera _mainCam;
    XRHandSubsystem _hands;

    bool _arActive;
    bool _prevSnap;
    float _lastToggleTime = -999f;
    const float COOLDOWN = 0.5f;                 // 500 ms

    // ──────────────────────────────────────────────────────────
    IEnumerator Start()
    {
        yield return null;                       // wait one frame for XR

        if (!logger) logger = GetComponent<HandJointLogger>();
        if (!logger) { Debug.LogError("SnapPredictor: HandJointLogger missing"); enabled = false; yield break; }

        _mainCam = Camera.main;
        _hands = XRGeneralSettings.Instance?.Manager?.activeLoader
                       ?.GetLoadedSubsystem<XRHandSubsystem>();

        if (!hudText) hudText = CreateHud();

#if UNITY_ANDROID || UNITY_STANDALONE
        if (!passthroughLayer)
            passthroughLayer = FindObjectOfType<OVRPassthroughLayer>();
        if (passthroughLayer) passthroughLayer.enabled = false;
#endif
        yield return LoadSvm();
    }

    void Update()
    {
        if (_svm == null) return;
        if (!logger.TryGetWindowFeatures(out _featBuf)) return;

        double prob = _svm.Predict(_featBuf);
        bool snap = prob > threshold;

        // float sim = ClickDetector.LastSimilarity;

        hudText.text =
        // Different Color when snap detected
          $"<b>Snap:</b> {(snap ? $"<color=#00FF7F>{prob:F2}</color>" : $"{prob:F2}")}\n";

        logger.TryGetPalmPose(out Pose palmPose);

        // ─── Rising-edge detection ────────────────────────────
        if (!_prevSnap && snap)
        {
            OnSnapDetected?.Invoke();            // notify listeners

            if (Time.time - _lastToggleTime > COOLDOWN)
            {
                _lastToggleTime = Time.time;
                TogglePassthrough();
                SpawnEffect(palmPose);
            }
        }
        _prevSnap = snap;
    }

    // ─── Passthrough toggle ──────────────────────────────────
    void TogglePassthrough()
    {
        if (!passthroughLayer) return;
        _arActive = !_arActive;
        passthroughLayer.enabled = _arActive;
    }

    // ─── Particle burst ──────────────────────────────────────
    void SpawnEffect(Pose palm)
    {
        if (!snapEffectPrefab) return;

        Vector3 pos = TryGetSnapPoint(out Vector3 p)
                      ? p
                      : palm.position;

        pos.y += particleYOffset;               // upward nudge

        var fx = Instantiate(snapEffectPrefab, pos, Quaternion.identity);
        fx.Play();
        Destroy(fx.gameObject, fx.main.duration + fx.main.startLifetime.constantMax);
    }

    bool TryGetSnapPoint(out Vector3 pos)
    {
        pos = Vector3.zero;
        if (_hands == null || !_hands.running) return false;

        XRHand h = logger.handNode == XRNode.LeftHand ? _hands.leftHand : _hands.rightHand;
        if (!h.isTracked) return false;

        if (h.GetJoint(XRHandJointID.ThumbTip).TryGetPose(out Pose th) &&
            h.GetJoint(XRHandJointID.MiddleTip).TryGetPose(out Pose mi))
        {
            pos = (th.position + mi.position) * 0.5f;
            return true;
        }
        return false;
    }

    // ─── HUD builder ─────────────────────────────────────────
    Text CreateHud()
    {
        // ── look for a panel called “PredictionMetrics” in the scene
        var host = GameObject.Find("PredictionMetrics");
        if (host)                      // ✅ found – put the text right in there
        {
            var txt = new GameObject("Snap Text").AddComponent<Text>();
            txt.transform.SetParent(host.transform, false);

            txt.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
            txt.fontSize = 26;
            txt.supportRichText = true;
            txt.alignment = TextAnchor.UpperCenter;
            txt.color = hudColor;
            txt.rectTransform.sizeDelta = new Vector2(270, 140);

            // keep a reference so Update() can write into it
            hudText = txt;
            _hudCanvas = host.GetComponentInParent<Canvas>();   // for position calc
            return txt;
        }

        // ── fallback to the old floating HUD so nothing breaks
        _hudCanvas = new GameObject("SnapHUD").AddComponent<Canvas>();
        _hudCanvas.renderMode = RenderMode.WorldSpace;
        _hudCanvas.sortingOrder = 99;
        _hudCanvas.transform.localScale = Vector3.one * 0.0012f;

        var t = new GameObject("Text").AddComponent<Text>();
        t.transform.SetParent(_hudCanvas.transform, false);
        t.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        t.fontSize = 90;
        t.supportRichText = true;
        t.alignment = TextAnchor.MiddleCenter;
        t.color = hudColor;
        t.rectTransform.sizeDelta = new Vector2(270, 140);

        hudText = t;
        return t;
    }

    // ─── SVM JSON loader ─────────────────────────────────────
    IEnumerator LoadSvm()
    {
        string path = Path.Combine(Application.streamingAssetsPath, svmJson);
        string json;

        if (path.Contains("://") || path.Contains("jar:"))
        {
            using var r = UnityWebRequest.Get(path);
            yield return r.SendWebRequest();
            if (r.result != UnityWebRequest.Result.Success)
            { Debug.LogError("SVM load error: " + r.error); yield break; }
            json = r.downloadHandler.text;
        }
        else
        {
            if (!File.Exists(path))
            { Debug.LogError("SVM file missing: " + path); yield break; }
            json = File.ReadAllText(path);
        }

        _svm = JsonUtility.FromJson<SvmModel>(json);
        if (!_svm.IsValid(out string why))
            Debug.LogError("SVM JSON malformed: " + why);
        else
            Debug.Log($"SnapPredictor: {_svm.nSV} SVs • γ={_svm.gamma}");
    }

    // ─── tiny RBF-SVM evaluator ───────────────────────────────
    [Serializable]
    class SvmModel
    {
        public float[] mean, scale, svFlat, dualCoef;
        public int nSV, featDim;
        public float intercept, gamma;

        public bool IsValid(out string why)
        {
            why = null;
            if (mean == null || scale == null || svFlat == null || dualCoef == null) { why = "null field"; return false; }
            if (mean.Length != scale.Length) { why = "mean/scale size"; return false; }
            if (svFlat.Length != nSV * featDim) { why = "svFlat size"; return false; }
            if (dualCoef.Length != nSV) { why = "dualCoef size"; return false; }
            return true;
        }

        public double Predict(double[] raw)
        {
            double[] x = new double[featDim];
            for (int i = 0; i < featDim; i++) x[i] = (raw[i] - mean[i]) / scale[i];

            double sum = 0;
            for (int s = 0; s < nSV; s++)
            {
                double d2 = 0; int baseIdx = s * featDim;
                for (int j = 0; j < featDim; j++)
                {
                    double diff = x[j] - svFlat[baseIdx + j];
                    d2 += diff * diff;
                }
                sum += dualCoef[s] * Math.Exp(-gamma * d2);
            }
            double z = sum + intercept;
            return 1.0 / (1.0 + Math.Exp(-z));
        }
    }
}
