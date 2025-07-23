using System;
using System.Collections;
using System.Globalization;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR;
using UnityEngine.XR.Hands;
using UnityEngine.XR.Interaction.Toolkit.Interactables;
using UnityEngine.XR.Management;

public class GestureReactionExperiment : MonoBehaviour
{
    [Header("Start buttons (3)")]
    public XRSimpleInteractable snapButton;
    public XRSimpleInteractable pinchButton;
    public XRSimpleInteractable clickButton;

    [Header("Stimulus panel & log")]
    public Image stimulusPanel;
    public TMP_Text logText;

    // ───────── timing ─────────
    [Header("Timing (s)")]
    public float waitMin = 1f;
    public float waitMax = 3f;

    // ───────── gesture detectors ─────────
    public SnapPredictor snapPredictor;
    public ClickDetector clickDetector;

    // ───────── pinch settings ─────────
    [Header("Pinch detection")]
    public XRNode pinchHand = XRNode.RightHand;   // which hand to test
    [Tooltip("Thumb‑Tip ↔ Index‑Tip distance (m) below this = pinch")]
    [Range(0.0f, 0.04f)] public float pinchThresh = 0.025f;

    // ───────── internal state ─────────
    enum Mode { None, Snap, Pinch, Click }
    Mode _mode = Mode.None;
    bool _running;
    float _stimulusTime;
    Coroutine _trial;

    XRHandSubsystem _hands;
    bool _prevPinch;

    void Awake()
    {
        if (!snapPredictor) snapPredictor = FindAnyObjectByType<SnapPredictor>();
        if (!clickDetector) clickDetector = FindAnyObjectByType<ClickDetector>();
        _hands = XRGeneralSettings.Instance?.Manager?.activeLoader
                     ?.GetLoadedSubsystem<XRHandSubsystem>();

        // wire three buttons -------------------------------------------------
        if (snapButton) snapButton.selectEntered.AddListener(_ => StartTrial(Mode.Snap));
        if (pinchButton) pinchButton.selectEntered.AddListener(_ => StartTrial(Mode.Pinch));
        if (clickButton) clickButton.selectEntered.AddListener(_ => StartTrial(Mode.Click));

        // subscribe to detectors --------------------------------------------
        if (snapPredictor) SnapPredictor.OnSnapDetected += OnSnap;
        if (clickDetector) ClickDetector.OnClickDetected += OnClick;
    }

    // ─────────────────── trial flow ───────────────────
    void StartTrial(Mode m)
    {
        if (_running) return;
        _mode = m;
        _trial = StartCoroutine(TrialRoutine());
    }

    IEnumerator TrialRoutine()
    {

        stimulusPanel.color = Color.red;
        yield return new WaitForSeconds(UnityEngine.Random.Range(waitMin, waitMax));

        stimulusPanel.color = Color.green;
        _running = true;

        _stimulusTime = Time.realtimeSinceStartup;

        const float TIMEOUT = 5f;
        for (float t = TIMEOUT; _running && t > 0f; t -= Time.deltaTime)
            yield return null;

        if (_running) Log($"MISS");
        FinishTrial();
    }

    void Update()
    {
        if (!_running || _mode != Mode.Pinch || _hands == null || !_hands.running)
            return;

        XRHand h = (pinchHand == XRNode.LeftHand) ? _hands.leftHand : _hands.rightHand;
        if (!h.isTracked) { _prevPinch = false; return; }

        bool thumbOK = h.GetJoint(XRHandJointID.ThumbTip).TryGetPose(out Pose thumb);
        bool indexOK = h.GetJoint(XRHandJointID.IndexTip).TryGetPose(out Pose index);
        if (!thumbOK || !indexOK) { _prevPinch = false; return; }

        float d = Vector3.Distance(thumb.position, index.position);
        bool pinchNow = d < pinchThresh;

        if (!_prevPinch && pinchNow)    // rising edge = pinch event
        {
            float lat = (Time.realtimeSinceStartup - _stimulusTime) * 1000f;
            Log($"{lat:F1} ms");
            FinishTrial();
        }
        _prevPinch = pinchNow;
    }

    void OnSnap()
    {
        if (!_running || _mode != Mode.Snap) return;
        float lat = (Time.realtimeSinceStartup - _stimulusTime) * 1000f;
        Log($"{lat:F1} ms");
        FinishTrial();
    }

    void OnClick()
    {
        if (!_running || _mode != Mode.Click) return;
        float lat = (Time.realtimeSinceStartup - _stimulusTime) * 1000f;
        Log($"{lat:F1} ms");
        FinishTrial();
    }

    void Log(string msg)
    {
        logText.text += $"{DateTime.Now:HH:mm:ss}  {_mode,-5}  {msg}\n";
    }

    void FinishTrial()
    {
        if (_trial != null) StopCoroutine(_trial);
        stimulusPanel.color = Color.red;
        _running = false;
        _mode = Mode.None;
        _prevPinch = false;
    }

    void OnDestroy()
    {
        if (snapPredictor) SnapPredictor.OnSnapDetected -= OnSnap;
        if (clickDetector) ClickDetector.OnClickDetected -= OnClick;
    }
}
