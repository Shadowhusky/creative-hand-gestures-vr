// Assets/Scripts/Editor/ClickFingerprint.cs
using UnityEngine;

[CreateAssetMenu(fileName = "ClickFingerprint", menuName = "Audio/Click Fingerprint")]
public class ClickFingerprint : ScriptableObject
{
  public int fftSize;          // e.g. 1024
  public float[] freqs;            // top-K peak bin indices  (K ≤ 8 keeps the file ~1 kB)
  public float[] mags;             // their magnitudes (normalised 0-1)
}
