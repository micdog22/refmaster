
import os, io, base64, typer
import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
from scipy.signal import savgol_filter, fftconvolve
import matplotlib.pyplot as plt
from jinja2 import Template

app = typer.Typer(add_completion=False)

def lufs_normalize(y, sr, target=-14.0):
    meter = pyln.Meter(sr)
    loud = meter.integrated_loudness(y.astype(np.float64))
    gain = target - loud
    factor = 10 ** (gain/20)
    return (y * factor).astype(np.float32), loud, gain

def avg_spectrum(y, sr, n_fft=4096, hop=1024):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))**2
    spec = np.mean(S, axis=1) + 1e-9
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return freqs, spec

def design_eq_curve(freqs, spec_src, spec_ref, smooth=31):
    curve = np.log10(spec_ref) - np.log10(spec_src)
    k = min(len(curve)-1, smooth if smooth%2==1 else smooth+1)
    if k >= 5:
        curve_s = savgol_filter(curve, k, 3)
    else:
        curve_s = curve
    return curve_s

def apply_eq(y, sr, curve, freqs):
    # FFT-based filtering using frequency-domain multiplication
    n = 1
    while n < len(y)*2:
        n *= 2
    Y = np.fft.rfft(y, n=n)
    F = np.interp(np.fft.rfftfreq(n, 1/sr), freqs, 10**(curve))
    Y2 = Y * F
    y2 = np.fft.irfft(Y2, n=n)[:len(y)]
    return y2.astype(np.float32)

def plot_curve(curve):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(curve)
    bio = io.BytesIO()
    fig.savefig(bio, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    return base64.b64encode(bio.getvalue()).decode("ascii")

@app.command()
def analyze(track: str, report: bool = typer.Option(False)):
    y, sr = librosa.load(track, sr=None, mono=True)
    freqs, spec = avg_spectrum(y, sr)
    y_n, lufs, gain = lufs_normalize(y, sr, -14.0)
    print(f"LUFS: {lufs:.2f}, ganho para -14 LUFS: {gain:.2f} dB")
    if report:
        png = plot_curve(np.log10(spec))
        os.makedirs("reports", exist_ok=True)
        out = os.path.join("reports", os.path.splitext(os.path.basename(track))[0] + "_refmaster.html")
        html = f"""<!doctype html><html><head><meta charset="utf-8"><title>RefMaster</title></head>
<body><h1>Relatório RefMaster</h1>
<p>Track: {os.path.basename(track)}</p>
<p>LUFS: {lufs:.2f}</p>
<img src="data:image/png;base64,{png}">
</body></html>"""
        with open(out,"w",encoding="utf-8") as f: f.write(html)
        print(f"Relatório: {out}")

@app.command()
def match(track: str, reference: str, lufs: float = -14.0, out: str = "matched.wav", report: bool = False):
    y, sr = librosa.load(track, sr=None, mono=True)
    r, sr2 = librosa.load(reference, sr=sr, mono=True)
    freqs, spec_src = avg_spectrum(y, sr)
    _, spec_ref = avg_spectrum(r, sr)
    curve = design_eq_curve(freqs, spec_src, spec_ref, smooth=51)
    y_eq = apply_eq(y, sr, curve, freqs)
    y_out, lufs_in, gain = lufs_normalize(y_eq, sr, target=lufs)
    sf.write(out, y_out, sr)
    print(f"Arquivo gerado: {out} | LUFS antes: {lufs_in:.2f} | alvo: {lufs:.2f}")
    if report:
        png = plot_curve(curve)
        os.makedirs("reports", exist_ok=True)
        rep = os.path.join("reports", os.path.splitext(os.path.basename(out))[0] + "_report.html")
        html = f"""<!doctype html><html><head><meta charset="utf-8"><title>RefMaster</title></head>
<body><h1>Relatório RefMaster</h1>
<p>Track: {os.path.basename(track)}</p>
<p>Reference: {os.path.basename(reference)}</p>
<p>Curva de EQ (log10 ganho) abaixo:</p>
<img src="data:image/png;base64,{png}">
</body></html>"""
        with open(rep,"w",encoding="utf-8") as f: f.write(html)
        print(f"Relatório: {rep}")

if __name__ == "__main__":
    app()
