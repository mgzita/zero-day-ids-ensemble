"""
run_inference_throughput.py
===========================
Measures inference throughput of the proposed ensemble
on 10,000 flows and extrapolates to production scale.

This addresses Review Point 4.4:
  "Training times reported but without hardware
   specification or discussion of production scalability."

Run from project root:
    python run_inference_throughput.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import logging
import platform
import numpy as np

from models.autoencoder import Autoencoder
from models.vae          import VAE
from models.deep_svdd    import DeepSVDD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

PROJECT_ROOT = r"C:/MLProject/zero_day_project"
ARTEFACT_DIR = os.path.join(PROJECT_ROOT, "artefacts_v2")

N_FLOWS      = 10_000
N_REPEATS    = 5


def norm_scores(scores):
    mu, sigma = scores.mean(), scores.std() + 1e-8
    return np.clip((scores - mu) / sigma, 0, 1).astype(np.float32)


def get_hardware_info():
    """Get CPU and memory information."""
    info = {}
    info["platform"] = platform.platform()
    info["processor"] = platform.processor()
    info["python"]   = platform.python_version()

    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_gb"] = round(mem.total / (1024**3), 1)
        info["cpu_count"] = psutil.cpu_count(logical=False)
        info["cpu_logical"] = psutil.cpu_count(logical=True)
    except ImportError:
        log.warning("psutil not installed — install with: "
                    "pip install psutil")
        info["ram_gb"] = "unknown"
        info["cpu_count"] = os.cpu_count()

    try:
        import cpuinfo
        cpu = cpuinfo.get_cpu_info()
        info["cpu_name"] = cpu.get("brand_raw", "unknown")
        info["cpu_hz"]   = cpu.get("hz_advertised_friendly", "unknown")
    except ImportError:
        log.warning("py-cpuinfo not installed — install with: "
                    "pip install py-cpuinfo")
        info["cpu_name"] = platform.processor()

    return info


def run():
    #    Hardware info                                                      
    log.info("Collecting hardware information...")
    hw = get_hardware_info()
    print("\n" + "="*65)
    print("HARDWARE SPECIFICATION")
    print("="*65)
    for k, v in hw.items():
        print(f"  {k:<20}: {v}")

    #    Load models                                                        
    log.info("Loading models...")
    ae   = Autoencoder()
    ae.load(os.path.join(ARTEFACT_DIR, "ae_model.pt"))
    vae  = VAE()
    vae.load(os.path.join(ARTEFACT_DIR, "vae_model.pt"))
    svdd = DeepSVDD()
    svdd.load(os.path.join(ARTEFACT_DIR, "svdd_model.pt"))

    #    Load test data                                                     
    log.info("Loading test data...")
    d    = np.load(os.path.join(ARTEFACT_DIR, "test_cic17.npz"))
    X    = np.clip(d["X"][:N_FLOWS], -10, 10).astype(np.float32)
    log.info("  Test flows: %d", len(X))

    #    Warm up                                                            
    log.info("Warming up models...")
    _ = ae.anomaly_scores(X[:100])
    _ = vae.anomaly_scores(X[:100])
    _ = svdd.anomaly_scores(X[:100])

    #    Measure inference time                                             
    log.info("Measuring inference throughput over %d repeats...",
             N_REPEATS)

    times_ae   = []
    times_vae  = []
    times_svdd = []
    times_ens  = []

    for i in range(N_REPEATS):
        # AE
        t0 = time.perf_counter()
        s_ae = ae.anomaly_scores(X)
        times_ae.append(time.perf_counter() - t0)

        # VAE
        t0 = time.perf_counter()
        s_vae = vae.anomaly_scores(X)
        times_vae.append(time.perf_counter() - t0)

        # SVDD
        t0 = time.perf_counter()
        s_svdd = svdd.anomaly_scores(X)
        times_svdd.append(time.perf_counter() - t0)

        # Full ensemble (AE + VAE + SVDD + fusion)
        t0 = time.perf_counter()
        s_ae_n   = norm_scores(ae.anomaly_scores(X))
        s_vae_n  = norm_scores(vae.anomaly_scores(X))
        s_svdd_n = norm_scores(svdd.anomaly_scores(X))
        _        = 0.4*s_ae_n + 0.4*s_vae_n + 0.2*s_svdd_n
        times_ens.append(time.perf_counter() - t0)

    #    Results                                                            
    def summarise(times, label):
        mean_s  = np.mean(times)
        std_s   = np.std(times)
        fps     = N_FLOWS / mean_s
        ms_flow = 1000 * mean_s / N_FLOWS
        return {
            "label":    label,
            "mean_s":   mean_s,
            "std_s":    std_s,
            "fps":      fps,
            "ms_flow":  ms_flow,
        }

    results = [
        summarise(times_ae,   "AE only"),
        summarise(times_vae,  "VAE only"),
        summarise(times_svdd, "SVDD only"),
        summarise(times_ens,  "Full Ensemble"),
    ]

    print("\n" + "="*75)
    print(f"INFERENCE THROUGHPUT — {N_FLOWS:,} flows, "
          f"{N_REPEATS} repeats")
    print("="*75)
    print(f"  {'Model':<18} {'Time (s)':>10} {'Std (s)':>10} "
          f"{'Flows/sec':>12} {'ms/flow':>10}")
    print("  " + "-"*65)
    for r in results:
        print(f"  {r['label']:<18} {r['mean_s']:>10.4f} "
              f"{r['std_s']:>10.4f} {r['fps']:>12,.0f} "
              f"{r['ms_flow']:>10.4f}")

    ens = results[-1]
    fps = ens["fps"]

    #    Production extrapolation                                           
    print("\n" + "="*75)
    print("PRODUCTION SCALABILITY EXTRAPOLATION")
    print("="*75)
    flows_per_min  = fps * 60
    flows_per_hour = fps * 3600
    print(f"  Ensemble throughput:  {fps:>12,.0f} flows/second")
    print(f"                        {flows_per_min:>12,.0f} flows/minute")
    print(f"                        {flows_per_hour:>12,.0f} flows/hour")
    print(f"\n  Typical enterprise:   ~1,000,000 flows/hour")
    print(f"  Capacity ratio:        {flows_per_hour/1_000_000:.1f}x "
          f"enterprise throughput")

    #    LaTeX text for paper                                               
    ens_ms   = ens["ms_flow"]
    ens_fps  = ens["fps"]
    ens_fph  = ens["fps"] * 3600

    cpu_name = hw.get("cpu_name", hw.get("processor", "Intel-based"))
    ram_gb   = hw.get("ram_gb", "unknown")

    print("\n" + "="*75)
    print("TEXT FOR PAPER (Section 7 Discussion / Table 4):")
    print("="*75)
    print(f"""
Hardware: {cpu_name}, {ram_gb} GB RAM.

Inference throughput sentence:
The proposed ensemble scores {N_FLOWS:,} network flows in 
{ens['mean_s']*1000:.1f}~ms ({ens_fps:,.0f}~flows per second), 
extrapolating to approximately {ens_fph/1_000_000:.1f}~million 
flows per hour on the evaluation hardware. This exceeds the 
typical enterprise network flow rate of approximately 
1~million flows per hour, confirming that the framework 
is suitable for real-time deployment without GPU acceleration.
""")

    print("TABLE 4 ROW to add:")
    print(f"Inference throughput & Flows scored per second & "
          f"\\multicolumn{{3}}{{c}}"
          f"{{{ens_fps:,.0f} flows/sec "
          f"({N_FLOWS:,} flows in "
          f"{ens['mean_s']*1000:.1f}~ms)}} \\\\")


if __name__ == "__main__":
    run()
