"""
PIXFORMWIN — Native Desktop Application
Starts the PIXFORM FastAPI backend in a background thread and
opens it in a native Edge WebView2 window (pywebview).
No browser required.
"""
import os, sys, threading, time, logging, argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pixformwin")

BASE_DIR    = Path(__file__).parent
BACKEND_DIR = BASE_DIR / "backend"
HOST        = "127.0.0.1"
PORT        = 8000
URL         = f"http://{HOST}:{PORT}"


def _resolve_device(profile: str) -> str:
    """Map a CLI profile string to a PIXFORM_DEVICE value."""
    alias = {"nvidia": "cuda", "cuda": "cuda", "mac": "mps", "mps": "mps", "cpu": "cpu"}
    if profile in alias:
        return alias[profile]
    # Fall back to saved preference written by install.ps1
    device_file = BASE_DIR / ".pixform_device"
    if device_file.exists():
        saved = device_file.read_text(encoding="ascii").strip()
        return saved if saved else "auto"
    return "auto"


def _start_backend() -> None:
    """Load backend/app.py and serve it with uvicorn (daemon thread)."""
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))

    import importlib.util, uvicorn

    spec = importlib.util.spec_from_file_location(
        "pixform_backend", BACKEND_DIR / "app.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pixform_backend"] = mod
    spec.loader.exec_module(mod)

    uvicorn.run(mod.app, host=HOST, port=PORT, log_level="warning")


def _wait_for_backend(timeout: int = 120) -> bool:
    """Poll /health until the backend is responding or timeout expires."""
    import urllib.request, urllib.error

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{URL}/health", timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PIXFORMWIN — Image to 3D native desktop app"
    )
    parser.add_argument(
        "profile",
        nargs="?",
        default="",
        help="Device profile: auto | nvidia | cuda | cpu | mac | mps",
    )
    args = parser.parse_args()

    device = _resolve_device(args.profile)
    if device and device != "auto":
        os.environ["PIXFORM_DEVICE"] = device
        logger.info(f"Device profile: {device}")

    # ── Launch backend ────────────────────────────────────────────────────────
    t = threading.Thread(target=_start_backend, daemon=True, name="pixform-backend")
    t.start()
    logger.info("Starting PIXFORMWIN backend…")

    if not _wait_for_backend():
        logger.error("Backend did not start within 120 s — check the log above.")
        sys.exit(1)
    logger.info("Backend ready — opening window")

    # ── Open native window ────────────────────────────────────────────────────
    try:
        import webview
    except ImportError:
        logger.error("pywebview is not installed. Run install.ps1 first.")
        sys.exit(1)

    window = webview.create_window(   # noqa: F841
        "PIXFORMWIN — Image to 3D",
        URL,
        width=1280,
        height=820,
        min_size=(900, 640),
        resizable=True,
        text_select=False,
    )
    webview.start(debug=False)


if __name__ == "__main__":
    main()
