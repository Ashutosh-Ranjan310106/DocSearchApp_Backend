import os
import sys
import time
import threading
import subprocess
import re

# ─────────────────────────────────────────────
# Terminal helpers
# ─────────────────────────────────────────────
WIDTH = 60

def _clr(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"

def cyan(t):   return _clr("96", t)
def green(t):  return _clr("92", t)
def yellow(t): return _clr("93", t)
def red(t):    return _clr("91", t)
def bold(t):   return _clr("1", t)
def dim(t):    return _clr("2", t)

def box(lines, colour=cyan):
    inner = WIDTH - 2
    top = "┌" + "─" * inner + "┐"
    bottom = "└" + "─" * inner + "┘"

    print(colour(top))
    for line in lines:
        plain = re.sub(r"\033\[[0-9;]*m", "", line)
        pad = inner - len(plain)
        print(colour("│") + line + " " * max(0, pad) + colour("│"))
    print(colour(bottom))

def banner():
    os.system("cls" if os.name == "nt" else "clear")
    box([
        "",
        bold(cyan("  DocSearch Backend Launcher")),
        dim("  FastAPI Service"),
        "",
    ])

def status(label, value, ok=True):
    icon = green("✔") if ok else red("✖")
    col = green if ok else red
    print(f"  {icon}  {bold(label):<15} {col(value)}")

def section(title):
    print()
    print(f"  {dim('────')} {yellow(title)}")

def spinner_wait(seconds, label):
    frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    end = time.time() + seconds
    i = 0

    while time.time() < end:
        frame = frames[i % len(frames)]
        sys.stdout.write(
            f"\r  {cyan(frame)} {label}..."
        )
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1

    sys.stdout.write("\r" + " " * WIDTH + "\r")

# ─────────────────────────────────────────────
# Windows ANSI support
# ─────────────────────────────────────────────
if os.name == "nt":
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
DOTENV = os.path.join(ROOT, ".env")

PYTHON = sys.executable

# ─────────────────────────────────────────────
# Load .env
# ─────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = "8000"

if os.path.exists(DOTENV):
    with open(DOTENV, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)

            key = key.strip()
            value = value.strip()

            if key == "HOST":
                HOST = value

            elif key == "PORT":
                PORT = value

# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────
env = os.environ.copy()

env["PYTHONPATH"] = os.path.join(
    ROOT,
    "venv",
    "Lib",
    "site-packages"
)

# ─────────────────────────────────────────────
# Log streaming
# ─────────────────────────────────────────────
_ERR_PAT = re.compile(
    r"\b(error|exception|traceback|fatal|failed|critical)\b",
    re.I,
)

_WARN_PAT = re.compile(
    r"\b(warning|warn|deprecated)\b",
    re.I,
)

log_lock = threading.Lock()
stop_event = threading.Event()

def format_log(name, line):
    tag = dim(f"[{name}]")

    if _ERR_PAT.search(line):
        return f"{tag} {red(line)}"

    if _WARN_PAT.search(line):
        return f"{tag} {yellow(line)}"

    return f"{tag} {line}"

def stream_output(proc, name):
    def reader(stream):
        if stream is None:
            return

        try:
            for raw in iter(stream.readline, b""):
                if stop_event.is_set():
                    break

                line = raw.decode(
                    "utf-8",
                    errors="replace"
                ).rstrip()

                if line:
                    with log_lock:
                        print(format_log(name, line))

        except:
            pass

    threading.Thread(
        target=reader,
        args=(proc.stdout,),
        daemon=True,
    ).start()

    threading.Thread(
        target=reader,
        args=(proc.stderr,),
        daemon=True,
    ).start()

# ─────────────────────────────────────────────
# Shutdown
# ─────────────────────────────────────────────
def stop_process(proc):
    if proc and proc.poll() is None:
        print()
        print(yellow("Stopping backend..."))

        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
banner()

if os.path.exists(DOTENV):
    status(".env", "found")
else:
    status(".env", "not found", False)

status("Host", HOST)
status("Port", PORT)

backend = None

try:
    section("Starting Backend")

    uvicorn_cmd = [
        PYTHON,
        "-m",
        "uvicorn",
        "backend.knowledge_rag:app",
        "--host",
        HOST,
        "--port",
        str(PORT),
    ]

    if os.path.exists(DOTENV):
        uvicorn_cmd += [
            "--env-file",
            DOTENV
        ]

    backend = subprocess.Popen(
        uvicorn_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=ROOT,
        env=env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )

    stream_output(backend, "Backend")

    status("Backend", "running")

    spinner_wait(1, "Waiting for FastAPI")

    print()

    box([
        "",
        green("  ✔ Backend is running"),
        dim(f"  http://{HOST}:{PORT}"),
        dim("  Press Ctrl+C to stop"),
        "",
    ], colour=green)

    print()
    section("Live Logs")
    print()

    backend.wait()

except KeyboardInterrupt:
    print()
    print(yellow("Keyboard interrupt received."))

finally:
    stop_event.set()

    if backend:
        stop_process(backend)

    print()

    box([
        "",
        bold("  Backend stopped."),
        "",
    ], colour=dim)

    print()