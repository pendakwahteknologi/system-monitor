#!/usr/bin/env python3
"""
ML/AI Training Monitor
Optimised for ASUS TUF Z890 + Intel Ultra 7 265K + AMD Radeon AI PRO R9700
Designed for vertical/narrow terminal view (side-by-side display)
"""

import psutil
import subprocess
import time
import os
from datetime import datetime
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.rule import Rule
from collections import deque
import re

console = Console()

# ── Sparkline characters (9 levels) ─────────────────────────────────
SPARK = " ▁▂▃▄▅▆▇█"

# ── Colour palette ───────────────────────────────────────────────────
C_HEADER   = "bold white on blue"
C_ACCENT   = "bold cyan"
C_VALUE    = "bold white"
C_DIM      = "dim"
C_OK       = "bold green"
C_WARN     = "bold yellow"
C_CRIT     = "bold red"
C_GPU      = "cyan"
C_CPU      = "green"
C_MEM      = "magenta"
C_DISK     = "blue"
C_NET      = "yellow"


class SystemMonitor:
    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.prev_net_io = psutil.net_io_counters()
        self.prev_disk_io = psutil.disk_io_counters()
        self.prev_time = time.time()

        # NVMe per-disk tracking
        perdisk = psutil.disk_io_counters(perdisk=True)
        self.nvme_dev = None
        for name in perdisk:
            if 'nvme' in name and 'p' not in name.split('nvme')[1]:
                self.nvme_dev = name
                break
        self.prev_nvme_io = perdisk.get(self.nvme_dev) if self.nvme_dev else None

        # History buffers for sparkline graphs (120 samples = 60s at 2Hz)
        self.gpu_history = deque(maxlen=120)
        self.cpu_history = deque(maxlen=120)
        self.power_history = deque(maxlen=120)
        self.temp_history = deque(maxlen=120)
        self.vram_history = deque(maxlen=120)
        self.cpu_temp_history = deque(maxlen=120)

    # ── Static info (cached once) ────────────────────────────────────

    def get_static_info(self):
        if hasattr(self, '_static'):
            return self._static

        s = {
            'hostname': 'Unknown', 'os': 'Unknown', 'kernel': 'Unknown',
            'rocm_version': 'Unknown', 'ip': 'N/A',
            'board_vendor': 'N/A', 'board_name': 'N/A',
            'bios_vendor': 'N/A', 'bios_version': 'N/A', 'bios_date': 'N/A',
            'docker_version': 'N/A', 'python_version': 'N/A',
            'cpu_model': 'N/A', 'cpu_max_mhz': 0, 'cpu_cache': 'N/A',
            'gpu_driver': 'N/A', 'gpu_vbios': 'N/A',
        }

        def _read(path):
            try:
                with open(path) as f:
                    return f.read().strip()
            except Exception:
                return None

        def _cmd(*args, timeout=1):
            try:
                r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
                return r.stdout.strip()
            except Exception:
                return ''

        # Host / OS
        s['hostname'] = _cmd('hostname')
        for line in (_read('/etc/os-release') or '').split('\n'):
            if line.startswith('PRETTY_NAME'):
                s['os'] = line.split('=')[1].strip().strip('"')
                break
        s['kernel'] = _cmd('uname', '-r')

        # Motherboard / BIOS
        s['board_vendor'] = _read('/sys/devices/virtual/dmi/id/board_vendor') or 'N/A'
        s['board_name'] = _read('/sys/devices/virtual/dmi/id/board_name') or 'N/A'
        s['bios_vendor'] = _read('/sys/devices/virtual/dmi/id/bios_vendor') or 'N/A'
        s['bios_version'] = _read('/sys/devices/virtual/dmi/id/bios_version') or 'N/A'
        s['bios_date'] = _read('/sys/devices/virtual/dmi/id/bios_date') or 'N/A'

        # ROCm
        rocm_ver = _read('/opt/rocm/.info/version')
        if rocm_ver:
            s['rocm_version'] = rocm_ver

        # Network IP
        ip_out = _cmd('hostname', '-I')
        if ip_out:
            s['ip'] = ip_out.split()[0]

        # Docker
        dv = _cmd('docker', '--version')
        m = re.search(r'(\d+\.\d+\.\d+)', dv)
        if m:
            s['docker_version'] = m.group(1)

        # Python
        pv = _cmd('python3', '--version')
        m = re.search(r'(\d+\.\d+\.\d+)', pv)
        if m:
            s['python_version'] = m.group(1)

        # CPU details from lscpu
        lscpu = _cmd('lscpu')
        m = re.search(r'CPU max MHz:\s*([\d.]+)', lscpu)
        if m:
            s['cpu_max_mhz'] = int(float(m.group(1)))
        m = re.search(r'Model name:\s*(.+)', lscpu)
        if m:
            s['cpu_model'] = m.group(1).strip()
        cache = _read('/proc/cpuinfo')
        if cache:
            m = re.search(r'cache size\s*:\s*(\d+)', cache)
            if m:
                s['cpu_cache'] = f"{int(m.group(1)) // 1024}MB"

        # GPU static info
        gpu_all = _cmd('rocm-smi', '--showall', timeout=3)
        m = re.search(r'Driver version:\s*(.+)', gpu_all)
        if m:
            s['gpu_driver'] = m.group(1).strip()
        m = re.search(r'VBIOS version:\s*(.+)', gpu_all)
        if m:
            s['gpu_vbios'] = m.group(1).strip()

        self._static = s
        return s

    # ── Dynamic metrics ──────────────────────────────────────────────

    def get_uptime(self):
        try:
            up = time.time() - psutil.boot_time()
            d, rem = divmod(int(up), 86400)
            h, rem = divmod(rem, 3600)
            m = rem // 60
            if d > 0:
                return f"{d}d {h}h {m}m"
            elif h > 0:
                return f"{h}h {m}m"
            return f"{m}m"
        except Exception:
            return 'N/A'

    def get_load(self):
        load = psutil.getloadavg()
        procs = len(psutil.pids())
        return {'load_1': load[0], 'load_5': load[1], 'load_15': load[2], 'procs': procs}

    def get_cpu_info(self):
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq()
        per_core = psutil.cpu_percent(percpu=True)
        self.cpu_history.append(cpu_percent)
        return {
            'percent': cpu_percent,
            'freq': cpu_freq.current if cpu_freq else 0,
            'cores': self.cpu_count,
            'per_core': per_core
        }

    def get_cpu_temp(self):
        """Get CPU package temp via sensors."""
        try:
            r = subprocess.run(['sensors', 'coretemp-isa-0000'],
                               capture_output=True, text=True, timeout=1)
            m = re.search(r'Package id 0:\s*\+([\d.]+)', r.stdout)
            temp = float(m.group(1)) if m else 0
            self.cpu_temp_history.append(temp)
            return temp
        except Exception:
            return 0

    def get_nvme_temp(self):
        """Get NVMe temp via sensors."""
        try:
            r = subprocess.run(['sensors', 'nvme-pci-0100'],
                               capture_output=True, text=True, timeout=1)
            m = re.search(r'Composite:\s*\+([\d.]+)', r.stdout)
            return float(m.group(1)) if m else 0
        except Exception:
            return 0

    def get_memory_info(self):
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            'ram_used': mem.used / (1024**3),
            'ram_total': mem.total / (1024**3),
            'ram_percent': mem.percent,
            'swap_used': swap.used / (1024**3),
            'swap_total': swap.total / (1024**3),
            'swap_percent': swap.percent
        }

    def get_disk_info(self):
        disk = psutil.disk_usage('/')
        curr_disk_io = psutil.disk_io_counters()
        curr_time = time.time()
        time_delta = max(curr_time - self.prev_time, 0.01)

        read_rate = (curr_disk_io.read_bytes - self.prev_disk_io.read_bytes) / time_delta / (1024**2)
        write_rate = (curr_disk_io.write_bytes - self.prev_disk_io.write_bytes) / time_delta / (1024**2)

        self.prev_disk_io = curr_disk_io
        self.prev_time = curr_time

        result = {
            'used': disk.used / (1024**3),
            'total': disk.total / (1024**3),
            'percent': disk.percent,
            'nvme_read_rate': 0, 'nvme_write_rate': 0,
            'nvme_total_read': 0, 'nvme_total_write': 0,
        }

        if self.nvme_dev:
            perdisk = psutil.disk_io_counters(perdisk=True)
            nvme_io = perdisk.get(self.nvme_dev)
            if nvme_io and self.prev_nvme_io:
                result['nvme_read_rate'] = (nvme_io.read_bytes - self.prev_nvme_io.read_bytes) / time_delta / (1024**2)
                result['nvme_write_rate'] = (nvme_io.write_bytes - self.prev_nvme_io.write_bytes) / time_delta / (1024**2)
                result['nvme_total_read'] = nvme_io.read_bytes / (1024**3)
                result['nvme_total_write'] = nvme_io.write_bytes / (1024**3)
            self.prev_nvme_io = nvme_io

        return result

    def get_network_info(self):
        curr_net_io = psutil.net_io_counters()
        curr_time = time.time()
        time_delta = max(curr_time - self.prev_time, 0.01)
        recv = (curr_net_io.bytes_recv - self.prev_net_io.bytes_recv) / time_delta / (1024**2)
        sent = (curr_net_io.bytes_sent - self.prev_net_io.bytes_sent) / time_delta / (1024**2)
        self.prev_net_io = curr_net_io
        return {'recv_rate': recv, 'sent_rate': sent}

    def get_docker_containers(self):
        try:
            r = subprocess.run(
                ['docker', 'ps', '--format', '{{.Names}}\t{{.Image}}\t{{.Status}}'],
                capture_output=True, text=True, timeout=2
            )
            containers = []
            for line in r.stdout.strip().split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        containers.append({
                            'name': parts[0], 'image': parts[1], 'status': parts[2]
                        })
            return containers
        except Exception:
            return []

    def get_gpu_info(self):
        gpu = {
            'name': 'ASUS TURBO Radeon AI PRO R9700', 'gfx': 'gfx1201',
            'temp_edge': 0, 'temp_junction': 0, 'temp_memory': 0,
            'power': 0, 'power_max': 300, 'gpu_use': 0,
            'vram_used': 0, 'vram_total': 32,
            'fan_percent': 0, 'fan_rpm': 0,
            'sclk': 0, 'mclk': 0, 'pcie': 'N/A', 'perf_level': 'N/A'
        }
        try:
            r = subprocess.run(['rocm-smi', '--showuse'], capture_output=True, text=True, timeout=1)
            m = re.search(r'GPU use \(%\):\s*(\d+)', r.stdout)
            if m: gpu['gpu_use'] = int(m.group(1))

            r = subprocess.run(['rocm-smi', '--showtemp'], capture_output=True, text=True, timeout=1)
            for key, pat in [('temp_edge', 'edge'), ('temp_junction', 'junction'), ('temp_memory', 'memory')]:
                m = re.search(rf'Sensor {pat}\).*?:\s*(\d+\.?\d*)', r.stdout, re.IGNORECASE)
                if m: gpu[key] = float(m.group(1))

            r = subprocess.run(['rocm-smi', '--showpower'], capture_output=True, text=True, timeout=1)
            m = re.search(r'Average Graphics Package Power.*?:\s*(\d+\.?\d*)', r.stdout)
            if m: gpu['power'] = float(m.group(1))

            r = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'], capture_output=True, text=True, timeout=1)
            m1 = re.search(r'VRAM Total Memory.*?:\s*(\d+)', r.stdout)
            m2 = re.search(r'VRAM Total Used Memory.*?:\s*(\d+)', r.stdout)
            if m1: gpu['vram_total'] = int(m1.group(1)) / (1024**3)
            if m2: gpu['vram_used'] = int(m2.group(1)) / (1024**3)

            r = subprocess.run(['rocm-smi', '--showfan'], capture_output=True, text=True, timeout=1)
            m = re.search(r'Fan Level:.*?\((\d+)%\)', r.stdout)
            if m: gpu['fan_percent'] = int(m.group(1))
            m = re.search(r'Fan RPM:\s*(\d+)', r.stdout)
            if m: gpu['fan_rpm'] = int(m.group(1))

            r = subprocess.run(['rocm-smi', '--showclocks'], capture_output=True, text=True, timeout=1)
            m = re.search(r'sclk clock level:.*?\((\d+)Mhz\)', r.stdout)
            if m: gpu['sclk'] = int(m.group(1))
            m = re.search(r'mclk clock level:.*?\((\d+)Mhz\)', r.stdout)
            if m: gpu['mclk'] = int(m.group(1))
            m = re.search(r'pcie clock level:.*?\(([\d.]+GT/s x\d+)\)', r.stdout)
            if m: gpu['pcie'] = m.group(1)

            r = subprocess.run(['rocm-smi', '--showproductname'], capture_output=True, text=True, timeout=1)
            m = re.search(r'GFX Version:\s*(.+)', r.stdout)
            if m: gpu['gfx'] = m.group(1).strip()

            r = subprocess.run(['rocm-smi', '--showperflevel'], capture_output=True, text=True, timeout=1)
            m = re.search(r'Performance Level:\s*(.+)', r.stdout)
            if m: gpu['perf_level'] = m.group(1).strip()
        except Exception:
            pass

        self.gpu_history.append(gpu['gpu_use'])
        self.power_history.append(gpu['power'])
        self.vram_history.append(
            (gpu['vram_used'] / gpu['vram_total'] * 100) if gpu['vram_total'] > 0 else 0
        )
        self.temp_history.append(gpu['temp_junction'])
        return gpu


# ── Rendering helpers ────────────────────────────────────────────────

def iw():
    """Inner width of a panel."""
    return max(console.width - 4, 30)


def bar(value, total, width, color="cyan"):
    """Full-width block progress bar."""
    if total <= 0:
        total = 1
    pct = min(value / total, 1.0)
    filled = int(width * pct)
    t = Text()
    t.append("█" * filled, style=f"bold {color}")
    t.append("░" * (width - filled), style="dim")
    return t


def labeled_bar(label, value, total, unit, width, color="cyan"):
    """Label row + bar row."""
    val_str = f"{value:.1f}/{total:.0f}{unit}" if isinstance(value, float) else f"{value}/{total}{unit}"
    pct = min(value / total * 100, 100) if total > 0 else 0
    pct_str = f" {pct:5.1f}%"

    line = Text()
    pad = width - len(label) - len(val_str) - len(pct_str)
    line.append(label, style=f"bold {color}")
    line.append(" " * max(pad, 1))
    line.append(val_str, style=C_VALUE)
    line.append(pct_str, style=C_WARN)

    return line, bar(value, total, width, color)


def sparkline(history, width, color="cyan"):
    """Sparkline graph from history data."""
    if not history:
        return Text("Collecting data...", style="dim italic")
    data = list(history)
    max_val = max(max(data), 1)
    if len(data) > width:
        step = len(data) / width
        data = [data[int(i * step)] for i in range(width)]
    elif len(data) < width:
        data = [0] * (width - len(data)) + data
    t = Text()
    for v in data:
        level = max(0, min(8, int(v / max_val * 8)))
        t.append(SPARK[level], style=f"bold {color}")
    return t


def info_row(label, value, width, lc=C_VALUE, vc=C_DIM):
    """Label left, value right."""
    t = Text()
    pad = width - len(label) - len(str(value))
    t.append(label, style=lc)
    t.append(" " * max(pad, 1))
    t.append(str(value), style=vc)
    return t


def separator(width, style="dim"):
    """Thin dotted separator line."""
    t = Text()
    t.append("─" * width, style=style)
    return t


def temp_color(temp, warn=80, crit=95):
    """Return style based on temperature."""
    if temp >= crit:
        return C_CRIT
    elif temp >= warn:
        return C_WARN
    elif temp >= 60:
        return "yellow"
    return C_OK


def spark_stats(label, history, unit, width, color):
    """Stats row + sparkline for a metric."""
    vals = list(history) or [0]
    rows = []
    t = Text()
    t.append(f"  {label}  ", style=f"bold {color}")
    t.append("min:", style="dim")
    t.append(f"{min(vals):4.0f}{unit}", style=color)
    t.append("  avg:", style="dim")
    t.append(f"{sum(vals)/len(vals):4.0f}{unit}", style=color)
    t.append("  max:", style="dim")
    t.append(f"{max(vals):4.0f}{unit}", style=color)
    t.append("  now:", style="dim")
    t.append(f"{vals[-1]:4.0f}{unit}", style=f"bold {color}")
    rows.append(t)
    rows.append(sparkline(history, width, color))
    return rows


# ── Main display ─────────────────────────────────────────────────────

def create_display(monitor):
    w = iw()

    static = monitor.get_static_info()
    uptime = monitor.get_uptime()
    load = monitor.get_load()
    cpu = monitor.get_cpu_info()
    cpu_temp = monitor.get_cpu_temp()
    nvme_temp = monitor.get_nvme_temp()
    mem = monitor.get_memory_info()
    disk = monitor.get_disk_info()
    net = monitor.get_network_info()
    gpu = monitor.get_gpu_info()
    containers = monitor.get_docker_containers()

    panels = []

    # ── HEADER ────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%H:%M:%S")
    dt = datetime.now().strftime("%a %Y-%m-%d")
    h = Text()
    h.append("  ML/AI TRAINING MONITOR", style="bold white")
    pad = w - 25 - len(dt) - len(ts) - 3
    h.append(" " * max(pad, 1))
    h.append(dt, style="dim white")
    h.append(" | ", style="dim")
    h.append(ts, style="bold green")
    panels.append(Panel(h, border_style="bright_blue", style="on grey3"))

    # ── SYSTEM + MOTHERBOARD ──────────────────────────────────────────
    rows = []
    rows.append(info_row("  Hostname", static['hostname'], w, C_VALUE, C_ACCENT))
    rows.append(info_row("  OS", static['os'], w, C_VALUE, C_ACCENT))
    rows.append(info_row("  Kernel", static['kernel'], w, C_VALUE, C_ACCENT))
    rows.append(separator(w))
    rows.append(info_row("  Motherboard", static['board_name'], w, C_VALUE, C_ACCENT))
    rows.append(info_row("  BIOS", f"{static['bios_vendor']} v{static['bios_version']} ({static['bios_date']})", w, C_VALUE, C_DIM))
    rows.append(separator(w))
    rows.append(info_row("  ROCm", static['rocm_version'], w, C_VALUE, C_ACCENT))
    rows.append(info_row("  Docker", static['docker_version'], w, C_VALUE, C_ACCENT))
    rows.append(info_row("  Python", static['python_version'], w, C_VALUE, C_ACCENT))
    rows.append(separator(w))
    rows.append(info_row("  IP Address", static['ip'], w, C_VALUE, C_ACCENT))
    la = f"{load['load_1']:.2f}  {load['load_5']:.2f}  {load['load_15']:.2f}"
    rows.append(info_row("  Load (1/5/15)", la, w, C_VALUE, C_DIM))
    rows.append(info_row("  Processes", str(load['procs']), w, C_VALUE, C_DIM))
    rows.append(info_row("  Uptime", uptime, w, C_VALUE, C_ACCENT))

    panels.append(Panel(
        Text("\n").join(rows),
        title="[bold white] System ",
        subtitle=f"[dim] {static['board_vendor']} {static['board_name']} ",
        border_style="white"
    ))

    # ── GPU ────────────────────────────────────────────────────────────
    g = []

    # Title
    title = Text()
    title.append(f"  {gpu['name']}", style="bold cyan")
    gfx = f"[{gpu['gfx']}]"
    pad = w - len(f"  {gpu['name']}") - len(gfx)
    title.append(" " * max(pad, 1))
    title.append(gfx, style="bold yellow")
    g.append(title)
    g.append(separator(w, "dim cyan"))

    # Bars
    l1, b1 = labeled_bar("  Compute", gpu['gpu_use'], 100, "%", w, "green")
    g.extend([l1, b1])

    l2, b2 = labeled_bar("  VRAM", gpu['vram_used'], gpu['vram_total'], "GB", w, "magenta")
    g.extend([l2, b2])

    l3, b3 = labeled_bar("  Power", gpu['power'], gpu['power_max'], "W", w, "yellow")
    g.extend([l3, b3])

    g.append(separator(w, "dim cyan"))

    # Temperature with colour-coded junction
    temp_row = Text()
    temp_row.append("  Temp", style=C_VALUE)
    tv = f"Edge {gpu['temp_edge']:.0f}°C | Junction {gpu['temp_junction']:.0f}°C | Mem {gpu['temp_memory']:.0f}°C"
    pad = w - 6 - len(tv)
    temp_row.append(" " * max(pad, 1))
    temp_row.append(f"Edge {gpu['temp_edge']:.0f}°C", style=temp_color(gpu['temp_edge']))
    temp_row.append(" | ", style="dim")
    temp_row.append(f"Junction {gpu['temp_junction']:.0f}°C", style=temp_color(gpu['temp_junction'], 90, 100))
    temp_row.append(" | ", style="dim")
    temp_row.append(f"Mem {gpu['temp_memory']:.0f}°C", style=temp_color(gpu['temp_memory']))
    g.append(temp_row)

    g.append(info_row("  Fan", f"{gpu['fan_percent']}% ({gpu['fan_rpm']} RPM)", w, C_VALUE, C_VALUE))
    g.append(info_row("  Clocks", f"GPU {gpu['sclk']}MHz | Mem {gpu['mclk']}MHz", w, C_VALUE, C_VALUE))
    g.append(separator(w, "dim cyan"))
    g.append(info_row("  PCIe", gpu['pcie'], w, C_VALUE, C_DIM))
    g.append(info_row("  Driver", f"amdgpu {static['gpu_driver']}", w, C_VALUE, C_DIM))
    g.append(info_row("  VBIOS", static['gpu_vbios'], w, C_VALUE, C_DIM))
    g.append(info_row("  Perf Level", gpu['perf_level'], w, C_VALUE, C_DIM))

    panels.append(Panel(Text("\n").join(g), title="[bold cyan] GPU ", border_style="cyan"))

    # ── LIVE GRAPHS ───────────────────────────────────────────────────
    spark = []
    spark.extend(spark_stats("GPU %  ", monitor.gpu_history, "%", w, "green"))
    spark.extend(spark_stats("Power  ", monitor.power_history, "W", w, "yellow"))
    spark.extend(spark_stats("Temp   ", monitor.temp_history, "°", w, "red"))
    spark.extend(spark_stats("VRAM % ", monitor.vram_history, "%", w, "magenta"))

    panels.append(Panel(
        Text("\n").join(spark),
        title="[bold green] Live Graphs ",
        subtitle="[dim] 60s rolling history ",
        border_style="green"
    ))

    # ── CPU ────────────────────────────────────────────────────────────
    c = []
    l_cpu, b_cpu = labeled_bar("  Usage", cpu['percent'], 100, "%", w, "green")
    c.extend([l_cpu, b_cpu])

    # Per-core with separators
    cores = cpu.get('per_core', [])
    if cores:
        num_cores = len(cores)
        sep_count = num_cores - 1
        bar_space = w - 2 - sep_count
        per = max(bar_space // num_cores, 1)

        core_line = Text()
        core_line.append("  ", style=C_VALUE)
        for i, pct in enumerate(cores):
            level = max(0, min(8, int(pct / 100 * 8)))
            col = "green" if pct < 50 else ("yellow" if pct < 80 else "red")
            core_line.append(SPARK[level] * per, style=f"bold {col}")
            if i < num_cores - 1:
                core_line.append("│", style="dim")
        c.append(core_line)

        num_line = Text()
        num_line.append("  ", style="dim")
        for i in range(num_cores):
            num_line.append(str(i).center(per), style="dim")
            if i < num_cores - 1:
                num_line.append(" ", style="dim")
        c.append(num_line)

    c.append(separator(w, "dim green"))

    # CPU details
    cpu_temp_str = f"{cpu_temp:.0f}°C"
    cpu_temp_style = temp_color(cpu_temp, 75, 95)
    ct_row = Text()
    ct_row.append("  CPU Temp", style=C_VALUE)
    pad = w - 10 - len(cpu_temp_str)
    ct_row.append(" " * max(pad, 1))
    ct_row.append(cpu_temp_str, style=cpu_temp_style)
    c.append(ct_row)

    c.append(info_row("  Frequency", f"{cpu['freq']:.0f}MHz (max {static['cpu_max_mhz']}MHz)", w, C_VALUE, C_DIM))
    c.append(info_row("  Topology", f"{cpu['cores']} cores | L3 {static['cpu_cache']}", w, C_VALUE, C_DIM))

    panels.append(Panel(
        Text("\n").join(c),
        title=f"[bold green] CPU - {static['cpu_model']} ",
        border_style="green"
    ))

    # ── MEMORY ────────────────────────────────────────────────────────
    m_rows = []
    l_ram, b_ram = labeled_bar("  RAM", mem['ram_used'], mem['ram_total'], "GB", w, "magenta")
    m_rows.extend([l_ram, b_ram])
    l_swp, b_swp = labeled_bar("  Swap", mem['swap_used'], mem['swap_total'], "GB", w, "blue")
    m_rows.extend([l_swp, b_swp])
    m_rows.append(separator(w, "dim magenta"))
    m_rows.append(info_row("  Config", "2x 24GB DDR5-8200 @ 6400 MT/s", w, C_VALUE, C_DIM))
    m_rows.append(info_row("  Module", "G.Skill F5-8200C4052G24G", w, C_VALUE, C_DIM))

    panels.append(Panel(Text("\n").join(m_rows), title="[bold magenta] Memory ", border_style="magenta"))

    # ── STORAGE + NETWORK ─────────────────────────────────────────────
    dn = []
    l_d, b_d = labeled_bar("  Storage", disk['used'], disk['total'], "GB", w, "blue")
    dn.extend([l_d, b_d])

    nvme_temp_str = f"{nvme_temp:.0f}°C"
    nvme_temp_style = temp_color(nvme_temp, 70, 80)
    nt_row = Text()
    nt_row.append("  NVMe I/O", style=C_VALUE)
    io_val = f"R {disk['nvme_read_rate']:.1f} MB/s | W {disk['nvme_write_rate']:.1f} MB/s | {nvme_temp_str}"
    pad = w - 10 - len(io_val)
    nt_row.append(" " * max(pad, 1))
    nt_row.append(f"R {disk['nvme_read_rate']:.1f} MB/s", style=C_VALUE)
    nt_row.append(" | ", style="dim")
    nt_row.append(f"W {disk['nvme_write_rate']:.1f} MB/s", style=C_VALUE)
    nt_row.append(" | ", style="dim")
    nt_row.append(nvme_temp_str, style=nvme_temp_style)
    dn.append(nt_row)

    dn.append(info_row("  Lifetime I/O", f"Read {disk['nvme_total_read']:.2f} GB | Written {disk['nvme_total_write']:.2f} GB", w, C_VALUE, C_DIM))
    dn.append(separator(w, "dim blue"))
    dn.append(info_row("  Network", f"Down {net['recv_rate']:.2f} MB/s | Up {net['sent_rate']:.2f} MB/s", w, C_VALUE, C_VALUE))

    panels.append(Panel(
        Text("\n").join(dn),
        title="[bold blue] WDC PC SN810 NVMe 512GB & Network ",
        border_style="blue"
    ))

    # ── DOCKER ────────────────────────────────────────────────────────
    if containers:
        dk = []
        for ctr in containers[:4]:
            status_color = C_OK if "Up" in ctr['status'] else C_DIM
            row = Text()
            row.append(f"  {ctr['name'][:20]}", style=C_VALUE)
            status_val = f"{ctr['status']}"
            pad = w - len(f"  {ctr['name'][:20]}") - len(status_val)
            row.append(" " * max(pad, 1))
            row.append(status_val, style=status_color)
            dk.append(row)

            img_row = Text()
            # Truncate image name to fit
            img = ctr['image']
            if len(img) > w - 4:
                img = "..." + img[-(w - 7):]
            img_row.append(f"    {img}", style=C_DIM)
            dk.append(img_row)

        panels.append(Panel(
            Text("\n").join(dk),
            title="[bold white] Docker Containers ",
            border_style="white"
        ))

    # ── Stack ─────────────────────────────────────────────────────────
    output = Table.grid(expand=True)
    output.add_column()
    for p in panels:
        output.add_row(p)
    return output


def main():
    console.print("\n[bold cyan]  Starting ML/AI Training Monitor...[/bold cyan]")
    console.print("[dim]  Optimised for ASUS TUF Z890 + AMD Radeon AI PRO R9700 + ROCm 7.2[/dim]")
    console.print("[dim]  Press Ctrl+C to exit[/dim]\n")
    time.sleep(0.5)

    monitor = SystemMonitor()

    try:
        with Live(create_display(monitor), refresh_per_second=2, console=console, screen=True) as live:
            while True:
                time.sleep(0.5)
                live.update(create_display(monitor))
    except KeyboardInterrupt:
        console.print("\n[bold green]  Monitor stopped[/bold green]")


if __name__ == "__main__":
    main()
