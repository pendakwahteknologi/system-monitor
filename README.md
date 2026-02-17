# System Monitor

Real-time terminal system monitor for ML/AI training workloads on Ubuntu 24.04.4, optimised for:

- ASUS TURBO Radeon™ AI PRO R9700 (ROCm) - https://www.asus.com/motherboards-components/graphics-cards/turbo/turbo-ai-pro-r9700-32g/
- Intel Core Ultra 7 265K
- Narrow vertical terminal layouts (side-by-side with training logs)

## What This Monitor Shows

- GPU utilisation, VRAM usage, power, temperatures, fan, clocks, PCIe link, performance level
- CPU utilisation, per-core activity view, frequency, package temperature
- RAM and swap utilisation
- NVMe storage usage, read/write throughput, NVMe temperature
- Network throughput
- Host and platform metadata (OS, kernel, board, BIOS, ROCm, Docker, Python, uptime, load)
- Up to 4 running Docker containers

## Repository Contents

- `system_monitor.py`: Main monitoring application (Rich TUI)
- `setup_monitor.sh`: Dependency install helper
- `monitor_requirements.txt`: Python dependencies

## Platform and Assumptions

This project is intended for Ubuntu Linux and a specific hardware/software setup.

- Tested target: Ubuntu Server 24.04.4
- GPU tooling: `rocm-smi` must be installed and available in `PATH`
- Sensor tooling: `sensors` (from `lm-sensors`) is used for CPU/NVMe temperatures
- Docker panel: optional; shown only when Docker is installed and containers are running

Important implementation assumptions in `system_monitor.py`:

- GPU model name and some labels are tuned for R9700-class ROCm systems.
- CPU and NVMe temperature parsing uses fixed sensor identifiers:
  - `coretemp-isa-0000`
  - `nvme-pci-0100`

If your sensor identifiers differ, temperature fields may show `0`.

## Requirements

Install these system dependencies first:

```bash
sudo apt update
sudo apt install -y python3 python3-pip lm-sensors
```

Optional but recommended:

```bash
sudo apt install -y tmux
```

If your workflow uses Docker:

```bash
sudo apt install -y docker.io
```

## Installation

### Option 1: Setup Script

```bash
bash setup_monitor.sh
```

### Option 2: Manual

```bash
pip3 install --user -r monitor_requirements.txt
```

## Usage

Run from this repository directory:

```bash
python3 system_monitor.py
```

If you want to run with `./system_monitor.py`, make it executable first:

```bash
chmod +x system_monitor.py
./system_monitor.py
```

## tmux Side-by-Side Workflow

Split your current tmux window and run monitor on the right:

```bash
tmux split-window -h 'python3 system_monitor.py'
```

Start a new tmux session with monitor side-by-side:

```bash
tmux new-session -s ml-train \; split-window -h 'python3 system_monitor.py' \; select-pane -L
```

## Running During Training

Example with two terminals:

```bash
# Terminal 1 (training)
cd ~/finetune-rocm
./scripts/run_training_in_docker.sh

# Terminal 2 (monitor)
cd /path/to/gpu-system-monitor
python3 system_monitor.py
```

## Background Run

```bash
python3 system_monitor.py > monitor.log 2>&1 &
```

## ROCm Quick Commands

```bash
rocm-smi
rocm-smi --showuse
rocm-smi --showtemp
rocm-smi --showpower
rocm-smi --showmeminfo vram
rocm-smi --showall
```

## Configuration Notes

Update rate is controlled in `system_monitor.py` inside `main()`:

```python
with Live(create_display(monitor), refresh_per_second=2, console=console, screen=True) as live:
```

- Increase `refresh_per_second` for smoother updates.
- Decrease it to reduce overhead.

Colour styling and panel composition are defined in `create_display()`.

## Troubleshooting

### `rocm-smi: command not found`

```bash
export PATH=/opt/rocm/bin:$PATH
```

If needed, use your installed ROCm versioned path.

### GPU values stay at `0`

- Confirm ROCm is functioning: `rocm-smi`
- Confirm workload is using GPU
- If inside Docker, verify GPU device/runtime access for the container

### CPU or NVMe temperature shows `0`

- Confirm `lm-sensors` is installed
- Run `sensors` and check actual chip labels
- Update sensor identifiers in `system_monitor.py` if they differ from expected values

### Permission issues with Docker

Add your user to Docker group and re-login:

```bash
sudo usermod -aG docker "$USER"
```

## Licence

MIT

## Metadata

- Created: 2026-02-16
- Target use case: Real-time monitoring during PyTorch pre-training and fine-tuning on ROCm
