#!/bin/bash
# Setup and run GPU Monitor

echo "Setting up GPU monitor..."

# Check if running in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Installing dependencies (user-local)..."
    pip3 install --user -r monitor_requirements.txt
else
    echo "Installing dependencies (venv)..."
    pip3 install -r monitor_requirements.txt
fi

echo ""
echo "Setup complete."
echo ""
echo "To run the monitor:"
echo "  ./system_monitor.py"
echo ""
echo "For side-by-side view with tmux:"
echo "  tmux split-window -h './system_monitor.py'"
echo ""
