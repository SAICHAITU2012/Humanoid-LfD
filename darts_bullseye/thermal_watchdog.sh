#!/bin/bash
# Thermal watchdog: protects the laptop during training/rendering.
# - Checks macOS thermal state every CHECK seconds (0=nominal 1=fair 2=serious 3=critical)
# - If state >= 2 ("serious"), touches STOP in the newest finetune run (clean pause)
# - Logs every state change + a heartbeat every HEARTBEAT checks
# - Exits after MAX_ITERATIONS checks (safety), or when killed
cd "$(dirname "$0")"
CHECK=60; HEARTBEAT=15; MAX_ITERATIONS=360   # 6 h max
LOG=thermal_watchdog.log
LAST="init"
echo "$(date '+%m-%d %H:%M:%S') watchdog started (check ${CHECK}s)" >> "$LOG"
for i in $(seq 1 "$MAX_ITERATIONS"); do
  T=$(osascript -l JavaScript -e 'ObjC.import("Foundation"); $.NSProcessInfo.processInfo.thermalState' 2>/dev/null)
  T=${T:-unknown}
  LOAD=$(uptime | sed -E 's/.*load averages: //' | cut -d' ' -f1)
  # protective stop if the machine reports serious/critical thermal state
  if [ "$T" = "2" ] || [ "$T" = "3" ]; then
    NEWEST=$(ls -d runs/finetune_2026* 2>/dev/null | tail -1)
    if [ -n "$NEWEST" ] && [ ! -f "$NEWEST/STOP" ]; then
      touch "$NEWEST/STOP"
      echo "$(date '+%m-%d %H:%M:%S') PROTECTIVE_STOP thermal=$T load=$LOAD -> $NEWEST/STOP" >> "$LOG"
    fi
  fi
  if [ "$T" != "$LAST" ]; then
    echo "$(date '+%m-%d %H:%M:%S') thermal_state=$T load=$LOAD" >> "$LOG"
    LAST=$T
  elif [ $((i % HEARTBEAT)) -eq 0 ]; then
    echo "$(date '+%m-%d %H:%M:%S') heartbeat thermal=$T load=$LOAD" >> "$LOG"
  fi
  sleep "$CHECK"
done
echo "$(date '+%m-%d %H:%M:%S') watchdog finished (max iterations)" >> "$LOG"
