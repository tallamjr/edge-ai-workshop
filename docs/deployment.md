# Deploying models to the board

**Board IP:** `192.168.1.236` (on the `NXP-NEUTRON` WiFi; your laptop must be on
the same WiFi). **View the stream at:** http://192.168.1.236:5000

All commands run from the repo root on the Mac. The compiled NPU models live in
`models/deploy/`. To rebuild them see `docs/model-preparation-notes.md`;
compiling for the NPU needs the x86 VM described in `docs/utm-vm-setup.md`. The
board app code (`board/`) must be synced once after the pose changes with
`make board-deploy-app`.

A safe restart helper (kills the running app without the launcher killing its own
shell, then relaunches in the background):

```bash
restart_board() {
  ssh root@192.168.1.236 'pkill -f "[m]ain.py"; sleep 2'
  ssh root@192.168.1.236 'cd /home/root/edge_ai_workshop/board; \
    nohup .venv/bin/python main.py >/tmp/app.log 2>&1 </dev/null & echo started'
}
```

## 1. Object detection (YOLOv8s, NPU) — ready

```bash
# sync app code + deploy corrected detection model + restart
make board-deploy-app BOARD_IP=192.168.1.236
make deploy BOARD_IP=192.168.1.236   # OUT_DIR defaults to models/deploy
ssh root@192.168.1.236 'pkill -f "[m]ain.py"; sleep 2'
ssh root@192.168.1.236 'cd /home/root/edge_ai_workshop/board; nohup .venv/bin/python main.py >/tmp/app.log 2>&1 </dev/null & echo started'
# view: http://192.168.1.236:5000
```

`board/config.json` already selects detection (`variant: s`, `task: detect`).

## 2. Pose / keypoints (YOLOv8s-pose): backbone on NPU, head on CPU

Pose now runs **mostly on the NPU** via a two-stage split: the int8 backbone/neck
runs on the Neutron NPU (230/233 ops, ~22 ms NPU latency), and only the small
float head runs on the CPU (where int8 would crush the confidence to zero). The
board decoder + skeleton overlay (`board/inference.py` / `board/overlay.py`,
gated on `task: pose`) consume the head output unchanged. How the split was built
and validated: `docs/model-preparation-notes.md`.

Deploy artifacts (board layout) live in `models/deploy/yolov8s-pose/`:
`pipeline.json` (stage 0 backbone NPU, stage 1 head CPU), `backbone_neutron.tflite`,
`head_float32.tflite`. The manifest sets `use_npu` per stage, so no `sed` of the
config is needed.

```bash
# sync app code (multi-stage chaining + pose decoder + overlay)
make board-deploy-app BOARD_IP=192.168.1.236
# place the split pipeline (manifest + both stage models) on the board
ssh root@192.168.1.236 'mkdir -p /opt/models/yolov8s-pose'
scp models/deploy/yolov8s-pose/pipeline.json \
    models/deploy/yolov8s-pose/backbone_neutron.tflite \
    models/deploy/yolov8s-pose/head_float32.tflite \
    root@192.168.1.236:/opt/models/yolov8s-pose/
# pose config (task: pose) — manifest controls per-stage NPU, no sed needed
scp board/config.pose.json root@192.168.1.236:/home/root/edge_ai_workshop/board/config.json
# restart
ssh root@192.168.1.236 'pkill -f "[m]ain.py"; sleep 2'
ssh root@192.168.1.236 'cd /home/root/edge_ai_workshop/board; nohup .venv/bin/python main.py >/tmp/app.log 2>&1 </dev/null & echo started'
# view: http://192.168.1.236:5000  (boxes + 17-point skeletons; backbone on NPU)
```

After restart, confirm the log shows two stages loading, the backbone via the
NPU delegate (`Stage 'backbone_npu': backbone_neutron.tflite ... NPU: True`) and
the head on CPU (`Stage 'head_cpu': head_float32.tflite ... NPU: False`), with no
`unresolved custom op: NeutronGraph` error (that error means the delegate did not
load; check `npu_delegate_path`).

To switch back to detection, re-run section 1 (it pushes the detection
`config.json`).

## Notes

- After any restart, confirm the app came up: `ssh root@192.168.1.236 'tail -20 /tmp/app.log'`
  (look for "Stream available at" and no `g2d_open` crash; `use_gstreamer` is
  already `false`).
- If the browser can't connect but `nc -z 192.168.1.236 5000` succeeds, it's the
  flaky WiFi, retry.
