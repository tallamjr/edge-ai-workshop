# UTM x86 Linux VM setup (for Apple Silicon Macs)

The NXP `neutron-converter` (and the rest of the eIQ Neutron SDK) is an
**x86-64 Linux ELF binary**. It cannot run on an arm64 Apple Silicon Mac. To
compile a quantised TFLite model for the i.MX95 Neutron NPU you therefore need
an **x86-64 Linux machine**. The cheapest way to get one on a Mac is an emulated
x86 VM under [UTM](https://mac.getutm.app/).

This guide reproduces the VM the workshop uses:
**Ubuntu Server 24.04 (x86_64), 4 vCPU, 8 GB RAM**, reachable from the Mac over
SSH, with the eIQ SDK copied in.

> The only thing this VM does is run `neutron-converter`. Everything else
> (Ultralytics export, int8 quantisation, validation, deploying to the board)
> happens on the Mac. See `docs/model-preparation-notes.md` for the full
> pipeline and `docs/deployment.md` for shipping to the board.

<!-- mtoc-start -->

- [0. What you need](#0-what-you-need)
- [1. Install UTM](#1-install-utm)
- [2. Create the VM (x86_64 EMULATED)](#2-create-the-vm-x86_64-emulated)
- [3. Install Ubuntu Server](#3-install-ubuntu-server)
- [4. Network so the Mac can reach the VM](#4-network-so-the-mac-can-reach-the-vm)
- [5. Put the eIQ SDK on the VM](#5-put-the-eiq-sdk-on-the-vm)
- [6. Verify](#6-verify)
- [7. Use it](#7-use-it)
- [Gotchas](#gotchas)

<!-- mtoc-end -->

## 0. What you need

- A Mac with Apple Silicon (M1/M2/M3/...).
- ~10 GB free disk for the VM, plus a few GB for models.
- The **Ubuntu Server 24.04 x86_64** ISO:
  https://releases.ubuntu.com/24.04/ (pick `ubuntu-24.04.x-live-server-amd64.iso`).
- The **eIQ Neutron SDK** (`eiq-neutron-sdk-linux-3.1.2.zip`, ~87 MB). It ships
  inside the NXP eIQ Toolkit and is not freely downloadable, so the simplest
  path is to copy the one already in this repo's `bin/` (gitignored) onto the VM
  in step 5. The eIQ Toolkit itself:
  https://www.nxp.com/design/design-center/software/eiq-ai-development-environment/eiq-toolkit-for-end-to-end-model-development-and-deployment:EIQ-TOOLKIT

---

## 1. Install UTM

```bash
brew install --cask utm
```

(or download the app from https://mac.getutm.app/).

---

## 2. Create the VM (x86_64 EMULATED)

The critical choice: on Apple Silicon, UTM offers **Virtualize** (fast, arm64
guests only) and **Emulate** (slower, any architecture). You must pick
**Emulate** so you can run an x86-64 guest.

1. UTM -> **Create a New Virtual Machine** -> **Emulate**.
2. **Operating System**: Linux. Browse to the Ubuntu Server `amd64` ISO.
3. **Architecture**: `x86_64`. System: leave the default machine (Standard PC /
   Q35). This is the step that makes it an x86 box; do not leave it on `aarch64`.
4. **Memory**: 8192 MB. **CPU Cores**: 4. (The converter is single-threaded for
   the heavy part; more cores mainly help the OS.)
5. **Storage**: 40 GB or more.
6. **Shared Directory**: optional, skip it; we use `scp` for transfers.
7. Name it (e.g. `neutron-x86`) and save.

> Emulated x86 runs under TCG (no hardware acceleration), so it is slow but
> perfectly adequate: compiling the pose backbone took about five minutes.

---

## 3. Install Ubuntu Server

Boot the VM and run the Ubuntu Server installer. Defaults are fine, with two
things to get right:

- Choose a username you will remember (the workshop VM uses `utm`). Every
  example in the docs uses `utm@<vm-ip>`; substitute your own user if different.
- On the **SSH Setup** screen, tick **Install OpenSSH server**. This is what
  lets the Mac drive the VM over `ssh`/`scp`.

After install, shut down, **remove the ISO** from the VM's drives in UTM (so it
boots from disk), and start it again.

---

## 4. Network so the Mac can reach the VM

UTM's default **Shared Network** mode gives the guest a private IP on a vmnet
subnet that the Mac can reach, plus NAT internet. That is all we need.

In the VM, find its IP:

```bash
ip -4 addr show scope global | awk '/inet/ {print $2}'
# e.g. 192.168.105.5/24   (yours will differ; this subnet is assigned by UTM)
```

From the Mac, confirm SSH works:

```bash
ssh utm@192.168.105.5        # use YOUR vm-ip and user
```

Notes:

- The guest IP is DHCP and can change across reboots. Re-check with `ip a`, or
  pin it: copy your Mac key over once with `ssh-copy-id utm@<vm-ip>` and add a
  `Host neutron-x86` block to `~/.ssh/config` so you can `ssh neutron-x86`.
- If `ssh` refuses, confirm `sudo systemctl status ssh` is active in the VM and
  that UTM's network is **Shared Network** (Settings -> Network), not Bridged.

---

## 5. Put the eIQ SDK on the VM

Copy the SDK zip from the Mac (this repo already has it in `bin/`) and unzip it
to the path the workshop commands expect, `~/edge-ai-workshop/bin/`:

```bash
# on the Mac, from the repo root
ssh utm@192.168.105.5 'mkdir -p ~/edge-ai-workshop/bin'
scp bin/eiq-neutron-sdk-linux-3.1.2.zip utm@192.168.105.5:~/edge-ai-workshop/bin/
ssh utm@192.168.105.5 \
  'cd ~/edge-ai-workshop/bin && unzip -q eiq-neutron-sdk-linux-3.1.2.zip && ls eiq-neutron-sdk-linux-3.1.2/bin'
```

The SDK needs no installation; the binaries are self-contained. You should see:

```
neutron-converter  neutron-runner  tflite-extractor
tflite-optimizer   tflite-profiler tflite-quantizer
```

---

## 6. Verify

```bash
ssh utm@192.168.105.5 \
  '~/edge-ai-workshop/bin/eiq-neutron-sdk-linux-3.1.2/bin/neutron-converter --help | head -5'
```

If it prints usage text, the VM is ready. (`neutron-converter` itself needs no
internet; it runs fully offline.)

---

## 7. Use it

You now have everything the "NPU compile (VM)" steps in the other docs assume.
The pattern is always: `scp` the quantised int8 TFLite to the VM, run
`neutron-converter`, `scp` the compiled model back.

```bash
# from the Mac, repo root. example: the detection model
scp models/work/yolov8s_full_integer_quant.tflite utm@192.168.105.5:/home/utm/
ssh utm@192.168.105.5 \
  '~/edge-ai-workshop/bin/eiq-neutron-sdk-linux-3.1.2/bin/neutron-converter \
     --input /home/utm/yolov8s_full_integer_quant.tflite \
     --output /home/utm/yolov8s_neutron.tflite --target imx95'
scp utm@192.168.105.5:/home/utm/yolov8s_neutron.tflite models/deploy/
```

A successful run reports the operator conversion ratio (e.g. `230 / 233 =
0.987`) and an NPU latency estimate. See `docs/model-preparation-notes.md` for
the full model pipeline and `docs/deployment.md` for deploying to the board.

---

## Gotchas

- **Emulate, not Virtualize.** Virtualize on Apple Silicon only runs arm64
  guests; the SDK is x86-64. If `uname -m` in the VM says `aarch64`, you built
  the wrong kind of VM.
- **Slow is fine.** TCG emulation has no hardware acceleration. A conversion
  taking minutes is normal and does not indicate a problem.
- **Guest IP drifts.** It is DHCP on UTM's vmnet; re-check with `ip a` after a
  reboot, or pin it via `~/.ssh/config`.
- **Internet is optional.** Shared Network usually provides NAT internet, but
  `neutron-converter` does not need it. If you want to `apt`/`pip` on the VM and
  it has no connectivity, check the UTM network mode is Shared Network.
- **Transfer via scp.** There is no shared clipboard or auto-mount by default;
  move files in and out with `scp` as shown above.
