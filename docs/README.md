# Docs

Working documentation for building and deploying models to the FRDM-IMX95
Neutron NPU from an Apple Silicon Mac.

## Read in this order

1. **[utm-vm-setup.md](utm-vm-setup.md)**: one-time setup of the x86 Linux VM
   that runs `neutron-converter` (the NPU compiler cannot run on arm64 macOS).
   Skip if you already have a working x86 Linux box.
2. **[model-preparation-notes.md](model-preparation-notes.md)**: the full
   model pipeline: Ultralytics export, int8 quantisation, NPU compile, and the
   pose backbone/head split that puts pose mostly on the NPU. The "why it is
   built this way" working log.
3. **[deployment.md](deployment.md)**: deploying the compiled models to the
   board and running the stream. Detection and pose, step by step.

## Reference

- **[iree-workflow.md](iree-workflow.md)**: IREE CPU-vs-NPU comparison notes
  (public IREE has no Neutron backend; CPU path only). Background, not part of
  the main deploy flow.

## The machines

| Machine | Role |
| --- | --- |
| **Mac** (Apple Silicon) | Ultralytics export, int8 quantisation, validation, deploy to board. Reaches the board and the internet. |
| **x86 Linux VM** (UTM) | Runs `neutron-converter` / eIQ SDK only. See utm-vm-setup.md. |
| **Board** (FRDM-IMX95) | Runs the Python TFLite pipeline with the Neutron delegate. |
