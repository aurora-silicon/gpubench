# Add a private CS2 replay

Keep the capture, identifying screenshots and receipt on the participating machines. One GFXReconstruct replay must reproduce consecutive frames of the known-good J616s build and the independently identified first-bad build. Index the frames in the trimmed replay. Record the capture-tool version, game build, demo tick, preset, kernel and both driver builds in the private evidence directory.

The receipt is a JSON object. Use the exact device name emitted by `gpubench-vk --info` and full Mesa commit IDs. Under `frames`, include at least two consecutive numbers. Each entry identifies same-sized screenshots of the **same frame** in the good and bad source replays. Specify a region as pixel coordinates `[left, top, right, bottom]`. On a different target device add `target_device` and a separate `target_good` image to each frame, enrolled on its accepted base. All file entries take a path relative to the receipt and its SHA-256.

```json
{
  "source_device": "<J616s device_name>",
  "target_device": "<J700 device_name>",
  "good_mesa": "<40 hex>", "bad_mesa": "<40 hex>",
  "trace": {"file":"scene.gfxr", "sha256":"<digest>"},
  "frames": {
    "2": {
      "good":{"file":"good/2.png","sha256":"<digest>"},
      "bad":{"file":"bad/2.png","sha256":"<digest>"},
      "target_good":{"file":"neo/2.png","sha256":"<digest>"},
      "roi":[100,80,150,120]
    },
    "3": {
      "good":{"file":"good/3.png","sha256":"<digest>"},
      "bad":{"file":"bad/3.png","sha256":"<digest>"},
      "target_good":{"file":"neo/3.png","sha256":"<digest>"},
      "roi":[100,80,150,120]
    }
  }
}
```

```sh
python3 tools/add-flicker-capture.py corpus/j700.json private/receipt.json private/combined.json
```

Pass the resulting machine-local file to the usual `--traces` argument. The importer retains the original corpus, checks every input digest and requires a source good/bad difference above the gate tolerance in the selected region on at least one frame. A missing or non-discriminating capture leaves CS2 unenrolled. The standard headless runner compares all supplied target frames to their own goldens.
