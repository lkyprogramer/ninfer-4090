# T00 source audit

Observed on 2026-09-28. This file records measurements. It does not change the tokenizer.

## Pin

`check_source_pin.py` against `source-lock.json` returned PASS, scope `BASE_COMMIT_AND_BLOBS_ONLY`. HEAD is `44a2c6c3392c9d51e76a895ed97bf1867530d252`, subject `feat(engine): measure board energy and attribute it to prefill and decode`. 13 unique blobs and 4 locked line ranges matched the pin. The working tree is dirty, so this check does not certify the image by itself.

Branch `b0-scanner` was created from that detached HEAD. The dirty overlay was left unstaged. The worktree was not reset.

## Overlay

`git diff` is 11 files, 125 insertions, 11 deletions. That matches the lock's review-reported size. SHA256 of `overlay.patch` is `579c0853f2ea690fd779861766f2530f213af2c990db023c3dfaeb2d01df01eb`.

The diff adds schema-20 prepare timers and size fields. It does not change `tokenizer.cpp` or `processor.cpp`. `Dockerfile.sm89-nobake` is untracked on both the laptop tree and `/home/hhtele/ninfer-4090`. It is the image build file, not part of the 11-file diff.

Fourteen files, including the 11 overlay files, the Dockerfile, and the untouched tokenizer sources, have identical SHA256 on the laptop tree and `/home/hhtele/ninfer-4090`. The remote directory is not a git checkout.

## Image

Running container `ninfer-4090-prod` uses image id `sha256:9c2dec8b9ff087d178f72b68251ff7ff18c77330833eaade5b40bf4c49eb312b`, tag `ninfer-4090:prepare-detail`. Rollback tag `ninfer-4090:44a2c6c` is still present as `sha256:2e62f8933bcf08fb2157549036b2ec781b0129d851fbb28f298cb013dcf690fc`. Launch script line is `IMAGE=ninfer-4090:prepare-detail`. No service was restarted for this audit.

## Resources

Six frontend objects were read from `/data/models/qwen/qwen38-ninfer/qwen3_8_27b.ninfer` using the v2 directory (`model_id` `qwen3.8-27b`, `weights_id` `groupwise-int`). Artifact SHA256 is `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e`. Full file hashes are in `resources-manifest.json`. Each hash starts with the prefix in the pack's `resources-reported.json`. Resource bytes are not committed.

`chat_template.jinja` contains `reasoning_effort` and `preserve_thinking`. It does not contain the literal identifiers `ReasoningEffort` or `ThinkingToggle`. The template default written in the file is `xhigh`. Production thinking level is still set by the compat proxy, not by this audit.

## What is review report versus remeasured

Image creation time, the 11-file overlay size, and the resource prefixes were previously reported. This audit remeasured the pin, the overlay diff, both full image ids, the artifact SHA256, the six resource SHA256 values, the template markers, and the laptop-to-host file match.
