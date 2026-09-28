# T00 result

Status: PASS for source pin, overlay identity, full image ids, and six artifact resources.

No product source was edited. No reviewer was run, because this card does not change code. The service was not restarted.

Commands that were run:

- `python3 docs/ninfer-44a2c6c-frontend-optimization-v2/tools/check_source_pin.py` on the local ninfer tree. Exit 0. Output `evidence/b0/t00/base-check.json`.
- `git status --porcelain=v1` and `git ls-files --others --exclude-standard`.
- `git diff` saved as `overlay.patch`.
- Read-only `docker image inspect` and `docker inspect ninfer-4090-prod` on 192.168.10.29.
- Read-only extract of six `frontend/*` resources from the `.ninfer` directory, then `resource_fingerprint.py`.
- `sha256sum` of the artifact.
- SHA256 compare of 14 files against `/home/hhtele/ninfer-4090`. Mismatches: 0.

Not run: official CMake configure. Host cmake is 3.25.1 and the project requires 3.28. That does not block this identity card.
