# Adding a response bank

A bank is one compressed NumPy archive `data/processed/<name>.npz` with:

| key | dtype / shape | meaning |
|---|---|---|
| `Y` | uint8 (J, I) | oracle verdict of system j on item i (1 = correct / resolved) |
| `systems` | str (J,) | unique system identifiers |
| `families` | str (J,) | family label; systems of the target's family never calibrate it |
| `items` | str (I,) | item identifiers |
| `meta` | JSON string | source, binarization rule, family rule, domain; optional `family_models` (dict system -> list of underlying models: systems sharing any model are treated as related) |
| `theta_true` | float (J,) | optional, synthetic banks only |

Then run, for example:

```bash
docker compose run --rm -T bench --phase mybank --datasets <name> --targets 60 --reps 5
docker compose run --rm -T analyze --phase mybank
```

The pilot S4 bank of the EOST project (240 items: GSM8K, GSM-Hard, SVAMP,
ASDiv; target Qwen3-8B; calibrators of stage S4B) can be added in exactly this
format; verdicts must come from the deterministic numeric oracle, and the
raw generations must stay untouched (derived matrices only).
