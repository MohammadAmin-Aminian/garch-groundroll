# Datasets

All files are MATLAB v5 `.mat` files readable by both implementations
(`garch_groundroll.load_dataset` in Python, `ggr_load_dataset` in MATLAB).
Each contains

| variable | meaning |
|---|---|
| `data` | gather, `nt x nx` (time samples x traces), AGC already applied |
| `dt` | sampling interval (s) |
| `dx` | trace spacing (m) |
| `description` | short text |
| `fk_mask` | *(optional)* f-k reject zone used in the paper, defined on `fftshift(fft2(data))` (1 = remove) |

| file | size | used for | paper figures | filter in the paper |
|---|---|---|---|---|
| *(generated)* `make_synthetic_shot()` / `ggr_make_synthetic()` | 1501 x 100 | method illustration with known truth | 2 | f-k / Ormsby |
| `shot_gather_502.mat` | 1500 x 502 | symmetric common-shot gather | 3-12 | f-k (`fk_mask`) |
| `field_shot.mat` | 1000 x 83 | field common-shot gather | 13-16 | f-k (`fk_mask`) |
| `stacked_section.mat` | 1000 x 281 | stacked section with ground roll | 17-18 | Ormsby |
| `crossline_3d.mat` | 700 x 273 | cross-line from a 3-D land survey | 19-20 | f-k |

## Sampling

The paper states 4 ms / 12.5 m for the stacked section; the same trace spacing
is used for the other gathers, whose f-k spectra in the paper were also
displayed with 12.5 m. The shot gathers use the 2 ms sampling of the original
processing scripts. Only the physical axis labels and the filter parameters
expressed in Hz or m/s depend on these values; change `dt`/`dx` (or pass
`--dt`/`--dx` to the command-line tool) if your records differ.

## Using your own data

Save a gather in the same format, e.g. in Python

```python
from garch_groundroll.io import save_mat
save_mat("data/my_gather.mat", gather, dt=0.002, dx=10.0, description="my line")
```

or in MATLAB

```matlab
data = myGather; dt = 0.002; dx = 10; description = 'my line';
save('data/my_gather.mat', 'data', 'dt', 'dx', 'description');
```

## Provenance

The gathers are the inputs of the published study (Aminian & Riahi, 2023) and
are shared to make its results reproducible.
