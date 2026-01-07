# Conv1D Reuse Analysis

A simplified reimplementation of the data reuse analysis from the MAESTRO paper (Kwon et al., IEEE Micro 2020).

This focuses specifically on Conv1D to understand the core concepts without the complexity of the full MAESTRO framework.

## What is this?

DNN accelerators achieve efficiency by reusing data - fetching once from expensive global memory and using it multiple times locally. Different "dataflows" (how you schedule computation across PEs and time) result in different reuse patterns.

This code analyzes those reuse patterns for Conv1D given:
- Layer parameters (filter size, input size)
- Hardware config (number of PEs)
- Mapping described using data-centric directives

## Data-Centric Directives

From the paper, mappings are described with two directive types:

```
TemporalMap(size, offset) Dimension   # data changes over time
SpatialMap(size, offset) Dimension    # data distributed across PEs
```

For Conv1D we have two dimensions:
- `S` - filter/weight dimension
- `X'` - output dimension

## Quick Start

```bash
python3 conv1d_reuse_analysis.py
```

This runs analysis on several example mappings and prints reuse statistics.

## Example Output

```
Layer Configuration:
  Filter size (S):  6
  Input size (X):   11
  Output size (X'): 6

Mapping (outer to inner):
  SpatialMap(1, 1) S
  TemporalMap(3, 3) X'

Weight Tensor:
  Temporal reuse factor:   2.00
  Spatial reuse factor:    1.00
  Reuse pattern:           Temporal multicast (weight-stationary style)
```

## Reuse Types

The paper identifies four reuse patterns:

| Type | What happens | Hardware needed |
|------|--------------|-----------------|
| Spatial Multicast | Same data to multiple PEs | Fanout/bus |
| Temporal Multicast | Same data reused over time | Buffer |
| Spatial Reduction | Partial sums across PEs | Reduction network |
| Temporal Reduction | Accumulate over time | Accumulator |

## Usage

```python
from conv1d_reuse_analysis import Conv1DLayer, Hardware, Mapping, MappingDirective
from conv1d_reuse_analysis import Conv1DReuseAnalyzer, DirectiveType, Dimension

# define layer
layer = Conv1DLayer(filter_size=64, input_size=256)

# hardware
hw = Hardware(num_pes=16)

# mapping: output-stationary style
mapping = Mapping([
    MappingDirective(DirectiveType.TEMPORAL, 8, 8, Dimension.S),
    MappingDirective(DirectiveType.SPATIAL, 4, 4, Dimension.X_OUT),
])

# analyze
analyzer = Conv1DReuseAnalyzer(layer, hw, mapping)
analyzer.print_analysis()

# or get raw results
results = analyzer.full_analysis()
print(results["Weight"].temporal_reuse_factor)
```

## Limitations

This is a teaching/learning implementation, not a production tool. Compared to real MAESTRO:

- Only Conv1D (no Conv2D, FC, etc.)
- No energy modeling
- No buffer size constraints
- Simplified reuse calculation
- Single-level hierarchy only

For the real thing, see [maestro.ece.gatech.edu](https://maestro.ece.gatech.edu/)

## Reference

```
@article{kwon2020maestro,
  title={MAESTRO: A Data-Centric Approach to Understand Reuse, Performance,
         and Hardware Cost of DNN Mappings},
  author={Kwon, Hyoukjun and Chatarasi, Prasanth and Sarkar, Vivek and
          Krishna, Tushar and Pellauer, Michael and Parashar, Angshuman},
  journal={IEEE Micro},
  year={2020}
}
```
