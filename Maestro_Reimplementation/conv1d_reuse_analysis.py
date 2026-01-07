"""
Simple Reuse Analysis for Conv1D based on MAESTRO (Kwon et al., 2020)

This implements the data-centric approach to analyze temporal and spatial
data reuse in Conv1D operations on DNN accelerators.

Conv1D operation:
    for s in range(S):           # filter dimension
        for x_out in range(X'):  # output dimension
            Output[x_out] += Weight[s] * Input[x_out + s]

Where:
    - S: Filter size
    - X: Input size
    - X' = X - S + 1: Output size
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple


class Dimension(Enum):
    """Dimensions in Conv1D"""
    S = "S"      # Filter/weight dimension
    X_OUT = "X'" # Output dimension


class DirectiveType(Enum):
    """Type of mapping directive"""
    TEMPORAL = "Temporal"
    SPATIAL = "Spatial"


@dataclass
class MappingDirective:
    """
    Data-centric mapping directive (Figure 3a in paper)

    - directive_type: TEMPORAL (changes over time) or SPATIAL (changes across PEs)
    - size: Number of data points mapped on each PE
    - offset: How mapping updates over time (temporal) or space (spatial)
    - dimension: Which dimension this directive maps
    """
    directive_type: DirectiveType
    size: int
    offset: int
    dimension: Dimension

    def __repr__(self):
        return f"{self.directive_type.value}Map({self.size}, {self.offset}) {self.dimension.value}"


@dataclass
class Conv1DLayer:
    """Conv1D layer parameters"""
    filter_size: int   # S
    input_size: int    # X

    @property
    def output_size(self) -> int:
        """X' = X - S + 1"""
        return self.input_size - self.filter_size + 1

    def total_macs(self) -> int:
        """Total multiply-accumulate operations"""
        return self.filter_size * self.output_size


@dataclass
class Hardware:
    """Simple hardware configuration"""
    num_pes: int


@dataclass
class Mapping:
    """
    Complete mapping specification using data-centric directives.
    Directives are ordered from slowest-changing (outermost) to fastest-changing (innermost).
    """
    directives: List[MappingDirective]

    def __repr__(self):
        return "\n".join([f"  {d}" for d in self.directives])


@dataclass
class ReuseAnalysisResult:
    """Results of reuse analysis for a tensor"""
    tensor_name: str
    total_size: int                    # Total unique data elements
    unique_elements_per_tile: int      # Elements needed per compute tile
    temporal_reuse_factor: float       # Reuse over time within same PE
    spatial_reuse_factor: float        # Reuse across PEs (multicast)
    total_reads_from_global: int       # Total reads from global buffer
    reuse_type: str                    # Description of reuse pattern


class Conv1DReuseAnalyzer:
    """
    Analyzes data reuse for Conv1D based on MAESTRO methodology.

    Key insight from paper: Data reuse is explicit in data space when using
    data-centric representation.
    """

    def __init__(self, layer: Conv1DLayer, hardware: Hardware, mapping: Mapping):
        self.layer = layer
        self.hardware = hardware
        self.mapping = mapping

    def _get_directive_for_dim(self, dim: Dimension) -> MappingDirective:
        """Find the directive that maps a specific dimension"""
        for d in self.mapping.directives:
            if d.dimension == dim:
                return d
        return None

    def _compute_iteration_counts(self) -> Dict[Dimension, int]:
        """Compute how many iterations needed for each dimension"""
        counts = {}

        for directive in self.mapping.directives:
            if directive.dimension == Dimension.S:
                dim_size = self.layer.filter_size
            else:  # X_OUT
                dim_size = self.layer.output_size

            if directive.directive_type == DirectiveType.SPATIAL:
                # Spatial dimension: ceil(dim_size / (size * num_pes)) temporal iterations
                spatial_coverage = directive.size * self.hardware.num_pes
                counts[directive.dimension] = max(1, (dim_size + spatial_coverage - 1) // spatial_coverage)
            else:
                # Temporal dimension: ceil(dim_size / size) iterations
                counts[directive.dimension] = max(1, (dim_size + directive.size - 1) // directive.size)

        return counts

    def analyze_weight_reuse(self) -> ReuseAnalysisResult:
        """
        Analyze reuse of weight/filter tensor.

        Weight is indexed by [s], so reuse depends on how S dimension is mapped.
        - If S is spatial: weights are distributed across PEs (no multicast for unique weights)
        - If S is temporal: same weight reused over time

        Weight reuse across X' dimension: same weight used for all output positions
        """
        s_directive = self._get_directive_for_dim(Dimension.S)
        x_directive = self._get_directive_for_dim(Dimension.X_OUT)

        total_weights = self.layer.filter_size

        if s_directive.directive_type == DirectiveType.SPATIAL:
            # Weights distributed across PEs
            weights_per_pe = s_directive.size
            # Temporal reuse: weights reused across all X' iterations
            x_iterations = self._compute_iteration_counts()[Dimension.X_OUT]
            temporal_reuse = x_iterations
            spatial_reuse = 1.0  # No spatial reuse (different weights per PE)
            reuse_type = "Temporal multicast (weight-stationary style)"
        else:
            # S is temporal: weights change over time
            weights_per_pe = s_directive.size
            # If X' is spatial, same weight broadcast to all PEs
            if x_directive.directive_type == DirectiveType.SPATIAL:
                spatial_reuse = min(self.hardware.num_pes,
                                   self.layer.output_size // x_directive.size)
                reuse_type = "Spatial multicast"
            else:
                spatial_reuse = 1.0
                reuse_type = "No spatial reuse"
            temporal_reuse = 1.0

        # Total reads from global buffer
        s_iters = self._compute_iteration_counts().get(Dimension.S, 1)
        total_reads = total_weights * s_iters // max(1, int(temporal_reuse))

        return ReuseAnalysisResult(
            tensor_name="Weight",
            total_size=total_weights,
            unique_elements_per_tile=weights_per_pe,
            temporal_reuse_factor=temporal_reuse,
            spatial_reuse_factor=spatial_reuse,
            total_reads_from_global=total_reads,
            reuse_type=reuse_type
        )

    def analyze_input_reuse(self) -> ReuseAnalysisResult:
        """
        Analyze reuse of input activation tensor.

        Input is indexed by [x' + s], creating a sliding window pattern.
        - Adjacent output positions share overlapping input elements
        - This creates "halo" reuse opportunities
        """
        s_directive = self._get_directive_for_dim(Dimension.S)
        x_directive = self._get_directive_for_dim(Dimension.X_OUT)

        total_inputs = self.layer.input_size

        # Input elements needed per output tile
        if x_directive.directive_type == DirectiveType.SPATIAL:
            # Multiple outputs computed in parallel need overlapping inputs
            outputs_per_iteration = x_directive.size * self.hardware.num_pes
            outputs_per_iteration = min(outputs_per_iteration, self.layer.output_size)
        else:
            outputs_per_iteration = x_directive.size

        # Input window for these outputs (with halo)
        inputs_per_tile = outputs_per_iteration + self.layer.filter_size - 1

        # Halo reuse: overlap between consecutive tiles
        halo_size = self.layer.filter_size - 1
        halo_reuse = halo_size / inputs_per_tile if inputs_per_tile > 0 else 0

        # Spatial reuse within a tile
        if x_directive.directive_type == DirectiveType.SPATIAL:
            # Adjacent PEs can share halo inputs via spatial multicast
            spatial_reuse = 1.0 + halo_reuse * (min(self.hardware.num_pes, outputs_per_iteration) - 1)
            reuse_type = "Spatial multicast (halo sharing)"
        else:
            spatial_reuse = 1.0
            reuse_type = "Temporal reuse (sliding window)"

        # Temporal reuse: same input used across filter dimension
        if s_directive.directive_type == DirectiveType.TEMPORAL:
            temporal_reuse = s_directive.size
        else:
            temporal_reuse = 1.0

        x_iters = self._compute_iteration_counts().get(Dimension.X_OUT, 1)
        s_iters = self._compute_iteration_counts().get(Dimension.S, 1)

        # Unique input reads (accounting for halo reuse between tiles)
        total_reads = x_iters * inputs_per_tile * s_iters // max(1, int(temporal_reuse))

        return ReuseAnalysisResult(
            tensor_name="Input",
            total_size=total_inputs,
            unique_elements_per_tile=inputs_per_tile,
            temporal_reuse_factor=temporal_reuse,
            spatial_reuse_factor=spatial_reuse,
            total_reads_from_global=total_reads,
            reuse_type=reuse_type
        )

    def analyze_output_reuse(self) -> ReuseAnalysisResult:
        """
        Analyze reuse of output activation tensor.

        Output is indexed by [x'], and partial sums are accumulated over s.
        - Temporal reduction: accumulate partial sums over time
        - Spatial reduction: accumulate partial sums across PEs
        """
        s_directive = self._get_directive_for_dim(Dimension.S)
        x_directive = self._get_directive_for_dim(Dimension.X_OUT)

        total_outputs = self.layer.output_size

        if x_directive.directive_type == DirectiveType.SPATIAL:
            outputs_per_pe = x_directive.size
        else:
            outputs_per_pe = x_directive.size

        # Reduction pattern depends on how S is mapped
        if s_directive.directive_type == DirectiveType.SPATIAL:
            # Different s values on different PEs -> spatial reduction needed
            spatial_reduction_factor = min(self.hardware.num_pes, self.layer.filter_size)
            temporal_reduction_factor = 1.0
            reuse_type = "Spatial reduction (partial sums across PEs)"
        else:
            # S changes over time -> temporal reduction (accumulate in local buffer)
            temporal_reduction_factor = s_directive.size
            spatial_reduction_factor = 1.0
            reuse_type = "Temporal reduction (output-stationary style)"

        # Output writes/reads
        s_iters = self._compute_iteration_counts().get(Dimension.S, 1)
        x_iters = self._compute_iteration_counts().get(Dimension.X_OUT, 1)

        # Each output needs to be written once final, but partial sums accumulate
        total_writes = total_outputs
        partial_sum_accesses = total_outputs * s_iters  # Read-modify-write for accumulation

        return ReuseAnalysisResult(
            tensor_name="Output",
            total_size=total_outputs,
            unique_elements_per_tile=outputs_per_pe,
            temporal_reuse_factor=temporal_reduction_factor,
            spatial_reuse_factor=spatial_reduction_factor,
            total_reads_from_global=partial_sum_accesses,
            reuse_type=reuse_type
        )

    def full_analysis(self) -> Dict[str, ReuseAnalysisResult]:
        """Run complete reuse analysis for all tensors"""
        return {
            "Weight": self.analyze_weight_reuse(),
            "Input": self.analyze_input_reuse(),
            "Output": self.analyze_output_reuse()
        }

    def print_analysis(self):
        """Print formatted analysis results"""
        print("=" * 70)
        print("CONV1D REUSE ANALYSIS (MAESTRO-style)")
        print("=" * 70)

        print(f"\nLayer Configuration:")
        print(f"  Filter size (S):  {self.layer.filter_size}")
        print(f"  Input size (X):   {self.layer.input_size}")
        print(f"  Output size (X'): {self.layer.output_size}")
        print(f"  Total MACs:       {self.layer.total_macs()}")

        print(f"\nHardware:")
        print(f"  Number of PEs: {self.hardware.num_pes}")

        print(f"\nMapping (outer to inner):")
        print(self.mapping)

        print(f"\n" + "-" * 70)
        print("REUSE ANALYSIS RESULTS")
        print("-" * 70)

        results = self.full_analysis()
        for tensor_name, result in results.items():
            print(f"\n{result.tensor_name} Tensor:")
            print(f"  Total size:              {result.total_size}")
            print(f"  Elements per tile:       {result.unique_elements_per_tile}")
            print(f"  Temporal reuse factor:   {result.temporal_reuse_factor:.2f}")
            print(f"  Spatial reuse factor:    {result.spatial_reuse_factor:.2f}")
            print(f"  Global buffer accesses:  {result.total_reads_from_global}")
            print(f"  Reuse pattern:           {result.reuse_type}")


def create_weight_stationary_mapping(num_pes: int) -> Mapping:
    """
    Weight-stationary mapping (Figure 3b in paper):
    - Weights stay in PE (spatially distributed across S)
    - Outputs computed temporally
    """
    return Mapping([
        MappingDirective(DirectiveType.SPATIAL, 1, 1, Dimension.S),
        MappingDirective(DirectiveType.TEMPORAL, 3, 3, Dimension.X_OUT),
    ])


def create_output_stationary_mapping(num_pes: int) -> Mapping:
    """
    Output-stationary mapping:
    - Outputs stay in PE (spatially distributed across X')
    - Weights streamed temporally
    """
    return Mapping([
        MappingDirective(DirectiveType.TEMPORAL, 1, 1, Dimension.S),
        MappingDirective(DirectiveType.SPATIAL, 3, 3, Dimension.X_OUT),
    ])


def create_input_stationary_mapping(num_pes: int) -> Mapping:
    """
    Input-stationary style mapping:
    - Maximize input reuse by keeping inputs in PE
    - Both S and X' temporal with overlapping input access
    """
    return Mapping([
        MappingDirective(DirectiveType.TEMPORAL, 2, 2, Dimension.S),
        MappingDirective(DirectiveType.TEMPORAL, 3, 3, Dimension.X_OUT),
    ])


# Example usage
if __name__ == "__main__":
    # Example from Figure 2: S=6, X=11, X'=9, 3 PEs
    layer = Conv1DLayer(filter_size=6, input_size=11)
    hardware = Hardware(num_pes=3)

    print("\n" + "=" * 70)
    print("EXAMPLE 1: Weight-Stationary Mapping (from Figure 2)")
    print("=" * 70)
    mapping_ws = create_weight_stationary_mapping(hardware.num_pes)
    analyzer_ws = Conv1DReuseAnalyzer(layer, hardware, mapping_ws)
    analyzer_ws.print_analysis()

    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: Output-Stationary Mapping")
    print("=" * 70)
    mapping_os = create_output_stationary_mapping(hardware.num_pes)
    analyzer_os = Conv1DReuseAnalyzer(layer, hardware, mapping_os)
    analyzer_os.print_analysis()

    print("\n\n" + "=" * 70)
    print("EXAMPLE 3: Input-Stationary Mapping")
    print("=" * 70)
    mapping_is = create_input_stationary_mapping(hardware.num_pes)
    analyzer_is = Conv1DReuseAnalyzer(layer, hardware, mapping_is)
    analyzer_is.print_analysis()

    # Larger example
    print("\n\n" + "=" * 70)
    print("EXAMPLE 4: Larger Layer (S=64, X=256, 16 PEs)")
    print("=" * 70)
    large_layer = Conv1DLayer(filter_size=64, input_size=256)
    large_hardware = Hardware(num_pes=16)
    large_mapping = Mapping([
        MappingDirective(DirectiveType.TEMPORAL, 8, 8, Dimension.S),
        MappingDirective(DirectiveType.SPATIAL, 4, 4, Dimension.X_OUT),
    ])
    analyzer_large = Conv1DReuseAnalyzer(large_layer, large_hardware, large_mapping)
    analyzer_large.print_analysis()
