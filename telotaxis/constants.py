"""Shared clocks and body numbers.

The 15 ms coupling interval is the one Fly.exe publishes. Membrane steps on
this CPU are 5 ms, so three of them fill one interval. The MaleCNS GeNN path
steps at 100 µs. That difference is declared here, not hidden.
"""

COUPLING_S = 0.015
NEURAL_DT = 0.005
PHYSICS_DT = 0.001
NEURAL_STEPS = 3
PHYSICS_STEPS = 15

ONSET_STAND_S = 1.5
ONSET_HOLD_S = 0.150

LOOM_HALF_DEG = 14.0
SIGMA_DEG = 18.0
SIGMA_GROW = 0.35

PEER_RADIUS_M = 0.20
NEAR_SURFACE_M = 0.15

WHEEL_RADIUS_M = 0.048
HALF_TRACK_M = 0.11
BODY_Z_M = WHEEL_RADIUS_M

MIN_CRUISE_MPS = 0.26
MAX_SPEED_MPS = 0.50
MAX_YAW_RATE = 0.80

# Filled in once a pathway is loaded. A frontal cue sits well above this,
# and a silent pathway sits at zero. Retuned in brain.probe if needed.
RATE_THRESHOLD_HZ = 8.0
RATE_TRACE_S = 0.040
SPIKE_VIS_S = 0.080

# Synaptic gain after each neuron's incoming mass is normalised to 1.
SYN_GAIN = 0.42
INPUT_GAIN = 2.8
TAU_M = 0.020
TAU_SYN = 0.020
TAU_ADAPT = 0.120
V_THRESHOLD = 1.0
ADAPT_JUMP = 0.08
REFRACTORY_STEPS = 1

MIN_SYNAPSES = 3
