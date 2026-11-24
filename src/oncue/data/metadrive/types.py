"""Reference tables for ``data.metadrive.description``: ScenarioNet type names and the subsets
MetaDrive 0.3.0.1 and CAT's ``_parse`` can read. Each table copies its source."""

from ...core.types import FeatureKind, LaneSignal, LaneType, RoadEdgeType, RoadLineType

# --- ScenarioNet ---
# From ``scenarionet/converter/waymo/type.py``; MetaDrive 0.4.3 and 0.3.0.1 know all of them.
AGENT_TYPE = {
    "vehicle": "VEHICLE",
    "pedestrian": "PEDESTRIAN",
    "cyclist": "CYCLIST",
    "other": "OTHER",
}
LANE_TYPE = {
    LaneType.UNDEFINED: "LANE_UNKNOWN",
    LaneType.FREEWAY: "LANE_FREEWAY",
    LaneType.SURFACE_STREET: "LANE_SURFACE_STREET",
    LaneType.BIKE_LANE: "LANE_BIKE_LANE",
}
ROAD_LINE_TYPE = {
    # ScenarioNet's name, though MetaDrive reads it as a road edge: what its planners saw.
    RoadLineType.UNKNOWN: "UNKNOWN",
    RoadLineType.BROKEN_SINGLE_WHITE: "ROAD_LINE_BROKEN_SINGLE_WHITE",
    RoadLineType.SOLID_SINGLE_WHITE: "ROAD_LINE_SOLID_SINGLE_WHITE",
    RoadLineType.SOLID_DOUBLE_WHITE: "ROAD_LINE_SOLID_DOUBLE_WHITE",
    RoadLineType.BROKEN_SINGLE_YELLOW: "ROAD_LINE_BROKEN_SINGLE_YELLOW",
    RoadLineType.BROKEN_DOUBLE_YELLOW: "ROAD_LINE_BROKEN_DOUBLE_YELLOW",
    RoadLineType.SOLID_SINGLE_YELLOW: "ROAD_LINE_SOLID_SINGLE_YELLOW",
    RoadLineType.SOLID_DOUBLE_YELLOW: "ROAD_LINE_SOLID_DOUBLE_YELLOW",
    RoadLineType.PASSING_DOUBLE_YELLOW: "ROAD_LINE_PASSING_DOUBLE_YELLOW",
}
ROAD_EDGE_TYPE = {
    RoadEdgeType.UNKNOWN: "UNKNOWN",
    RoadEdgeType.BOUNDARY: "ROAD_EDGE_BOUNDARY",
    RoadEdgeType.MEDIAN: "ROAD_EDGE_MEDIAN",
}
POLYGON_TYPE = {
    FeatureKind.CROSSWALK: "CROSSWALK",
    FeatureKind.SPEED_BUMP: "SPEED_BUMP",
    FeatureKind.DRIVEWAY: "DRIVEWAY",
}
SIGNAL_STATE = {
    LaneSignal.UNKNOWN: "LANE_STATE_UNKNOWN",
    LaneSignal.ARROW_STOP: "LANE_STATE_ARROW_STOP",
    LaneSignal.ARROW_CAUTION: "LANE_STATE_ARROW_CAUTION",
    LaneSignal.ARROW_GO: "LANE_STATE_ARROW_GO",
    LaneSignal.STOP: "LANE_STATE_STOP",
    LaneSignal.CAUTION: "LANE_STATE_CAUTION",
    LaneSignal.GO: "LANE_STATE_GO",
    LaneSignal.FLASHING_STOP: "LANE_STATE_FLASHING_STOP",
    LaneSignal.FLASHING_CAUTION: "LANE_STATE_FLASHING_CAUTION",
}
STOP_SIGN_TYPE = "STOP_SIGN"
TRAFFIC_LIGHT_TYPE = "TRAFFIC_LIGHT"
COORDINATE_WAYMO = "waymo"  # metadata coordinate: right-handed, x forward, as WOMD
DATASET_NAME = "waymo"  # ScenarioNet's dataset name for WOMD
KMH_PER_MPH = 1.609344  # from ScenarioNet's ``mph_to_kmh``

# --- MetaDrive 0.3.0.1 and CAT ---
# MetaDrive 0.3.0.1 as bundled with SAGE and SEAL (CAT downloads its own copy).
VERSION_0301_TAG = "MetaDrive v0.3.0.1"  # its DATA_VERSION, written by its WOMD converter
# Types CAT's and SAGE's ``_parse`` know (cat@dcd9152 advgen/adv_generator.py); any other
# raises a KeyError there, so OTHER agents are dropped.
CAT_MAP_TYPES = frozenset(
    {
        "LANE_FREEWAY",
        "LANE_SURFACE_STREET",
        "LANE_BIKE_LANE",
        "ROAD_LINE_BROKEN_SINGLE_WHITE",
        "ROAD_LINE_SOLID_SINGLE_WHITE",
        "ROAD_LINE_SOLID_DOUBLE_WHITE",
        "ROAD_LINE_BROKEN_SINGLE_YELLOW",
        "ROAD_LINE_BROKEN_DOUBLE_YELLOW",
        "ROAD_LINE_SOLID_SINGLE_YELLOW",
        "ROAD_LINE_SOLID_DOUBLE_YELLOW",
        "ROAD_LINE_PASSING_DOUBLE_YELLOW",
        "ROAD_EDGE_BOUNDARY",
        "ROAD_EDGE_MEDIAN",
        "STOP_SIGN",
        "CROSSWALK",
        "SPEED_BUMP",
    }
)
CAT_AGENT_TYPES = frozenset({"VEHICLE", "PEDESTRIAN", "CYCLIST"})
# Lanes MetaDrive 0.3.0.1 builds; ``ScenarioBlock`` skips shorter polylines.
LANE_TYPES_0301 = frozenset({"LANE_FREEWAY", "LANE_SURFACE_STREET", "LANE_BIKE_LANE"})
MIN_LANE_POINTS_0301 = 2
