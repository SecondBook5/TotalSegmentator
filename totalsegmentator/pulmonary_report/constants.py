LANDMARK_DEFINITIONS = {
    1: ("pul_annulus", "proximal MPA"),
    2: ("pul_sinotubular_junction", "sinotub. junc."),
    3: ("pul_bifurcation", "bifurcation"),
    4: ("pul_left_start", "left PA"),
    5: ("pul_right_start", "right PA"),
}

REPORT_FRAMES = 24
REPORT_SMOOTHING = 100


def create_landmarks():
    return {
        index: {
            "name": name,
            "display_name": display_name,
            "data": None,
            "empty": False,
        }
        for index, (name, display_name) in LANDMARK_DEFINITIONS.items()
    }
