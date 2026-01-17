ARCHS = {
    "tiny": {
        "stem_ch": 32,
        "stages": [(32, 2, False), (64, 2, True), (128, 2, True), (256, 1, True)],
    },
    "base": {
        "stem_ch": 64,
        "stages": [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)],
    },
    "deeper": {
        "stem_ch": 64,
        "stages": [(64, 3, False), (128, 3, True), (256, 3, True), (512, 2, True)],
    },
    "wider": {
        "stem_ch": 80,
        "stages": [(80, 2, False), (160, 2, True), (320, 2, True), (640, 1, True)],
    },
}
