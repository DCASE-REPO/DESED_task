"""
we store here a dict where we define the encodings for all classes in DESED task.
"""

from collections import OrderedDict

classes_labels_desed = OrderedDict(
    {
        "Alarm_bell_ringing": 0,
        "Blender": 1,
        "Cat": 2,
        "Dishes": 3,
        "Dog": 4,
        "Electric_shaver_toothbrush": 5,
        "Frying": 6,
        "Running_water": 7,
        "Speech": 8,
        "Vacuum_cleaner": 9,
    }
)


classes_labels_maestro_real = OrderedDict(
    {
        "cutlery and dishes": 0,
        "furniture dragging": 1,
        "people talking": 2,
        "children voices": 3,
        "coffee machine": 4,
        "footsteps": 5,
        "large_vehicle": 6,
        "car": 7,
        "brakes_squeaking": 8,
        "cash register beeping": 9,
        "announcement": 10,
        "shopping cart": 11,
        "metro leaving": 12,
        "metro approaching": 13,
        "door opens/closes": 14,
        "wind_blowing": 15,
        "birds_singing": 16,
    }
)

classes_labels_maestro_synth = OrderedDict(
    {
        "car_horn": 0,
        "children_voices": 1,
        "engine_idling": 2,
        "siren": 3,
        "street_music": 4,
        "dog_bark": 5,
    }
)

classes_labels_maestro_real_eval = {
    "birds_singing",
    "car",
    "people talking",
    "footsteps",
    "children voices",
    "wind_blowing",
    "brakes_squeaking",
    "large_vehicle",
    "cutlery and dishes",
    "metro approaching",
    "metro leaving",
}

maestro_desed_alias = {
    "people talking": "Speech",
    "children voices": "Speech",  # both synth and real
    "announcement": "Speech",
    "cutlery and dishes": "Dishes",
    # these are from synth
    "dog_bark": "Dog",
}
