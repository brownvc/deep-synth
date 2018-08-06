from data.house import *

"""
Just filters by room type
A room may have multiple types, all must match
This removes dual function rooms e.g. bedroom+kitchen
"""
def room_type_criteria(types):
    if not isinstance(types, list): types = [types]
    def wrapper(room, _):
        return all(t in types for t in room.roomTypes) and len(room.roomTypes)>0
    return wrapper

