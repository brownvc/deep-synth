from data.house import *
import utils

"""
Good houses
The list of good houses should be included in filters/ already
In addition, if a house is not scaled to meters, we reject those
"""
def good_house_criteria(house):
    data_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f"{data_dir}/good_houses", 'r') as f:
        good_houses = [s[:-1] for s in f.readlines()[1:]]
    if house.scaleToMeters != 1: return False
    if not house.id in good_houses: return False
    return True

