from enum import Enum

class GestureCategories(Enum):
    BEATS = 0
    WISH_WASH = 1
    THIS_THAT = 2
    TIME_SIDE_SWIPE = 3
    TIME_FORWARD_SWIPE = 4
    MIXING = 5
    BIG_OPENING = 6


LABELED_GESTURES = [
    {
        id: 73867
        cat: GestureCategories.BEATS
    },
    {
        id: 73848
        cat: GestureCategories.THIS_THAT
    },
    {
        id: 88132
        cat: GestureCategories.TIME_SIDE_SWIPE
    },
    {
        id: 88224
        cat: GestureCategories.TIME_SIDE_SWIPE
    },
    {
        id: 88148
        cat: GestureCategories.MIXING
    },
    {
        id: 88130
        cat: GestureCategories.WISH_WASH
    },
    {
        id: 88131
        cat: GestureCategories.WISH_WASH
    },
    {
        id: 84939
        cat: GestureCategories.BIG_OPENING
    }
]
