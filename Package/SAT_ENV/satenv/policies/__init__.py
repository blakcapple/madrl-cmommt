from .escape import Escape
from .random_move import RandomMove
from .rlpolicy import RLPolicy
from .acmommt import ACMOMMT

policy_dict = {
    'escape': Escape,
    'random': RandomMove,
    'a_cmommt': ACMOMMT,
    'rl': RLPolicy,
}


