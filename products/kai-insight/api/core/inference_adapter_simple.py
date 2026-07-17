'''Simple inference adapter importing KaiMind directly from the repo root.
'''
import sys
import os

# Add the repo root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../../../')

from kai_mind import KaiMind

_kai = None

def get_kai() -> KaiMind:
    global _kai
    if _kai is None:
        _kai = KaiMind()
    return _kai

def infer(state: list, goal: str, depth: int = 3) -> dict:
    '''Call Kai to compute expected free energy and select policy.
    This is a simplified version that just calls select_policy with the goal.
    '''
    kai = get_kai()
    # The select_policy method expects a list of candidate choices
    # For inference, we'll just pass the goal as the single candidate
    candidate = goal if goal else 'Market_data_gen'
    policy, info = kai.select_policy([candidate])
    return {
        'policy': policy,
        'efe': info.get('G'),
        'risk': info.get('risk'),
        'epistemic': info.get('epistemic'),
        'mode': kai._mode if hasattr(kai, '_mode') else 'unknown',
        'vfe': kai.vfe,
        'ricci': kai.ricci,
        'trend': kai.vfe_trend(),
        'info': info,
    }
