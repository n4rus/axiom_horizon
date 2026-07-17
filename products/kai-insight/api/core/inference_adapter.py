'''Kai reasoning API adapter - standalone implementation.
This adapter uses direct imports from the repository root to avoid path issues.
'''

import sys
import os
import math

# Global variables
_kai = None

# Define a simplified standalone version for testing if KaiMind can't be imported
class MockKaiMind:
    def __init__(self):
        self._mode = 'explore'
        self._mode_bias = 0.5
        self._prime_goal = 'Market_data_gen'
        self._last_efe = {'G': 0.1, 'risk': 0.2, 'epistemic': 0.3, 'n': 1, 'beta_explore': 0.8}
        self.vfe_history = [0.0]
        self._vfe_history = deque([0.0], maxlen=500)
        self._vfe_ledger = deque([(0, 0.0)], maxlen=5000)
        self._last_efe = None
        
    def vfe_trend(self, window=500):
        pts = list(self._vfe_ledger)[-window:]
        n = len(pts)
        if n < 3:
            return 0.0
        xs = list(range(n))
        ys = [p[1] for p in pts]
        mx = sum(xs) / n
        my = sum(ys) / n
        vx = sum((x - mx) ** 2 for x in xs)
        if vx <= 1e-12:
            return 0.0
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        return cov / vx

    def select_policy(self, candidates):
        '''Simplified policy selection'''
        policy = candidates[0] if candidates else 'Market_data_gen'
        info = {
            'G': 0.5, 'risk': 0.3, 'epistemic': 0.4, 'n': len(candidates),
            'beta_explore': round((self._mode_bias + 1.0) / 2.0, 3)
        }
        self._last_efe = info
        return policy, info

def get_kai():
    global _kai
    if _kai is None:
        try:
            # Try to import the real KaiMind
            from kai_mind import KaiMind
            _kai = KaiMind()
        except ImportError:
            print("⚠️  Warning: kai_mind import failed, using mock implementation")
            _kai = MockKaiMind()
    return _kai

def infer(state: list, goal: str, depth: int = 3) -> dict:
    '''Call Kai to compute expected free energy and select policy.
    State must be an embedding vector from the brain model.
    Goal is an attractor label (prime goal name).
    Returns:
        result = {
            'policy': <chosen_action_style> (list),
            'efe': <expected_free_energy>
            'risk': <risk>
            'epistemic': <epistemic>
            'mode': <current cognitive mode>
            'vfe': <variational_free_energy>
            'ricci': <ricci_curvature>
            'trend': <vfe_trend>
        }
    '''
    from collections import deque  # Import here to use in MockKaiMind
    
    kai = get_kai()
    
    # Handle state parameter - could be embedding or just context
    if not isinstance(state, list):
        state = []
    
    # Determine goal to use
    candidate = goal if goal else 'Market_data_gen'
    
    try:
        policy, info = kai.select_policy([candidate])
        
        # Extract info fields with safe defaults
        efe = info.get('G') if info else None
        risk = info.get('risk') if info else None
        epistemic = info.get('epistemic') if info else None
        mode = getattr(kai, '_mode', 'unknown')
        
        # Get VFE, Ricci, and Trend values with safe defaults
        vfe_val = getattr(kai, 'vfe', lambda: 0.0)()
        ricci_val = getattr(kai, 'ricci', lambda: 0.0)()
        trend_val = getattr(kai, 'vfe_trend', lambda: 0.0)()
        
        # For MockKaiMind, ensure the attributes exist
        if hasattr(kai, 'vfe'):
            vfe_val = kai.vfe
        if hasattr(kai, 'ricci'):
            ricci_val = kai.ricci
        if hasattr(kai, 'vfe_trend'):
            trend_val = kai.vfe_trend()
        
        return {
            'policy': policy,
            'efe': efe,
            'risk': risk,
            'epistemic': epistemic,
            'mode': mode,
            'vfe': vfe_val,
            'ricci': ricci_val,
            'trend': trend_val,
            'info': info,
            '_fallback_used': isinstance(type(kai).__name__, str) and type(kai).__name__ == 'MockKaiMind'
        }
    except Exception as e:
        # Return a valid response even if something fails
        import traceback
        print(f"⚠️  Inference error: {e}")
        return {
            'policy': candidate,
            'efe': 0.5,
            'risk': 0.3,
            'epistemic': 0.4,
            'mode': 'explore',
            'vfe': 0.1,
            'ricci': 4.0,
            'trend': -0.01,
            'info': {'G': 0.5, 'risk': 0.3, 'epistemic': 0.4},
            '_fallback_used': True,
            'error': str(e)
        }
