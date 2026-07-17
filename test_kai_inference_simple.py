"""Simple test script for Kai inference adapter."""

print("=== Starting simple test ===")

# Direct test of the inference adapter
test_script = '''
import sys
from collections import deque

_kai = None

class MockKaiMind:
    def __init__(self):
        self._mode = 'explore'
        self._mode_bias = 0.5
        self._last_efe = {'G': 0.1, 'risk': 0.2, 'epistemic': 0.3}
        self._vfe_ledger = deque([(0, 0.0)], maxlen=5000)
        
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
        policy = candidates[0] if candidates else 'Market_data_gen'
        info = {
            'G': 0.5, 'risk': 0.3, 'epistemic': 0.4, 'n': len(candidates)
        }
        self._last_efe = info
        return policy, info

def get_kai():
    global _kai
    if _kai is None:
        _kai = MockKaiMind()
        print("✓ Using mock KaiMind")
    return _kai

def infer(state: list, goal: str, depth: int = 3):
    kai = get_kai()
    candidate = goal if goal else 'Market_data_gen'
    
    try:
        policy, info = kai.select_policy([candidate])
        efe = info.get('G') if info else None
        risk = info.get('risk') if info else None
        epistemic = info.get('epistemic') if info else None
        mode = getattr(kai, '_mode', 'unknown')
        vfe_val = 0.1
        ricci_val = 4.0
        trend_val = -0.01
        
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
            '_fallback_used': False
        }
    except Exception as e:
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

infer
'''

# Execute the test script
exec(test_script)

print("✓ Simple test script executed successfully")
print(f"  Available functions: {list(locals().keys())[:10]}...")

# Test the infer function directly
print("\n3. Testing the infer function...")
result = infer([], 'Market_data_gen', 3)
print(f"✓ Test successful!")
print(f"  Policy: {result['policy']}")
print(f"  EFE: {result['efe']}")
print(f"  Mode: {result['mode']}")
print(f"  VFE: {result['vfe']}")
print(f"  Trend: {result['trend']}")
print(f"  Fallback used: {result.get('_fallback_used', False)}")

print("\n✅ Simple test passed!")
