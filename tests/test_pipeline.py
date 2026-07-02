import unittest
import math

def calculate_clt_sample_mean(sample_values):
    """Reflected logic: calculates sample mean for CLT simulations."""
    if not sample_values:
        return 0.0
    return round(sum(sample_values) / len(sample_values), 2)

def calculate_confidence_interval(mean, std_dev, n, confidence=0.95):
    """Reflected logic: calculates z-based confidence intervals."""
    if n <= 0 or std_dev <= 0:
        return mean, mean
    # z-value for 95% is 1.96
    z = 1.96 if confidence == 0.95 else 2.58
    margin = z * (std_dev / math.sqrt(n))
    return round(mean - margin, 2), round(mean + margin, 2)

class TestWalmartPipeline(unittest.TestCase):
    def test_sample_mean(self):
        self.assertEqual(calculate_clt_sample_mean([10, 20, 30]), 20.0)
        self.assertEqual(calculate_clt_sample_mean([]), 0.0)
        
    def test_confidence_interval(self):
        lower, upper = calculate_confidence_interval(100, 15, 100, 0.95)
        self.assertEqual(lower, 97.06) # 100 - 1.96 * 1.5
        self.assertEqual(upper, 102.94)

if __name__ == '__main__':
    unittest.main()
