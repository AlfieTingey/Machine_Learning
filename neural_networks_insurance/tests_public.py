from part2_tests import tests_part2
from part3_tests import tests_part3, AImarket
from part2_claim_classifier import ClaimClassifier
from part3_pricing_model import PricingModel


# Check for part2 load function
try:
    from part2_claim_classifier import load_model as part2_loader
except ImportError:
    part2_loader = None

# Check for part3 load function
try:
    from part3_pricing_model import load_model as part3_loader
except ImportError:
    part3_loader = None

# Check for part3 linear load function
try:
    from part3_pricing_model import load_model_linear as part3_loader_linear
except ImportError:
    part3_loader_linear = None

if __name__ == "__main__":
    print('\nPART 2 test output:\n')
    test_results_part2 = tests_part2(is_public=True, load_function=part2_loader)
    print('\nPART 3 test output:\n')
    test_results_part3 = tests_part3(is_public=True, load_function=part3_loader, load_function_linear=part3_loader_linear)
    test_results = test_results_part2 + test_results_part3
    print(test_results)
