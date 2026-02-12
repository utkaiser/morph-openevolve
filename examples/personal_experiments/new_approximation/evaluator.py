import math
import importlib.util
import os

def evaluate(program_path):
    """
    Evaluates how well the program approximates the sine function.
    """
    try:
        # Load the evolved program
        spec = importlib.util.spec_from_file_location("evolved_code", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, 'approximate_function'):
            return {"combined_score": 0.0, "error": "Function 'approximate_function' not found"}
        
        approx_func = module.approximate_function
        
        # Dataset: (x, sin(x)) pairs over a range (e.g., 0 to 8)
        test_points = [i * (8 / 20) for i in range(21)]
        
        total_error = 0.0
        max_error = 0.0
        
        for x in test_points:
            expected = math.sin(x)
            try:
                actual = approx_func(x)
                error = abs(actual - expected)
                total_error += error**2
                max_error = max(max_error, error)
            except Exception as e:
                return {"combined_score": 0.0, "error": f"Execution error: {str(e)}"}
        
        mse = total_error / len(test_points)
        
        # Score calculation: 1.0 / (1.0 + MSE)
        # Higher score is better (max 1.0)
        score = 1.0 / (1.0 + mse)
        
        return {
            "combined_score": score,
            "mse": mse,
            "max_error": max_error,
            "complexity": len(open(program_path).read())
        }
        
    except Exception as e:
        return {"combined_score": 0.0, "error": str(e)}
