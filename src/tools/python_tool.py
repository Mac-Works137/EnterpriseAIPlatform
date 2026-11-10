
"""Safe numerical computation tool for the enterprise platform."""

import json
import math
import statistics

from langchain_core.tools import tool


@tool
def calculate_statistics(
    values: list[float],
    operation: str,
) -> str:
    """Calculate numerical statistics for a list of numbers.

    Supported operations:
    - mean: arithmetic average
    - median: middle value
    - standard_deviation: population standard deviation
    - minimum: smallest value
    - maximum: largest value
    - sum: total of all values
    - count: number of values
    - percentage_change: percentage change from the first value
      to the last value; requires at least two values and a nonzero
      first value.

    Use this tool for numerical calculations that are not best handled
    by a SQL query or document retrieval.
    """
    if not values:
        return json.dumps({"error": "Provide at least one number."})

    if len(values) > 10000:
        return json.dumps({
            "error": "A maximum of 10,000 values is allowed."
        })

    try:
        numbers = [float(value) for value in values]
    except (TypeError, ValueError):
        return json.dumps({"error": "All values must be numeric."})

    if not all(math.isfinite(number) for number in numbers):
        return json.dumps({
            "error": "Values must be finite numbers."
        })

    operation = operation.strip().lower()

    operations = {
        "mean": lambda: statistics.mean(numbers),
        "median": lambda: statistics.median(numbers),
        "standard_deviation": lambda: statistics.pstdev(numbers),
        "minimum": lambda: min(numbers),
        "maximum": lambda: max(numbers),
        "sum": lambda: sum(numbers),
        "count": lambda: len(numbers),
    }

    if operation == "percentage_change":
        if len(numbers) < 2:
            return json.dumps({
                "error": "Percentage change requires at least two values."
            })

        if numbers[0] == 0:
            return json.dumps({
                "error": "The first value cannot be zero."
            })

        result = ((numbers[-1] - numbers[0]) / abs(numbers[0])) * 100

    elif operation in operations:
        result = operations[operation]()

    else:
        return json.dumps({
            "error": (
                "Unsupported operation. Choose mean, median, "
                "standard_deviation, minimum, maximum, sum, count, "
                "or percentage_change."
            )
        })

    return json.dumps({
        "operation": operation,
        "count": len(numbers),
        "result": result,
    })


def main() -> None:
    test_cases = [
        ("mean", [10, 20, 30, 40, 50]),
        ("standard_deviation", [10, 20, 30, 40, 50]),
        ("percentage_change", [100, 125]),
    ]

    for operation, values in test_cases:
        print(f"\nOperation: {operation}")
        print(f"Values: {values}")
        print(
            calculate_statistics.invoke({
                "values": values,
                "operation": operation,
            })
        )


if __name__ == "__main__":
    main()
