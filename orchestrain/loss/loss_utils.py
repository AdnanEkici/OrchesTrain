from __future__ import annotations


def calculate_equation(equation: str, **kwargs) -> float:
    """Exec the euqation string and return the result

    Parameters
    ----------
    equation : str
        string of equation

    Returns
    -------
    float
        the result of the equation

    Raises
    ------
    Exception
        if result is None

    """
    equation_str = f"result = {equation}"
    loc = {}
    loc.update(kwargs)
    exec(equation_str, globals(), loc)
    result = loc["result"]

    if result is None:
        raise Exception(f"equation: {equation} is not valid!")

    return result
