import numpy as np
import sympy as sp

def parse_function(func_str):
    """
    Convierte la cadena en una función evaluable (f) con sympy,
    devuelve también la expresión simbólica y la variable x.
    """
    x = sp.symbols('x', real=True)
    expr = sp.sympify(func_str)
    f = sp.lambdify(x, expr, 'numpy')
    return f, expr, x

# -------------------------
# Métodos Numéricos
# -------------------------

def biseccion(func_str, a, b, tol=0.1, max_iter=50, mode='error'):
    """
    Retorna la tabla de iteraciones para la Bisección.
    """
    f, _, _ = parse_function(func_str)
    results = []
    iteracion = 0
    xr = a
    error = 100.0

    while (mode=='error' and error > tol) or (mode=='iteraciones' and iteracion < max_iter):
        xr_old = xr
        xr = (a + b) / 2.0
        f_xr = f(xr)
        f_a = f(a)
        if iteracion > 0:
            error = abs((xr - xr_old)/xr)*100
        else:
            error = 100.0

        results.append({
            'iteracion': iteracion,
            'a': a,
            'xr': xr,
            'b': b,
            'f(xr)': f_xr,
            'error': error
        })

        if f_a * f_xr < 0:
            b = xr
        else:
            a = xr

        iteracion += 1
        if iteracion >= max_iter and mode=='error':
            break

    return results

def biseccion_plot_data(func_str, a, b, xr):
    """
    Genera datos para Plotly (x_vals, y_vals, etc.) en la Bisección.
    """
    f, _, _ = parse_function(func_str)
    x_vals = np.linspace(a, b, 300)
    y_vals = f(x_vals)
    y_xr = f(xr)
    return {
        "x_vals": x_vals.tolist(),
        "y_vals": y_vals.tolist(),
        "xr": xr,
        "y_xr": y_xr
    }

def falsa_posicion(func_str, a, b, tol=0.1, max_iter=50, mode='error'):
    f, _, _ = parse_function(func_str)
    results = []
    iteracion = 0
    xr = a
    error = 100.0

    while (mode=='error' and error > tol) or (mode=='iteraciones' and iteracion < max_iter):
        xr_old = xr
        f_a = f(a)
        f_b = f(b)
        if (f_a - f_b) == 0:
            break
        xr = b - (f_b*(a-b)) / (f_a - f_b)
        f_xr = f(xr)
        if iteracion > 0:
            error = abs((xr - xr_old)/xr)*100
        else:
            error = 100.0
        results.append({
            'iteracion': iteracion,
            'a': a,
            'xr': xr,
            'b': b,
            'f(xr)': f_xr,
            'error': error
        })
        if f_a*f_xr < 0:
            b = xr
        else:
            a = xr
        iteracion += 1
        if iteracion >= max_iter and mode=='error':
            break
    return results

def falsa_posicion_plot_data(func_str, a, b, xr):
    f, _, _ = parse_function(func_str)
    x_vals = np.linspace(a, b, 300)
    y_vals = f(x_vals)
    y_xr = f(xr)
    return {
        "x_vals": x_vals.tolist(),
        "y_vals": y_vals.tolist(),
        "xr": xr,
        "y_xr": y_xr
    }

def fixed_point(func_str, x0, tol=0.1, max_iter=50, mode='error', g_str=None):
    """
    Método del punto fijo.
    Si no se proporciona g_str, usar g(x) = x + f(x) directamente.
    """
    f, expr, x = parse_function(func_str)

    if g_str and g_str.strip() != "":
        # Usa el g(x) proporcionado
        g_expr = sp.sympify(g_str)
        g = sp.lambdify(x, g_expr, 'numpy')
        used_g = g_str
    else:
        # Evitar solve() en funciones complejas => forzamos g(x) = x + f(x)
        g_expr = x + expr
        g = sp.lambdify(x, g_expr, 'numpy')
        used_g = "x + f(x)"

    results = []
    iteracion = 0
    xi = x0
    error = 100.0

    while (mode=='error' and error > tol) or (mode=='iteraciones' and iteracion < max_iter):
        xi_old = xi
        xi = g(xi_old)
        if iteracion > 0:
            error = abs((xi - xi_old)/xi)*100
        else:
            error = 100.0

        results.append({
            'iteracion': iteracion,
            'xi': xi,
            'error': error
        })

        iteracion += 1
        if iteracion >= max_iter and mode=='error':
            break

    return results, used_g

def fixed_point_plot_data(func_str, g_str, x0, results):
    f, _, x = parse_function(func_str)
    # Último valor
    xf = results[-1]['xi']
    yxf = f(xf)

    # Generamos un rango alrededor de x0 y xf
    min_x = min(x0, xf) - 5
    max_x = max(x0, xf) + 5
    x_vals = np.linspace(min_x, max_x, 300)
    y_f = f(x_vals)

    # Si se proporcionó g(x), graficamos
    if g_str and g_str.strip() != "":
        g_expr = sp.sympify(g_str)
        g = sp.lambdify(x, g_expr, 'numpy')
    else:
        g_expr = x + sp.sympify(func_str)
        g = sp.lambdify(x, g_expr, 'numpy')

    y_g = g(x_vals)

    return {
        "x_vals": x_vals.tolist(),
        "y_f": y_f.tolist(),
        "y_g": y_g.tolist(),
        "xf": xf,
        "yxf": yxf
    }

def newton_raphson(func_str, x0, tol=0.1, max_iter=50, mode='error'):
    f, expr, x = parse_function(func_str)
    deriv_expr = sp.diff(expr, x)
    f_deriv = sp.lambdify(x, deriv_expr, 'numpy')
    results = []
    iteracion = 0
    xi = x0
    error = 100.0

    while (mode=='error' and error > tol) or (mode=='iteraciones' and iteracion < max_iter):
        f_xi = f(xi)
        f_deriv_xi = f_deriv(xi)
        if f_deriv_xi == 0:
            raise ZeroDivisionError("La derivada se hizo 0. Considera el método de la secante.")
        xi_new = xi - f_xi/f_deriv_xi
        if iteracion > 0:
            error = abs((xi_new - xi)/xi_new)*100
        else:
            error = 100.0

        results.append({
            'iteracion': iteracion,
            'xi': xi_new,
            'error': error
        })

        xi = xi_new
        iteracion += 1
        if iteracion >= max_iter and mode=='error':
            break

    return results

def newton_raphson_plot_data(func_str, x0, results):
    f, expr, x = parse_function(func_str)
    deriv_expr = sp.diff(expr, x)
    f_deriv = sp.lambdify(x, deriv_expr, 'numpy')

    # Tomar el último xi
    x_final = results[-1]['xi']
    y_final = f(x_final)

    min_x = min(x0, x_final) - 5
    max_x = max(x0, x_final) + 5
    x_vals = np.linspace(min_x, max_x, 300)
    y_vals = f(x_vals)

    # Si quieres graficar las tangentes en cada iteración,
    # podrías generar puntos extra. Para simplificar, solo dibujamos f(x) y el último punto.
    return {
        "x_vals": x_vals.tolist(),
        "y_vals": y_vals.tolist(),
        "x_final": x_final,
        "y_final": y_final
    }

def secante(func_str, x0, x1, tol=0.1, max_iter=50, mode='error'):
    f, _, _ = parse_function(func_str)
    results = []
    error = 100.0
    xi_0 = x0
    xi_1 = x1
    iteracion = 0

    while (mode=='error' and error > tol) or (mode=='iteraciones' and iteracion < max_iter):
        f_x0 = f(xi_0)
        f_x1 = f(xi_1)
        if f_x1 - f_x0 == 0:
            break
        xi_new = xi_1 - f_x1*(xi_1 - xi_0)/(f_x1 - f_x0)
        if iteracion > 0:
            error = abs((xi_new - xi_1)/xi_new)*100
        else:
            error = 100.0

        results.append({
            'iteracion': iteracion,
            'xi': xi_new,
            'error': error
        })
        xi_0 = xi_1
        xi_1 = xi_new
        iteracion += 1
        if iteracion >= max_iter and mode=='error':
            break

    return results

def secante_plot_data(func_str, x0, x1, results):
    f, _, _ = parse_function(func_str)
    x_final = results[-1]['xi']
    y_final = f(x_final)

    min_x = min(x0, x1, x_final) - 5
    max_x = max(x0, x1, x_final) + 5
    x_vals = np.linspace(min_x, max_x, 300)
    y_vals = f(x_vals)

    return {
        "x_vals": x_vals.tolist(),
        "y_vals": y_vals.tolist(),
        "x_final": x_final,
        "y_final": y_final
    }

# -------------------------
# Calculadora
# -------------------------

def evaluate_expression(expr_str):
    expr = sp.sympify(expr_str)
    return expr.evalf()
