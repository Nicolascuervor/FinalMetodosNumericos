import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import os

def parse_function(func_str):
    """
    Convierte la cadena en una función evaluable (f) con sympy,
    devuelve también la expresión simbólica y la variable x.
    """
    x = sp.Symbol('x')

    # Reemplazar cualquier referencia a math con sympy
    func_str = func_str.replace('math.', '')

    try:
        # Convertir la cadena a expresión simbólica
        expr = sp.sympify(func_str, locals={'x': x, 'exp': sp.exp})

        # Crear función lambda
        f = sp.lambdify(x, expr, {'exp': np.exp, 'numpy': np})

        return f, expr, x
    except Exception as e:
        raise ValueError(f"Error al parsear la función: {e}")

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
    # Parsear la función
    f, expr, x = parse_function(func_str)

    # Calcular la derivada simbólicamente
    try:
        deriv_expr = sp.diff(expr, x)
        print("Derivada simbólica:", deriv_expr)  # Imprimir la derivada simbólica

        # Crear función de derivada
        df_x = sp.lambdify(x, deriv_expr, {'exp': np.exp, 'numpy': np})
    except Exception as e:
        raise ValueError(f"Error al calcular la derivada: {e}")

    # Inicializar variables
    results = []
    xi = x0
    iteracion = 0
    error = 100.0  # Iniciar con 100% de error

    while (mode == 'error' and error > tol) or (mode == 'iteraciones' and iteracion < max_iter):
        # Calcular valores de la función y su derivada
        f_xi = f(xi)
        f_deriv_xi = df_x(xi)

        print(f"Iteración {iteracion}:")
        print(f"xi = {xi}")
        print(f"f(xi) = {f_xi}")
        print(f"f'(xi) = {f_deriv_xi}")

        # Verificar si la derivada es cero para evitar división por cero
        if abs(f_deriv_xi) < 1e-10:
            print("Derivada muy cercana a cero")
            break

        # Calcular nuevo punto
        xi_new = xi - (f_xi / f_deriv_xi)

        # Calcular error
        if iteracion > 0:
            error = abs((xi_new - xi) / xi_new) * 100
        else:
            error = 100.0

        # Almacenar resultados
        results.append({
            'iteracion': iteracion,
            'xi': xi_new,
            'f(xi)': f_xi,
            'f_deriv_xi': f_deriv_xi,
            'error': error
        })

        print(f"Error = {error}%")
        print("-" * 30)

        # Actualizar para próxima iteración
        xi = xi_new
        iteracion += 1

        # Condición de parada adicional
        if iteracion >= max_iter and mode == 'error':
            break

    return results

def newton_raphson_plot_data(func_str, x0, results):
    """
    Genera datos para graficar el método de Newton-Raphson.
    """
    f, expr, x = parse_function(func_str)
    deriv_expr = sp.diff(expr, x)
    f_deriv = sp.lambdify(x, deriv_expr, 'numpy')

    # Manejar diferentes estructuras de resultados
    try:
        # Primero intentar con 'x'
        x_final = results[-1]['x']
    except (KeyError, TypeError):
        try:
            # Si falla, intentar con 'xi'
            x_final = results[-1]['xi']
        except (KeyError, TypeError):
            # Si ambos fallan, usar el punto inicial
            x_final = x0

    # Calcular valor de la función en el punto final
    y_final = f(x_final)

    # Rango de x para graficar
    min_x = min(x0, x_final) - abs(x_final) * 0.5
    max_x = max(x0, x_final) + abs(x_final) * 0.5
    x_vals = np.linspace(min_x, max_x, 300)
    y_vals = f(x_vals)

    img_path = os.path.join("static", "newton_raphson.png")

    x_final = results[-1]['xi'] if results else x0

    return {
        "x_vals": x_vals.tolist(),
        "y_vals": y_vals.tolist(),
        "x_final": x_final,
        "y_final": y_final,
        "xr": x_final,  # Para que coincida con la plantilla
        "y_xr": y_final  # Para que coincida con la plantilla
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
