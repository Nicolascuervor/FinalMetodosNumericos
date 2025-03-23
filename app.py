from flask import Flask, render_template, request, redirect, url_for, flash
from methods import (
    biseccion, biseccion_plot_data,
    falsa_posicion, falsa_posicion_plot_data,
    fixed_point, fixed_point_plot_data,
    newton_raphson, newton_raphson_plot_data,
    secante, secante_plot_data,
    evaluate_expression
)

app = Flask(__name__)
app.secret_key = 'clave_secreta'

@app.route('/')
def index():
    return render_template('index.html')

# -------------------------
# Bisección
# -------------------------
@app.route('/biseccion', methods=['GET', 'POST'])
def metodo_biseccion():
    if request.method == 'POST':
        func_str = request.form['funcion']
        a = float(request.form['a'])
        b = float(request.form['b'])
        modo = request.form['modo']
        valor = float(request.form['valor'])

        # Dependiendo de 'modo', se usan iteraciones o error
        if modo == 'iteraciones':
            results = biseccion(func_str, a, b, tol=1e-6, max_iter=int(valor), mode='iteraciones')
        else:  # modo == 'error'
            results = biseccion(func_str, a, b, tol=valor, max_iter=50, mode='error')

        # Obtenemos el último xr
        xr = results[-1]['xr']
        # Datos para la gráfica
        plot_data = biseccion_plot_data(func_str, a, b, xr)

        return render_template('biseccion.html',
                               resultados=results,
                               funcion=func_str,
                               plot_data=plot_data)
    return render_template('biseccion_form.html')

# -------------------------
# Falsa Posición
# -------------------------
@app.route('/falsa_posicion', methods=['GET', 'POST'])
def metodo_falsa_posicion():
    if request.method == 'POST':
        func_str = request.form['funcion']
        a = float(request.form['a'])
        b = float(request.form['b'])
        modo = request.form['modo']
        valor = float(request.form['valor'])

        if modo == 'iteraciones':
            results = falsa_posicion(func_str, a, b, tol=1e-6, max_iter=int(valor), mode='iteraciones')
        else:
            results = falsa_posicion(func_str, a, b, tol=valor, max_iter=50, mode='error')

        xr = results[-1]['xr']
        plot_data = falsa_posicion_plot_data(func_str, a, b, xr)

        return render_template('falsa_posicion.html',
                               resultados=results,
                               funcion=func_str,
                               plot_data=plot_data)
    return render_template('falsa_posicion_form.html')

# -------------------------
# Punto Fijo
# -------------------------
@app.route('/fixed_point', methods=['GET', 'POST'])
def metodo_fixed_point():
    if request.method == 'POST':
        func_str = request.form['funcion']
        x0 = float(request.form['x0'])
        modo = request.form['modo']
        valor = float(request.form['valor'])
        g_str = request.form.get('g', None)

        if modo == 'iteraciones':
            results, used_g = fixed_point(func_str, x0, tol=1e-6, max_iter=int(valor),
                                          mode='iteraciones', g_str=g_str)
        else:
            results, used_g = fixed_point(func_str, x0, tol=valor, max_iter=50,
                                          mode='error', g_str=g_str)

        plot_data = fixed_point_plot_data(func_str, g_str, x0, results)

        return render_template('fixed_point.html',
                               resultados=results,
                               funcion=func_str,
                               used_g=used_g,
                               plot_data=plot_data)
    return render_template('fixed_point_form.html')

# -------------------------
# Newton-Raphson
# -------------------------
@app.route('/newton_raphson', methods=['GET', 'POST'])
def metodo_newton_raphson():
    if request.method == 'POST':
        func_str = request.form['funcion']
        x0 = float(request.form['x0'])
        modo = request.form['modo']
        valor = float(request.form['valor'])

        try:
            if modo == 'iteraciones':
                results = newton_raphson(func_str, x0, tol=1e-6, max_iter=int(valor),
                                         mode='iteraciones')
            else:
                results = newton_raphson(func_str, x0, tol=valor, max_iter=50,
                                         mode='error')
        except ZeroDivisionError as e:
            flash(str(e))
            return redirect(url_for('metodo_secante'))

        plot_data = newton_raphson_plot_data(func_str, x0, results)

        return render_template('newton_raphson.html',
                               resultados=results,
                               funcion=func_str,
                               plot_data=plot_data)
    return render_template('newton_raphson_form.html')

# -------------------------
# Secante
# -------------------------
@app.route('/secante', methods=['GET', 'POST'])
def metodo_secante():
    if request.method == 'POST':
        func_str = request.form['funcion']
        x0 = float(request.form['x0'])
        x1 = float(request.form['x1'])
        modo = request.form['modo']
        valor = float(request.form['valor'])

        results = []
        try:
            if modo == 'iteraciones':
                results = secante(func_str, x0, x1, tol=1e-6, max_iter=int(valor),
                                  mode='iteraciones')
            else:
                results = secante(func_str, x0, x1, tol=valor, max_iter=50,
                                  mode='error')
        except ZeroDivisionError as e:
            flash(str(e))
            return redirect(url_for('metodo_secante'))

        plot_data = secante_plot_data(func_str, x0, x1, results)

        return render_template('secante.html',
                               resultados=results,
                               funcion=func_str,
                               plot_data=plot_data)
    return render_template('secante_form.html')

# -------------------------
# Calculadora
# -------------------------
@app.route('/calculator', methods=['GET', 'POST'])
def calculator():
    result = None
    if request.method == 'POST':
        expr_str = request.form['expression']
        try:
            result = evaluate_expression(expr_str)
        except Exception as e:
            flash(f"Error al evaluar la expresión: {e}")
    return render_template('calculator.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
