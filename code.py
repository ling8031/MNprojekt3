import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

def get_routes_from_files(directory_path: str):
    routes = []
    files = glob.glob(os.path.join(directory_path, '*.txt'))

    for file in files:
        try:
            df = pd.read_csv(file)
            df.columns = ['Distance_m', 'Elevation_m']
            routes.append((file, df))
        except Exception as e:
            print(f"Błąd wczytywania pliku {file}: {e}")
    return routes

def take_nodes(data, number):
    indices = np.linspace(0, len(data) - 1, number, dtype=int)
    return data.iloc[indices]

def lagrange_interpolation(data, node_number):
    nodes = take_nodes(data, node_number)
    x_nodes = nodes['Distance_m'].values
    y_nodes = nodes['Elevation_m'].values

    a, b = x_nodes.min(), x_nodes.max()

    # skala do [-1, 1]
    x_scaled = (2 * x_nodes - (b + a)) / (b - a)

    def lagrange(x_eval):
        x_eval = np.asarray(x_eval)

        is_scalar = False
        if x_eval.ndim == 0:
            x_eval = x_eval[np.newaxis] 
            is_scalar = True

        x_eval_scaled = (2 * x_eval - (b + a)) / (b - a)
        result = np.zeros_like(x_eval_scaled, dtype=float)

        n = len(x_scaled)
        for k, x in enumerate(x_eval_scaled):
            total = 0
            for i in range(n):
                term = y_nodes[i]
                for j in range(n):
                    if i != j:
                        term *= (x - x_scaled[j]) / (x_scaled[i] - x_scaled[j])
                total += term
            result[k] = total

        return result[0] if is_scalar else result

    return lagrange


import numpy as np

class CubicSpline:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = len(x) - 1          # liczba segmentów
        self.h = np.diff(x)          # różnice między węzłami

        # współczynniki
        self.a = y.copy()            # a_i = f(x_i)
        self.b = np.zeros(self.n)    
        self.c = np.zeros(self.n )
        self.d = np.zeros(self.n)   

        self._compute_coefficients()

    def _compute_coefficients(self):
        n = self.n
        h = self.h
        y = self.y
        
        # liczba niewiadomych: 4*n (a_i,b_i,c_i,d_i dla i=0..n-1)
        size = 4 * n
        A = np.zeros((size, size))
        b = np.zeros(size)

        # warunki interpolacji
        for i in range(n):
            # S_i(x_i) = a_i = y_i
            A[2*i, 4*i] = 1
            b[2*i] = y[i]

            # S_i(x_{i+1}) = a_i + b_i h_i + c_i h_i^2 + d_i h_i^3 = y_{i+1}
            A[2*i + 1, 4*i] = 1
            A[2*i + 1, 4*i + 1] = h[i]
            A[2*i + 1, 4*i + 2] = h[i]**2
            A[2*i + 1, 4*i + 3] = h[i]**3
            b[2*i + 1] = y[i+1]

        # równania ciągłości 1. pochodnej
        for i in range(n - 1):
            A[2*n + i, 4*i + 1] = 1
            A[2*n + i, 4*i + 2] = 2 * h[i]
            A[2*n + i, 4*i + 3] = 3 * h[i]**2
            A[2*n + i, 4*(i+1) + 1] = -1
            b[2*n + i] = 0

        # równania ciągłości 2. pochodnej
        for i in range(n - 1):
            row = 3*n - 1 + i 

            A[row, 4*i + 2] = 2
            A[row, 4*i + 3] = 6 * h[i]
            A[row, 4*(i+1) + 2] = -2

            b[row] = 0

        # druga pochodna na końcach równa 0
        # S''_0(x0) = 0
        A[-2, 2] = 2
        b[-2] = 0
        # S''_{n-1}(x_n) = 0
        A[-1, 4*(n-1) + 2] = 2
        A[-1, 4*(n-1) + 3] = 6 * h[n-1]
        b[-1] = 0

        # rozwiązanie
        coeffs = np.linalg.solve(A, b)

        # wypełnienie współczynników a,b,c,d
        for i in range(n):
            self.a[i] = coeffs[4*i]
            self.b[i] = coeffs[4*i + 1]
            self.c[i] = coeffs[4*i + 2]
            self.d[i] = coeffs[4*i + 3]


    def evaluate(self, x_eval):
        x_eval = np.array(x_eval)
        results = np.zeros_like(x_eval, dtype=float)

        for i in range(len(x_eval)):
            x_val = x_eval[i]
            # segment w którym leży x_val
            idx = np.searchsorted(self.x, x_val) - 1
            if idx < 0:
                idx = 0
            elif idx >= self.n:
                idx = self.n - 1

            dx = x_val - self.x[idx]

            # wartość splajnu w punkcie
            results[i] = (
                self.a[idx] +
                self.b[idx] * dx +
                self.c[idx] * dx ** 2 +
                self.d[idx] * dx ** 3
            )
        return results


def plot_spline_from_data(data: pd.DataFrame, number_of_nodes: int, label: str = "Spline kubiczny", save_path: str = None):
    nodes = take_nodes(data, number_of_nodes)

    x_nodes = nodes['Distance_m'].values
    y_nodes = nodes['Elevation_m'].values

    spline = CubicSpline(x_nodes, y_nodes)

    x_dense = np.linspace(min(x_nodes), max(x_nodes), 500)
    y_dense = spline.evaluate(x_dense)

    plt.figure(figsize=(10, 6))
    plt.plot(data['Distance_m'], data['Elevation_m'], 'o', alpha=0.3, label='Oryginalne dane')
    plt.plot(x_nodes, y_nodes, 'o', label=f'Węzły ({number_of_nodes})')
    plt.plot(x_dense, y_dense, '-', label=label)
    plt.xlabel('Dystans (m)')
    plt.ylabel('Wysokość (m)')
    plt.legend()
    plt.grid(True)
    plt.title('Interpolacja spline\'em kubicznym')

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Wykres zapisany do pliku: {save_path}")
        plt.close()
    else:
        plt.show()

def plot_lagrange_from_data(data: pd.DataFrame, number_of_nodes: int, label: str = "Lagrange interpolacja", save_path: str = None):
    nodes = take_nodes(data, number_of_nodes)

    x_nodes = nodes['Distance_m'].values
    y_nodes = nodes['Elevation_m'].values

    lagrange_func = lagrange_interpolation(data, number_of_nodes)

    x_dense = np.linspace(min(x_nodes), max(x_nodes), 500)
    y_dense = np.array([lagrange_func(x) for x in x_dense])

    plt.figure(figsize=(10, 6))
    plt.plot(data['Distance_m'], data['Elevation_m'], 'o', alpha=0.3, label='Oryginalne dane')
    plt.plot(x_nodes, y_nodes, 'o', label=f'Węzły ({number_of_nodes})')
    plt.plot(x_dense, y_dense, '-', label=label)
    plt.xlabel('Dystans (m)')
    plt.ylabel('Wysokość (m)')
    plt.legend()
    plt.grid(True)
    plt.title('Interpolacja metodą Lagrange’a')

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Wykres zapisany do pliku: {save_path}")
        plt.close()
    else:
        plt.show()

def take_nodes_chebyshev2(data: pd.DataFrame, number: int):
    a = data['Distance_m'].min()
    b = data['Distance_m'].max()
    indices = []

    for i in range(number):
        x_i = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(i * np.pi / (number - 1))
        # najbliższy punkt w danych
        closest_index = (np.abs(data['Distance_m'] - x_i)).idxmin()
        indices.append(closest_index)

    return data.loc[sorted(set(indices))]

def plot_analisis():
    import os
    os.makedirs("analiza_wykresy", exist_ok=True)

    routes = get_routes_from_files('./dane')
    node_counts = [5, 10, 15, 20]

    for filename, df in routes:
        base_name = os.path.splitext(os.path.basename(filename))[0]

        for count in node_counts:
            plot_spline_from_data(
                data=df,
                number_of_nodes=count,
                label=f"Spline ({count} węzłów)",
                save_path=f"analiza_wykresy/{base_name}_spline_{count}.png"
            )

            plot_lagrange_from_data(
                data=df,
                number_of_nodes=count,
                label=f"Lagrange ({count} węzłów)",
                save_path=f"analiza_wykresy/{base_name}_lagrange_{count}.png"
            )

        for count in [10, 15, 20, 25]:
            nodes = take_nodes_chebyshev2(df, count)
            x_nodes = nodes['Distance_m'].values
            y_nodes = nodes['Elevation_m'].values

            lagrange_func = lagrange_interpolation(nodes, count)
            x_dense = np.linspace(min(x_nodes), max(x_nodes), 500)
            y_dense = np.array([lagrange_func(x) for x in x_dense])

            plt.figure(figsize=(10, 6))
            plt.plot(df['Distance_m'], df['Elevation_m'], 'o', alpha=0.3, label='Oryginalne dane')
            plt.plot(x_nodes, y_nodes, 'o', label=f'Węzły Czebyszewa II ({count})')
            plt.plot(x_dense, y_dense, '-', label='Lagrange (Czebyszew II)')
            plt.xlabel('Dystans (m)')
            plt.ylabel('Wysokość (m)')
            plt.title(f'{base_name} – Lagrange (Czebyszew II, {count} węzłów)')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"analiza_wykresy/{base_name}_lagrange_chebyshev2_{count}.png")
            plt.close()

    print("Wszystkie wykresy wygenerowane do folderu 'analiza_wykresy'")


def main():
    plot_analisis()

if __name__ == "__main__":
    main()