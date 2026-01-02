import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    from typing import Tuple
    return (np,)


@app.cell
def _(np):
    y = np.array([1, 2, 3,4,5])
    np.mean((y-y.mean())**2)
    return (y,)


@app.cell
def _(y):
    print(y.shape)
    return


@app.cell
def _(y):
    (y.std())**2
    return


@app.cell
def _(np, y):
    def calculate_variance(y:np.array) -> np.float32:
        y_mean = y.mean()
        variation = y - y_mean
        variance = np.mean(
            np.mean(
                variation**2
            )
        )
        return variance
    calculate_variance(y)
    return


@app.cell
def _(y):
    y.index
    return


@app.cell
def _(np, y):
    def split_data(X: np.array, y) -> tuple[tuple[np.array, np.float16], tuple[np.array, np.float16]]:
        return (X, y), (X, y)
    split_data(y, 2)
    return


@app.cell
def _(np):
    a = np.arange(20).reshape(4, 5)
    b = np.arange(5, 9)
    print(a)
    print(b)

    mask = a[:, 2]  <= 2
    return a, mask


@app.cell
def _(a, mask):
    a[~mask]
    return


@app.cell
def _(np):
    c = np.array([1,2,3,4,2,3,1,4,5,6,71,4,21,4,5])
    np.unique(c).shape
    return


if __name__ == "__main__":
    app.run()
