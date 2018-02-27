import Classes as Cl
import plotly.graph_objs as go
from plotly.offline import plot
from  Individual import Individual
import numpy as np


def main():
    Cl.load_files_to_genetic_algorithm(1)

    start_genetic_algorithm(0.12, 0.75, 15, 100, 100, 1, 0, 5)
    pass


def start_genetic_algorithm(p_m, p_x, tour, pop_size, generations_count, combine_point_count, matrix_index, genetic_iterations):
    best_average = np.zeros(shape=(genetic_iterations, generations_count))
    average_average = np.zeros(shape=(genetic_iterations, generations_count))
    worst_average = np.zeros(shape=(genetic_iterations, generations_count))
    for i in range(genetic_iterations):
        genetic = Cl.GeneticAlgorithm(p_m, p_x, tour, pop_size, generations_count, combine_point_count, matrix_index)
        genetic.start_evolution()
        best_average[i] = genetic.best_of_generations
        average_average[i] = genetic.average_of_generations
        worst_average[i] = genetic.worst_of_generations

    best_average = np.average(best_average, axis=0)
    average_average = np.average(average_average, axis=0)
    worst_average = np.average(worst_average, axis=0)
    show_chart(generations_count, best_average, average_average, worst_average)
    pass


def show_chart(x_count, y_best, y_average, y_worst):
    axis_x = np.linspace(0, x_count)
    trace_best = go.Scatter(
        x=axis_x,
        y=y_best,
        mode='lines+markers',
        name='best',
        line=dict(
            color=('rgb(255, 0, 0)')
        )

    )
    trace_average = go.Scatter(
        x=axis_x,
        y=y_average,
        mode='lines+markers',
        name='average',
        line=dict(
            color=('rgb(0, 255, 0)')
        )
    )
    trace_worst = go.Scatter(
        x=axis_x,
        y=y_worst,
        mode='lines+markers',
        name='worst',
        line=dict(
            color=('rgb(0, 0, 255)')
        )
    )
    plot([trace_average, trace_best, trace_worst], 'chart.html')
    pass


if __name__ == "__main__":
    main()
