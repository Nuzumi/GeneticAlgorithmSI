import Classes as Cl
import plotly.graph_objs as go
from plotly.offline import plot
import numpy as np


def main():
    Cl.load_files_to_genetic_algorithm(5)
    a = Cl.GeneticAlgorithm(0.08, 0.7, 5, 100, 200, 1, 4)
    a.start_evolution()
    axis_x = np.linspace(0, a.generations_count)
    axis_y_best = a.best_of_generations
    axis_y_average = a.average_of_generations
    axis_y_worst = a.worst_of_generations
    trace_best = go.Scatter(
        x=axis_x,
        y=axis_y_best,
        mode='lines+markers',
        name='best',
        line=dict(
            color=('rgb(255, 0, 0)')
        )

    )
    trace_average = go.Scatter(
        x=axis_x,
        y=axis_y_average,
        mode='lines+markers',
        name='average',
        line=dict(
            color=('rgb(0, 255, 0)')
        )
    )
    trace_worst = go.Scatter(
        x=axis_x,
        y=axis_y_worst,
        mode='lines+markers',
        name='worst',
        line=dict(
            color=('rgb(0, 0, 255)')
        )
    )
    plot([trace_average, trace_best, trace_worst], 'chart.html')
    pass


def start_genetic_algorithm(p_m, p_x, tour, pop_size, generations_count, combine_point_count, matrix_index, genetic_iterations):
    genetic = Cl.GeneticAlgorithm(p_m, p_x, tour, pop_size, generations_count, combine_point_count, matrix_index)
    best_average = np.zeros(shape=(generations_count, genetic_iterations))
    average_average = np.zeros(shape=(generations_count, genetic_iterations))
    worst_average = np.zeros(shape=(generations_count, genetic_iterations))
    for i in range(genetic_iterations):
        genetic.start_evolution()
        best_average[i] = genetic.best_of_generations
        average_average[i] = genetic.average_of_generations
        worst_average[i] = genetic.worst_of_generations

    np.sum(best_average, axis=0)
    np.sum(average_average, axis=0)
    np.sum(worst_average, axis=0)
    best_average /= genetic_iterations
    average_average /= genetic_iterations
    worst_average /= genetic_iterations
    pass


def show_chart():
    pass


if __name__ == "__main__":
    main()
