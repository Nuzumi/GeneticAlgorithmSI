import Classes as Cl
import plotly.graph_objs as go
from plotly.offline import plot
from  Individual import Individual
import numpy as np


def main():
    Cl.load_files_to_genetic_algorithm(1)
    #start_genetic_algorithm(0.12, 0.75, 15, 100, 100, 2, 0, 50)
    #start_genetic_algorithm_points(0.10, 0.8, 15, 100, 100, np.arange(1,10,1), 0, 50)
    #start_genetic_algorithm_pm(np.arange(0, 0.5, 0.05), 0.75, 15, 100, 100, 2, 0, 50)
    #start_genetic_algorithm_px(0.15, np.arange(0, 1, 0.1), 15, 100, 100, 2, 0, 50)
    start_genetic_algorithm_tour(0.15, 0.8 , np.arange(5, 50, 5), 100, 100, 2, 0, 50)
    start_genetic_algorithm_popsize(0.15, 0.8, 15, np.arange(50, 551, 50), 100, 2, 0, 50)

    pass


def start_genetic_algorithm(p_m, p_x, tour, pop_size, generations_count, combine_point_count, matrix_index, genetic_iterations):
    title = Cl.GeneticAlgorithm(p_m, p_x, tour, pop_size, generations_count, combine_point_count, matrix_index)
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
    show_chart(generations_count, best_average, average_average, worst_average, title)

    pass


def start_genetic_algorithm_pm(p_m, p_x, tour, pop_size, generations_count, combine_point_count, matrix_index, genetic_iterations):
    trace_best = np.zeros(shape=(genetic_iterations, generations_count))
    trace_bests = []
    axis_x = np.linspace(0, generations_count)

    for i in range(len(p_m)):
        trace_best = np.zeros(shape=(genetic_iterations, generations_count))
        for j in range(genetic_iterations):
            genetic = Cl.GeneticAlgorithm(p_m[i], p_x, tour, pop_size, generations_count, combine_point_count, matrix_index)
            genetic.start_evolution()
            trace_best[j] = genetic.best_of_generations

        best_average = np.average(trace_best, axis=0)
        trace = go.Scatter(
                x=axis_x,
                y=best_average,
                mode='lines+markers',
                name='Mutacja: ' + str(p_m[i]),
            )
        trace_bests.append(trace)

    layout = dict(title='Wpływ parametru mutacji',
                  xaxis=dict(title='Generacja'),
                  yaxis=dict(title='Droga')
                  )

    chart = dict(data=trace_bests, layout=layout)

    show_chart_params(chart, 'Mutacja')

    pass


def start_genetic_algorithm_px(p_m, p_x, tour, pop_size, generations_count, combine_point_count, matrix_index, genetic_iterations):
    trace_best = np.zeros(shape=(genetic_iterations, generations_count))
    trace_bests = []
    axis_x = np.linspace(0, generations_count)

    for i in range(len(p_x)):
        trace_best = np.zeros(shape=(genetic_iterations, generations_count))
        for j in range(genetic_iterations):
            genetic = Cl.GeneticAlgorithm(p_m, p_x[i], tour, pop_size, generations_count, combine_point_count, matrix_index)
            genetic.start_evolution()
            trace_best[j] = genetic.best_of_generations

        best_average = np.average(trace_best, axis=0)
        trace = go.Scatter(
                x=axis_x,
                y=best_average,
                mode='lines+markers',
                name='Krzyżowanie: ' + str(p_x[i]),
            )
        trace_bests.append(trace)

    layout = dict(title='Wpływ parametru krzyżowania',
                  xaxis=dict(title='Generacja'),
                  yaxis=dict(title='Droga')
                  )

    chart = dict(data=trace_bests, layout=layout)

    show_chart_params(chart, 'Krzyżowanie')
    pass


def start_genetic_algorithm_tour(p_m, p_x, tour, pop_size, generations_count, combine_point_count, matrix_index, genetic_iterations):
    trace_best = np.zeros(shape=(genetic_iterations, generations_count))
    trace_bests = []
    axis_x = np.linspace(0, generations_count)

    for i in range(len(tour)):
        trace_best = np.zeros(shape=(genetic_iterations, generations_count))
        for j in range(genetic_iterations):
            genetic = Cl.GeneticAlgorithm(p_m, p_x, tour[i], pop_size, generations_count, combine_point_count, matrix_index)
            genetic.start_evolution()
            trace_best[j] = genetic.best_of_generations

        best_average = np.average(trace_best, axis=0)
        trace = go.Scatter(
                x=axis_x,
                y=best_average,
                mode='lines+markers',
                name='Pojedynek: ' + str(tour[i]),
            )
        trace_bests.append(trace)

    layout = dict(title='Wpływ ilości osób w pojedynku',
                  xaxis=dict(title='Generacja'),
                  yaxis=dict(title='Droga')
                  )

    chart = dict(data=trace_bests, layout=layout)

    show_chart_params(chart, 'Tour')
    pass


def start_genetic_algorithm_popsize(p_m, p_x, tour, pop_size, generations_count, combine_point_count, matrix_index, genetic_iterations):
    trace_best = np.zeros(shape=(genetic_iterations, generations_count))
    trace_bests = []
    axis_x = np.linspace(0, generations_count)

    for i in range(len(pop_size)):
        trace_best = np.zeros(shape=(genetic_iterations, generations_count))
        for j in range(genetic_iterations):
            genetic = Cl.GeneticAlgorithm(p_m, p_x, tour, pop_size[i], generations_count, combine_point_count, matrix_index)
            genetic.start_evolution()
            trace_best[j] = genetic.best_of_generations

        best_average = np.average(trace_best, axis=0)
        trace = go.Scatter(
                x=axis_x,
                y=best_average,
                mode='lines+markers',
                name='Populacja: ' + str(pop_size[i]),
            )
        trace_bests.append(trace)

    layout = dict(title='Wpływ wielkości populacji',
                  xaxis=dict(title='Generacja'),
                  yaxis=dict(title='Droga')
                  )

    chart = dict(data=trace_bests, layout=layout)

    show_chart_params(chart, 'Populacja')
    pass


def start_genetic_algorithm_points(p_m, p_x, tour, pop_size, generations_count, combine_point_count, matrix_index, genetic_iterations):

    trace_bests = []
    axis_x = np.linspace(0, generations_count)

    for i in range(len(combine_point_count)):
        trace_best = np.zeros(shape=(genetic_iterations, generations_count))
        for j in range(genetic_iterations):
            genetic = Cl.GeneticAlgorithm(p_m, p_x, tour, pop_size, generations_count, combine_point_count[i], matrix_index)
            genetic.start_evolution()
            trace_best[j] = genetic.best_of_generations

        best_average = np.average(trace_best, axis=0)
        trace = go.Scatter(
                x=axis_x,
                y=best_average,
                mode='lines+markers',
                name='Punkty: ' + str(combine_point_count[i]),
            )
        trace_bests.append(trace)

    layout = dict(title='Wpływ ilości punktów krzyżowania',
                  xaxis=dict(title='Generacje'),
                  yaxis=dict(title='Droga')
                  )

    chart = dict(data=trace_bests, layout=layout)

    show_chart_params(chart, 'Ilość punktów')
    pass


def show_chart(x_count, y_best, y_average, y_worst, title):
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

    layout = dict(title=str(title),
                  xaxis=dict(title='Generacje'),
                  yaxis=dict(title='Droga')
                  )

    chart = dict(data=[trace_average, trace_best, trace_worst], layout=layout)
    plot(chart, 'chart.html')
    pass


def show_chart_params(chart, title):

    plot(chart, title + '.html')
    pass


if __name__ == "__main__":
    main()
