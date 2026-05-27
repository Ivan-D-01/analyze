import random
import math
import matplotlib.pyplot as plt

POPULATION_SIZE = 4
GENERATIONS = 50
MUTATION_RATE = 0.1
NUM_CITIES = 5
SECTION_POINT = 2

def generate_cities(n):
    return [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]

def distance(city1, city2):
    return math.hypot(city1[0] - city2[0], city1[1] - city2[1])

def grade(route, cities):
    # оценивает длину маршрута
    total = 0
    for i in range(len(route)):
        total += distance(cities[route[i]], cities[route[(i + 1) % len(route)]])
    return total

def create_population(size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(size)]

def selection(population, cities):
    # возвращает лучшую половину популяций городов
    population.sort(key=lambda x: grade(x, cities))
    return population[:len(population)//2]

def crossover(parent1, parent2):
    # хечбек
    point = random.randint(1, len(parent1) - 1)
    child = parent1[:point]
    for gene in parent2:
        if gene not in child:
            child.append(gene)
    return child

def mutate(route):
    # мутант
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

def next_generation(current_gen, cities):
    selected = selection(current_gen, cities)
    new_population = selected.copy()
    
    while len(new_population) < POPULATION_SIZE:
        parent1, parent2 = random.sample(selected, 2)
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_population.append(child)

    if len(current_gen) > 0:
        best = min(current_gen, key=lambda x: grade(x, cities))
        new_population[-1] = best  

    return new_population

def genetic_algorithm(cities):
    population = create_population(POPULATION_SIZE, NUM_CITIES)
    best_route = None
    best_grade = 999999
    history = []

    for generation in range(GENERATIONS):
        population = next_generation(population, cities)
        current_best = min(population, key=lambda x: grade(x, cities))
        current_grade = grade(current_best, cities)

        if current_grade < best_grade:
            best_grade = current_grade
            best_route = current_best

        history.append(best_grade)
        print(f"Поколение {generation+1}: Лучшая длина = {best_grade:.2f}")

        if len(history) > int(GENERATIONS * 0.5) and abs(history[-1] - history[int(GENERATIONS * 0.5)]) < 0.1:
            print("Алгоритм сошёлся.")
            break

    return best_route, best_grade, history

def plot_route(cities, route, title=""):
    plt.figure(figsize=(8, 6))
    xs = [city[0] for city in cities]
    ys = [city[1] for city in cities]
    plt.scatter(xs, ys, color='red')

    for i in range(len(route)):
        start = cities[route[i]]
        end = cities[route[(i + 1) % len(route)]]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'k-')

    plt.title(title)
    plt.grid(True)
    plt.show()

def generate_specific_cities():
    return [(0,0),(2,3),(5,2),(6,6),(8,3)]

if __name__ == "__main__":
    #cities = generate_cities(NUM_CITIES)
    cities = generate_specific_cities()
    best_route, best_length, history = genetic_algorithm(cities)

    print(f"\nЛучший маршрут найден с длиной: {best_length:.2f}")
    plot_route(cities, best_route, f"Кратчайший маршрут (Длина: {best_length:.2f})")

    plt.plot(history)
    plt.title("Изменение длины лучшего маршрута")
    plt.xlabel("Поколение")
    plt.ylabel("Длина маршрута")
    plt.grid(True)
    plt.show()