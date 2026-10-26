def simple_genetic_algorithm_weights(
    population_size: int, number_iter: int,
    crossover_prob: float, mutation_prob: float,
    item_values: list, item_weights: list, weight_limit: int):
    """
    SGA for bonus task
    """
    if item_values == None or item_weights == None:
        return
    

    stats = {
        "value": {
            "avg": [],
            "best": []
        },
        "weight": {
            "avg": [],
            "best": []
        },
        "num_items": {
            "avg": [],
            "best": []
        },
        "item_value": {
            "avg": [],
            "best": []
        },
        "item_weight": {
            "avg": [],
            "best": []
        },
        "v_w_ratio": {
            "avg": [],
            "best": []
        },
    }
    
    gene_len = len(item_values)
    population = initialization_weight(
        pop_size=population_size,
        gene_len=gene_len,
        item_weights=item_weights,
        weight_lim=weight_limit
    )

    for _ in range(number_iter):
        fitness_values = fitness(population, item_values)

        pop = np.array(population)
        num_items = pop.sum(axis=1)
        num_items[num_items == 0] = 1

        # Get best individual
        best_idx = np.argmax(fitness_values)
        best_individual = pop[best_idx]

        values  = pop @ item_values # total value per individual
        weights = pop @ item_weights # total weight per individual
        counts  = pop.sum(axis=1) # real number of items per individual
        safe    = np.where(counts == 0, 1, counts) # only used for dividing

        per_individual = {
            "value":       values,
            "weight":      weights,
            "num_items":   counts,
            "item_value":  values / safe,
            "item_weight": weights / safe,
            "v_w_ratio":   values / np.where(weights == 0, 1, weights),
        }

        best_idx = np.argmax(fitness_values)
        for name, arr in per_individual.items():
            stats[name]["avg"].append(arr.mean())
            stats[name]["best"].append(arr[best_idx])
    

        parents = parent_selection_task_3(population, fitness_values)

        offspring = crossover_weight(parents, crossover_prob, item_weights, weight_limit)
        offspring = mutation_weight(offspring, mutation_prob, item_weights, weight_limit)
        offspring_fitness_values = fitness(offspring, item_values)

        survivors = survivor_selection(
            population,
            offspring,
            fitness_values,
            offspring_fitness_values
        )
        population = survivors

    return population, fitness(population, item_values), stats


def plot_stat_over_generations(ax, avg_data: list[float], best_data: list[float],
                               graph_name: str, y_label: str):
    gens = range(1, len(avg_data) + 1)

    ax.plot(gens, avg_data, color="tab:blue", alpha=0.8, linestyle="--", label="Average individual")
    ax.plot(gens, best_data, color="tab:red", alpha=0.8, linestyle="-", linewidth=2, label="Best individual")
    ax.set_xlabel("Generation")
    ax.set_ylabel(y_label)
    ax.set_title(graph_name)
    ax.legend()
    ax.grid(True)


def test_sga_weight(
            pop_size: int = 30, num_iter: int = 500,
            pc: float = 0.9, pm: float = 0.1,
            item_values: list = None, item_weights: list = None,
            weight_limit: int = WEIGHT_LIMIT,
            show_plot: bool = False, show_stat: bool = True) -> dict:
    """
    function to test the weight SGA
    """
    report = simple_genetic_algorithm_weights_extra(
        population_size=pop_size,
        number_iter=num_iter,
        crossover_prob=pc,
        mutation_prob=pm,
        item_values=item_values,
        item_weights=item_weights,
        weight_limit=weight_limit
    )

    population         = report[0]
    population_fitness = report[1]
    stats              = report[2]

    best_idx = np.argmax(population_fitness)
    best_individual = population[best_idx]

    final_index = num_iter - 1

    if show_stat:
        print("=" * 60)
        print(f"Test with weight limit = {weight_limit}")
        print(f"Population size: {pop_size} | Iterations: {num_iter} | PC: {pc} | PM: {pm}")
        print(f"Average fitness of final population: {np.mean(population_fitness):,.2f}")
        print(f"Best individual")
        print(f"Best value: {stats["value"]["best"][final_index]:,}")
        print(f"Best sum of weight: {stats["weight"]["best"][final_index]:,}")
        print(f"Number of items: {stats["num_items"]["best"][final_index]}")
        print(f"Average value per selected item: {stats["value"]["best"][final_index] / stats["num_items"]["best"][final_index]:,.2f}")
        print(f"Average weight per selected item: {stats["weight"]["best"][final_index] / stats["num_items"]["best"][final_index]:,.2f}")
        print("=" * 60 + "\n")
        
        #test_weight(population, item_weights, print_each=False)

        if show_plot:
            plots = [
                ("value",       "Total value",              "Value"),
                ("weight",      "Total weight",             "Weight"),
                ("num_items",   "Number of items",          "Items"),
                ("item_value",  "Average item value",       "Value per item"),
                ("item_weight", "Average item weight",      "Weight per item"),
                ("v_w_ratio",   "Value-to-weight ratio",    "Value / weight"),
            ]

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            for ax, (key, title, y_label) in zip(axes.flat, plots):
                plot_stat_over_generations(ax, stats[key]["avg"], stats[key]["best"], title, y_label)

            fig.suptitle(f"Weight-limited SGA (W = {weight_limit:,}): best vs. average individual", fontsize=14)
            fig.tight_layout()
            plt.show()
    
    return stats, best_individual


population_size = 30
number_iter = 1000
crossover_prob = 0.9
mutation_prob = 0.1

report = test_sga_weight(
    population_size, number_iter,
    crossover_prob, mutation_prob,
    item_values, item_weights, 
    WEIGHT_LIMIT, 
    show_plot=True, show_stat=True
)
