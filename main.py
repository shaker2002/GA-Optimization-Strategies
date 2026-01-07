from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import random

dataset = "diabetes"
df = pd.read_csv(f"{dataset}.csv")
print(df.shape)

if dataset == "breast-cancer":
    df = df.iloc[:, 1:]  # This selects all rows and all columns except the first one

    # Move the second column to the last
    cols = list(df.columns)
    second_col = cols[0]  # Assuming the current second column is now the first
    cols = cols[1:] + [second_col]  # Reorder columns
    df = df[cols]


class GA:
    def __init__(self, dataframe, population_size=10):
        self.dataframe = dataframe
        self.array_size = dataframe.shape[1]-1
        self.population_size = population_size
        self.population = []

        for i in range(population_size):
            self.population.append(self.gen_array())
        self.population = np.array(self.population)

        self.fittest = None

    def mutation(self, parent):

        child = parent

        num_bits_to_flip = random.randint(1, self.array_size)

        flip_indices = random.sample(range(self.array_size), num_bits_to_flip)

        for idx in flip_indices:
            child[idx] = 1 - child[idx]
        return list(child)

    def crossover(self, parent1_index, parent2_index, c_type="n_point"):
        if c_type == "1_point":

            parent1 = self.population[parent1_index]
            parent2 = self.population[parent2_index]

            crossover_point = random.randint(1, self.array_size - 1)

            offspring = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

            return offspring.tolist()

        elif c_type == "n_point":

            parent1 = self.population[parent1_index]
            parent2 = self.population[parent2_index]

            n = random.randint(1, self.array_size//2)
            crossover_points = sorted(random.sample(range(1, self.array_size-1), n))

            offspring = []

            prev_point = 0
            p1 = True
            for point in crossover_points:
                offspring.extend(parent1[prev_point:point] if p1 else parent2[prev_point:point])
                p1 = not p1
                prev_point = point

            if p1:
                offspring.extend(parent1[prev_point:])
            else:
                offspring.extend(parent2[prev_point:])
            return offspring

        elif c_type == "uniform":

            parent1 = self.population[parent1_index]
            parent2 = self.population[parent2_index]

            offspring = self.gen_array()

            offspring[0] = parent1[0]
            offspring[-1] = parent2[-1]

            return offspring.tolist()

    def parent_selection(self, ps_type="uniform"):
        if ps_type == "uniform":
            parent1_index, parent2_index = random.sample(range(len(self.population)), 2)
            return parent1_index, parent2_index
        elif ps_type == "fps":
            fitness = self.evaluation()
            fitness_sum = sum(fitness)
            probabilities = [f / fitness_sum for f in fitness]
            parent1_index, parent2_index = np.random.choice(range(len(self.population)), size=2, p=probabilities, replace=False)
            return parent1_index, parent2_index

    def survival_selection(self, ss_type="m+l"):
        if ss_type == "elitism":
            # keep to 10% of population
            fitness = self.evaluation()
            sorted_indev_score = sorted(list(zip(self.population, fitness)), key=lambda x: x[1], reverse=True)
            sorted_indev = [indev for indev, fit in sorted_indev_score]
            num_indevs = round(len(self.population)*0.1)
            return sorted_indev[:num_indevs]
        elif ss_type == "genitor":
            # remove worst 10% of population (keep to 90%)
            fitness = self.evaluation()
            sorted_indev_score = sorted(list(zip(self.population, fitness)), key=lambda x: x[1], reverse=True)
            sorted_indev = [indev for indev, fit in sorted_indev_score]
            num_indevs = round(len(self.population) * 0.9)
            return sorted_indev[:num_indevs]
        elif ss_type == "m,l":
            # # don't keep anything form the population
            return []
        elif ss_type == "m+l":
            # keep to best 50% of the population
            fitness = self.evaluation()
            sorted_indev_score = sorted(list(zip(self.population, fitness)), key=lambda x: x[1], reverse=True)
            sorted_indev = [indev for indev, fit in sorted_indev_score]
            num_indevs = round(len(self.population) * 0.5)
            return sorted_indev[:num_indevs]

    def evaluation(self, algo="knn"):

        X = self.dataframe.iloc[:, :-1].values
        y = self.dataframe.iloc[:, -1].values

        accuracies = []

        for individual in self.population:
            individual = np.array(individual)
            selected_features = X[:, individual == 1]
            if selected_features.shape[1] == 0:
                accuracies.append(0)
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                selected_features, y, test_size=0.2, random_state=42
            )
            if algo == "knn":
                knn = KNeighborsClassifier(n_neighbors=3)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                accuracies.append(accuracy_score(y_test, y_pred))
            elif algo == "dt":
                dtc = DecisionTreeClassifier(random_state=42)
                dtc.fit(X_train, y_train)
                y_pred = dtc.predict(X_test)
                accuracies.append(accuracy_score(y_test, y_pred))
        return accuracies

    def gen_array(self):
        return np.random.randint(0, 2, size=self.array_size)

    def run_until_threshold(self, accuracy_threshold, ss_method="m+l", ps_method="fps", c_method="n_point", evaluation_algo="knn", show_progress=True):
        num_of_gens = 1
        max_accuracy = 0
        while True:
            off_spring = []
            accuracies = self.evaluation(algo=evaluation_algo)
            max_accuracy = max(accuracies)
            max_gene = self.population[accuracies.index(max_accuracy)]
            self.fittest = max_gene
            if show_progress:
                print(f"Max Accuracy: {max_accuracy}/ gen: {num_of_gens}")
                print("##################################")

            if max_accuracy >= accuracy_threshold:
                print("Threshold reached.")
                print(f"the threshold was reached in {num_of_gens} generations, with accuracy {max_accuracy}")
                break
            else:
                num_of_gens += 1
            off_spring = self.survival_selection(ss_type=ss_method)
            while len(off_spring) < len(self.population):
                # Generate a random number to decide whether to apply crossover or mutation
                if random.random() < 0.5:
                    # crossover
                    parent1_index, parent2_index = self.parent_selection(ps_type=ps_method)
                    child = self.crossover(parent1_index, parent2_index, c_type=c_method)
                    if random.random() < 0.1:
                        # mutation
                        child = self.mutation(child)
                        if sum(child) != 0:
                            off_spring.append(child)
                    else:
                        if sum(child) != 0:
                            off_spring.append(child)
                if len(off_spring) <= len(self.population):
                    if random.random() < 0.1:
                        # mutation
                        parent, _ = self.parent_selection(ps_type=ps_method)
                        child = self.mutation(self.population[parent])
                        if sum(child) != 0:
                            off_spring.append(child)
                if len(self.population) == len(off_spring) == self.population_size:
                    self.population = off_spring
        return num_of_gens, max_accuracy

    def run_gens(self, num_of_gens, ss_method="m+l", ps_method="fps", c_method="n_point", evaluation_algo="knn", show_progress=True):
        gen = 1
        max_accuracy = 0
        for i in range(num_of_gens):
            off_spring = []
            accuracies = self.evaluation(algo=evaluation_algo)
            max_accuracy = max(accuracies)
            max_gene = self.population[accuracies.index(max_accuracy)]
            self.fittest = max_gene
            if show_progress:
                print(f"Max Accuracy: {max_accuracy}/ gen: {gen}")
                print("##################################")
            off_spring = self.survival_selection(ss_type=ss_method)
            while len(off_spring) < len(self.population):
                # Generate a random number to decide whether to apply crossover or mutation
                if random.random() < 0.5:
                    # crossover
                    parent1_index, parent2_index = self.parent_selection(ps_type=ps_method)
                    child = self.crossover(parent1_index, parent2_index, c_type=c_method)
                    if random.random() < 0.1:
                        # mutation
                        child = self.mutation(child)
                        if sum(child) != 0:
                            off_spring.append(child)
                    else:
                        if sum(child) != 0:
                            off_spring.append(child)
                if len(off_spring) <= len(self.population):
                    if random.random() < 0.1:
                        # mutation
                        parent, _ = self.parent_selection(ps_type=ps_method)
                        child = self.mutation(self.population[parent])
                        if sum(child) != 0:
                            off_spring.append(child)
                if len(self.population) == len(off_spring) == self.population_size:
                    self.population = off_spring

            gen += 1
        print(f"Max Accuracy: {max_accuracy}/ gen: {gen}")


algos = ["knn", "dt"]
ss_methods = ["elitism", "genitor", "m,l", "m+l"]
ps_methods = ["uniform", "fps"]
crossover_methods = ["uniform", "n_point", "1_point"]

columns = ['evaluation_algo', 'survival_selection', 'parent_selection', 'crossover_method', 'number_of_generations', 'accuracy', 'num_features']
df2 = pd.DataFrame(columns=columns)
for algo in algos:
    for ss in ss_methods:
        for ps in ps_methods:
            for crossover in crossover_methods:
                print(f"running.....\nalgo = {algo} / survival_selection = {ss} / parent_selection = {ps} / crossover_method = {crossover}")
                g = GA(df)
                gen, acc = g.run_until_threshold(0.74, evaluation_algo=algo, ss_method=ss, ps_method=ps, c_method=crossover, show_progress=False)
                print(g.fittest)
                num_features = list(g.fittest).count(1)
                print(num_features)
                row = [algo, ss, ps, crossover, gen, acc, num_features]
                row_df = pd.DataFrame([row], columns=columns)
                df2 = pd.concat([df2, row_df], ignore_index=True)


df2.to_csv(f"{dataset}_results.csv")
