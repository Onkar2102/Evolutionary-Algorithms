import numpy as np
import pandas as pd
from pdb import set_trace

class my_GA:
    # Tuning with Genetic Algorithm for model parameters

    def __init__(self, model, data_X, data_y, decision_boundary, obj_func, generation_size=100, selection_rate=0.5,
                 mutation_rate=0.01, crossval_fold=5, max_generation=100, max_life=3):
        # Initialize the Genetic Algorithm parameters and attributes
        # inputs:
        # model: class object of the learner under tuning, e.g. my_DT
        # data_X: training data independent variables (pd.Dataframe, csr_matrix or np.array)
        # data_y: training data dependent variables (pd.Series or list)
        # decision_boundary: a dictionary of boundaries of each decision variable,
        # e.g. decision_boundary = {"criterion": ("gini", "entropy"), "max_depth": [1, 16], "min_impurity_decrease": [0, 0.1]} for my_DT means:
        # the first argument criterion can be chosen as either "gini" or "entropy"
        # the second argument max_depth can be any number 1 <= max_depth < 16
        # the third argument min_impurity_decrease can be any number 0 <= min_impurity_decrease < 0.1
        # obj_func: generate objectives, all objectives are higher the better
        # generation_size: number of points in each generation
        # selection_rate: percentage of survived points after selection, only affect single objective
        # mutation_rate: probability of being mutated for each decision in each point
        # crossval_fold: number of fold in cross-validation (for evaluation)
        # max_generation: early stopping rule, stop when reached
        # max_life: stopping rule, stop when max_life consecutive generations do not improve
        self.model = model
        self.data_X = data_X
        self.data_y = data_y
        # self.decision_keys stores keys of decision_boundary
        self.decision_keys = list(decision_boundary.keys())
        # self.decision_boundary stores values of decision_boundary
        self.decision_boundary = list(decision_boundary.values())
        self.obj_func = obj_func
        self.generation_size = int(generation_size)
        self.selection_rate = selection_rate  # applies only to single-objective
        self.mutation_rate = mutation_rate
        self.crossval_fold = int(crossval_fold)
        self.max_generation = int(max_generation)
        self.max_life = int(max_life)
        self.life = self.max_life
        self.iter = 0
        self.generation = []
        self.pf_best = []
        self.evaluated = {None: -1}

    def initialize(self):
        # Randomly generate generation_size points and add them to self.generation
        # If boundary in self.decision_boundary is integer, the generated
        # value must also be an integer.
        self.generation = []
        for _ in range(self.generation_size):
            x = []
            for boundary in self.decision_boundary:
                if type(boundary) == list:
                    val = np.random.random() * (boundary[1] - boundary[0]) + boundary[0]
                    if type(boundary[0]) == int:
                        val = round(val)
                    x.append(val)
                else:
                    x.append(boundary[np.random.randint(len(boundary))])
            self.generation.append(tuple(x))
        ######################
        # Check if the size of the generation is correct
        assert (len(self.generation) == self.generation_size)
        return self.generation

    def evaluate(self, decision):
        # Evaluate a certain point
        # decision: tuple of decisions
        # Avoid repetitive evaluations
        # Write your own code below
        if decision not in self.evaluated:
            # Evaluate with self.crossval_fold fold cross-validation on self.data_X and self.data_y
            dec_dict = {key: decision[i] for i, key in enumerate(self.decision_keys)}
            clf = self.model(**dec_dict)
            # Write your own code below
            # Cross-validation:
            indices = [i for i in range(len(self.data_y))]
            np.random.shuffle(indices)
            size = int(np.ceil(len(self.data_y) / float(self.crossval_fold)))
            objs_crossval = []
            for fold in range(self.crossval_fold):
                start = int(fold * size)
                end = start + size
                test_indices = indices[start:end]
                train_indices = indices[:start] + indices[end:]
                X_train = self.data_X.loc[train_indices]
                X_train.index = range(len(X_train))
                X_test = self.data_X.loc[test_indices]
                X_test.index = range(len(X_test))
                y_train = self.data_y.loc[train_indices]
                y_train.index = range(len(y_train))
                y_test = self.data_y.loc[test_indices]
                y_test.index = range(len(y_test))
                clf.fit(X_train, y_train)
                predictions = clf.predict(X_test)
                try:
                    pred_proba = clf.predict_proba(X_test)
                except:
                    pred_proba = None
                actuals = y_test
                objs = np.array(self.obj_func(predictions, actuals, pred_proba))
                objs_crossval.append(objs)
            # Take a mean of each fold of the cross-validation result
            # objs_crossval should become a 1-d array of the same size as objs
            objs_crossval = np.mean(objs_crossval, axis=0)
            self.evaluated[decision] = objs_crossval
        return self.evaluated[decision]

    def is_better(self, a, b):
        # Check if decision a binary dominates decision b
        # Return 0 if a == b,
        # Return 1 if a binary dominates b,
        # Return -1 if a does not binary dominate b.
        if a == b:
            return 0
        obj_a = self.evaluate(a)
        obj_b = self.evaluate(b)
        # Write your own code below
        dominates = True
        for a_val, b_val in zip(obj_a, obj_b):
            if a_val < b_val:
                dominates = False
                break

        if dominates:
            return 1
        else:
            return -1

    def compete(self, pf_new, pf_best):
        # Compare and merge two pareto frontiers
        modified = False
        # Create a list to hold points to be added to pf_best
        points_to_add = []

        # Check if points in pf_new dominate or are non-dominated compared to points in pf_best
        for new_point in pf_new:
            dominated_by_pf_best = False
            for best_point in pf_best:
                if self.is_better(best_point, new_point) == 1:
                    # new_point is dominated by an existing point in pf_best
                    dominated_by_pf_best = True
                    break

            if not dominated_by_pf_best:
                # new_point is non-dominated, prepare to add it to pf_best
                points_to_add.append(new_point)

        # Remove dominated points from pf_best and add non-dominated new points
        for add_point in points_to_add:
            # Check if any point in pf_best is dominated by the new point
            for best_point in pf_best.copy():
                if self.is_better(add_point, best_point) == 1:
                    # Remove dominated point from pf_best
                    pf_best.remove(best_point)
                    modified = True

            # Add the new non-dominated point if it's not already in pf_best
            if add_point not in pf_best:
                pf_best.append(add_point)
                modified = True

        return modified

    def select(self):
        # Select which points will survive based on the objectives
        # Update the following:
        # self.pf = pareto frontier (undominated points from self.generation)
        # self.generation = survived points

        # single-objective:
        if len(self.evaluate(self.generation[0])) == 1:
            selected = np.argsort([self.evaluate(x)[0] for x in self.generation])[::-1][
                       :int(np.ceil(self.selection_rate * self.generation_size))]
            self.pf = [self.generation[selected[0]]]
            self.generation = [self.generation[i] for i in selected]
        # multi-objective:
        else:
            self.pf = []
            for x in self.generation:
                if not np.array([self.is_better(y, x) == 1 for y in self.generation]).any():
                    self.pf.append(x)
            # remove duplicates
            self.pf = list(set(self.pf))
            # Add the second batch of undominated points into the next generation if only one point in self.pf
            if len(self.pf) == 1:
                self.generation.remove(self.pf[0])
                next_pf = []
                for x in self.generation:
                    if not np.array([self.is_better(y, x) == 1 for y in self.generation]).any():
                        next_pf.append(x)
                next_pf = list(set(next_pf))
                self.generation = self.pf + next_pf
            else:
                self.generation = self.pf[:]

    def crossover(self):
        # Randomly select two points in self.generation
        # and generate a new point
        # Repeat until self.generation_size points are generated
        # Write your own code below
        def cross(a, b):
            new_point = []
            for i in range(len(a)):
                if np.random.random() < 0.5:
                    new_point.append(a[i])
                else:
                    new_point.append(b[i])
            return tuple(new_point)

        to_add = []
        for _ in range(self.generation_size - len(self.generation)):
            ids = np.random.choice(len(self.generation), 2, replace=False)
            new_point = cross(self.generation[ids[0]], self.generation[ids[1]])
            to_add.append(new_point)
        self.generation.extend(to_add)
        ######################
        # Check if the size of the generation is correct
        assert (len(self.generation) == self.generation_size)
        return self.generation

    def mutate(self):
        # Uniform random mutation:
        # Each decision value in each point of self.generation
        # has the same probability self.mutation_rate of being mutated
        # to a random valid value
        # If the boundary in self.decision_boundary is an integer, the mutated
        # value must also be an integer.
        # Write your own code below
        for i, x in enumerate(self.generation):
            new_x = list(x)
            for j in range(len(x)):
                if np.random.random() < self.mutation_rate:
                    boundary = self.decision_boundary[j]
                    if type(boundary) == list:
                        val = np.random.uniform(boundary[0], boundary[1])
                        if type(boundary[0]) == int:
                            val = int(round(val))
                        new_x[j] = val
                    else:
                        new_x[j] = np.random.choice(boundary)
            self.generation[i] = tuple(new_x)

        return self.generation

    def tune(self):
        # Main function of my_GA
        # Stop when self.iter == self.max_generation or self.life == 0
        # Return the best pareto frontier pf_best (list of decisions that never get binary dominated by any candidate evaluated)
        self.initialize()
        while self.life > 0 and self.iter < self.max_generation:
            self.select()
            # If any better than the current best
            if self.compete(self.pf, self.pf_best):
                self.life = self.max_life
            else:
                self.life -= 1
            self.iter += 1
            self.crossover()
            self.mutate()
        return self.pf_best
