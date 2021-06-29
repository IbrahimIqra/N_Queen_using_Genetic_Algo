# Importing numpy & time
import numpy as np
import time

"""Genetic_Algo(population, mutation_chance)"""

def genetic_algo(population, mutation_prob, max_gen_num):

  population_fitness = fitness(population)

  gen_num = 0

  for i in range(max_gen_num):

    # empty array for new_gen with 0 row of num_of_queens length
    # np.empty() posa
    new_gen = np.empty((0, num_of_queens), int)

    for j in range(population_size):

      # selecting a father from the old population based on fitness values
      # population_fitness is a class variable
      while (True):
        father = parents_selection(population, population_fitness)
        mother = parents_selection(population, population_fitness)

        if (np.array_equal(father, mother)):
          continue
        else:
          break

      child = reproduce(father, mother)

      rnd_prob = np.random.uniform(0, 1)
      # rounding it to 2 decimal points
      rnd_prob = round(rnd_prob, 2)

      # for higher mutation_prob this condition will be true more freqently
      # for lower mutation_prob this condition will be true less freqently
      if (mutation_prob >= rnd_prob):
        child = mutation(child)

      new_gen = np.append(new_gen, [child], axis=0)

    print("Generation:", (i + 1))
    # xtra print("New generation: \n",new_gen,end='\n')

    new_gen_fitness = fitness(new_gen)
    print(f"Max Fitness of this generation: {max(new_gen_fitness)}\n")
    # xtra print(f"Fitness of the new_gen board-wise: \n{new_gen_fitness}\n")
    # xtra print(f"Selection_Prob of the new_gen board-wise: \n{selection_prob(new_gen_fitness)} ")

    population = new_gen
    population_fitness = new_gen_fitness

    paisi = False

    n = num_of_queens
    max_chaos = int((n * (n - 1)) / 2)

    if (max_chaos in population_fitness):
      for (dna_strand, dna_fit) in zip(population, population_fitness):
        if (dna_fit == max_chaos):
          print(
              f"\nBest DNA_strand found in the population of {i+1}th generation: {dna_strand}")
          print(f"Population in each generation was {len(population)}")
          print(
              f"Mutation probability in each generation was {mutation_prob}\n")

          print(f"{num_of_queens}x{num_of_queens} Board")
          print(f"BEST Possible Fitness of a board: {max_chaos}")
          print(f"Fitness of this board: {dna_fit}")
          print(f"Attacking pairs in this board: {max_chaos-dna_fit}")
          print(
              f"\nThe DNA_strand converted to BOARD:\n{translate(dna_strand)}")
          paisi = True
          break

    if (paisi):
      break

    if (i == max_gen_num - 1):
      max_fit = max(population_fitness)
      max_fit_id = np.where(population_fitness == max_fit)
      # np.where() posa that's why the line below
      max_fit_id = max_fit_id[0][0]
      dna_strand = population[max_fit_id]

      print("\n************SORRY, COULDN't FIND THE BEST POSSIBLE DNA_Strand...***************")
      print("*****Try again with MORE generations and MORE population for better result*****\n")

      print(
          f"Best DNA_Strand in the population of {i+1}th generation: {dna_strand}")
      print(f"Population in each generation was {len(population)}")
      print(f"Mutation probability in each generation was {mutation_prob}\n")

      print(f"{num_of_queens}x{num_of_queens} Board")
      print(f"BEST Possible Fitness of a board: {max_chaos}")
      print(f"Fitness of this board: {max_fit}")
      print(f"Total attacking pairs in this board: {max_chaos-max_fit}")
      print(f"\nThe DNA_strand converted to BOARD:\n{translate(dna_strand)}")


"""Fitness(population)"""

def fitness(population):

  fitness_list = []

  n = len(population[0])
  max_chaos = (n * (n - 1)) / 2

  for single in population:

    board = translate(single)
    fit = 0
    for i in range(len(board)):
      # for ith horizontal lines
      qs = np.where(board[i] == 'Q')
      for q in qs:
        if len(q) > 1:
          fit += (len(q) - 1)
      # for ith vertical lines
      qs = np.where(board[:, i] == 'Q')
      for q in qs:
        if len(q) > 1:
          fit += (len(q) - 1)

    diag_rnge = len(board) - 2

    for i in range(-diag_rnge, diag_rnge + 1):

      # from left to right diagonal
      # xtra print(f"\n {board.diagonal(i)}")
      diag = board.diagonal(i)
      qs1 = np.where(diag == 'Q')
      for q in qs1:
        if len(q) > 1:
          fit += (len(q) - 1)

      # from right to left diagonal
      # xtra print(f"\n {board[::-1,:].diagonal(i)}")
      diag = board[::-1, :].diagonal(i)
      qs2 = np.where(diag == 'Q')
      for q in qs2:
        if len(q) > 1:
          fit += (len(q) - 1)

    fitness_list.append(int(max_chaos - fit))
    # xtra print(board)
    # xtra print(fit, end='\n\n')

  return np.array(fitness_list)


"""Translate(dna_strand)"""

def translate(dna_strand):

  # the user can also enter the whole population
  # instead of a single dna_strand to get a list of boards

  # this means if the arg is whole population
  if (dna_strand.ndim == 2):
    population = dna_strand
    return np.array([translate(dna) for dna in population])

  # if the arg is a single dna_strand
  size = len(dna_strand)
  board = np.full([size, size], '|')
  # translate
  for i in range(size):
    board[dna_strand[i]][i] = 'Q'

  return board


"""Parents_Selection(population, population_fitness)"""

def parents_selection(population, population_fitness):

  sum_of_fit = np.sum(population_fitness)

  all_fit_prob = [(single_fit / sum_of_fit)
                  for single_fit in population_fitness]

  popu_id = np.arange(len(population))

  chosen_id = np.random.choice(popu_id, size=1, replace=True, p=all_fit_prob)

  chosen_one = population[chosen_id]
  # at this stage chosen_one is a 2d array with 1 row and 8 column
  chosen_one = chosen_one[0]

  return chosen_one


"""Reproduce(father, mother)"""

def reproduce(father, mother):

  # xtra print(f"Chosen father from the population: {father}")
  # xtra print(f"Chosen mother from the population: {mother}")

  # num_of_queens is a class variable
  cross_id = np.random.randint(low=2, high=num_of_queens - 1)

  # xtra print(cross_id)

  # xtra print(father[:cross_id])
  # xtra print(mother[cross_id:])

  child = np.concatenate((father[:cross_id], mother[cross_id:]), axis=0)

  # xtra print(child)

  return child


"""Mutation(child)"""


def mutation(child):
  mutation_index = np.random.randint(low=0, high=len(child))
  mutated_value = np.random.randint(low=0, high=len(child))

  child[mutation_index] = mutated_value

  mutated_child = child

  #print("MUTATION!!!!!\nMutated Child: ",mutated_child)

  return mutated_child


"""Finally, Testing out the Genetic Algorithm"""

num_of_queens = int(input("Enter the value of N for the N-Queen problem\n"))
population_size = int(
    input("Enter the size of the population (for 8 queens 50 is enough)\n"))

while (True):
  mutation_prob = float(input(
      "Enter the probability of mutation (0.25 is optimal) (value between 0 and 1)\n"))
  if (0 <= mutation_prob <= 1):
    break
  else:
    print("OUT OF RANGE!!!")

max_gen_num = int(
    input("Enter the max number of generations (for 8 queens 3000 is suggested)\n"))

# randomized population
# size = (row x col)
# low = 0, high = 8 means rands between 0 to 7
population = np.random.randint(
    low=0, high=num_of_queens, size=(population_size, num_of_queens))

# starting the counter
start_time = time.time()
genetic_algo(population, mutation_prob, max_gen_num)
# ending the counter
end_time = time.time()

exec_time = round((end_time - start_time), 2)

if (exec_time > 3600):
  print(f"\nExecution time: {round(exec_time/3600,2)} hours\n")
elif (exec_time > 60):
  print(f"\nExecution time: {round(exec_time/60,2)} minutes\n")
else:
  print(f"\nExecution time: {exec_time} seconds\n")

print("\nTry to experiment with different values.. Have fun ;)")
