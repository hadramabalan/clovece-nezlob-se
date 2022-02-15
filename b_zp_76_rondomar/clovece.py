from random import Random
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
sns.set_style('whitegrid')

def random_roll():
    roll = np.random.choice(np.arange(12), p=[0.166667, 0.166667, 0.166667, 0.166667, 0.166667, 0, 0.027778, 0.027778, 0.027778, 0.027777, 0.027777, 0.027777])
    return roll


def simulate_game(rseed=None, max_roll=6):
    """
    simuluje hru clovece, vraci pocet kol
    """
    rand = Random(rseed)
    position = 0
    turns = 0

    while position < 21:
        turns += 1
        #na startu trikrat hodime kostkou, v pripade aspon jedne sestky jdeme na policko jedna, jinak zustaneme na nule
        if position == 0:
            for i in range(0, 3):
                roll = rand.randint(1, max_roll)
                if roll == 6:
                    position = 1
                    break

        else:
            roll = random_roll()

            #pak se pohneme o vysledny hod
            position += roll
    return turns


# vytvoreni grafu pro pocet kol pro vyhru ve 100 000 simulacich
sim_games = [simulate_game() for i in range(100000)]
plt.hist(sim_games, bins=range(40), density=True)
plt.xlabel('Počet kol pro výhru')
plt.ylabel('Procento her')
plt.title('Simulovaná délka hry člověče, nezlob se')
plt.show()


#Markovova matice
def markov_matrix():
    matrix = np.zeros((22, 22))
    matrix[0,0] = 0.578704
    matrix[0,1] = 0.421296
    for i in range(15):
        matrix[1 + i, 2+i:7+i] = 0.166667
    for i in range(9):
        matrix[1 + i, 8+i:14+i] = 0.027778
    matrix[10, 17:21] = 0.027778
    matrix[11, 18:21] = 0.027778
    matrix[12, 19:21] = 0.027778
    matrix[13, 20] = 0.027778
    matrix[10,21] = 0.055556
    matrix[11,21] = 0.083334
    matrix[12,21] = 0.111112
    matrix[13,21] = 0.13889
    matrix[14,21] = 0.166667
    matrix[15,21] = 0.166667
    matrix[16,21] = 0.333334
    matrix[17,21] = 0.5
    matrix[18,21] = 0.666667
    matrix[19,21] = 0.833334
    matrix[20,21] = 1
    matrix[21,21] = 1
    return matrix

def cl_probability(n):
    """stavový vektor po n kolech"""
    matrix = markov_matrix()
    v_0 = np.zeros(22)
    v_0[0] = 1
    return np.linalg.matrix_power(matrix, n) @ v_0

probs = [cl_probability(i)[-1] for i in range(40)]

#Vytvoření závěrečného grafu
sns.distplot(sim_games, hist=True, kde=True,
             bins=range(40),
             hist_kws={'edgecolor':'black'},
             kde_kws={'bw':0.6, 'linewidth': 2})
plt.plot(np.arange(1, 40), np.diff(probs))
plt.title('člověče, nezlob se: Procento výher v kole ')
plt.xlabel('Počet kol')
plt.ylabel('Procento dokončených her')
plt.show()


