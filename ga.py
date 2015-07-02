from random import random
import math
import expr
import fitness
from copy import copy
import textextraction as tx

def lg(x):
    return math.log(x) / math.log(2.0)

OPCODES = sorted({"add","call","cmp","mov","jnz","jmp","jz","lea","pop","push",
        "retn","sub","test","xor"})


GENE_LENGTH = int(math.ceil(lg(len(OPCODES))))


def get_binary(opcode):
    '''
    Return a binary string representing an opcode. Gene size is decided as
    minimum number of bits required to represent all the opcodes in the
    above table.

    '''
    token = OPCODES.index(opcode)
    val = bin(token)[2:]    # bin returns the binary rep of a number prefixed with a '0b' - the [2:] takes...
    val = ('0' * (GENE_LENGTH - len(val))) + val   # ...everything after the prefix.
    return val


CROSSOVER_RATE              = 0.7
MUTATION_RATE               = 0.001
POP_SIZE                    = 100

# Not used at all here, but was there in Fup's C++ code.
CHROMO_LENGTH               = 320

# Number of times to try before we give up :P
MAX_ALLOWABLE_GENERATIONS   = 400

OPERATORS = ['+', '-', '*', '/']

# ==============================================================#
# Chromosome                                                    #
# - The class that represents a chromosome                      #
# Corresponds to chromo_type in the C++ code.                   #
# ==============================================================#
class Chromosome:
    def __init__(self, bits="", fitness=0.0):
        self.bits = bits
        self.fitness = fitness

    def length(self):
        return len(self.bits)
#----------------------------------------------------------------
def crossover(offspring1, offspring2):
    '''Takes two Chromosome objects, and crosses over their parts
    as determined by the crossover rate.

    *Note: I've used the greater of the lengths of the two
    chromosomes to determine the crossover position. Fup's code
    uses CHROMO_LENGTH to do the same.
    '''
    if random() < CROSSOVER_RATE:
        cpos = int(random() * max(offspring1.length(), offspring2.length()))
        offspring1.bits, offspring2.bits = (
                                  offspring1.bits[:cpos] + offspring2.bits[cpos:],
                                  offspring2.bits[:cpos] + offspring1.bits[cpos:]
                                 )
        return True

    return False
#----------------------------------------------------------------
def mutate(chromosome):
    '''Steps through each "bit" in the chromosome, and as determined
    by the mutation rate, flips the bit.
    '''
    result = []
    ret = False
    for bit in chromosome.bits:
        prob = random()
        if prob < MUTATION_RATE:
            result.append(int(not int(bit)))
            ret = True
        else:
            result.append(int(bit))

    chromosome.bits = ''.join(map(str, result))
    return ret
#----------------------------------------------------------------
def decode(chromosome):
    '''Decode a chromosome into a sequence of opcodes.'''
    binary_opcodes = [chromosome.bits[i:i+GENE_LENGTH] for i in xrange(0, chromosome.length(), GENE_LENGTH)]
    opcode_sequence = []
    for binary_opcode in binary_opcodes:
        opcode_idx = int(binary_opcode, 2)
        if opcode_idx >= len(OPCODES):
            continue
        opcode_sequence.append(OPCODES[opcode_idx])
    return opcode_sequence
#----------------------------------------------------------------
def decode_as_str(chromosome):
    '''return the decoded chromosome as an str'''
    decoded = decode(chromosome)
    return ' '.join(map(str, decoded))
#----------------------------------------------------------------
def encode(seq):
    '''Encode an opcode sequence into a binary string'''
    encoded = []
    for opcode in seq:
        encoded.append(get_binary(opcode))
    return ''.join(encoded)
#----------------------------------------------------------------
def evaluate_fitness(chromosome, target):
    '''Evaluate an opcode sequence. Right now, seek to make the opcode sequence
       ``target`` opcodes long
    '''
    changed = False
    seq = decode(chromosome)
    uniq_opcodes = set(seq)
    n_common = float(len(uniq_opcodes.intersection(target)))
    n_total = float(len(target))
    fitness = n_common / n_total
    chromosome.fitness = fitness
    # PLAY WITH THIS VALUE 0.8
    if fitness >= 0.8:
        return False
    return True
#----------------------------------------------------------------
def roulette_select(total_fitness, population):
    fitness_slice = random() * total_fitness
    fitness_so_far = 0.0

    for phenotype in population:
        fitness_so_far += phenotype.fitness

        if fitness_so_far >= fitness_slice:
            return phenotype

    return None
#----------------------------------------------------------------
def get_random_bits(length):
    '''Return a random string of bits of given length'''
    result = ''
    for i in xrange(length):
        if random() < 0.5:
            result += '0'
        else:
            result += '1'

    return result
#----------------------------------------------------------------
def ga_main(target):
    soln_found = False
    population = []
    # Generate the initial population:
    for i in xrange(POP_SIZE):
        bits = get_random_bits(CHROMO_LENGTH)
        c = Chromosome(bits)
        population.append(c)

    gens_required = 0
    while(not soln_found):
        total_fitness = 0.0
        # Assign fitnesses
        for phenotype in population:
            if evaluate_fitness(phenotype, target) == False: # solution found
                print("***Solution found in {n} generations: {s} "
                      .format(n=gens_required, s=','.join(decode(phenotype))))
                soln_found = True
                break

            total_fitness += phenotype.fitness

        if soln_found:
            break
        # Create new population
        tmp = []
        cpop = 0
        while cpop < POP_SIZE:
            c1 = None
            while c1 == None:
                c1 = roulette_select(total_fitness, population)

            c2 = None
            while c2 == None:
                c2 = roulette_select(total_fitness, population)

            c1, c2 = copy(c1), copy(c2) # Required, as the original population will
                                        # be affected otherwise.

            crossover(c1, c2)
            mutate(c1)
            mutate(c2)

            tmp.append(c1)
            tmp.append(c2)

            cpop += 2

        population = tmp[:]

        gens_required += 1
        if gens_required > MAX_ALLOWABLE_GENERATIONS:
            print("***No solution found in this run!!")
            return False

    return True
#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    while True: # As fup says, repeat till the user gets bored!
        target = raw_input('The target file: ')
        if not target.split():
            target = '100%.asm'
        opcodes = [opcode for opcode, in tx.n_gram_opcodes(target, 1).keys()]
        ga_main(opcodes)
#---------------------------------------------------------------------------------------------------
