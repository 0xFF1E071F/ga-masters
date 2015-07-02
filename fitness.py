import math


# The placeholder value for 0 counts
epsilon = 0.0001


def opcode_llr(opcode, freq_table_before, freq_table_after):

    '''
    Args:
        opcode: A single opcode mnemonic, e.g., 'mov'

        freq_table_before: The frequency table for opcode trigrams *before*
                           extraction.

        freq_table_after: The frequency table for opcode trigrams *after*
                          extraction.

    The keys for both tables are tuples of string. So, each is of the form

        {
            ('mov', 'mov', 'mov'): 5.0,
            ('mov', 'jmp', 'mov'): 7.0,
            ...
        }

    '''
    t_b = len(freq_table_before) or epsilon
    t_a = len(freq_table_after) or epsilon

    # Compute the opcode counts when occurring in positions 0, 1, 2
    opcode_counts = [epsilon, epsilon, epsilon]
    for triplet in freq_table_after.keys():
        for i, comp in enumerate(triplet):
            if comp == opcode:
                opcode_counts[i] += 1

    f1 = opcode_counts[0]
    f2 = opcode_counts[1]
    f3 = opcode_counts[2]

    return (f1 + f2 + f3) * math.log(float(t_b) / t_a)
