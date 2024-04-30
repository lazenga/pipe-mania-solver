# pipe.py: Template para implementação do projeto de Inteligência Artificial 2023/2024.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 20:
# 99052 André Lazenga
# 102598 Alexandre Miguel Piedade Ramos

import sys
import numpy as np
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)


class PipeManiaState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = PipeManiaState.state_id
        PipeManiaState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de PipeMania."""

    V_TL, V_TR, V_BL, V_BR = 'VC', 'VD', 'VE', 'VB'
    F_T, F_B, F_L, F_R = 'FC', 'FB', 'FE', 'FD'
    L_V, L_H = 'LV', 'LH'
    B_T, B_B, B_L, B_R = 'BC', 'BB', 'BE', 'BD'
    
    def __init__(self, matrix, size) -> None:
        self.board = matrix
        self.size = size
        self.locked = []

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.board[row, col]
    
    def set_value(self, row, col, value):
        """Modifica o valor guardado em (row, col) para value."""
        self.board[row, col] = value

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""

        return None if row == 0 else self.board[row - 1, col], \
            None if row == self.size - 1 else self.board[row + 1, col]

    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""

        return None if col == 0 else self.board[row, col - 1], \
            None if col == self.size - 1 else self.board[row, col + 1]

    def simplify_sides(self):
        """
        Simplifica os lados da board.
        """
        corners = [(0, 0), (0, self.size - 1), (self.size - 1, 0), (self.size - 1, self.size - 1)]

        for corner, (row, col) in enumerate(corners):
            if self.board[row, col][0] == 'V':
                value = self.V_BR if corner == 0 else self.V_BL \
                                    if corner == 1 else self.V_TR if corner == 2 else self.V_TL
                self.set_value(row, col, value)
                self.locked.append((row, col))
                
        for row in range(self.size):
            if row == 0 or row == self.size - 1:
                for col in range(1, self.size - 1):
                    if self.board[row, col][0] == 'L' or self.board[row, col][0] == 'B':
                        value = self.L_H if self.board[row, col][0] == 'L' else self.B_B if row == 0 else self.B_T
                        self.set_value(row, col, value)
                        self.locked.append((row, col))
            else:
                if self.board[row, 0][0] == 'L' or self.board[row, 0][0] == 'B':
                    value = self.L_V if self.board[row, 0][0] == 'L' else self.B_R
                    self.set_value(row, 0, value)
                    self.locked.append((row, 0))

                if self.board[row, self.size - 1][0] == 'L' or self.board[row, self.size - 1][0] == 'B':
                    value = self.L_V if self.board[row, self.size - 1][0] == 'L' else self.B_L
                    self.set_value(row, self.size - 1, value)
                    self.locked.append((row, self.size - 1))
    
    def deep_copy(self):
        """
        Devolve uma cópia da board.
        """
        copy = np.array([row.copy() for row in self.board])

        return Board(copy, self.size)
    
    def is_objective(self):
        """
        Verifica se a board é uma solução.
        """
        # TODO
        pass

    def __str__(self):
        return '\n'.join(['\t'.join(row) for row in self.board]) + '\n'

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.
        """
        line = sys.stdin.readline().strip().split('\t')
        
        n = len(line)
        first = True
        matrix = np.full((n, n), None)

        for i in range(n):
            if not first:
                line = sys.stdin.readline().strip().split('\t')
            
            matrix[i] = line
            first = False

        return Board(matrix, n)


class PipeMania(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        # TODO
        pass

    def actions(self, state: PipeManiaState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # TODO
        pass

    def result(self, state: PipeManiaState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        # TODO
        pass

    def goal_test(self, state: PipeManiaState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        # TODO
        pass

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    board = Board.parse_instance()

    board.simplify_sides()
    print(board)
    
    # TODO:
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    pass
