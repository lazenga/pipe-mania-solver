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
    breadth_first_tree_search,
)

class PipeManiaState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = PipeManiaState.state_id
        PipeManiaState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

class Board:
    """Representação interna de um tabuleiro de PipeMania."""

    V_TL, V_TR, V_DL, V_DR = 'VC', 'VD', 'VE', 'VB'
    F_T, F_D, F_L, F_R = 'FC', 'FB', 'FE', 'FD'
    L_V, L_H = 'LV', 'LH'
    B_T, B_D, B_L, B_R = 'BC', 'BB', 'BE', 'BD'

    LEFT = {V_TL, V_DL, F_L, L_H, B_T, B_D, B_L}
    RIGHT = {V_TR, V_DR, F_R, L_H, B_T, B_D, B_R}
    TOP = {V_TL, V_TR, F_T, L_V, B_T, B_R, B_L}
    DOWN = {V_DL, V_DR, F_D, L_V, B_D, B_R, B_L}
    
    def __init__(self, matrix, size, not_locked, valid = True, num_lock = 0, locked = set()) -> None:
        self.board = matrix
        self.size = size
        self.locked = locked
        self.not_locked = not_locked
        self.num_lock = num_lock
        self.valid = valid

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição da board."""
        return self.board[row, col] if row != None and col != None else None
    
    def set_value(self, row, col, value):
        """Modifica o valor guardado em (row, col) para value."""
        self.board[row, col] = value
        self.locked.add((row, col))
        self.not_locked.remove((row, col))
        self.num_lock += 1
    
    def adjacent_vertical_pos(self, row: int, col: int):
        """Devolve as posições adjacentes verticais."""
        return (None if row == 0 else row - 1, col), (None if row == self.size - 1 else row + 1, col)

    def adjacent_horizontal_pos(self, row: int, col: int):
        """Devolve as posições adjacentes horizontais."""
        return (row, None if col == 0 else col - 1), (row, None if col == self.size - 1 else col + 1)
    
    def is_pos_valid(self, row: int, col: int) -> bool:
        """Verifica se a posição está dentro dos limites da board."""
        return 0 <= row < self.size and 0 <= col < self.size

    def get_orientations(self, pos, adj_h, adj_v):      #TODO simplify
        """Devolve as possiveis orientações da peça."""
        row, col = pos
        curr = self.get_value(row, col)
        possible = {self.V_DL, self.V_DR, self.V_TL, self.V_TR} \
                if curr.startswith('V') else {self.B_T, self.B_D, self.B_L, self.B_R} \
                if curr.startswith('B') else {self.F_D, self.F_T, self.F_L, self.F_R} \
                if curr.startswith('F') else {self.L_V, self.L_H}
        
        if pos[0] == 0:
            possible = possible - self.TOP
        if pos[0] == self.size - 1:
            possible = possible - self.DOWN
        if pos[1] == 0:
            possible = possible - self.LEFT
        if pos[1] == self.size - 1:
            possible = possible - self.RIGHT

        if adj_h[0] in self.locked and self.board[adj_h[0][0], adj_h[0][1]] in self.RIGHT:
            if curr.startswith('V'):
                temp = {self.V_DL} if row == 0 else {self.V_TL} if row == self.size - 1 else {self.V_DL, self.V_TL}
                possible = possible.intersection(temp)

            elif curr.startswith('B'):
                possible = possible.intersection({self.B_D, self.B_L, self.B_T})

            elif curr.startswith('L'):
                possible = possible.intersection({self.L_H})
            
            elif curr.startswith('F'):
                possible = possible.intersection({self.F_L})
                if self.board[adj_h[0][0], adj_h[0][1]].startswith('F'):
                    return set()

        elif adj_h[0] in self.locked:
            if curr.startswith('V'):
                temp = {self.V_DR} if row == 0 else {self.V_TR} if row == self.size - 1 else {self.V_DR, self.V_TR}
                possible = possible.intersection(temp)

            elif curr.startswith('B'):
                possible = possible.intersection({self.B_R})

            elif curr.startswith('L'):
                possible = possible.intersection({self.L_V})
            
            elif curr.startswith('F'):
                possible = possible.intersection({self.F_T, self.F_D, self.F_R})

        if adj_h[1] in self.locked and self.board[adj_h[1][0], adj_h[1][1]] in self.LEFT:
            if curr.startswith('V'):
                temp = {self.V_DR} if row == 0 else {self.V_TR} if row == self.size - 1 else {self.V_DR, self.V_TR}
                possible = possible.intersection(temp)
            
            elif curr.startswith('B'):
                possible = possible.intersection({self.B_D, self.B_R, self.B_T})
            
            elif curr.startswith('L'):
                possible = possible.intersection({self.L_H})
            
            elif curr.startswith('F'):
                possible = possible.intersection({self.F_R})
                if self.board[adj_h[1][0], adj_h[1][1]].startswith('F'):
                    return set()

        elif adj_h[1] in self.locked:
            if curr.startswith('V'):
                temp = {self.V_DL} if row == 0 else {self.V_TL} if row == self.size - 1 else {self.V_DL, self.V_TL}
                possible = possible.intersection(temp)

            elif curr.startswith('B'):
                possible = possible.intersection({self.B_L})

            elif curr.startswith('L'):
                possible = possible.intersection({self.L_V})
            
            elif curr.startswith('F'):
                possible = possible.intersection({self.F_T, self.F_D, self.F_L})

        if adj_v[0] in self.locked and self.board[adj_v[0][0], adj_v[0][1]] in self.DOWN:
            if curr.startswith('V'):
                temp = {self.V_TR} if col == 0 else {self.V_TL} if col == self.size - 1 else {self.V_TR, self.V_TL}
                possible = possible.intersection(temp)
            
            elif curr.startswith('B'):
                possible = possible.intersection({self.B_T, self.B_R, self.B_L})
            
            elif curr.startswith('L'):
                possible = possible.intersection({self.L_V})
            
            elif curr.startswith('F'):
                possible = possible.intersection({self.F_T})
                if self.board[adj_v[0][0], adj_v[0][1]].startswith('F'):
                    return set()

        elif adj_v[0] in self.locked:
            if curr.startswith('V'):
                temp = {self.V_DR} if col == 0 else {self.V_DL} if col == self.size - 1 else {self.V_DR, self.V_DL}
                possible = possible.intersection(temp)

            elif curr.startswith('B'):
                possible = possible.intersection({self.B_D})

            elif curr.startswith('L'):
                possible = possible.intersection({self.L_H})
            
            elif curr.startswith('F'):
                possible = possible.intersection({self.F_L, self.F_D, self.F_R})

        if adj_v[1] in self.locked and self.board[adj_v[1][0], adj_v[1][1]] in self.TOP:
            if curr.startswith('V'):
                temp = {self.V_DR} if col == 0 else {self.V_DL} if col == self.size - 1 else {self.V_DR, self.V_DL}
                possible = possible.intersection(temp)
            
            elif curr.startswith('B'):
                possible = possible.intersection({self.B_D, self.B_R, self.B_L})
            
            elif curr.startswith('L'):
                possible = possible.intersection({self.L_V})
            
            elif curr.startswith('F'):
                possible = possible.intersection({self.F_D})
                if self.board[adj_v[1][0], adj_v[1][1]].startswith('F'):
                    return set()

        elif adj_v[1] in self.locked:
            if curr.startswith('V'):
                temp = {self.V_TR} if col == 0 else {self.V_TL} if col == self.size - 1 else {self.V_TR, self.V_TL}
                possible = possible.intersection(temp)

            elif curr.startswith('B'):
                possible = possible.intersection({self.B_T})

            elif curr.startswith('L'):
                possible = possible.intersection({self.L_H})
            
            elif curr.startswith('F'):
                possible = possible.intersection({self.F_T, self.F_L, self.F_R})
        
        return possible
    
    def simplify_board(self):   #TODO can be optimized to check only the neighbors
        """Simplifica a board e faz forward check."""
        modified = True
        while modified:
            modified = False
            not_locked = self.not_locked.copy()
            for row, col in not_locked:
                adj_h = self.adjacent_horizontal_pos(row, col)
                adj_v = self.adjacent_vertical_pos(row, col)
                possible = self.get_orientations((row, col), adj_h, adj_v)

                if len(possible) == 1:
                    self.set_value(row, col, possible.pop())
                    modified = True
                elif len(possible) == 0:
                    return False
        return True

    def action_finder(self):
        """Procura as próximas ações."""
        if self.num_lock == self.size ** 2:
            return []
        
        row, col = self.not_locked[0]
        adj_h = self.adjacent_horizontal_pos(row, col)
        adj_v = self.adjacent_vertical_pos(row, col)
        possible = self.get_orientations((row, col), adj_h, adj_v)

        return [(row, col, piece) for piece in possible]

    def deep_copy(self):
        """Devolve uma cópia da board."""
        copy = np.copy(self.board)

        return Board(copy, self.size, self.not_locked.copy(), self.valid, self.num_lock, self.locked.copy())
    
    def is_objective(self):
        """Verifica se a board é uma solução."""
        if self.num_lock != self.size ** 2:
            return False
        
        direction_map = {self.V_DL: {'D', 'L'}, self.V_DR: {'D', 'R'}, self.V_TL: {'T', 'L'},
                         self.V_TR: {'T', 'R'}, self.L_H: {'L', 'R'}, self.L_V: {'T', 'D'},
                         self.F_D: {'D'}, self.F_L: {'L'}, self.F_R: {'R'}, self.F_T: {'T'},
                         self.B_D: {'L', 'D', 'R'}, self.B_L: {'T', 'L', 'D'},
                         self.B_T: {'L', 'T', 'R'}, self.B_R: {'T', 'R', 'D'}}
        directions = {'T': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
        reverse_direction = {'T': 'D', 'D': 'T', 'L': 'R', 'R': 'L'}
        
        visited = [[False for _ in range(self.size)] for _ in range(self.size)]
        stack = [(0, 0)]
        visited[0][0] = True
        num_visited, valid = 0, True

        while stack:
            if not valid:
                break
            row, col = stack.pop()
            num_visited += 1
            for direction in direction_map[self.get_value(row, col)]:
                adj_row, adj_col = row + directions[direction][0], col + directions[direction][1]
                if self.is_pos_valid(adj_row, adj_col):
                    if reverse_direction[direction] in direction_map[self.get_value(adj_row, adj_col)]:
                        if not visited[adj_row][adj_col]:
                            visited[adj_row][adj_col] = True
                            stack.append((adj_row, adj_col))
                    else:
                        valid = False
                else:
                    valid = False
        
        return valid and num_visited == self.size ** 2

    def __str__(self):
        return '\n'.join(['\t'.join(row) for row in self.board])

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board."""       
        matrix = []

        for line in sys.stdin:
            pieces = line.strip().split('\t')
            matrix.append(pieces)

        matrix = np.array(matrix)
        not_locked = [(row, col) for row in range(len(matrix)) for col in range(len(matrix))]

        return Board(matrix, len(matrix), not_locked)
    
class PipeMania(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = PipeManiaState(board)

    def actions(self, state: PipeManiaState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        if not state.board.valid:
            return []
        return state.board.action_finder()

    def result(self, state: PipeManiaState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        board = state.board.deep_copy()
        new_state = PipeManiaState(board)
        new_state.board.set_value(action[0], action[1], action[2])

        return new_state

    def goal_test(self, state: PipeManiaState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        if not state.board.simplify_board():
            state.board.valid = False
            return False
        return state.board.is_objective()

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        #return node.state.board.total_con - node.state.board.num_con
    
if __name__ == "__main__":
    board = Board.parse_instance()
    challenge = PipeMania(board)
    final_node = depth_first_tree_search(challenge)
    print(final_node.state.board)
