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
    
    def __init__(self, matrix, size, total_con, not_locked, num_islands = 0, islands = {}, simplified = False, locked = set()) -> None:
        self.board = matrix
        self.size = size
        self.locked = locked
        self.not_locked = not_locked
        self.total_con = total_con
        self.simplified = simplified
        self.islands = islands
        self.num_islands = num_islands
        self.num_con = self.count_connected()

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.board[row, col] if row != None and col != None else None
    
    def set_value(self, row, col, value):
        """Modifica o valor guardado em (row, col) para value."""
        before = self.get_num_connections(row, col)
        self.board[row, col] = value
        self.locked.add((row, col))
        self.not_locked.remove((row, col))
        after = self.get_num_connections(row, col)
        self.num_con += (after - before) * 2
    
    def adjacent_vertical_pos(self, row: int, col: int):
        """Devolve as posições adjacentes verticais."""
        return (None if row == 0 else row - 1, col), (None if row == self.size - 1 else row + 1, col)

    def adjacent_horizontal_pos(self, row: int, col: int):
        """Devolve as posições adjacentes horizontais."""
        return (row, None if col == 0 else col - 1), (row, None if col == self.size - 1 else col + 1)

    def simplify_board(self):
        """Simplifica os lados da board."""
        corners = [(0, 0), (0, self.size - 1), (self.size - 1, 0), (self.size - 1, self.size - 1)]

        if not self.simplified:
            for corner, (row, col) in enumerate(corners):
                if self.board[row, col][0] == 'V':
                    value = self.V_DR if corner == 0 else self.V_DL \
                                        if corner == 1 else self.V_TR if corner == 2 else self.V_TL
                    self.set_value(row, col, value)
                    self.locked.add((row, col))
                    
            for row in range(self.size):
                if row == 0 or row == self.size - 1:
                    for col in range(1, self.size - 1):
                        if self.board[row, col][0] == 'L' or self.board[row, col][0] == 'B':
                            value = self.L_H if self.board[row, col][0] == 'L' else self.B_D if row == 0 else self.B_T
                            self.set_value(row, col, value)
                            self.locked.add((row, col))
                else:
                    if self.board[row, 0][0] == 'L' or self.board[row, 0][0] == 'B':
                        value = self.L_V if self.board[row, 0][0] == 'L' else self.B_R
                        self.set_value(row, 0, value)
                        self.locked.add((row, 0))

                    if self.board[row, self.size - 1][0] == 'L' or self.board[row, self.size - 1][0] == 'B':
                        value = self.L_V if self.board[row, self.size - 1][0] == 'L' else self.B_L
                        self.set_value(row, self.size - 1, value)
                        self.locked.add((row, self.size - 1))
        
        modified = True
        while modified:
            modified = False
            not_locked = list(self.not_locked)
            for row, col in not_locked:
                adj_h = self.adjacent_horizontal_pos(row, col)
                adj_v = self.adjacent_vertical_pos(row, col)
                possible = self.get_orientations((row, col), adj_h, adj_v)

                if len(possible) == 1:
                    self.set_value(row, col, possible.pop())
                    modified = True
        
        self.simplified = True
        self.not_locked = sorted(self.not_locked)

    def get_orientations(self, pos, adj_h, adj_v):      #TODO simplify
        """Devolve as possiveis orientações da peça."""
        row, col = pos
        curr = self.get_value(row, col)
        possible = {self.V_DL, self.V_DR, self.V_TL, self.V_TR} \
                if curr.startswith('V') else {self.B_T, self.B_D, self.B_L, self.B_R} \
                if curr.startswith('B') else {self.F_D, self.F_T, self.F_L, self.F_R} \
                if curr.startswith('F') else {self.L_V, self.L_H}

        if adj_h[0] in self.locked and self.board[adj_h[0][0], adj_h[0][1]] in self.RIGHT:
            if curr.startswith('V'):
                temp = {self.V_DL} if row == 0 else {self.V_TL} if row == self.size - 1 else {self.V_DL, self.V_TL}
                possible = possible.intersection(temp)

            if curr.startswith('B'):
                possible = possible.intersection({self.B_D, self.B_L, self.B_T})

            if curr.startswith('L'):
                possible = possible.intersection({self.L_H})
            
            if curr.startswith('F'):
                possible = possible.intersection({self.F_L})
                if self.board[adj_h[0][0], adj_h[0][1]].startswith('F'):
                    possible = set()

        elif adj_h[0] in self.locked:
            if curr.startswith('V'):
                temp = {self.V_DR} if row == 0 else {self.V_TR} if row == self.size - 1 else {self.V_DR, self.V_TR}
                possible = possible.intersection(temp)

            if curr.startswith('B'):
                possible = possible.intersection({self.B_R})

            if curr.startswith('L'):
                possible = possible.intersection({self.L_V})
            
            if curr.startswith('F'):
                possible = possible.intersection({self.F_T, self.F_D, self.F_R})

        if adj_h[1] in self.locked and self.board[adj_h[1][0], adj_h[1][1]] in self.LEFT:
            if curr.startswith('V'):
                temp = {self.V_DR} if row == 0 else {self.V_TR} if row == self.size - 1 else {self.V_DR, self.V_TR}
                possible = possible.intersection(temp)
            
            if curr.startswith('B'):
                possible = possible.intersection({self.B_D, self.B_R, self.B_T})
            
            if curr.startswith('L'):
                possible = possible.intersection({self.L_H})
            
            if curr.startswith('F'):
                possible = possible.intersection({self.F_R})
                if self.board[adj_h[1][0], adj_h[1][1]].startswith('F'):
                    possible = set()

        elif adj_h[1] in self.locked:
            if curr.startswith('V'):
                temp = {self.V_DL} if row == 0 else {self.V_TL} if row == self.size - 1 else {self.V_DL, self.V_TL}
                possible = possible.intersection(temp)

            if curr.startswith('B'):
                possible = possible.intersection({self.B_L})

            if curr.startswith('L'):
                possible = possible.intersection({self.L_V})
            
            if curr.startswith('F'):
                possible = possible.intersection({self.F_T, self.F_D, self.F_L})

        if adj_v[0] in self.locked and self.board[adj_v[0][0], adj_v[0][1]] in self.DOWN:
            if curr.startswith('V'):
                temp = {self.V_TR} if col == 0 else {self.V_TL} if col == self.size - 1 else {self.V_TR, self.V_TL}
                possible = possible.intersection(temp)
            
            if curr.startswith('B'):
                possible = possible.intersection({self.B_T, self.B_R, self.B_L})
            
            if curr.startswith('L'):
                possible = possible.intersection({self.L_V})
            
            if curr.startswith('F'):
                possible = possible.intersection({self.F_T})
                if self.board[adj_v[0][0], adj_v[0][1]].startswith('F'):
                    possible = set()

        elif adj_v[0] in self.locked:
            if curr.startswith('V'):
                temp = {self.V_DR} if col == 0 else {self.V_DL} if col == self.size - 1 else {self.V_DR, self.V_DL}
                possible = possible.intersection(temp)

            if curr.startswith('B'):
                possible = possible.intersection({self.B_D})

            if curr.startswith('L'):
                possible = possible.intersection({self.L_H})
            
            if curr.startswith('F'):
                possible = possible.intersection({self.F_L, self.F_D, self.F_R})

        if adj_v[1] in self.locked and self.board[adj_v[1][0], adj_v[1][1]] in self.TOP:
            if curr.startswith('V'):
                temp = {self.V_DR} if col == 0 else {self.V_DL} if col == self.size - 1 else {self.V_DR, self.V_DL}
                possible = possible.intersection(temp)
            
            if curr.startswith('B'):
                possible = possible.intersection({self.B_D, self.B_R, self.B_L})
            
            if curr.startswith('L'):
                possible = possible.intersection({self.L_V})
            
            if curr.startswith('F'):
                possible = possible.intersection({self.F_D})
                if self.board[adj_v[1][0], adj_v[1][1]].startswith('F'):
                    possible = set()

        elif adj_v[1] in self.locked:
            if curr.startswith('V'):
                temp = {self.V_TR} if col == 0 else {self.V_TL} if col == self.size - 1 else {self.V_TR, self.V_TL}
                possible = possible.intersection(temp)

            if curr.startswith('B'):
                possible = possible.intersection({self.B_T})

            if curr.startswith('L'):
                possible = possible.intersection({self.L_H})
            
            if curr.startswith('F'):
                possible = possible.intersection({self.F_T, self.F_L, self.F_R})
        
        return possible

    def action_finder(self):
        """Procura as próximas ações."""
        if len(self.not_locked) == 0:
            return []
        
        row, col = self.not_locked[0]

        adj_h = self.adjacent_horizontal_pos(row, col)
        adj_v = self.adjacent_vertical_pos(row, col)
            
        possible = self.get_orientations((row, col), adj_h, adj_v)

        return [(row, col, piece) for piece in possible]
    
    def get_island(self, pos):
        """Devolve a ilha a que a peça pertence ou None caso não pertença a nenhuma."""
        for island in self.islands:
            if pos in self.islands[island]:
                return island
    
        return None
    
    def count_islands(self):
        """Conta o número de ilhas."""
        id = 0
        self.num_islands = 0
        self.islands = {}

        for row in range(self.size):
            for col in range(self.size):
                islands = set()
                pos = (row, col)
                curr = self.get_value(row, col)
                adj_h = self.adjacent_horizontal_pos(row, col)
                adj_v = self.adjacent_vertical_pos(row, col)
                adj_h_i = tuple([self.get_island(temp) for temp in adj_h])
                adj_v_i = tuple([self.get_island(temp) for temp in adj_v])
                adj_h_v = tuple([self.get_value(temp[0], temp[1]) for temp in adj_h])
                adj_v_v = tuple([self.get_value(temp[0], temp[1]) for temp in adj_v])
                
                if curr in self.LEFT and adj_h_v[0] != None and adj_h_v[0] in self.RIGHT:
                    if adj_h_i[0] == None:
                        islands.add(id)
                        self.islands[id] = {pos, adj_h[0]}
                        id += 1
                        self.num_islands += 1
                    else:
                        islands.add(adj_h_i[0])

                if curr in self.TOP and adj_v_v[0] != None and adj_v_v[0] in self.DOWN:
                    if adj_v_i[0] == None:
                        islands.add(id)
                        self.islands[id] = {pos, adj_v[0]}
                        id += 1
                        self.num_islands += 1
                    else:
                        islands.add(adj_v_i[0])

                if curr in self.DOWN and adj_v_v[1] != None and adj_v_v[1] in self.TOP:
                    if adj_v_i[1] == None:
                        islands.add(id)
                        self.islands[id] = {pos, adj_v[1]}
                        id += 1
                        self.num_islands += 1
                    else:
                        islands.add(adj_v_i[1])

                if curr in self.RIGHT and adj_h_v[1] != None and adj_h_v[1] in self.LEFT:
                    if adj_h_i[1] == None:
                        islands.add(id)
                        self.islands[id] = {pos, adj_h[1]}
                        id += 1
                        self.num_islands += 1
                    else:
                        islands.add(adj_h_i[1])

                if len(islands) == 1:
                    self.islands[islands.pop()].add(pos)

                elif len(islands) == 0 and pos in self.locked:
                    self.islands[id] = {pos}
                    id += 1
                    self.num_islands += 1

                else:
                    self.merge_islands(pos, islands)

    def merge_islands(self, pos, islands):
        """Dá merge a duas ou mais ilhas."""
        new_id = min(islands)
        islands.remove(new_id)

        for island in islands:
            self.islands[new_id].update(self.islands[island])
            del self.islands[island]
        
        self.islands[new_id].add(pos)
        self.num_islands -= len(islands)
    
    def get_num_connections(self, row, col) -> int:
        """Devolve o número de entradas conectadas da peça dada."""
        curr = self.get_value(row, col)
        adj_h = tuple([self.get_value(pos[0], pos[1]) for pos in self.adjacent_horizontal_pos(row, col)])
        adj_v = tuple([self.get_value(pos[0], pos[1]) for pos in self.adjacent_vertical_pos(row, col)])

        count = ((1 if curr in self.LEFT and adj_h[0] in self.RIGHT else 0) + \
            (1 if curr in self.RIGHT and adj_h[1] in self.LEFT else 0) + \
            (1 if curr in self.TOP and adj_v[0] in self.DOWN else 0) + \
            (1 if curr in self.DOWN and adj_v[1] in self.TOP else 0))
    
        return count

    def count_connected(self) -> int:
        """Devolve o total de entradas conectadas."""        
        return sum([self.get_num_connections(row, col) for row in range(self.size) for col in range(self.size)])

    def deep_copy(self):
        """Devolve uma cópia da board."""
        copy = np.array([row.copy() for row in self.board])

        return Board(copy, self.size, self.total_con, self.not_locked.copy(), self.num_islands, self.islands.copy(), \
                     self.simplified, self.locked.copy())
    
    def is_objective(self):     #TODO make it more efficient 
        """Verifica se a board é uma solução."""
        if (self.total_con - self.num_con) == 0:
            self.count_islands()

        return (self.total_con - self.num_con) == 0 and self.num_islands == 1

    def __str__(self):
        return '\n'.join(['\t'.join(row) for row in self.board])

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board."""       
        matrix = []
        total_con = 0

        for line in sys.stdin:
            pieces = line.strip().split('\t')
            total_con += sum([3 if piece.startswith('B') else 1 if piece.startswith('F') else 2 for piece in pieces])
            matrix.append(pieces)

        matrix = np.array(matrix)
        not_locked = set([(row, col) for row in range(len(matrix)) for col in range(len(matrix))])

        return Board(matrix, len(matrix), total_con, not_locked)


class PipeMania(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = PipeManiaState(board)

    def actions(self, state: PipeManiaState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
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
        state.board.simplify_board()

        return state.board.is_objective()

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        return node.state.board.total_con - node.state.board.num_con
    
if __name__ == "__main__":
    board = Board.parse_instance()
    challenge = PipeMania(board)
    final_node = depth_first_tree_search(challenge)
    print(final_node.state.board)
