import math
import time
from copy import deepcopy
import random

import cchess


class GameController(object):
    """Game Controller.

    A generic interface that all game controllers must abide by so that they
    can be put on trial against one another. Each game is conducted between
    a pair of concrete GameController instances representing the two players.
    """

    def get_next_move(self, state:cchess.Board):
        assert not state.is_game_over()


class RandomGameController(GameController):
    """Random Game Controller.

    This game controller will play any game by simply selecting moves at random.
    It serves as a benchmark for the performance of the MCTSGameController.
    """

    def get_next_move(self, state:cchess.Board):
        super(RandomGameController, self).get_next_move(state)
        return random.choice(list(state.legal_moves))


class MCTSNode(object):
    """Monte Carlo Tree Node.

    Each node encapsulates a particular game state, the moves that
    are possible from that state and the strategic information accumulated
    by the tree search as it progressively samples the game space.

    """

    def __init__(self, state:cchess.Board, parent=None, move=None):
        self.parent = parent
        self.move = move
        self.state = state

        self.plays = 0
        self.score = 0

        self.pending_moves = list(state.legal_moves)
        self.children = []

    def select_child_ucb(self,c=1.42):
        # Note that each node's plays count is equal
        # to the sum of its children's plays

        # def win_ucb(child:MCTSNode):
        #     win_ratio = child.score / child.plays \
        #         + math.sqrt(2 * math.log(self.plays) / child.plays)
        #     return win_ratio
    
        if self.state.turn == cchess.RED:
            current_palyer = 1
        else:
            current_palyer = -1
        def ucb(child:MCTSNode):
            win_ratio = current_palyer*(child.score / child.plays) \
                + c*math.sqrt(math.log(self.plays) / child.plays)
            return win_ratio
        
        # ucb = win_ucb if self.state.turn == cchess.RED else loss_ucb

        return max(self.children, key=ucb)

    def expand_move(self, move):
        self.pending_moves.remove(move) # raises KeyError

        child_state = self.state.copy()
        child_state.push(move)

        child = MCTSNode(state=child_state, parent=self, move=move)
        self.children.append(child)
        return child

    def get_score(self, result):
        if result == 0.5:
            return 0

        if result == cchess.RED:
            return 1
        else:
            return -1

    def __repr__(self):
        s = 'ROOT\n' if self.parent is None else ''

        children_moves = [c.move for c in self.children]

        s += """Score ratio: {score} / {plays}
Pending moves: {pending_moves}
Children's moves: {children_moves}
State:
{state}\n""".format(children_moves=children_moves, **self.__dict__)

        return s


class MCTSGameController(GameController):
    """Game controller that uses MCTS to determine the next move.

    This is the class which implements the Monte Carlo Tree Search algorithm.
    It builds a game tree of MCTSNodes and samples the game space until a set
    time has elapsed.

    """

    def select(self):
        node = self.root_node

        # Descend until we find a node that has pending moves, or is terminal
        while node.pending_moves == [] and node.children != []:
            node = node.select_child_ucb()

        return node
    
    def expand(self, node:MCTSNode):
        assert node.pending_moves != []

        move = random.choice(node.pending_moves)
        return node.expand_move(move)

    def simulate(self, state:cchess.Board, max_iterations=5000):
        state = state.copy()

        # move = random.choice(list(state.legal_moves))
        while not state.is_game_over():
            move = random.choice(list(state.legal_moves))
            state.push(move)
            
            max_iterations -= 1
            if max_iterations <= 0:
                return 0.5 # raise exception? (game too deep to simulate)

        result = state.result()
        if result == '1/2-1/2':
            result = 0.5
        elif result == '1-0':
            result = 1
        else:
            result = 0
        return result
        
    def update(self, node:MCTSNode, result):
        while node is not None:
            node.plays += 1
            node.score += node.get_score(result)
            # print(node.state.turn,result)
            node = node.parent

    def get_next_move(self, state:cchess.Board, time_allowed=2.0,iters = -1):
        super(MCTSGameController, self).get_next_move(state)

        # Create new tree (TODO: Preserve some state for better performance?)
        self.root_node = MCTSNode(state)
        iterations = 0

        start_time = time.time()
        while time.time() < start_time + time_allowed:
            node = self.select()

            if node.pending_moves != []:
                node = self.expand(node)

            result = self.simulate(node.state)
            self.update(node, result)

            iterations += 1
            if iterations == iters:
                break

        # Return most visited node's move
        # # func = max if self.root_node.state.turn == cchess.RED else min
        # s = [(n.score,n.plays) for n in self.root_node.children]
        # print(s)
        # best_child = max(self.root_node.children, key=lambda n:n.plays)
        # # best_child = self.root_node.select_child_ucb()
        # print(1-best_child.score/best_child.plays,best_child.plays,best_child.parent.plays)
        # return best_child.move

        best_child = self.root_node.select_child_ucb(0)
        print(best_child.score,'/',best_child.plays,'=',best_child.score/best_child.plays)
        return best_child.move
    
