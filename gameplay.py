import cocos
from cocos.scene import Scene
from cocos.layer import Layer
from cocos.menu import MenuItem, Menu
from pyglet.window import key
from game.utils import Color
from cocos.euclid import Vector2

import cchess

from game.utils import *
from game.pieces import Piece


WINDOW_WIDTH = 1366
WINDOW_HEIGHT = 768


class PlayerVsAIScene(cocos.layer.Layer):
    is_event_handler = True
    def __init__(self):
        super(PlayerVsAIScene, self).__init__()
        self.add(cocos.layer.ColorLayer(0,255,0,255))

    def on_key_press(self, symbol, modifiers):
        print(symbol)
        if symbol == key.ENTER:
            cocos.director.director.pop()

class ChessBoard(cocos.layer.Layer):
    is_event_handler = True
    def __init__(self):
        super(ChessBoard, self).__init__()
        self.gap = WINDOW_HEIGHT//13
        self.board = cchess.Board()
        self.pieces:list[list[Piece]] = [[None] * 9 for _ in range(10)]  # Assuming 10x9 board
        self.alive_pieces:list[Piece] = []
        self.selected_piece = None


        self.border = 20
        self.rows = 9
        self.cols = 8
        self.width = 2*self.border + self.cols*self.gap
        self.height = 4*self.border + self.rows*self.gap
        self.position = 100,30
        self.pos0 = self.border,2*self.border

        self.set_pieces()
        self.calc_legal_moves()
        self.draw_board()
        self.draw_pieces()

    def calc_legal_moves(self):
        self.legal_moves = list(self.board.legal_moves)
        for p in self.alive_pieces:
            p.legal_moves = []
        for move in self.legal_moves:
            print(move)
            src = self.square2coord(move.from_square)
            dst = self.square2coord(move.to_square)
            self.pieces[src[0]][src[1]].legal_moves.append(dst)

        for p in self.alive_pieces:
            print(cchess.PIECE_NAMES[p.piece_type],p.legal_moves)



    @property
    def turn(self,):
        return self.board.turn
    
    def move(self,move: cchess.Move):
        # if move not in self.legal_moves:
        #     raise "Can't move"


        self.board.push(move)
        for p in self.alive_pieces:
            super().remove(p)
        self.alive_pieces = []
        self.pieces = [[None] * 9 for _ in range(10)]
        
        self.set_pieces()
        self.calc_legal_moves()
        self.draw_pieces()

        # print("move.from_square",move.from_square)
        # src = self.square2coord(move.from_square)
        # dst = self.square2coord(move.to_square)
        # if self.pieces[dst[0]][dst[1]]:
        #     self.remove(self.pieces[dst[0]][dst[1]])
        # self.pieces[dst[0]][dst[1]] = self.pieces[src[0]][src[1]]
        # self.pieces[dst[0]][dst[1]].position = self.coord2pos(dst)
        # self.pieces[src[0]][src[1]] = None
        # self.board.push(move)

        # self.calc_legal_moves()



    def remove(self, obj):
        self.alive_pieces.remove(obj)
        return super().remove(obj)


    def square2coord(self,sq):
        print("SQ",sq)
        return cchess.square_rank(sq),cchess.square_file(sq)
    
    def reset(self):
        pass

    def set_pieces(self):
        for sq,p in self.board.piece_map().items():
            r, c = cchess.square_rank(sq),cchess.square_file(sq)
            self.pieces[r][c] = Piece(p.piece_type,p.color,(r,c))
            self.pieces[r][c].position = self.coord2pos((r,c))
            self.pieces[r][c].coord = r,c
            self.alive_pieces.append(self.pieces[r][c])

    def draw_pieces(self):
        for p in self.alive_pieces:
                self.add(p)


    def coord2pos(self,coord):
        # coord: (row,col)
        # pos: (x,y)
        return self.pos0[0]+coord[1]*self.gap, self.pos0[1] + (self.rows-coord[0])*self.gap

    def pos2coord(self,pos):
        pos = pos[0]- self.pos0[0], pos[1]-self.pos0[1]
        coord = pos[1] // self.gap, pos[0] // self.gap
        return self.rows - self.coord[0], coord[1]

    def draw_board(self):
        x0, y0 = self.pos0
        rows = 9
        cols = 8
        height = rows*self.gap
        width = cols*self.gap
        for row in range(rows + 1):
            line = cocos.draw.Line((x0,y0+row*self.gap),(x0+width,y0+row*self.gap),Color.GREY,2)
            self.add(line)

            for col in range(cols + 1):
                line = cocos.draw.Line((col * self.gap + x0, y0),
                    (col * self.gap + x0, height + y0),Color.GREY,2)
                self.add(line)

        palaceCoors = [
            (
                (x0 + self.gap * 3, y0),
                (x0 + self.gap * 5, y0 + self.gap * 2),
            ),
            (
                (x0 + self.gap * 5, y0),
                (x0 + self.gap * 3, y0 + self.gap * 2),
            ),
            (
                (x0 + self.gap * 3, y0 + self.gap * 7),
                (x0 + self.gap * 5, y0 + self.gap * 9),
            ),
            (
                (x0 + self.gap * 5, y0 + self.gap * 7),
                (x0 + self.gap * 3, y0 + self.gap * 9),
            ),
        ]
        for col in range(9, 0, -1):
            x = x0+(col-1) * self.gap
            y = y0+-1.5*self.border
            label = cocos.text.Label(str(10-col), font_size=13, anchor_x='center', anchor_y='center')
            label.position = (x, y)
            self.add(label)

        for col in range(1, 10, 1):
            x = x0+(col-1) * self.gap
            y = y0+height+1.5*self.border
            label = cocos.text.Label(str(col), font_size=13, anchor_x='center', anchor_y='center')
            label.position = (x, y)
            self.add(label)

        for point1, point2 in palaceCoors:
            line = cocos.draw.Line(point1, point2, Color.GREY ,2)
            self.add(line)


        # Draw the river
        line = cocos.layer.ColorLayer(0,0,0,255,width=width-2,height=self.gap-2)
        line.position = (x0+1,y0+1+rows//2*self.gap)
        self.add(line)


        # Draw the border
        self.add(RectangleLayer((x0-self.border,y0-2*self.border),
                                self.width,
                                self.height,
                                Color.GREY,2))
        
    def coord_near_pos(self,pos):
        new = round(pos[0]//self.gap), round(pos[1]//self.gap)
        return self.rows -new[1], new[0]
        
    def is_clicked(self,pos,side):
        pos = pos[0]- self.position[0], pos[1]- self.position[1]
        print('board',pos)
        for p in self.alive_pieces:
            if p.side == side and self.is_close(p.position,pos):
                print(cchess.PIECE_NAMES[p.piece_type],p.side)
                return True
        return False
    
    def is_close(self,pos1,pos2):
        if (pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2 <= 15**2:
            return True
        else:
            return False
    
    def select_piece(self,pos):
        pos = pos[0]- self.position[0], pos[1]- self.position[1]
        coord = self.coord_near_pos(pos)
        self.selected_piece = self.pieces[coord[0]][coord[1]]

    def deselect_piece(self):
        self.selected_piece = None

    def can_move(self,pos):
        if self.selected_piece is None:
            return False
        pos = pos[0]- self.position[0], pos[1]- self.position[1]
        coord = self.coord_near_pos(pos)
        dst_pos = self.coord2pos(coord)
        print(coord)
        if self.is_close(pos,dst_pos) and coord in self.selected_piece.legal_moves:
            self.move(cchess.Move.from_uci(f'{self.selected_piece.coord[0]}{self.selected_piece.coord[1]}{coord[0]}{coord[1]}'))
            return True
        else:
            return False
        
# Add your ChessboardLayer and AIChessboardLayer implementations here...

if __name__ == "__main__":
    cocos.director.director.init(width =WINDOW_WIDTH,height = WINDOW_HEIGHT)
    cocos.director.director.run(cocos.scene.Scene(ChessBoard()))
