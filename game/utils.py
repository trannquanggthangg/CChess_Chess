import pyglet
import os

DIR = os.path.dirname(__file__)

# Width and height of the application
WIN_WIDTH = 1200
WIN_HEIGHT = 900

# Red side, blue side indicator
RED_SIDE = RED_TURN = 1
BLACK_SIDE = BLACK_TURN = 0

import cocos
from math import cos, sin, pi

class CircleLayer(cocos.layer.Layer):
    def __init__(self, center, radius, color, segments=100):
        super(CircleLayer, self).__init__()
        self.center = center
        self.radius = radius
        self.color = color
        self.segments = segments

        self.draw_circle()

    def draw_circle(self):
        delta_theta = 2 * pi / self.segments

        for i in range(self.segments):
            theta1 = i * delta_theta
            theta2 = (i + 1) * delta_theta

            x1 = self.center[0] + self.radius * cos(theta1)
            y1 = self.center[1] + self.radius * sin(theta1)

            x2 = self.center[0] + self.radius * cos(theta2)
            y2 = self.center[1] + self.radius * sin(theta2)

            line = cocos.draw.Line((x1, y1), (x2, y2), self.color)
            self.add(line)

class RectangleLayer(cocos.layer.Layer):
            def __init__(self, pos, width, height, color,stroke_width=1):
                super(RectangleLayer, self).__init__()
                self.draw_rectangle(width, height, color,stroke_width)
                self.position = pos[0],pos[1]

            def draw_rectangle(self, width, height, color,stroke_width):
                x, y = 0,0
                left_top = (x, y)
                right_top = (x + width, y)
                right_bottom = (x + width, y + height)
                left_bottom = (x, y + height)

                self.add(cocos.draw.Line(left_top, right_top, color,stroke_width))
                self.add(cocos.draw.Line(right_top, right_bottom, color,stroke_width))
                self.add(cocos.draw.Line(right_bottom, left_bottom, color,stroke_width))
                self.add(cocos.draw.Line(left_bottom, left_top, color,stroke_width))

class PieceType:
    [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
               SOLDIER, CANNON, FLAG, CHARIOT, HORSE, ELEPHANT, ADVISOR, GENERAL] = range(1, 15)

PIECE_SYMBOLS = [None, "p", "n", "b", "r", "q", "k", 
                 "s", "c", "f", "x", 'h', 'e', 'a', 'g']

PIECE_NAMES = [None, "pawn", "knight", "bishop", "rook", "queen", "king",
               "soldier", "cannon", "flag", "chariot", "horse", "elephant", "advisor","general"]

class Color:
    RED = (255, 0, 0,255)
    GREEN = (0, 255, 0,255)
    BLUE = (0, 255, 0,255)
    YELLOW = (255, 255, 0,255)
    WHITE = (255, 255, 255,255)
    BLACK = (0, 0, 0,255)
    PURPLE = (128, 0, 128,255)
    ORANGE = (255, 165, 0,255)
    GREY = (128, 128, 128,255)
    TURQUOISE = (64, 224, 208,255)


class ChessImages:
    @classmethod
    def get_piece_image(cls,piece_type,side):
        name = ("RED_" if side == RED_SIDE else "BLUE_") + PIECE_NAMES[piece_type].upper()
        return getattr(cls,name)
        
    RED_CHARIOT = pyglet.image.load(os.path.join(DIR,"../images/pieces/red-car.png"))
    RED_CANNON = pyglet.image.load(os.path.join(DIR,"../images/pieces/red-cannon.png"))
    RED_HORSE = pyglet.image.load(os.path.join(DIR,"../images/pieces/red-horse.png"))
    RED_ELEPHANT = pyglet.image.load(os.path.join(DIR,"../images/pieces/red-elephant.png"))
    RED_SOLDIER = pyglet.image.load(os.path.join(DIR,"../images/pieces/red-pawn.png"))
    RED_ADVISOR = pyglet.image.load(os.path.join(DIR,"../images/pieces/red-bodyguard.png"))
    RED_GENERAL = pyglet.image.load(os.path.join(DIR,"../images/pieces/red-king.png"))
    RED_FLAG = pyglet.image.load(os.path.join(DIR,"../images/pieces/red-flag.png"))


    BLUE_CHARIOT = pyglet.image.load(os.path.join(DIR,"../images/pieces/blue-car.png"))
    BLUE_CANNON = pyglet.image.load(os.path.join(DIR,"../images/pieces/blue-cannon.png"))
    BLUE_HORSE = pyglet.image.load(os.path.join(DIR,"../images/pieces/blue-horse.png"))
    BLUE_ELEPHANT = pyglet.image.load(os.path.join(DIR,"../images/pieces/blue-elephant.png"))
    BLUE_SOLDIER = pyglet.image.load(os.path.join(DIR,"../images/pieces/blue-pawn.png"))
    BLUE_ADVISOR = pyglet.image.load(os.path.join(DIR,"../images/pieces/blue-bodyguard.png"))
    BLUE_GENERAL = pyglet.image.load(os.path.join(DIR,"../images/pieces/blue-king.png"))
    BLUE_FLAG = pyglet.image.load(os.path.join(DIR,"../images/pieces/blue-flag.png"))

    BLUE_KING = pyglet.image.load(os.path.join(DIR,"../images/chess_pieces/bkw.png"))
    BLUE_QUEEN = pyglet.image.load(os.path.join(DIR,"../images/chess_pieces/bqw.png"))
    BLUE_KNIGHT = pyglet.image.load(os.path.join(DIR,"../images/chess_pieces/bnw.png"))
    BLUE_BISHOP = pyglet.image.load(os.path.join(DIR,"../images/chess_pieces/bbw.png"))
    BLUE_PAWN = pyglet.image.load(os.path.join(DIR,"../images/chess_pieces/bpw.png"))
    BLUE_ROOK = pyglet.image.load(os.path.join(DIR,"../images/chess_pieces/brw.png"))

    RED_KING = pyglet.image.load(os.path.join(DIR,"../images/chess_pieces/rkw.png"))
    RED_QUEEN = pyglet.image.load(os.path.join(DIR,"../images/chess_pieces/rqw.png"))
    RED_KNIGHT = pyglet.image.load(os.path.join(DIR,"../images/chess_pieces/rnw.png"))
    RED_BISHOP = pyglet.image.load(os.path.join(DIR,"../images/chess_pieces/rbw.png"))
    RED_PAWN = pyglet.image.load(os.path.join(DIR,"../images/chess_pieces/rpw.png"))
    RED_ROOK = pyglet.image.load(os.path.join(DIR,"../images/chess_pieces/rrw.png"))

    
